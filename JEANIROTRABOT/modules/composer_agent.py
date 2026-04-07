"""
JEANIROTRABOT - Composer Agent (MiroFish ReportAgent-style)
Mengumpulkan sinyal dari semua specialist agents dan membuat keputusan final.

Hybrid mode:
  - Consensus (>=3-1 dengan score kuat): keputusan langsung, tanpa LLM
  - Conflict (2-2 atau 3-1 lemah):       LLM call sekali untuk resolve
  - Fallback (LLM error/unavailable):    majority vote
  - Safe (>=3 agents error):             HOLD tanpa eksekusi
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from modules.specialist_agents import AgentSignal

logger = logging.getLogger("JEANIROTRABOT.composer")

CONFLICT_SCORE_THRESHOLD = 0.30   # avg score minimum untuk consensus tanpa LLM
LLM_DEFAULT_TIMEOUT = 10.0        # detik


# ─────────────────────────────────────────────
# Output Contract
# ─────────────────────────────────────────────

@dataclass
class FinalDecision:
    """Output ComposerAgent yang diteruskan ke TradingEngine untuk eksekusi."""
    action: str           # "BUY" | "SELL" | "HOLD"
    confidence: float
    score: float
    reason: str
    mode: str             # "CONSENSUS" | "CONFLICT_LLM" | "MAJORITY_FALLBACK" | "SAFE_HOLD"
    vote_summary: str     # e.g. "3 BUY, 1 HOLD"
    sl_points: int = 0
    tp_points: int = 0
    lot_size: float = 0.0
    ticket: int = 0
    new_sl: float = 0.0
    new_tp: float = 0.0


# ─────────────────────────────────────────────
# ComposerAgent
# ─────────────────────────────────────────────

class ComposerAgent:
    """
    Menggabungkan sinyal specialist agents -> keputusan trading final.
    Dipanggil dengan ai_agent=None untuk consensus-only mode.
    """

    def __init__(self, llm_timeout: float = LLM_DEFAULT_TIMEOUT):
        self._llm_timeout = llm_timeout

    def decide(
        self,
        signals: list[AgentSignal],
        market_context: dict,
        ai_agent=None,
    ) -> FinalDecision:
        """
        Entrypoint utama. Evaluasi sinyal -> consensus atau conflict -> FinalDecision.
        market_context: dict yang sama yang dikirim ke AIAgent.analyze_multi()
        ai_agent: instance AIAgent (atau None untuk no-LLM mode)
        """
        valid   = [s for s in signals if not s.error]
        errored = [s for s in signals if s.error]

        vote_summary = self._build_vote_summary(signals)

        # Safe mode: terlalu banyak agent error
        if len(errored) >= 3:
            logger.warning(f"[Composer] SAFE_HOLD — {len(errored)} agents errored")
            return FinalDecision(
                action="HOLD", confidence=0.0, score=0.0,
                reason=f"Safe mode: {len(errored)}/4 agents errored",
                mode="SAFE_HOLD", vote_summary=vote_summary,
            )

        if not valid:
            return FinalDecision(
                action="HOLD", confidence=0.0, score=0.0,
                reason="No valid agent signals",
                mode="SAFE_HOLD", vote_summary="",
            )

        buy_sigs  = [s for s in valid if s.action == "BUY"]
        sell_sigs = [s for s in valid if s.action == "SELL"]
        hold_sigs = [s for s in valid if s.action == "HOLD"]
        n = len(valid)

        # ── Consensus check ──
        # 4-0 atau 3-1 dengan score kuat
        if len(buy_sigs) >= 3:
            avg_score = sum(abs(s.score) for s in buy_sigs) / len(buy_sigs)
            if len(buy_sigs) == n or avg_score >= CONFLICT_SCORE_THRESHOLD:
                return self._build_consensus("BUY", buy_sigs, vote_summary)

        if len(sell_sigs) >= 3:
            avg_score = sum(abs(s.score) for s in sell_sigs) / len(sell_sigs)
            if len(sell_sigs) == n or avg_score >= CONFLICT_SCORE_THRESHOLD:
                return self._build_consensus("SELL", sell_sigs, vote_summary)

        if len(hold_sigs) >= 3:
            return self._build_consensus("HOLD", hold_sigs, vote_summary)

        # ── Conflict — gunakan LLM jika tersedia ──
        if ai_agent is not None and ai_agent.enabled:
            logger.info(f"[Composer] CONFLICT ({vote_summary}) -> calling LLM")
            return self._llm_decide(signals, market_context, ai_agent, vote_summary)

        # ── Fallback: majority vote tanpa LLM ──
        logger.info(f"[Composer] CONFLICT ({vote_summary}) -> majority fallback (no LLM)")
        return self._majority_fallback(valid, vote_summary)

    # ── Internal helpers ──

    def _build_consensus(
        self, action: str, agreeing: list[AgentSignal], vote_summary: str
    ) -> FinalDecision:
        avg_score = sum(s.score for s in agreeing) / len(agreeing)
        avg_conf  = sum(s.confidence for s in agreeing) / len(agreeing)
        top_reasons = [r for s in agreeing for r in s.reasons][:4]
        reason_str = "; ".join(top_reasons)
        logger.info(f"[Composer] CONSENSUS {action} ({vote_summary}) score={avg_score:+.2f}")
        return FinalDecision(
            action=action,
            confidence=round(avg_conf, 3),
            score=round(avg_score, 3),
            reason=f"Consensus ({vote_summary}): {reason_str}",
            mode="CONSENSUS",
            vote_summary=vote_summary,
        )

    def _majority_fallback(
        self, valid: list[AgentSignal], vote_summary: str
    ) -> FinalDecision:
        buy_sigs  = [s for s in valid if s.action == "BUY"]
        sell_sigs = [s for s in valid if s.action == "SELL"]

        if len(buy_sigs) > len(sell_sigs):
            majority = buy_sigs
            action = "BUY"
        elif len(sell_sigs) > len(buy_sigs):
            majority = sell_sigs
            action = "SELL"
        else:
            return FinalDecision(
                action="HOLD", confidence=0.30, score=0.0,
                reason=f"True tie — HOLD ({vote_summary})",
                mode="MAJORITY_FALLBACK", vote_summary=vote_summary,
            )

        avg_score = sum(s.score for s in majority) / len(majority)
        avg_conf  = sum(s.confidence for s in majority) / len(majority) * 0.80
        return FinalDecision(
            action=action,
            confidence=round(avg_conf, 3),
            score=round(avg_score, 3),
            reason=f"Majority fallback ({vote_summary})",
            mode="MAJORITY_FALLBACK",
            vote_summary=vote_summary,
        )

    def _llm_decide(
        self,
        signals: list[AgentSignal],
        market_context: dict,
        ai_agent,
        vote_summary: str,
    ) -> FinalDecision:
        agent_lines = "\n".join(
            f"  {s.agent_name}: {s.action} score={s.score:+.2f} conf={s.confidence:.2f}"
            f" — {'; '.join(s.reasons[:2])}"
            for s in signals if not s.error
        )
        conflict_summary = (
            f"=== AGENT SIGNALS (CONFLICT: {vote_summary}) ===\n"
            f"{agent_lines}\n"
            f"The specialist agents disagree. Resolve this conflict and decide."
        )

        try:
            result = ai_agent.analyze_multi(
                conflict_summary=conflict_summary,
                **{k: v for k, v in market_context.items()},
            )
            action     = result.get("action", "HOLD")
            confidence = float(result.get("confidence", 0.5))
            reason     = result.get("reason", "LLM resolved")

            logger.info(f"[Composer] LLM resolved: {action} conf={confidence:.2f}")
            return FinalDecision(
                action=action,
                confidence=confidence,
                score=confidence if action == "BUY" else (-confidence if action == "SELL" else 0.0),
                reason=f"LLM resolved conflict ({vote_summary}): {reason}",
                mode="CONFLICT_LLM",
                vote_summary=vote_summary,
                sl_points=result.get("sl_points", 0),
                tp_points=result.get("tp_points", 0),
                lot_size=result.get("lot_size", 0.0),
                ticket=result.get("ticket", 0),
                new_sl=result.get("new_sl", 0.0),
                new_tp=result.get("new_tp", 0.0),
            )
        except Exception as e:
            logger.warning(f"[Composer] LLM error: {e} — fallback to majority vote")
            return self._majority_fallback([s for s in signals if not s.error], vote_summary)

    @staticmethod
    def _build_vote_summary(signals: list[AgentSignal]) -> str:
        buy  = sum(1 for s in signals if s.action == "BUY")
        sell = sum(1 for s in signals if s.action == "SELL")
        hold = sum(1 for s in signals if s.action == "HOLD")
        return f"{buy} BUY, {sell} SELL, {hold} HOLD"
