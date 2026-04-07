"""
JEANIROTRABOT - Specialist Agent Pool (MiroFish OASIS-style)
4 specialist agents dengan kepribadian trading berbeda.
Setiap agent menganalisis indikator spesifik → AgentSignal.
SpecialistPool menjalankan semua agent dan mengumpulkan sinyal.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("JEANIROTRABOT.specialists")


# ─────────────────────────────────────────────
# Data Contract
# ─────────────────────────────────────────────

@dataclass
class AgentSignal:
    """Output standar setiap specialist agent."""
    agent_name: str
    action: str           # "BUY" | "SELL" | "HOLD"
    score: float          # -1.0 (kuat SELL) → +1.0 (kuat BUY)
    confidence: float     # 0.0 → 1.0
    reasons: list[str] = field(default_factory=list)
    error: bool = False
    error_msg: str = ""


# ─────────────────────────────────────────────
# Base Class
# ─────────────────────────────────────────────

class BaseSpecialistAgent:
    """Interface seragam untuk semua specialist agent."""
    NAME = "BaseAgent"

    def analyze(
        self,
        indicators: dict,
        tick: Optional[dict] = None,
        symbol_info: Optional[dict] = None,
    ) -> AgentSignal:
        raise NotImplementedError


# ─────────────────────────────────────────────
# TrendAgent — SMA / EMA crossover
# ─────────────────────────────────────────────

class TrendAgent(BaseSpecialistAgent):
    """
    Menganalisis arah tren melalui alignment EMA9 / SMA20 / SMA50.
    BUY  jika EMA9 > SMA20 > SMA50 (full uptrend alignment).
    SELL jika EMA9 < SMA20 < SMA50 (full downtrend alignment).
    Bonus score untuk golden/death cross yang baru terjadi.
    """
    NAME = "TrendAgent"

    def analyze(self, indicators: dict, tick=None, symbol_info=None) -> AgentSignal:
        try:
            sma20      = indicators.get("sma20")
            sma50      = indicators.get("sma50")
            ema9       = indicators.get("ema9")
            sma20_prev = indicators.get("sma20_prev")
            sma50_prev = indicators.get("sma50_prev")

            if any(v is None for v in [sma20, sma50, ema9, sma20_prev, sma50_prev]):
                return AgentSignal(self.NAME, "HOLD", 0.0, 0.0,
                                   ["Insufficient data for trend analysis"],
                                   error=True, error_msg="None indicators")

            score = 0.0
            confidence = 0.3
            reasons = []

            # Full alignment check
            if ema9 > sma20 > sma50:
                score = 0.65
                confidence = 0.72
                action = "BUY"
                reasons.append(f"EMA9({ema9:.5f})>SMA20({sma20:.5f})>SMA50({sma50:.5f}) uptrend")
            elif ema9 < sma20 < sma50:
                score = -0.65
                confidence = 0.72
                action = "SELL"
                reasons.append(f"EMA9({ema9:.5f})<SMA20({sma20:.5f})<SMA50({sma50:.5f}) downtrend")
            else:
                score = 0.0
                confidence = 0.28
                action = "HOLD"
                reasons.append("No clear EMA/SMA alignment")

            # Golden cross bonus: sma20 baru saja cross above sma50
            # Hanya berlaku jika arah dasarnya BUY atau HOLD (tidak override SELL)
            if sma20_prev < sma50_prev and sma20 > sma50:
                score = min(1.0, score + 0.20)
                confidence = min(1.0, confidence + 0.10)
                reasons.append("Golden cross detected")
                if action == "HOLD":
                    action = "BUY"
            # Death cross bonus: sma20 baru saja cross below sma50
            # Hanya berlaku jika arah dasarnya SELL atau HOLD (tidak override BUY)
            elif sma20_prev > sma50_prev and sma20 < sma50:
                score = max(-1.0, score - 0.20)
                confidence = min(1.0, confidence + 0.10)
                reasons.append("Death cross detected")
                if action == "HOLD":
                    action = "SELL"

            score = max(-1.0, min(1.0, score))
            confidence = max(0.0, min(1.0, confidence))

            logger.debug(f"[TrendAgent] {action} score={score:+.2f} conf={confidence:.2f}")
            return AgentSignal(self.NAME, action, score, confidence, reasons)

        except Exception as e:
            logger.warning(f"TrendAgent error: {e}")
            return AgentSignal(self.NAME, "HOLD", 0.0, 0.0,
                               [f"Error: {e}"], error=True, error_msg=str(e))


# ─────────────────────────────────────────────
# MomentumAgent — RSI / MACD / Stochastic
# ─────────────────────────────────────────────

class MomentumAgent(BaseSpecialistAgent):
    """
    Menganalisis kecepatan & kekuatan pergerakan harga.
    Indikator: RSI(14) bobot 30%, MACD(12,26,9) bobot 40%, Stochastic(14,3) bobot 30%.
    BUY jika RSI oversold + MACD positif + Stoch oversold-cross.
    SELL jika RSI overbought + MACD negatif + Stoch overbought-cross.
    """
    NAME = "MomentumAgent"

    def analyze(self, indicators: dict, tick=None, symbol_info=None) -> AgentSignal:
        try:
            rsi            = indicators.get("rsi")
            macd_hist      = indicators.get("macd_hist")
            macd_hist_prev = indicators.get("macd_hist_prev")
            macd           = indicators.get("macd")
            stoch_k        = indicators.get("stoch_k")
            stoch_d        = indicators.get("stoch_d")
            stoch_k_prev   = indicators.get("stoch_k_prev")
            stoch_d_prev   = indicators.get("stoch_d_prev")

            if any(v is None for v in [rsi, macd_hist, stoch_k, stoch_d]):
                return AgentSignal(self.NAME, "HOLD", 0.0, 0.0,
                                   ["Insufficient momentum data"],
                                   error=True, error_msg="None indicators")

            reasons = []

            # ── RSI signal (30% weight) ──
            if rsi < 40:
                rsi_score = (40 - rsi) / 40.0
                reasons.append(f"RSI={rsi:.1f} oversold")
            elif rsi > 60:
                rsi_score = -((rsi - 60) / 40.0)
                reasons.append(f"RSI={rsi:.1f} overbought")
            else:
                rsi_score = 0.0
                reasons.append(f"RSI={rsi:.1f} neutral")

            # ── MACD signal (40% weight) ──
            macd_ref = abs(macd) if macd and macd != 0 else 1e-5
            if macd_hist > 0:
                macd_score = min(1.0, macd_hist / macd_ref)
                reasons.append("MACD histogram positive")
                if macd_hist_prev is not None and macd_hist_prev <= 0:
                    macd_score = min(1.0, macd_score + 0.30)
                    reasons.append("MACD bullish crossover")
            elif macd_hist < 0:
                macd_score = max(-1.0, -min(1.0, abs(macd_hist) / macd_ref))
                reasons.append("MACD histogram negative")
                if macd_hist_prev is not None and macd_hist_prev >= 0:
                    macd_score = max(-1.0, macd_score - 0.30)
                    reasons.append("MACD bearish crossover")
            else:
                macd_score = 0.0

            # ── Stochastic signal (30% weight) ──
            if stoch_k < 20:
                stoch_score = 0.60
                reasons.append(f"Stoch %K={stoch_k:.1f} oversold")
                if stoch_k_prev is not None and stoch_d_prev is not None:
                    if stoch_k_prev <= stoch_d_prev and stoch_k > stoch_d:
                        stoch_score = 0.90
                        reasons.append("Stoch bullish cross")
            elif stoch_k > 80:
                stoch_score = -0.60
                reasons.append(f"Stoch %K={stoch_k:.1f} overbought")
                if stoch_k_prev is not None and stoch_d_prev is not None:
                    if stoch_k_prev >= stoch_d_prev and stoch_k < stoch_d:
                        stoch_score = -0.90
                        reasons.append("Stoch bearish cross")
            else:
                stoch_score = 0.0

            # ── Weighted final score ──
            final_score = (rsi_score * 0.30) + (macd_score * 0.40) + (stoch_score * 0.30)
            final_score = max(-1.0, min(1.0, final_score))
            confidence = min(1.0, abs(final_score) + 0.30)

            if final_score >= 0.20:
                action = "BUY"
            elif final_score <= -0.20:
                action = "SELL"
            else:
                action = "HOLD"

            logger.debug(f"[MomentumAgent] {action} score={final_score:+.2f}")
            return AgentSignal(self.NAME, action, final_score, confidence, reasons)

        except Exception as e:
            logger.warning(f"MomentumAgent error: {e}")
            return AgentSignal(self.NAME, "HOLD", 0.0, 0.0,
                               [f"Error: {e}"], error=True, error_msg=str(e))


# ─────────────────────────────────────────────
# VolatilityAgent — Bollinger Bands / ATR
# ─────────────────────────────────────────────

class VolatilityAgent(BaseSpecialistAgent):
    """
    Menganalisis kondisi pasar dan potensi breakout via Bollinger Bands + ATR.
    BUY  jika harga dekat lower band (price_position ≤ 0.15) dengan BB menyempit.
    SELL jika harga dekat upper band (price_position ≥ 0.85).
    ATR filter: pasar terlalu flat → turunkan confidence 50%.
    """
    NAME = "VolatilityAgent"

    def analyze(self, indicators: dict, tick=None, symbol_info=None) -> AgentSignal:
        try:
            bb_upper      = indicators.get("bb_upper")
            bb_lower      = indicators.get("bb_lower")
            bb_width      = indicators.get("bb_width")
            bb_width_prev = indicators.get("bb_width_prev")
            atr           = indicators.get("atr")
            atr_avg       = indicators.get("atr_avg")
            close         = indicators.get("close")

            if any(v is None for v in [bb_upper, bb_lower, bb_width, close]):
                return AgentSignal(self.NAME, "HOLD", 0.0, 0.0,
                                   ["Insufficient volatility data"],
                                   error=True, error_msg="None indicators")

            reasons = []
            bb_range = bb_upper - bb_lower

            if bb_range <= 0:
                return AgentSignal(self.NAME, "HOLD", 0.0, 0.25,
                                   ["BB range is zero"])

            price_position = (close - bb_lower) / bb_range

            if price_position <= 0.15:
                score = 0.60
                confidence = 0.65
                action = "BUY"
                reasons.append(f"Price near BB lower ({price_position:.2f}) — potential bounce")
            elif price_position >= 0.85:
                score = -0.60
                confidence = 0.65
                action = "SELL"
                reasons.append(f"Price near BB upper ({price_position:.2f}) — potential reversal")
            else:
                score = 0.0
                confidence = 0.30
                action = "HOLD"
                reasons.append(f"Price in BB middle zone ({price_position:.2f})")

            # BB squeeze bonus
            if bb_width_prev is not None and action != "HOLD":
                if bb_width < bb_width_prev:
                    score = max(-1.0, min(1.0, score * 1.15))
                    confidence = min(1.0, confidence + 0.08)
                    reasons.append("BB squeezing")

            # ATR filter: pasar terlalu flat
            if atr is not None and atr_avg is not None and atr_avg > 0:
                if atr < 0.30 * atr_avg:
                    confidence *= 0.50
                    reasons.append(f"Low ATR ({atr:.5f} < 30% avg) — flat market")

            score = max(-1.0, min(1.0, score))
            confidence = max(0.0, min(1.0, confidence))

            logger.debug(f"[VolatilityAgent] {action} score={score:+.2f}")
            return AgentSignal(self.NAME, action, score, confidence, reasons)

        except Exception as e:
            logger.warning(f"VolatilityAgent error: {e}")
            return AgentSignal(self.NAME, "HOLD", 0.0, 0.0,
                               [f"Error: {e}"], error=True, error_msg=str(e))


# ─────────────────────────────────────────────
# VolumeAgent — OBV / Volume SMA
# ─────────────────────────────────────────────

class VolumeAgent(BaseSpecialistAgent):
    """
    Mengkonfirmasi sinyal harga dengan analisis volume.
    OBV rising + harga rising → konfirmasi bullish.
    OBV falling + harga falling → konfirmasi bearish.
    Divergence OBV vs harga → sinyal lemah.
    Volume spike → boost confidence; volume lesu → kurangi confidence.
    """
    NAME = "VolumeAgent"

    def analyze(self, indicators: dict, tick=None, symbol_info=None) -> AgentSignal:
        try:
            obv        = indicators.get("obv")
            obv_prev   = indicators.get("obv_prev")
            volume     = indicators.get("volume")
            volume_sma = indicators.get("volume_sma")
            close      = indicators.get("close")
            close_prev = indicators.get("close_prev")

            if any(v is None for v in [obv, obv_prev, close, close_prev]):
                return AgentSignal(self.NAME, "HOLD", 0.0, 0.0,
                                   ["Insufficient volume data"],
                                   error=True, error_msg="None indicators")

            reasons = []
            obv_rising   = obv > obv_prev
            price_rising = close > close_prev

            if obv_rising and price_rising:
                score = 0.60
                confidence = 0.70
                action = "BUY"
                reasons.append("OBV rising with price — bullish confirmation")
            elif not obv_rising and not price_rising:
                score = -0.60
                confidence = 0.70
                action = "SELL"
                reasons.append("OBV falling with price — bearish confirmation")
            elif obv_rising and not price_rising:
                score = 0.20
                confidence = 0.40
                action = "BUY"
                reasons.append("OBV rising, price falling — bullish divergence")
            else:
                score = -0.20
                confidence = 0.35
                action = "HOLD"
                reasons.append("OBV falling, price rising — bearish divergence warning")

            # Volume spike / lesu adjustment
            if volume is not None and volume_sma is not None and volume_sma > 0:
                if volume > 2.0 * volume_sma:
                    confidence = min(1.0, confidence + 0.10)
                    reasons.append(f"Volume spike ({volume:.0f} > 2× SMA)")
                elif volume < 0.5 * volume_sma:
                    confidence = max(0.0, confidence - 0.15)
                    reasons.append(f"Low volume ({volume:.0f} < 0.5× SMA)")

            score = max(-1.0, min(1.0, score))
            confidence = max(0.0, min(1.0, confidence))

            logger.debug(f"[VolumeAgent] {action} score={score:+.2f}")
            return AgentSignal(self.NAME, action, score, confidence, reasons)

        except Exception as e:
            logger.warning(f"VolumeAgent error: {e}")
            return AgentSignal(self.NAME, "HOLD", 0.0, 0.0,
                               [f"Error: {e}"], error=True, error_msg=str(e))


# ─────────────────────────────────────────────
# Specialist Pool
# ─────────────────────────────────────────────

class SpecialistPool:
    """
    Menjalankan semua specialist agents secara sekuensial.
    Agent yang error return HOLD — tidak menghentikan agent lain.
    """

    def __init__(self):
        self._agents: list[BaseSpecialistAgent] = [
            TrendAgent(),
            MomentumAgent(),
            VolatilityAgent(),
            VolumeAgent(),
        ]

    def analyze(
        self,
        indicators: dict,
        tick: Optional[dict] = None,
        symbol_info: Optional[dict] = None,
    ) -> list[AgentSignal]:
        """Return list AgentSignal dari semua 4 agent."""
        signals = []
        for agent in self._agents:
            try:
                signal = agent.analyze(indicators, tick, symbol_info)
            except Exception as e:
                logger.error(f"Uncaught error in {agent.NAME}: {e}")
                signal = AgentSignal(agent.NAME, "HOLD", 0.0, 0.0,
                                     [f"Uncaught: {e}"], error=True, error_msg=str(e))
            signals.append(signal)
        return signals
