"""
JEANIROTRABOT - AI Composer Agent (Meta-Orchestrator)
Agent tertinggi yang mengendalikan semua agent lain.
Memiliki self-adaptive commander prompt yang berubah sesuai kondisi pasar.
Target: Capai profit 70% dari modal awal.
"""

import logging
import json
import time
from datetime import datetime
from typing import Optional, Callable

logger = logging.getLogger("JEANIROTRABOT.composer")

# Market modes yang bisa dipilih Composer
MARKET_MODES = {
    "AGGRESSIVE": {
        "description": "Pasar trending kuat — ambil peluang maksimal",
        "confidence_threshold": 0.55,
        "lot_multiplier": 1.3,
        "tp_multiplier": 1.5,
        "color": "#ff6600",
    },
    "CONSERVATIVE": {
        "description": "Pasar sideways/choppy — kurangi risiko",
        "confidence_threshold": 0.62,  # Fix D: 0.72 → 0.62
        "lot_multiplier": 0.7,
        "tp_multiplier": 0.8,
        "color": "#00aaff",
    },
    "NEWS_DRIVEN": {
        "description": "Volatilitas tinggi akibat berita — tunggu kejelasan",
        "confidence_threshold": 0.65,  # Fix D: 0.75 → 0.65
        "lot_multiplier": 0.5,
        "tp_multiplier": 1.0,
        "color": "#ffaa00",
    },
    "CAPITAL_PRESERVE": {
        "description": "Mendekati target 70% — jaga modal",
        "confidence_threshold": 0.68,  # Fix D: 0.80 → 0.68
        "lot_multiplier": 0.3,
        "tp_multiplier": 0.7,
        "color": "#00ff88",
    },
    "RECOVERY": {
        "description": "Drawdown tinggi — hanya setup sangat kuat",
        "confidence_threshold": 0.65,  # Fix D: 0.78 → 0.65
        "lot_multiplier": 0.25,
        "tp_multiplier": 1.2,
        "color": "#ff4444",
    },
    "STANDBY": {
        "description": "Tidak ada peluang — scan pasar baru",
        "confidence_threshold": 0.62,  # Fix D: 0.70 → 0.62
        "lot_multiplier": 0.3,          # Fix D: 0.0 → 0.3 (tidak blokir total)
        "tp_multiplier": 1.0,
        "color": "#888888",
    },
}

# Template commander prompt dasar (Composer akan memodifikasi ini)
COMPOSER_BASE_PROMPT = """Kamu adalah AI Composer — orchestrator utama sistem trading JEANIROTRABOT.
Kamu mengendalikan semua agent trading dan bertanggung jawab penuh atas hasil trading.

TUJUAN UTAMA: Capai profit {profit_target:.0f}% dari modal awal {initial_balance:.2f} {currency}.
Progress saat ini: {current_profit_pct:.1f}% dari target {profit_target:.0f}% (Profit: {current_profit:.2f}).

=== STATUS PASAR SAAT INI ===
{market_summary}

=== HASIL NEWS AGENT ===
{news_summary}

=== HASIL RESEARCH AGENT ===
{research_summary}

=== RIWAYAT TRADING HARI INI ===
Profit/Loss hari ini: {daily_pnl:.2f} {currency}
Jumlah trade terbuka: {open_positions}
Trade dilakukan hari ini: {trade_count}
Win rate hari ini: {win_rate:.0f}%
Drawdown saat ini: {drawdown_pct:.1f}%

=== TUGAS COMPOSER ===
Berdasarkan SEMUA data di atas, tentukan:
1. market_mode yang paling tepat ({modes})
2. trading_focus: simbol mana yang paling menjanjikan (maksimal 3)
3. adjusted_confidence: threshold confidence yang sesuai kondisi
4. adjusted_lot_multiplier: pengali lot size (0.0 = no trade, 2.0 = double)
5. dynamic_system_prompt: instruksi spesifik untuk trading AI berdasarkan kondisi ini
6. agent_directives: arahkan news_agent dan research_agent
7. reasoning: jelaskan logika keputusanmu

PENTING:
- Jika profit_pct >= 60% dari target, WAJIB gunakan CAPITAL_PRESERVE
- Jika drawdown >= 3%, WAJIB gunakan RECOVERY
- Jika ada berita high-impact, pertimbangkan NEWS_DRIVEN
- Jika tidak ada peluang sama sekali, gunakan STANDBY dan trigger SCAN_MARKET

Respond dalam JSON ONLY:
{{
  "market_mode": "{mode_options}",
  "trading_focus": ["SYM1", "SYM2"],
  "adjusted_confidence": 0.60-0.90,
  "adjusted_lot_multiplier": 0.0-2.0,
  "dynamic_system_prompt": "Instruksi khusus untuk trading AI...",
  "agent_directives": {{
    "news_agent": "run_now|skip",
    "research_agent": "run_now|skip"
  }},
  "reasoning": "Penjelasan singkat keputusan Composer"
}}"""


class ComposerAgent:
    """
    Meta-Orchestrator AI yang mengendalikan semua agent lain.
    Self-adaptive commander prompt berubah sesuai kondisi pasar.
    """

    def __init__(self, ai_agent, config):
        self._ai = ai_agent
        self._config = config
        self._last_run: float = 0
        self._interval_minutes: int = config.get_int("COMPOSER_INTERVAL", 5)  # Fix E: 10 → 5
        self._profit_target: float = config.get_float("PROFIT_TARGET_PERCENT", 70.0)
        self._initial_balance: float = config.get_float("INITIAL_BALANCE_SNAPSHOT", 0.0)

        # Current state
        self._market_mode: str = "CONSERVATIVE"
        self._trading_focus: list = []
        self._confidence_threshold: float = 0.65
        self._lot_multiplier: float = 1.0
        self._dynamic_prompt: str = ""
        self._agent_directives: dict = {"news_agent": "skip", "research_agent": "skip"}
        self._last_reasoning: str = ""

        # Callbacks
        self._log_callback: Optional[Callable[[str], None]] = None
        self._on_mode_change: Optional[Callable[[str, dict], None]] = None

        # Stats tracking
        self._trade_count_today: int = 0
        self._wins_today: int = 0

    def set_log_callback(self, cb: Callable[[str], None]):
        self._log_callback = cb

    def set_mode_change_callback(self, cb: Callable[[str, dict], None]):
        self._on_mode_change = cb

    def _log(self, msg: str):
        logger.info(msg)
        if self._log_callback:
            try:
                self._log_callback(f"[COMPOSER] {msg}")
            except Exception:
                pass

    def set_initial_balance(self, balance: float):
        if self._initial_balance <= 0:
            self._initial_balance = balance
            self._config.set("INITIAL_BALANCE_SNAPSHOT", str(balance))
            self._log(f"Modal awal dicatat: {balance:.2f}")

    def record_trade(self, won: bool):
        self._trade_count_today += 1
        if won:
            self._wins_today += 1

    def reset_daily_stats(self):
        self._trade_count_today = 0
        self._wins_today = 0

    def is_due(self) -> bool:
        elapsed = time.time() - self._last_run
        return elapsed >= (self._interval_minutes * 60)

    @property
    def market_mode(self) -> str:
        return self._market_mode

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @property
    def lot_multiplier(self) -> float:
        return self._lot_multiplier

    @property
    def dynamic_prompt(self) -> str:
        return self._dynamic_prompt

    @property
    def agent_directives(self) -> dict:
        return self._agent_directives

    def get_mode_params(self) -> dict:
        return MARKET_MODES.get(self._market_mode, MARKET_MODES["CONSERVATIVE"])

    def run(self, account_info, market_summary: str = "",
            news_summary: str = "", research_summary: str = "",
            open_positions: int = 0, daily_pnl: float = 0.0) -> dict:
        """
        Jalankan Composer cycle. Analisis kondisi pasar dan update parameter semua agent.
        Returns: directive dict untuk trading engine.
        """
        if not self._ai or not self._ai.enabled:
            return self._default_directive()

        if self._initial_balance <= 0 and account_info:
            self.set_initial_balance(account_info.balance)

        # Hitung metrics
        current_balance = account_info.balance if account_info else 0.0
        current_profit = current_balance - self._initial_balance if self._initial_balance > 0 else 0.0
        current_profit_pct = (current_profit / self._initial_balance * 100) if self._initial_balance > 0 else 0.0
        currency = account_info.currency if account_info else "USD"
        drawdown_pct = 0.0
        if account_info and self._initial_balance > 0:
            drawdown_pct = max(0, (self._initial_balance - account_info.equity) / self._initial_balance * 100)

        win_rate = (self._wins_today / self._trade_count_today * 100) if self._trade_count_today > 0 else 0.0

        mode_options = "|".join(MARKET_MODES.keys())

        # Build composer prompt
        prompt = COMPOSER_BASE_PROMPT.format(
            profit_target=self._profit_target,
            initial_balance=self._initial_balance if self._initial_balance > 0 else current_balance,
            currency=currency,
            current_profit_pct=current_profit_pct,
            current_profit=current_profit,
            market_summary=market_summary or "Tidak ada data pasar.",
            news_summary=news_summary or "Tidak ada berita.",
            research_summary=research_summary or "Tidak ada research.",
            daily_pnl=daily_pnl,
            open_positions=open_positions,
            trade_count=self._trade_count_today,
            win_rate=win_rate,
            drawdown_pct=drawdown_pct,
            modes=mode_options,
            mode_options=mode_options,
        )

        try:
            result = self._ai.analyze_raw(prompt)
            if isinstance(result, dict):
                self._apply_directive(result, current_profit_pct, drawdown_pct)
                self._last_run = time.time()
                return result
        except Exception as e:
            logger.error(f"Composer run error: {e}")

        return self._default_directive()

    def _apply_directive(self, directive: dict, profit_pct: float, drawdown_pct: float):
        """Terapkan directive dari Composer ke state internal."""
        old_mode = self._market_mode

        # Override paksa berdasarkan kondisi kritis
        if profit_pct >= self._profit_target * 0.85:
            directive["market_mode"] = "CAPITAL_PRESERVE"
        elif drawdown_pct >= 3.0:
            directive["market_mode"] = "RECOVERY"

        mode = directive.get("market_mode", "CONSERVATIVE").upper()
        if mode not in MARKET_MODES:
            mode = "CONSERVATIVE"

        self._market_mode = mode
        mode_params = MARKET_MODES[mode]

        # Ambil dari directive atau gunakan defaults mode
        self._confidence_threshold = directive.get(
            "adjusted_confidence",
            mode_params["confidence_threshold"]
        )
        self._lot_multiplier = directive.get(
            "adjusted_lot_multiplier",
            mode_params["lot_multiplier"]
        )
        self._dynamic_prompt = directive.get("dynamic_system_prompt", "")
        self._agent_directives = directive.get(
            "agent_directives",
            {"news_agent": "skip", "research_agent": "skip"}
        )
        self._last_reasoning = directive.get("reasoning", "")
        trading_focus = directive.get("trading_focus", [])
        if trading_focus:
            self._trading_focus = trading_focus

        if mode != old_mode:
            self._log(f"Mode berubah: {old_mode} → {mode} | {mode_params['description']}")
            if self._on_mode_change:
                self._on_mode_change(mode, mode_params)

        self._log(
            f"Mode={mode} | Conf={self._confidence_threshold:.2f} | "
            f"LotMult={self._lot_multiplier:.2f} | Focus={self._trading_focus} | "
            f"Reasoning: {self._last_reasoning[:80]}"
        )

    def _default_directive(self) -> dict:
        """Directive default jika Composer tidak bisa berjalan."""
        return {
            "market_mode": "CONSERVATIVE",
            "trading_focus": [],
            "adjusted_confidence": 0.65,
            "adjusted_lot_multiplier": 1.0,
            "dynamic_system_prompt": "",
            "agent_directives": {"news_agent": "skip", "research_agent": "skip"},
            "reasoning": "Composer tidak aktif — menggunakan default.",
        }

    def get_status_text(self) -> str:
        mode_params = MARKET_MODES.get(self._market_mode, {})
        return (
            f"Mode: {self._market_mode} | {mode_params.get('description', '')}\n"
            f"Confidence min: {self._confidence_threshold:.0%} | "
            f"Lot multiplier: {self._lot_multiplier:.1f}x\n"
            f"Fokus: {', '.join(self._trading_focus) if self._trading_focus else 'semua simbol'}\n"
            f"Alasan: {self._last_reasoning[:120]}"
        )

    def get_trading_focus(self) -> list[str]:
        return self._trading_focus

    def get_last_reasoning(self) -> str:
        return self._last_reasoning
