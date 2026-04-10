"""
JEANIROTRABOT - Trading Engine Module (Full Autonomous)
Orchestrates AI decisions, risk checks, and order execution.
Supports 13 actions — all without user confirmation.
Composer Agent mengendalikan semua agent lain secara adaptif.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd

from modules.mt5_connector import MT5Connector, OrderResult
from modules.risk_manager import RiskManager
from modules.ai_agent import AIAgent
from modules.specialist_agents import SpecialistPool, AgentSignal
from modules.composer_agent import ComposerAgent, FinalDecision

logger = logging.getLogger("JEANIROTRABOT.engine")


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rsi = pd.Series(np.nan, index=series.index)
    valid = avg_gain.notna() & avg_loss.notna()
    both_zero = valid & (avg_gain == 0) & (avg_loss == 0)
    loss_zero = valid & (avg_loss == 0) & (avg_gain > 0)
    normal = valid & (avg_loss > 0)
    rsi[both_zero] = 50.0
    rsi[loss_zero] = 100.0
    rsi[normal] = 100.0 - (100.0 / (1.0 + avg_gain[normal] / avg_loss[normal]))
    return rsi


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, Signal line, Histogram."""
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K and %D."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    stoch_k = 100.0 * (close - lowest_low) / denom
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Upper band, Mid (SMA), Lower band, Width."""
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    width = upper - lower
    return upper, mid, lower, width


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


def _safe_last(series: pd.Series) -> Optional[float]:
    """Ambil nilai terakhir series, return None jika NaN."""
    val = series.iloc[-1]
    return float(val) if not pd.isna(val) else None


def _safe_prev(series: pd.Series, n: int = 1) -> Optional[float]:
    """Ambil nilai sebelumnya, return None jika NaN atau index OOB."""
    if len(series) <= n:
        return None
    val = series.iloc[-1 - n]
    return float(val) if not pd.isna(val) else None


def compute_all_indicators(df: pd.DataFrame) -> dict:
    """
    Hitung semua indikator dari DataFrame OHLCV.
    Return dict dengan semua nilai terakhir (float atau None jika NaN).
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    sma20 = compute_sma(close, 20)
    sma50 = compute_sma(close, 50)
    ema9 = compute_ema(close, 9)
    rsi = compute_rsi(close, 14)
    macd, macd_signal, macd_hist = compute_macd(close)
    stoch_k, stoch_d = compute_stochastic(high, low, close)
    bb_upper, bb_mid, bb_lower, bb_width = compute_bollinger_bands(close)
    atr = compute_atr(high, low, close)
    atr_ma = atr.rolling(window=20).mean()
    obv = compute_obv(close, volume)
    volume_sma = volume.rolling(window=20).mean()

    return {
        "sma20":          _safe_last(sma20),
        "sma50":          _safe_last(sma50),
        "ema9":           _safe_last(ema9),
        "rsi":            _safe_last(rsi),
        "macd":           _safe_last(macd),
        "macd_signal":    _safe_last(macd_signal),
        "macd_hist":      _safe_last(macd_hist),
        "macd_hist_prev": _safe_prev(macd_hist),
        "stoch_k":        _safe_last(stoch_k),
        "stoch_d":        _safe_last(stoch_d),
        "stoch_k_prev":   _safe_prev(stoch_k),
        "stoch_d_prev":   _safe_prev(stoch_d),
        "bb_upper":       _safe_last(bb_upper),
        "bb_mid":         _safe_last(bb_mid),
        "bb_lower":       _safe_last(bb_lower),
        "bb_width":       _safe_last(bb_width),
        "bb_width_prev":  _safe_prev(bb_width),
        "atr":            _safe_last(atr),
        "atr_avg":        _safe_last(atr_ma),
        "obv":            _safe_last(obv),
        "obv_prev":       _safe_prev(obv),
        "volume":         _safe_last(volume),
        "volume_sma":     _safe_last(volume_sma),
        "close":          _safe_last(close),
        "close_prev":     _safe_prev(close),
        "sma20_prev":     _safe_prev(sma20),
        "sma50_prev":     _safe_prev(sma50),
        "ema9_prev":      _safe_prev(ema9),
    }


class TradingEngine:
    """Full autonomous trading: AI decides, engine executes without confirmation.
    Composer Agent mengorkestrasi semua agent lain secara adaptif."""

    def __init__(
        self,
        connector,  # MT5Connector atau ExchangeConnector
        risk_manager: RiskManager,
        ai_agent: AIAgent,
        config,
    ):
        self.connector = connector
        self.risk = risk_manager
        self.ai = ai_agent
        self.config = config

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._symbols: list[str] = ["EURUSD"]
        self._timeframe = "M5"
        self._lot_size = 0.01
        self._log_callback: Optional[Callable[[str], None]] = None
        self._interval  = config.get_int("TRADE_INTERVAL", 2)    # detik (dari 5)
        self._cache_ttl = config.get_int("DECISION_CACHE_TTL", 30)  # detik
        self._decision_cache: dict = {}  # {symbol: (FinalDecision, timestamp)}

        # Self-configure state
        self._session_end_time: Optional[str] = config.get("AI_SESSION_END_TIME", "") or None
        self._trading_ended_today: bool = False
        self._last_date: str = ""
        self._daily_pnl: float = 0.0
        self._trade_count_today: int = 0

        # MiroFish-style specialist pool + MiroFish ComposerAgent
        self._pool = SpecialistPool()
        self._mirofish_composer = ComposerAgent(
            llm_timeout=config.get_float("LLM_CONFLICT_TIMEOUT", 5.0)
        )

        # Optional orchestration agents (diinisialisasi dari GUI atau di sini)
        self._composer = None   # MarketModeComposer (meta-orchestrator, HEAD)
        self._news_agent = None
        self._research_agent = None

        # Composer state cache (for MarketModeComposer)
        self._composer_confidence_threshold: float = 0.6
        self._composer_lot_multiplier: float = 1.0
        self._composer_dynamic_prompt: str = ""
        self._composer_last_run: float = 0
        self._composer_interval: int = config.get_int("COMPOSER_INTERVAL", 10) * 60  # detik

        # News/Research intervals
        self._news_last_run: float = 0
        self._news_interval: int = config.get_int("NEWS_AGENT_INTERVAL", 30) * 60
        self._research_last_run: float = 0
        self._research_interval: int = config.get_int("RESEARCH_AGENT_INTERVAL", 60) * 60

        # Cached summaries pentru Composer
        self._news_summary: str = ""
        self._research_summary: str = ""
        self._market_summary: str = ""

        # Symbol cache pentru background scan (Fix I)
        self._available_symbols_cache: list = []
        self._agent_threads: list = []

        # MiroFish signals/decision tracking
        self._last_signals: list[AgentSignal] = []
        self._last_decision: Optional[FinalDecision] = None
        self._signal_callback: Optional[Callable[[list, object], None]] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def set_symbols(self, symbols: list[str]):
        self._symbols = symbols if symbols else ["EURUSD"]

    def set_symbol(self, symbol: str):
        """Backward compatible — sets single symbol."""
        self._symbols = [symbol]

    def set_timeframe(self, timeframe: str):
        self._timeframe = timeframe

    def set_lot_size(self, lot: float):
        self._lot_size = lot

    def _get_cached_decision(self, symbol: str):
        """Return cached FinalDecision jika masih valid, else None."""
        entry = self._decision_cache.get(symbol)
        if entry is None:
            return None
        decision, ts = entry
        if time.time() - ts > self._cache_ttl:
            del self._decision_cache[symbol]
            return None
        return decision

    def _cache_decision(self, symbol: str, decision) -> None:
        """Simpan decision ke cache dengan timestamp sekarang."""
        self._decision_cache[symbol] = (decision, time.time())

    def set_log_callback(self, cb: Callable[[str], None]):
        self._log_callback = cb

    def set_composer(self, composer):
        self._composer = composer
        if composer:
            composer.set_log_callback(self._log_callback)

    def set_news_agent(self, news_agent):
        self._news_agent = news_agent

    def set_research_agent(self, research_agent):
        self._research_agent = research_agent

    def set_signal_callback(self, cb: Callable[[list, object], None]):
        """Callback dipanggil setiap cycle dengan (signals, decision) untuk GUI update."""
        self._signal_callback = cb

    def _log(self, msg: str):
        logger.info(msg)
        if self._log_callback:
            try:
                self._log_callback(msg)
            except Exception:
                pass

    def _log_agent_signals(self, symbol: str, signals: list[AgentSignal], decision: FinalDecision):
        """Log sinyal semua agent + keputusan composer dan update GUI."""
        for sig in signals:
            status = "ERROR" if sig.error else sig.action
            self._log(
                f"[{sig.agent_name}] {status} score={sig.score:+.2f} "
                f"conf={sig.confidence:.2f} — {'; '.join(sig.reasons[:2])}"
            )
        self._log(
            f"[COMPOSER] {decision.mode} -> {decision.action} "
            f"score={decision.score:+.2f} conf={decision.confidence:.2f} "
            f"({decision.vote_summary})"
        )
        self._last_signals = signals
        self._last_decision = decision
        if self._signal_callback:
            try:
                self._signal_callback(signals, decision)
            except Exception:
                pass

    def _execute_decision(
        self,
        decision: FinalDecision,
        symbol: str,
        symbol_info: Optional[dict],
        total_positions: int,
        symbol_pos_count: int,
    ):
        """Eksekusi FinalDecision dari ComposerAgent — mapping ke MT5 orders."""
        action = decision.action
        confidence = decision.confidence

        if action == "HOLD":
            return

        if action in ("BUY", "SELL"):
            if confidence < 0.6:
                self._log(f"Skipping {action}: confidence {confidence:.2f} < 0.6")
                return

            ok, msg = self.risk.can_open_position(total_positions, symbol_pos_count)
            if not ok:
                self._log(f"Skipping {action}: {msg}")
                return

            if decision.lot_size > 0:
                lot = self.risk.validate_lot_size(decision.lot_size)
            elif symbol_info:
                sl_pts = decision.sl_points if decision.sl_points > 0 else self.risk.default_sl_points
                lot = self.risk.calculate_lot_size(
                    equity=0,
                    sl_points=sl_pts,
                    tick_value=symbol_info.get("trade_tick_value", 1),
                    tick_size=symbol_info.get("trade_tick_size", 1),
                )
            else:
                lot = self.risk.validate_lot_size(self._lot_size)

            sl = decision.sl_points if decision.sl_points > 0 else self.risk.default_sl_points
            tp = decision.tp_points if decision.tp_points > 0 else self.risk.default_tp_points

            self._log(f"AUTO-EXECUTE [{decision.mode}]: {action} {lot} {symbol} SL={sl} TP={tp}")
            order_result = self.connector.send_market_order(
                symbol=symbol, order_type=action.lower(),
                volume=lot, sl_points=sl, tp_points=tp,
            )
            if order_result.success:
                self.risk.record_order()
                self._log(f"Order filled: ticket={order_result.ticket} @ {order_result.price}")
            else:
                self._log(f"Order failed: {order_result.comment}")

        elif action == "CLOSE":
            ticket = decision.ticket
            if not ticket:
                self._log("CLOSE action missing ticket.")
                return
            self._log(f"AUTO-CLOSE: ticket #{ticket}")
            close_result = self.connector.close_position(ticket)
            if close_result.success:
                self._log(f"Closed #{ticket} @ {close_result.price}")
            else:
                self._log(f"Close failed: {close_result.comment}")

        elif action == "CLOSE_ALL":
            self._log(f"AUTO-CLOSE_ALL: {symbol}")
            results = self.connector.close_positions_by_symbol(symbol)
            for r in results:
                status = f"Closed #{r.ticket} @ {r.price}" if r.success else f"Failed: {r.comment}"
                self._log(f"  {status}")

        elif action == "MODIFY_SL_TP":
            ticket = decision.ticket
            if not ticket:
                self._log("MODIFY_SL_TP missing ticket.")
                return
            self._log(f"AUTO-MODIFY: #{ticket} SL={decision.new_sl} TP={decision.new_tp}")
            mod_result = self.connector.modify_position_sl_tp(ticket, decision.new_sl, decision.new_tp)
            if mod_result.success:
                self._log(f"Position #{ticket} SL/TP modified.")
            else:
                self._log(f"Modify failed: {mod_result.comment}")

    def start(self) -> tuple[bool, str]:
        if self.running:
            return False, "Engine already running."
        if not self.connector.connected:
            return False, "Not connected to MT5/Exchange."

        account = self.connector.get_account_info()
        if account:
            self.risk.set_initial_equity(account.equity)
            # Catat modal awal ke Composer
            if self._composer:
                self._composer.set_initial_balance(account.balance)

        # Reset daily state saat start
        self._trading_ended_today = False
        self._last_date = datetime.now().strftime("%Y-%m-%d")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Start background agent threads (Fix A)
        self._start_background_agents()

        # Proactive startup symbol discovery (Fix O)
        threading.Thread(target=self._startup_symbol_discovery, daemon=True).start()

        symbols_str = ", ".join(self._symbols)
        self._log(f"Trading engine started: [{symbols_str}] {self._timeframe}")
        return True, "Engine started."

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._thread = None
        self._log("Trading engine stopped.")

    def emergency_stop(self):
        self.stop()
        self._log("EMERGENCY STOP – closing all positions...")
        results = self.connector.close_all_positions()
        for r in results:
            if r.success:
                self._log(f"  Closed ticket {r.ticket} @ {r.price}")
            else:
                self._log(f"  Failed to close: {r.comment}")
        self.risk.deactivate("Emergency stop triggered.")
        self._log("Emergency stop complete.")

    def _run_loop(self):
        consecutive_errors = 0
        while not self._stop_event.is_set():
            try:
                # Reset daily state di hari baru
                today = datetime.now().strftime("%Y-%m-%d")
                if today != self._last_date:
                    self._reset_daily_state()
                    self._last_date = today

                # Cek batas waktu sesi yang diset AI
                if self._session_end_time:
                    now_str = datetime.now().strftime("%H:%M")
                    if now_str >= self._session_end_time:
                        self._log(f"Sesi trading berakhir (AI set {self._session_end_time}). Menunggu hari berikutnya.")
                        self._trading_ended_today = True
                        self._session_end_time = None

                # Skip jika trading sudah diakhiri hari ini
                if self._trading_ended_today:
                    if self._stop_event.wait(timeout=60):
                        break
                    continue

                # Composer/News/Research berjalan di background threads (Fix A)
                self._cycle()
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                self._log(f"Engine error: {e}")
                logger.exception("Trading cycle error")
                backoff = min(self._interval * (2 ** consecutive_errors), 60)
                if self._stop_event.wait(timeout=backoff):
                    break
                continue
            if self._stop_event.wait(timeout=self._interval):
                break

    # ── Background agent threads (Fix A) ──

    def _start_background_agents(self):
        """Start Composer/News/Research sebagai background daemon threads."""
        self._agent_threads = []
        for target in [self._composer_loop, self._news_loop, self._research_loop]:
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._agent_threads.append(t)

    def _composer_loop(self):
        """Background loop untuk Composer Agent (Fix A + Fix P)."""
        # Refresh symbol cache saat start (Fix J)
        threading.Thread(target=self._refresh_symbols_cache, daemon=True).start()
        # Force run pertama langsung (Fix P)
        self._composer_last_run = 0
        self._stop_event.wait(timeout=3)
        while not self._stop_event.is_set():
            try:
                self._run_composer_if_due()
            except Exception as e:
                logger.error(f"Composer bg error: {e}")
            self._stop_event.wait(timeout=30)

    def _news_loop(self):
        """Background loop untuk News Agent."""
        while not self._stop_event.is_set():
            try:
                self._run_news_agent_if_due()
            except Exception as e:
                logger.error(f"News bg error: {e}")
            self._stop_event.wait(timeout=60)

    def _research_loop(self):
        """Background loop untuk Research Agent."""
        while not self._stop_event.is_set():
            try:
                self._run_research_agent_if_due()
            except Exception as e:
                logger.error(f"Research bg error: {e}")
            self._stop_event.wait(timeout=120)

    # ── Periodic agent runners ──

    def _run_composer_if_due(self):
        if not self._composer or not self.ai.enabled:
            return
        if not self._composer.is_due():
            return
        try:
            account = self.connector.get_account_info()
            all_positions = self.connector.get_positions()
            open_count = len(all_positions)

            directive = self._composer.run(
                account_info=account,
                market_summary=self._market_summary,
                news_summary=self._news_summary,
                research_summary=self._research_summary,
                open_positions=open_count,
                daily_pnl=self._daily_pnl,
            )

            # Terapkan directive Composer ke engine
            self._composer_confidence_threshold = directive.get("adjusted_confidence", 0.65)
            self._composer_lot_multiplier = directive.get("adjusted_lot_multiplier", 1.0)
            self._composer_dynamic_prompt = directive.get("dynamic_system_prompt", "")

            # Update simbol fokus jika ada
            focus = directive.get("trading_focus", [])
            if focus:
                self._log(f"Composer fokus: {', '.join(focus)}")

            # Jalankan agent sesuai directive
            agent_dirs = directive.get("agent_directives", {})
            if agent_dirs.get("news_agent") == "run_now":
                self._run_news_agent_now()
            if agent_dirs.get("research_agent") == "run_now":
                self._run_research_agent_now()

        except Exception as e:
            logger.error(f"Composer run error: {e}")

    def _run_news_agent_if_due(self):
        if not self._news_agent:
            return
        if not self.config.get("NEWS_AGENT_ENABLED", "false").lower() == "true":
            return
        elapsed = time.time() - self._news_last_run
        if elapsed >= self._news_interval:
            self._run_news_agent_now()

    def _run_news_agent_now(self):
        if not self._news_agent:
            return
        try:
            news = self._news_agent.get_news(self._symbols, max_items=15)
            if news:
                analysis = self._news_agent.analyze_news_with_ai(news, self.ai)
                self._news_summary = analysis.get("summary", "")
                self._log(
                    f"News Agent: sentimen={analysis.get('sentiment','?')} "
                    f"signal={analysis.get('signal','?')} | {self._news_summary[:80]}"
                )
            self._news_last_run = time.time()
        except Exception as e:
            logger.error(f"News agent error: {e}")

    def _run_research_agent_if_due(self):
        if not self._research_agent:
            return
        if not self.config.get("RESEARCH_AGENT_ENABLED", "false").lower() == "true":
            return
        elapsed = time.time() - self._research_last_run
        if elapsed >= self._research_interval:
            self._run_research_agent_now()

    def _run_research_agent_now(self):
        if not self._research_agent:
            return
        try:
            report = self._research_agent.run_full_research(self._symbols[:5], self.ai)
            self._research_summary = self._research_agent.get_market_summary_text()
            self._log(f"Research Agent selesai: {self._research_summary[:100]}")
            self._research_last_run = time.time()
        except Exception as e:
            logger.error(f"Research agent error: {e}")

    # ── Self-configure helpers ──

    def _reset_daily_state(self):
        """Reset state harian saat hari berganti."""
        self._trading_ended_today = False
        self._session_end_time = None
        self._daily_pnl = 0.0
        self._trade_count_today = 0
        if self._composer:
            self._composer.reset_daily_stats()
        self._log("State harian direset untuk hari baru.")

    def _get_scannable_symbols(self, max_symbols: int = 20) -> list[str]:
        """Ambil daftar simbol dari cache (Fix I) — tidak blocking."""
        if self._available_symbols_cache:
            return self._available_symbols_cache[:max_symbols]
        return self._symbols

    def _refresh_symbols_cache(self):
        """Jalan di background thread — pre-cache daftar simbol tersedia (Fix I)."""
        try:
            all_syms = self.connector.get_symbols()
            if all_syms:
                self._available_symbols_cache = all_syms[:50]
                self._log(f"Symbol cache: {len(self._available_symbols_cache)} simbol tersedia")
        except Exception as e:
            logger.debug(f"Symbol cache refresh error: {e}")

    def _background_symbol_scan(self):
        """Jalankan SCAN_MARKET di background thread (Fix I)."""
        available = self._get_scannable_symbols(max_symbols=30)
        if available:
            self._request_symbol_selection(available)

    def _startup_symbol_discovery(self):
        """Jalankan saat startup — discover semua simbol dan pilihkan AI (Fix O)."""
        try:
            # Refresh cache dulu
            self._refresh_symbols_cache()

            if self.config.get("AUTO_DISCOVER_SYMBOLS", "true").lower() != "true":
                return

            max_count = self.config.get_int("MAX_AUTO_SYMBOLS", 10)
            available = self._available_symbols_cache[:max_count * 3] if self._available_symbols_cache else []

            if not available:
                self._log("Startup scan: menggunakan default symbol list")
                return

            self._log(f"Startup scan: {len(available)} simbol tersedia — memilih terbaik...")

            # Tunggu AI siap (maks 10 detik)
            for _ in range(10):
                if self.ai.enabled:
                    break
                self._stop_event.wait(timeout=1)

            if self.ai.enabled:
                self._request_symbol_selection(available)
            else:
                self._auto_select_symbol_mix(available, max_count)
        except Exception as e:
            logger.error(f"Startup symbol discovery error: {e}")

    def _auto_select_symbol_mix(self, available: list, max_count: int):
        """Pilih campuran otomatis forex + crypto + metals tanpa AI (Fix O)."""
        forex_kw  = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]
        crypto_kw = ["BTC", "ETH", "BNB", "XRP", "LTC", "ADA", "DOT", "SOL"]
        metals_kw = ["XAU", "XAG", "GOLD", "SILVER"]

        selected = []
        per_group = max(1, max_count // 3)

        for sym in available:
            up = sym.upper()
            for kw in crypto_kw:
                if kw in up and len([s for s in selected if any(c in s.upper() for c in crypto_kw)]) < per_group:
                    selected.append(sym)
                    break
        for sym in available:
            up = sym.upper()
            for kw in metals_kw:
                if kw in up and sym not in selected and len(selected) < per_group * 2:
                    selected.append(sym)
                    break
        for sym in available:
            up = sym.upper()
            for kw in forex_kw:
                if kw in up and sym not in selected and len(selected) < max_count:
                    selected.append(sym)
                    break

        if selected:
            self._symbols = selected
            self._log(f"Auto-selected symbols: {', '.join(selected)}")

    def _request_symbol_selection(self, available_symbols: list[str]):
        """Kirim daftar simbol ke AI untuk dipilih, proses SELECT_SYMBOLS response."""
        if not self.ai.enabled or not available_symbols:
            return
        try:
            prompt = (
                f"=== SCAN MARKET ===\n"
                f"Simbol tersedia ({len(available_symbols)}): {', '.join(available_symbols[:30])}\n\n"
                f"Simbol aktif saat ini: {', '.join(self._symbols)}\n\n"
                f"Berdasarkan kondisi pasar saat ini, pilih 2-5 simbol terbaik untuk trading hari ini. "
                f"Prioritaskan simbol dengan volatilitas cukup dan spread rendah.\n\n"
                f"Respond JSON: {{\"action\": \"SELECT_SYMBOLS\", \"new_symbols\": [\"SYM1\",\"SYM2\"], "
                f"\"confidence\": 0.8, \"reason\": \"penjelasan\"}}"
            )
            result = self.ai.analyze_raw(prompt)
            if result.get("action") == "SELECT_SYMBOLS":
                new_syms = result.get("new_symbols", [])
                if new_syms:
                    self._symbols = new_syms
                    self._log(f"AI memilih simbol baru: {', '.join(new_syms)}")
        except Exception as e:
            logger.error(f"_request_symbol_selection error: {e}")

    def _scale_in_position(self, ticket, scale_volume: float):
        """Tambah volume ke posisi existing (buka order baru searah)."""
        try:
            positions = self.connector.get_positions()
            pos = next((p for p in positions if str(p.get("ticket", "")) == str(ticket)), None)
            if not pos:
                self._log(f"SCALE_IN: posisi #{ticket} tidak ditemukan")
                return
            symbol = pos["symbol"]
            order_type = pos["type"].lower()
            vol = self.risk.validate_lot_size(scale_volume)
            result = self.connector.send_market_order(
                symbol=symbol, order_type=order_type,
                volume=vol, sl_points=self.risk.default_sl_points,
                tp_points=self.risk.default_tp_points,
            )
            if result.success:
                self.risk.record_order()
                self._trade_count_today += 1
                self._log(f"SCALE_IN #{ticket}: +{vol} {symbol} @ {result.price}")
            else:
                self._log(f"SCALE_IN failed: {result.comment}")
        except Exception as e:
            logger.error(f"_scale_in_position error: {e}")

    def _scale_out_position(self, ticket, close_percent: int):
        """Tutup sebagian posisi (jika connector mendukung partial close)."""
        try:
            positions = self.connector.get_positions()
            pos = next((p for p in positions if str(p.get("ticket", "")) == str(ticket)), None)
            if not pos:
                self._log(f"SCALE_OUT: posisi #{ticket} tidak ditemukan")
                return
            total_vol = pos.get("volume", 0.01)
            partial_vol = round(total_vol * (close_percent / 100), 2)
            partial_vol = max(0.01, partial_vol)

            if partial_vol >= total_vol:
                # Close seluruhnya jika partial >= total
                result = self.connector.close_position(ticket)
                if result.success:
                    self._log(f"SCALE_OUT #{ticket}: full close {total_vol} @ {result.price}")
            else:
                # MT5 tidak mendukung partial close native — close dan reopen sisa
                symbol = pos["symbol"]
                order_type = pos["type"].lower()
                close_result = self.connector.close_position(ticket)
                if close_result.success:
                    remaining = round(total_vol - partial_vol, 2)
                    if remaining >= 0.01:
                        reopen = self.connector.send_market_order(
                            symbol=symbol, order_type=order_type,
                            volume=remaining,
                            sl_points=self.risk.default_sl_points,
                            tp_points=self.risk.default_tp_points,
                        )
                        self._log(
                            f"SCALE_OUT #{ticket}: tutup {partial_vol}, reopen {remaining} "
                            f"ticket={reopen.ticket if reopen.success else 'gagal'}"
                        )
                    else:
                        self._log(f"SCALE_OUT #{ticket}: full close {total_vol}")
        except Exception as e:
            logger.error(f"_scale_out_position error: {e}")

    def _cycle(self):
        """Full autonomous cycle: scan all symbols, review positions, execute AI decisions."""
        # 1. Global risk check
        account = self.connector.get_account_info()
        if not account:
            self._log("Failed to get account info, skipping cycle.")
            return

        can_trade, msg = self.risk.can_trade(account.equity)
        if not can_trade:
            self._log(f"Risk check failed: {msg}")
            return

        # Gather global context once
        all_positions = self.connector.get_positions()
        risk_status = self.risk.get_status(account.equity)
        total_positions = len(all_positions)

        # 2. Scan each symbol
        for symbol in self._symbols:
            if self._stop_event.is_set():
                break
            try:
                self._process_symbol(
                    symbol=symbol,
                    account=account,
                    all_positions=all_positions,
                    risk_status=risk_status,
                    total_positions=total_positions,
                )
            except Exception as e:
                self._log(f"Error processing {symbol}: {e}")
                logger.exception(f"Symbol processing error: {symbol}")

        # 3. Position review phase (check all open positions across all symbols)
        if self.config.get("POSITION_REVIEW_ENABLED", "true").lower() == "true":
            self._review_positions(account, all_positions, risk_status)

    def _process_symbol(self, symbol: str, account, all_positions: list,
                        risk_status: dict, total_positions: int):
        """Process satu symbol: fetch data -> indicators -> specialist pool -> composer -> execute."""
        # 1. Fetch OHLCV (minimal 60 candle untuk semua indikator)
        df = self.connector.get_ohlcv(symbol, self._timeframe, count=150)
        if df is None or len(df) < 60:
            self._log(f"[{symbol}] Insufficient data ({len(df) if df is not None else 0} candles)")
            return

        # 2. Compute all indicators
        indicators = compute_all_indicators(df)

        # Validasi indikator kritis
        if indicators["sma20"] is None or indicators["rsi"] is None:
            self._log(f"[{symbol}] Core indicators NaN, skipping cycle")
            return

        # 3. Get market data
        tick = self.connector.get_tick(symbol)
        if not tick:
            return

        symbol_info = self.connector.get_symbol_info(symbol)
        symbol_positions = [p for p in all_positions if p["symbol"] == symbol]

        # 3b. Decision cache — skip pipeline jika HOLD masih berlaku
        cached = self._get_cached_decision(symbol)
        if cached is not None and cached.action == "HOLD":
            self._log(f"[{symbol}] Cache HIT: HOLD (mode={cached.mode}) — skip pipeline")
            return  # HOLD berulang tidak perlu recompute agents

        # 4. Run specialist pool (MiroFish agent pool style)
        signals = self._pool.analyze(indicators, tick, symbol_info)

        # 5. Build market context untuk LLM conflict resolution
        ohlcv_summary = df.tail(10).to_string()
        market_context = {
            "symbol":          symbol,
            "timeframe":       self._timeframe,
            "ohlcv_summary":   ohlcv_summary,
            "sma20":           indicators["sma20"],
            "sma50":           indicators["sma50"] or 0.0,
            "rsi":             indicators["rsi"],
            "bid":             tick["bid"],
            "ask":             tick["ask"],
            "account_equity":  account.equity,
            "account_balance": account.balance,
            "free_margin":     account.free_margin,
            "open_positions":  symbol_positions,
            "spread":          symbol_info.get("spread", 0) if symbol_info else 0,
            "risk_status":     risk_status,
            "symbol_info":     symbol_info,
        }

        # 6. MiroFish ComposerAgent decide (consensus / conflict / fallback)
        ai_agent = self.ai if self.ai.enabled else None
        decision = self._mirofish_composer.decide(signals, market_context, ai_agent)

        # 6b. Simpan keputusan ke cache
        self._cache_decision(symbol, decision)

        # 7. Log + notify GUI
        self._log_agent_signals(symbol, signals, decision)

        # 8. Execute MiroFish decision (maps BUY/SELL/HOLD/CLOSE/MODIFY to orders)
        self._execute_decision(decision, symbol, symbol_info,
                               total_positions, len(symbol_positions))

        # 9. Also run direct AI analyze (HEAD path) for SCAN_MARKET / self-configure actions
        #    and as a second opinion when AI is enabled and MarketModeComposer dynamic prompt is set.
        if self.ai.enabled and self._composer_dynamic_prompt:
            original_prompt = self.ai.system_prompt
            self.ai.system_prompt = self._composer_dynamic_prompt
            result = self.ai.analyze(
                symbol=symbol,
                timeframe=self._timeframe,
                ohlcv_summary=ohlcv_summary,
                sma20=indicators["sma20"] or 0.0,
                sma50=indicators["sma50"] or 0.0,
                rsi=indicators["rsi"] or 50.0,
                bid=tick["bid"], ask=tick["ask"],
                account_equity=account.equity,
                account_balance=account.balance,
                free_margin=account.free_margin,
                open_positions=symbol_positions,
                spread=symbol_info.get("spread", 0) if symbol_info else 0,
                risk_status=risk_status,
                symbol_info=symbol_info,
            )
            self.ai.system_prompt = original_prompt
            # Only execute self-configure actions from this path (not BUY/SELL — MiroFish handles those)
            self_configure_actions = {
                "SCAN_MARKET", "SELECT_SYMBOLS", "SET_SESSION_END",
                "EARLY_TP", "END_DAY", "SCALE_IN", "SCALE_OUT",
            }
            if result.get("action") in self_configure_actions:
                self._log(
                    f"AI SELF-CFG [{symbol}]: {result['action']} "
                    f"(conf={result['confidence']:.2f}) – {result['reason']}"
                )
                self._execute_ai_decision(result, symbol, symbol_info, total_positions,
                                          len(symbol_positions))

    def _execute_ai_decision(self, result: dict, symbol: str, symbol_info: dict,
                              total_positions: int, symbol_pos_count: int):
        """Execute any AI decision autonomously — no confirmation needed. 13 actions supported."""
        action = result.get("action", "HOLD")
        confidence = result.get("confidence", 0.0)

        if action == "HOLD":
            return

        # Gunakan threshold dari Composer jika aktif, cap di 0.68 (Fix B)
        BASE_THRESHOLD = 0.60
        if self._composer:
            threshold = min(self._composer_confidence_threshold, 0.68)
        else:
            threshold = BASE_THRESHOLD

        if action in ("BUY", "SELL"):
            if confidence < threshold:
                self._log(f"Skipping {action}: confidence {confidence:.2f} < {threshold:.2f}")
                return

            # Position limit check
            ok, msg = self.risk.can_open_position(total_positions, symbol_pos_count)
            if not ok:
                self._log(f"Skipping {action}: {msg}")
                return

            # Lot size: AI custom > dynamic calculation > default, dengan Composer multiplier
            if "lot_size" in result:
                base_lot = self.risk.validate_lot_size(result["lot_size"])
            elif symbol_info:
                sl_pts = result.get("sl_points", self.risk.default_sl_points)
                base_lot = self.risk.calculate_lot_size(
                    equity=0,
                    sl_points=sl_pts,
                    tick_value=symbol_info.get("trade_tick_value", 1),
                    tick_size=symbol_info.get("trade_tick_size", 1),
                )
            else:
                base_lot = self.risk.validate_lot_size(self._lot_size)

            # Terapkan lot multiplier dari Composer (Fix C: STANDBY tidak blokir total)
            lot_mult = self._composer_lot_multiplier if self._composer else 1.0
            if lot_mult <= 0:
                # STANDBY: scan background, gunakan lot minimal (bukan stop total)
                self._log("Composer STANDBY — scan pasar di background, trading lot minimal")
                threading.Thread(target=self._background_symbol_scan, daemon=True).start()
                lot_mult = 0.3
            lot = self.risk.validate_lot_size(base_lot * lot_mult)

            sl = result.get("sl_points", self.risk.default_sl_points)
            tp = result.get("tp_points", self.risk.default_tp_points)

            self._log(f"AUTO-EXECUTE: {action} {lot} {symbol} SL={sl} TP={tp} (mult={lot_mult:.1f}x)")
            order_result = self.connector.send_market_order(
                symbol=symbol, order_type=action.lower(),
                volume=lot, sl_points=sl, tp_points=tp,
            )
            if order_result.success:
                self.risk.record_order()
                self._trade_count_today += 1
                if self._composer:
                    self._composer.record_trade(won=False)  # update saat close
                self._log(f"Order filled: ticket={order_result.ticket} @ {order_result.price}")
            else:
                self._log(f"Order failed: {order_result.comment}")

        elif action == "CLOSE":
            ticket = result.get("ticket")
            if not ticket:
                self._log("CLOSE action missing ticket number.")
                return
            self._log(f"AUTO-CLOSE: ticket #{ticket}")
            close_result = self.connector.close_position(ticket)
            if close_result.success:
                self._log(f"Position #{ticket} closed @ {close_result.price}")
            else:
                self._log(f"Close failed: {close_result.comment}")

        elif action == "CLOSE_ALL":
            self._log(f"AUTO-CLOSE_ALL: {symbol}")
            results = self.connector.close_positions_by_symbol(symbol)
            for r in results:
                if r.success:
                    self._log(f"  Closed #{r.ticket} @ {r.price}")
                else:
                    self._log(f"  Failed: {r.comment}")

        elif action == "MODIFY_SL_TP":
            ticket = result.get("ticket")
            if not ticket:
                self._log("MODIFY_SL_TP action missing ticket number.")
                return
            new_sl = result.get("new_sl", 0.0)
            new_tp = result.get("new_tp", 0.0)
            self._log(f"AUTO-MODIFY: ticket #{ticket} SL={new_sl} TP={new_tp}")
            mod_result = self.connector.modify_position_sl_tp(ticket, new_sl, new_tp)
            if mod_result.success:
                self._log(f"Position #{ticket} SL/TP modified.")
            else:
                self._log(f"Modify failed: {mod_result.comment}")

        # ── NEW: Self-configure actions ──

        elif action == "SCAN_MARKET":
            self._log("AI SCAN_MARKET — scan pasar di background thread")
            threading.Thread(target=self._background_symbol_scan, daemon=True).start()

        elif action == "SELECT_SYMBOLS":
            new_syms = result.get("new_symbols", [])
            if new_syms:
                self._symbols = [s.strip() for s in new_syms if s.strip()]
                self._log(f"AI SELECT_SYMBOLS: trading sekarang di {', '.join(self._symbols)}")

        elif action == "SET_SESSION_END":
            end_time = result.get("session_end", "")
            if end_time:
                self._session_end_time = end_time
                self._log(f"AI SET_SESSION_END: sesi trading berakhir jam {end_time}")

        elif action == "EARLY_TP":
            if result.get("all"):
                self._log("AI EARLY_TP: menutup semua posisi (ambil profit awal)")
                results_all = self.connector.close_all_positions()
                for r in results_all:
                    if r.success:
                        self._log(f"  EARLY_TP closed #{r.ticket} @ {r.price}")
            elif result.get("ticket"):
                ticket = result["ticket"]
                self._log(f"AI EARLY_TP: menutup posisi #{ticket}")
                r = self.connector.close_position(ticket)
                if r.success:
                    self._log(f"  EARLY_TP closed #{ticket} @ {r.price}")
                else:
                    self._log(f"  EARLY_TP failed: {r.comment}")
            else:
                self._log("EARLY_TP: tidak ada ticket atau all=true, skip.")

        elif action == "END_DAY":
            self._trading_ended_today = True
            self._log(f"AI END_DAY: tidak ada peluang pasar. Trading dihentikan untuk hari ini. [{symbol}]")

        elif action == "SCALE_IN":
            ticket = result.get("ticket")
            vol = result.get("scale_volume", 0.01)
            if ticket:
                self._scale_in_position(ticket, vol)
            else:
                self._log("SCALE_IN: ticket tidak ada, skip.")

        elif action == "SCALE_OUT":
            ticket = result.get("ticket")
            pct = result.get("close_percent", 50)
            if ticket:
                self._scale_out_position(ticket, pct)
            else:
                self._log("SCALE_OUT: ticket tidak ada, skip.")

    def _execute_basic_signal(self, signal: str, symbol: str,
                               total_positions: int, symbol_pos_count: int):
        """Execute basic strategy signal (fallback when AI is disabled)."""
        ok, msg = self.risk.can_open_position(total_positions, symbol_pos_count)
        if not ok:
            self._log(f"Skipping {signal}: {msg}")
            return

        lot = self.risk.validate_lot_size(self._lot_size)
        sl = self.risk.default_sl_points
        tp = self.risk.default_tp_points

        self._log(f"Executing {signal} {lot} {symbol} SL={sl} TP={tp}")
        result = self.connector.send_market_order(
            symbol=symbol, order_type=signal.lower(),
            volume=lot, sl_points=sl, tp_points=tp,
        )
        if result.success:
            self.risk.record_order()
            self._log(f"Order filled: ticket={result.ticket} @ {result.price}")
        else:
            self._log(f"Order failed: {result.comment}")

    def _review_positions(self, account, all_positions: list, risk_status: dict):
        """Review existing positions — AI can decide to close or modify them."""
        if not self.ai.enabled or not all_positions:
            return

        # Group positions by symbol for context
        symbols_with_positions = set(p["symbol"] for p in all_positions)

        for symbol in symbols_with_positions:
            if self._stop_event.is_set():
                break
            if symbol in self._symbols:
                continue  # Already analyzed in main cycle

            # For positions in symbols NOT in the main scan list,
            # still let AI review them
            try:
                symbol_positions = [p for p in all_positions if p["symbol"] == symbol]
                tick = self.connector.get_tick(symbol)
                if not tick:
                    continue

                symbol_info = self.connector.get_symbol_info(symbol)

                # Minimal data analysis for position review
                df = self.connector.get_ohlcv(symbol, self._timeframe, count=50)
                ohlcv_summary = ""
                sma20 = sma50 = rsi = 0.0
                if df is not None and len(df) >= 20:
                    df["SMA20"] = compute_sma(df["Close"], 20)
                    df["SMA50"] = compute_sma(df["Close"], 50)
                    df["RSI"] = compute_rsi(df["Close"], 14)
                    latest = df.iloc[-1]
                    sma20 = latest["SMA20"] if not pd.isna(latest["SMA20"]) else 0.0
                    sma50 = latest["SMA50"] if not pd.isna(latest["SMA50"]) else 0.0
                    rsi = latest["RSI"] if not pd.isna(latest["RSI"]) else 50.0
                    ohlcv_summary = df.tail(5).to_string()

                result = self.ai.analyze(
                    symbol=symbol, timeframe=self._timeframe,
                    ohlcv_summary=ohlcv_summary,
                    sma20=sma20, sma50=sma50, rsi=rsi,
                    bid=tick["bid"], ask=tick["ask"],
                    account_equity=account.equity,
                    account_balance=account.balance,
                    free_margin=account.free_margin,
                    open_positions=symbol_positions,
                    spread=symbol_info.get("spread", 0) if symbol_info else 0,
                    risk_status=risk_status,
                    symbol_info=symbol_info,
                )

                if result["action"] != "HOLD":
                    self._log(
                        f"AI REVIEW [{symbol}]: {result['action']} "
                        f"(conf={result['confidence']:.2f}) – {result['reason']}"
                    )
                    self._execute_ai_decision(
                        result, symbol, symbol_info,
                        len(all_positions), len(symbol_positions)
                    )

            except Exception as e:
                self._log(f"Position review error for {symbol}: {e}")

    def _basic_strategy(self, df: pd.DataFrame, sma20: float, sma50: float, rsi: float) -> str:
        """Simple SMA crossover + RSI filter (fallback when AI disabled)."""
        prev = df.iloc[-2]
        prev_sma20 = prev.get("SMA20", None)
        prev_sma50 = prev.get("SMA50", None)

        if prev_sma20 is None or prev_sma50 is None:
            return "HOLD"
        if pd.isna(prev_sma20) or pd.isna(prev_sma50):
            return "HOLD"

        if prev_sma20 <= prev_sma50 and sma20 > sma50 and rsi < 70:
            return "BUY"
        if prev_sma20 >= prev_sma50 and sma20 < sma50 and rsi > 30:
            return "SELL"

        return "HOLD"
