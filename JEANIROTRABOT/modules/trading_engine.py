"""
JEANIROTRABOT - Trading Engine Module
Orchestrates signal generation, risk checks, and order execution.
"""

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd

from modules.mt5_connector import MT5Connector, OrderResult
from modules.risk_manager import RiskManager
from modules.ai_agent import AIAgent

logger = logging.getLogger("JEANIROTRABOT.engine")


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    # Handle division: when avg_loss==0 → RSI=100, when both 0 → RSI=50
    rsi = pd.Series(np.nan, index=series.index)
    valid = avg_gain.notna() & avg_loss.notna()
    both_zero = valid & (avg_gain == 0) & (avg_loss == 0)
    loss_zero = valid & (avg_loss == 0) & (avg_gain > 0)
    normal = valid & (avg_loss > 0)
    rsi[both_zero] = 50.0
    rsi[loss_zero] = 100.0
    rsi[normal] = 100.0 - (100.0 / (1.0 + avg_gain[normal] / avg_loss[normal]))
    return rsi


class TradingEngine:
    """Core trading loop: fetch data -> compute indicators -> AI/signal -> risk check -> execute."""

    def __init__(
        self,
        connector: MT5Connector,
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
        self._symbol = "EURUSD"
        self._timeframe = "M5"
        self._lot_size = 0.01
        self._log_callback: Optional[Callable[[str], None]] = None
        self._interval = 5  # seconds between cycles

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def set_symbol(self, symbol: str):
        self._symbol = symbol

    def set_timeframe(self, timeframe: str):
        self._timeframe = timeframe

    def set_lot_size(self, lot: float):
        self._lot_size = lot

    def set_log_callback(self, cb: Callable[[str], None]):
        self._log_callback = cb

    def _log(self, msg: str):
        logger.info(msg)
        if self._log_callback:
            try:
                self._log_callback(msg)
            except Exception:
                pass

    def start(self) -> tuple[bool, str]:
        """Start the trading loop in a background thread."""
        if self.running:
            return False, "Engine already running."
        if not self.connector.connected:
            return False, "Not connected to MT5."

        # Set initial equity for risk management
        account = self.connector.get_account_info()
        if account:
            self.risk.set_initial_equity(account.equity)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._log(f"Trading engine started: {self._symbol} {self._timeframe}")
        return True, "Engine started."

    def stop(self):
        """Stop the trading loop gracefully."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._thread = None
        self._log("Trading engine stopped.")

    def emergency_stop(self):
        """Stop engine and close all positions."""
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
        """Main trading loop running in background thread."""
        consecutive_errors = 0
        while not self._stop_event.is_set():
            try:
                self._cycle()
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                self._log(f"Engine error: {e}")
                logger.exception("Trading cycle error")
                # Exponential backoff on consecutive errors (max 60s)
                backoff = min(self._interval * (2 ** consecutive_errors), 60)
                if self._stop_event.wait(timeout=backoff):
                    break
                continue
            # Use Event.wait instead of time.sleep for responsive shutdown
            if self._stop_event.wait(timeout=self._interval):
                break

    def _cycle(self):
        """Single trading cycle."""
        # 1. Check risk
        account = self.connector.get_account_info()
        if not account:
            self._log("Failed to get account info, skipping cycle.")
            return

        can_trade, msg = self.risk.can_trade(account.equity)
        if not can_trade:
            self._log(f"Risk check failed: {msg}")
            return

        # 2. Fetch OHLCV data
        df = self.connector.get_ohlcv(self._symbol, self._timeframe, count=100)
        if df is None or len(df) < 50:
            self._log(f"Insufficient data for {self._symbol} {self._timeframe}")
            return

        # 3. Compute indicators
        df["SMA20"] = compute_sma(df["Close"], 20)
        df["SMA50"] = compute_sma(df["Close"], 50)
        df["RSI"] = compute_rsi(df["Close"], 14)

        latest = df.iloc[-1]
        sma20 = latest["SMA20"]
        sma50 = latest["SMA50"]
        rsi = latest["RSI"]

        if pd.isna(sma20) or pd.isna(sma50) or pd.isna(rsi):
            return

        tick = self.connector.get_tick(self._symbol)
        if not tick:
            return

        # 4. Determine signal
        signal = "HOLD"

        if self.ai.enabled:
            # Use AI agent for decision
            ohlcv_summary = df.tail(10).to_string()
            result = self.ai.analyze(
                symbol=self._symbol,
                timeframe=self._timeframe,
                ohlcv_summary=ohlcv_summary,
                sma20=sma20,
                sma50=sma50,
                rsi=rsi,
                bid=tick["bid"],
                ask=tick["ask"],
            )
            self._log(f"AI: {result['action']} (conf={result['confidence']:.2f}) – {result['reason']}")
            if result["confidence"] >= 0.6:
                signal = result["action"]
        else:
            # Simple SMA crossover + RSI strategy
            signal = self._basic_strategy(df, sma20, sma50, rsi)

        # 5. Execute if signal is BUY or SELL
        if signal in ("BUY", "SELL"):
            # Prevent duplicate/pyramid positions on same symbol
            if self.connector.has_open_position(self._symbol):
                self._log(f"Skipping {signal}: already have open position on {self._symbol}")
                return

            lot = self.risk.validate_lot_size(self._lot_size)
            sl = self.risk.default_sl_points
            tp = self.risk.default_tp_points

            self._log(f"Executing {signal} {lot} {self._symbol} SL={sl} TP={tp}")
            result = self.connector.send_market_order(
                symbol=self._symbol,
                order_type=signal.lower(),
                volume=lot,
                sl_points=sl,
                tp_points=tp,
            )
            if result.success:
                self.risk.record_order()
                self._log(f"Order filled: ticket={result.ticket} @ {result.price}")
            else:
                self._log(f"Order failed: {result.comment}")

    def _basic_strategy(self, df: pd.DataFrame, sma20: float, sma50: float, rsi: float) -> str:
        """Simple SMA crossover + RSI filter strategy."""
        prev = df.iloc[-2]
        prev_sma20 = prev.get("SMA20", None)
        prev_sma50 = prev.get("SMA50", None)

        if prev_sma20 is None or prev_sma50 is None:
            return "HOLD"
        if pd.isna(prev_sma20) or pd.isna(prev_sma50):
            return "HOLD"

        # Bullish crossover: SMA20 crosses above SMA50 + RSI not overbought
        if prev_sma20 <= prev_sma50 and sma20 > sma50 and rsi < 70:
            return "BUY"

        # Bearish crossover: SMA20 crosses below SMA50 + RSI not oversold
        if prev_sma20 >= prev_sma50 and sma20 < sma50 and rsi > 30:
            return "SELL"

        return "HOLD"
