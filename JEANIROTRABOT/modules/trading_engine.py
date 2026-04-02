"""
JEANIROTRABOT - Trading Engine Module (Full Autonomous)
Orchestrates AI decisions, risk checks, and order execution.
Supports: BUY, SELL, CLOSE, CLOSE_ALL, MODIFY_SL_TP — all without user confirmation.
Multi-symbol scanning and position review built-in.
"""

import logging
import threading
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
    """Full autonomous trading: AI decides, engine executes without confirmation."""

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
        self._symbols: list[str] = ["EURUSD"]
        self._timeframe = "M5"
        self._lot_size = 0.01
        self._log_callback: Optional[Callable[[str], None]] = None
        self._interval = 5

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
        if self.running:
            return False, "Engine already running."
        if not self.connector.connected:
            return False, "Not connected to MT5."

        account = self.connector.get_account_info()
        if account:
            self.risk.set_initial_equity(account.equity)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
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
        """Process a single symbol: fetch data → AI analysis → execute."""
        # Fetch OHLCV
        df = self.connector.get_ohlcv(symbol, self._timeframe, count=100)
        if df is None or len(df) < 50:
            return

        # Compute indicators
        df["SMA20"] = compute_sma(df["Close"], 20)
        df["SMA50"] = compute_sma(df["Close"], 50)
        df["RSI"] = compute_rsi(df["Close"], 14)

        latest = df.iloc[-1]
        sma20 = latest["SMA20"]
        sma50 = latest["SMA50"]
        rsi = latest["RSI"]

        if pd.isna(sma20) or pd.isna(sma50) or pd.isna(rsi):
            return

        tick = self.connector.get_tick(symbol)
        if not tick:
            return

        # Get symbol info
        symbol_info = self.connector.get_symbol_info(symbol)

        # Symbol-specific positions
        symbol_positions = [p for p in all_positions if p["symbol"] == symbol]

        # Determine signal
        if self.ai.enabled:
            spread = symbol_info.get("spread", 0) if symbol_info else 0
            ohlcv_summary = df.tail(10).to_string()
            result = self.ai.analyze(
                symbol=symbol,
                timeframe=self._timeframe,
                ohlcv_summary=ohlcv_summary,
                sma20=sma20, sma50=sma50, rsi=rsi,
                bid=tick["bid"], ask=tick["ask"],
                account_equity=account.equity,
                account_balance=account.balance,
                free_margin=account.free_margin,
                open_positions=symbol_positions,
                spread=spread,
                risk_status=risk_status,
                symbol_info=symbol_info,
            )
            self._log(
                f"AI [{symbol}]: {result['action']} (conf={result['confidence']:.2f}) "
                f"– {result['reason']}"
            )
            self._execute_ai_decision(result, symbol, symbol_info, total_positions,
                                       len(symbol_positions))
        else:
            # Fallback: basic SMA crossover strategy
            signal = self._basic_strategy(df, sma20, sma50, rsi)
            if signal in ("BUY", "SELL"):
                self._execute_basic_signal(signal, symbol, total_positions,
                                           len(symbol_positions))

    def _execute_ai_decision(self, result: dict, symbol: str, symbol_info: dict,
                              total_positions: int, symbol_pos_count: int):
        """Execute any AI decision autonomously — no confirmation needed."""
        action = result.get("action", "HOLD")
        confidence = result.get("confidence", 0.0)

        if action == "HOLD":
            return

        if action in ("BUY", "SELL"):
            if confidence < 0.6:
                self._log(f"Skipping {action}: confidence {confidence:.2f} < 0.6")
                return

            # Position limit check
            ok, msg = self.risk.can_open_position(total_positions, symbol_pos_count)
            if not ok:
                self._log(f"Skipping {action}: {msg}")
                return

            # Lot size: AI custom > dynamic calculation > default
            if "lot_size" in result:
                lot = self.risk.validate_lot_size(result["lot_size"])
            elif symbol_info:
                sl_pts = result.get("sl_points", self.risk.default_sl_points)
                lot = self.risk.calculate_lot_size(
                    equity=0,  # Will use config-based
                    sl_points=sl_pts,
                    tick_value=symbol_info.get("trade_tick_value", 1),
                    tick_size=symbol_info.get("trade_tick_size", 1),
                )
            else:
                lot = self.risk.validate_lot_size(self._lot_size)

            sl = result.get("sl_points", self.risk.default_sl_points)
            tp = result.get("tp_points", self.risk.default_tp_points)

            self._log(f"AUTO-EXECUTE: {action} {lot} {symbol} SL={sl} TP={tp}")
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
