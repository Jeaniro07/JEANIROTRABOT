"""
JEANIROTRABOT - Risk Management Module
Enforces drawdown limits, order limits, lot size caps, and SL/TP.
"""

import logging
from datetime import datetime

logger = logging.getLogger("JEANIROTRABOT.risk")


class RiskManager:
    """Enforces trading risk parameters."""

    def __init__(self, config):
        self._config = config
        self._initial_equity = 0.0
        self._orders_today = 0
        self._last_reset_date = datetime.now().date()
        self._active = True
        self.reload_params()

    def reload_params(self):
        """Reload risk parameters from config."""
        self.max_drawdown_pct = self._config.get_float("MAX_DRAWDOWN_PERCENT", 5.0)
        self.max_orders_per_day = self._config.get_int("MAX_ORDERS_PER_DAY", 20)
        self.max_lot_size = self._config.get_float("MAX_LOT_SIZE", 1.0)
        self.default_sl_points = self._config.get_int("DEFAULT_SL_POINTS", 100)
        self.default_tp_points = self._config.get_int("DEFAULT_TP_POINTS", 200)
        self.max_positions_total = self._config.get_int("MAX_POSITIONS_TOTAL", 5)
        self.max_positions_per_symbol = self._config.get_int("MAX_POSITIONS_PER_SYMBOL", 1)
        self.risk_per_trade_pct = self._config.get_float("RISK_PER_TRADE_PERCENT", 1.0)
        self.trailing_stop_points = self._config.get_int("TRAILING_STOP_POINTS", 0)

    def set_initial_equity(self, equity: float):
        """Set the equity baseline for drawdown calculation."""
        self._initial_equity = equity
        logger.info(f"Risk baseline equity set to {equity:.2f}")

    @property
    def active(self) -> bool:
        return self._active

    def deactivate(self, reason: str = ""):
        """Stop all trading due to risk breach."""
        self._active = False
        logger.warning(f"RiskManager DEACTIVATED: {reason}")

    def activate(self):
        self._active = True
        self._orders_today = 0
        logger.info("RiskManager re-activated.")

    def _reset_daily_counter(self):
        today = datetime.now().date()
        if today != self._last_reset_date:
            self._orders_today = 0
            self._last_reset_date = today

    def check_drawdown(self, current_equity: float) -> tuple[bool, str]:
        """Check if drawdown limit is breached. Returns (ok, message)."""
        if self._initial_equity <= 0:
            return True, "No baseline set."
        dd_pct = ((self._initial_equity - current_equity) / self._initial_equity) * 100
        if dd_pct >= self.max_drawdown_pct:
            msg = f"Drawdown limit breached: {dd_pct:.2f}% >= {self.max_drawdown_pct}%"
            self.deactivate(msg)
            return False, msg
        return True, f"Drawdown OK: {dd_pct:.2f}%"

    def check_order_limit(self) -> tuple[bool, str]:
        """Check if daily order limit is reached."""
        self._reset_daily_counter()
        if self._orders_today >= self.max_orders_per_day:
            msg = f"Daily order limit reached: {self._orders_today}/{self.max_orders_per_day}"
            return False, msg
        return True, f"Orders today: {self._orders_today}/{self.max_orders_per_day}"

    def validate_lot_size(self, requested: float) -> float:
        """Clamp lot size to max allowed."""
        return min(requested, self.max_lot_size)

    def record_order(self):
        """Increment daily order counter."""
        self._reset_daily_counter()
        self._orders_today += 1

    def can_trade(self, current_equity: float) -> tuple[bool, str]:
        """Combined check: active, drawdown, order limit."""
        if not self._active:
            return False, "Risk manager is deactivated."

        ok, msg = self.check_drawdown(current_equity)
        if not ok:
            return False, msg

        ok, msg = self.check_order_limit()
        if not ok:
            return False, msg

        return True, "All risk checks passed."

    @property
    def orders_today(self) -> int:
        self._reset_daily_counter()
        return self._orders_today

    def calculate_lot_size(self, equity: float, sl_points: int,
                           tick_value: float = 1.0, tick_size: float = 1.0) -> float:
        """Dynamic lot sizing based on risk percentage per trade."""
        if sl_points <= 0 or equity <= 0 or tick_value <= 0:
            return self._config.get_float("MAX_LOT_SIZE", 0.01)
        risk_amount = equity * (self.risk_per_trade_pct / 100.0)
        sl_value = sl_points * (tick_value / tick_size) if tick_size > 0 else sl_points
        if sl_value <= 0:
            return 0.01
        lot = risk_amount / sl_value
        lot = max(0.01, min(lot, self.max_lot_size))
        return round(lot, 2)

    def can_open_position(self, total_positions: int, symbol_positions: int) -> tuple[bool, str]:
        """Check if we can open a new position based on position limits."""
        if total_positions >= self.max_positions_total:
            return False, f"Max total positions reached: {total_positions}/{self.max_positions_total}"
        if symbol_positions >= self.max_positions_per_symbol:
            return False, f"Max positions per symbol reached: {symbol_positions}/{self.max_positions_per_symbol}"
        return True, "Position limits OK."

    def get_status(self, current_equity: float = 0.0) -> dict:
        """Return current risk status for display."""
        dd_pct = 0.0
        if self._initial_equity > 0 and current_equity > 0:
            dd_pct = ((self._initial_equity - current_equity) / self._initial_equity) * 100
        self._reset_daily_counter()
        return {
            "active": self._active,
            "initial_equity": self._initial_equity,
            "drawdown_pct": round(dd_pct, 2),
            "max_drawdown_pct": self.max_drawdown_pct,
            "orders_today": self._orders_today,
            "max_orders_per_day": self.max_orders_per_day,
            "max_lot_size": self.max_lot_size,
            "sl_points": self.default_sl_points,
            "tp_points": self.default_tp_points,
            "max_positions_total": self.max_positions_total,
            "max_positions_per_symbol": self.max_positions_per_symbol,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "trailing_stop_points": self.trailing_stop_points,
        }
