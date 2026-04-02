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

    def get_status(self, current_equity: float = 0.0) -> dict:
        """Return current risk status for display."""
        dd_pct = 0.0
        if self._initial_equity > 0 and current_equity > 0:
            dd_pct = ((self._initial_equity - current_equity) / self._initial_equity) * 100
        self._reset_daily_counter()
        return {
            "active": self._active,
            "drawdown_pct": round(dd_pct, 2),
            "max_drawdown_pct": self.max_drawdown_pct,
            "orders_today": self._orders_today,
            "max_orders_per_day": self.max_orders_per_day,
            "max_lot_size": self.max_lot_size,
            "sl_points": self.default_sl_points,
            "tp_points": self.default_tp_points,
        }
