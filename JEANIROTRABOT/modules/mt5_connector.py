"""
JEANIROTRABOT - MetaTrader 5 Connector Module
Handles MT5 initialization, login, account info, market data, and order execution.
"""

import logging
import threading
import time
from typing import Optional
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger("JEANIROTRABOT.mt5")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 library not installed. MT5 features disabled.")


@dataclass
class AccountInfo:
    login: int = 0
    server: str = ""
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    profit: float = 0.0
    currency: str = "USD"
    leverage: int = 0
    name: str = ""


@dataclass
class OrderResult:
    success: bool = False
    ticket: int = 0
    volume: float = 0.0
    price: float = 0.0
    comment: str = ""
    retcode: int = 0


class MT5Connector:
    """Manages the connection and communication with MetaTrader 5."""

    # Timeframe mapping
    TIMEFRAMES = {
        "M1": None, "M5": None, "M15": None, "M30": None,
        "H1": None, "H4": None, "D1": None,
    }

    def __init__(self):
        self._connected = False
        self._login_id = 0
        self._server = ""
        self._lock = threading.Lock()  # Thread safety for all MT5 API calls
        self._init_timeframes()

    def _init_timeframes(self):
        if MT5_AVAILABLE:
            self.TIMEFRAMES = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
            }

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self, server: str, login: int, password: str) -> tuple[bool, str]:
        """Initialize MT5 and login. Returns (success, message)."""
        if not MT5_AVAILABLE:
            return False, "MetaTrader5 library not installed."

        with self._lock:
            if not mt5.initialize():
                error = mt5.last_error()
                msg = f"MT5 initialize failed: {error}"
                logger.error(msg)
                return False, msg

            authorized = mt5.login(login=login, password=password, server=server)
            if not authorized:
                error = mt5.last_error()
                msg = f"MT5 login failed: {error}"
                logger.error(msg)
                mt5.shutdown()
                return False, msg

        self._connected = True
        self._login_id = login
        self._server = server
        logger.info(f"Connected to MT5: server={server}, login={login}")
        return True, "Connected successfully."

    def disconnect(self) -> None:
        """Shutdown MT5 connection."""
        with self._lock:
            if MT5_AVAILABLE and self._connected:
                mt5.shutdown()
            self._connected = False
        logger.info("Disconnected from MT5.")

    def get_account_info(self) -> Optional[AccountInfo]:
        """Fetch current account information."""
        if not self._connected or not MT5_AVAILABLE:
            return None
        with self._lock:
            info = mt5.account_info()
        if info is None:
            return None
        return AccountInfo(
            login=info.login,
            server=info.server,
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            free_margin=info.margin_free,
            profit=info.profit,
            currency=info.currency,
            leverage=info.leverage,
            name=info.name,
        )

    def get_symbols(self) -> list[str]:
        """Get list of available symbols."""
        if not self._connected or not MT5_AVAILABLE:
            return []
        with self._lock:
            symbols = mt5.symbols_get()
        if symbols is None:
            return []
        return sorted([s.name for s in symbols if s.visible])

    def get_ohlcv(self, symbol: str, timeframe_str: str, count: int = 200) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for charting."""
        if not self._connected or not MT5_AVAILABLE:
            return None

        tf = self.TIMEFRAMES.get(timeframe_str)
        if tf is None:
            logger.error(f"Unknown timeframe: {timeframe_str}")
            return None

        with self._lock:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning(f"No data for {symbol} {timeframe_str}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "tick_volume": "Volume"
        }, inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def get_tick(self, symbol: str) -> Optional[dict]:
        """Get latest tick for a symbol."""
        if not self._connected or not MT5_AVAILABLE:
            return None
        with self._lock:
            tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {"bid": tick.bid, "ask": tick.ask, "last": tick.last, "time": tick.time}

    def has_open_position(self, symbol: str) -> bool:
        """Check if there's already an open position for a symbol."""
        positions = self.get_positions()
        return any(p["symbol"] == symbol for p in positions)

    def send_market_order(
        self, symbol: str, order_type: str, volume: float,
        sl_points: int = 0, tp_points: int = 0, comment: str = "JEANIROTRABOT"
    ) -> OrderResult:
        """Send a market order (buy or sell) without confirmation dialog."""
        if not self._connected or not MT5_AVAILABLE:
            return OrderResult(success=False, comment="Not connected to MT5.")

        with self._lock:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return OrderResult(success=False, comment=f"Symbol {symbol} not found.")

            if not symbol_info.visible:
                mt5.symbol_select(symbol, True)

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return OrderResult(success=False, comment="Failed to get tick data.")

            point = symbol_info.point
            min_lot = symbol_info.volume_min
            # Enforce minimum lot size from broker
            if volume < min_lot:
                volume = min_lot

            if order_type.lower() == "buy":
                trade_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                sl = price - sl_points * point if sl_points > 0 else 0.0
                tp = price + tp_points * point if tp_points > 0 else 0.0
            elif order_type.lower() == "sell":
                trade_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                sl = price + sl_points * point if sl_points > 0 else 0.0
                tp = price - tp_points * point if tp_points > 0 else 0.0
            else:
                return OrderResult(success=False, comment=f"Invalid order type: {order_type}")

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": trade_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 123456,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)

        if result is None:
            return OrderResult(success=False, comment="order_send returned None.")

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            msg = f"Order failed: retcode={result.retcode}, comment={result.comment}"
            logger.warning(msg)
            return OrderResult(
                success=False, retcode=result.retcode, comment=result.comment
            )

        logger.info(f"Order executed: {order_type} {volume} {symbol} @ {result.price}, ticket={result.order}")
        return OrderResult(
            success=True,
            ticket=result.order,
            volume=volume,
            price=result.price,
            comment=result.comment,
            retcode=result.retcode,
        )

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        if not self._connected or not MT5_AVAILABLE:
            return []
        with self._lock:
            positions = mt5.positions_get()
        if positions is None:
            return []
        return [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == 0 else "SELL",
                "volume": p.volume,
                "price_open": p.price_open,
                "price_current": p.price_current,
                "profit": p.profit,
                "sl": p.sl,
                "tp": p.tp,
            }
            for p in positions
        ]

    def get_history_orders(self, days: int = 1) -> list[dict]:
        """Get order history for the last N days."""
        if not self._connected or not MT5_AVAILABLE:
            return []
        from datetime import datetime, timedelta
        date_from = datetime.now() - timedelta(days=days)
        date_to = datetime.now()
        with self._lock:
            deals = mt5.history_deals_get(date_from, date_to)
        if deals is None:
            return []
        return [
            {
                "ticket": d.ticket,
                "time": d.time,
                "symbol": d.symbol,
                "type": "BUY" if d.type == 0 else "SELL",
                "volume": d.volume,
                "price": d.price,
                "profit": d.profit,
                "comment": d.comment,
            }
            for d in deals
        ]

    def close_position(self, ticket: int, comment: str = "AI_CLOSE") -> OrderResult:
        """Close a specific position by ticket number."""
        if not self._connected or not MT5_AVAILABLE:
            return OrderResult(success=False, comment="Not connected to MT5.")

        positions = self.get_positions()
        pos = None
        for p in positions:
            if p["ticket"] == ticket:
                pos = p
                break
        if pos is None:
            return OrderResult(success=False, comment=f"Position {ticket} not found.")

        close_type = "sell" if pos["type"] == "BUY" else "buy"
        return self.send_market_order(
            symbol=pos["symbol"],
            order_type=close_type,
            volume=pos["volume"],
            comment=comment,
        )

    def close_positions_by_symbol(self, symbol: str, comment: str = "AI_CLOSE") -> list[OrderResult]:
        """Close all positions for a specific symbol."""
        results = []
        positions = self.get_positions()
        for pos in positions:
            if pos["symbol"] == symbol:
                result = self.close_position(pos["ticket"], comment)
                results.append(result)
        return results

    def modify_position_sl_tp(self, ticket: int, sl: float = 0.0, tp: float = 0.0) -> OrderResult:
        """Modify SL/TP on an existing position."""
        if not self._connected or not MT5_AVAILABLE:
            return OrderResult(success=False, comment="Not connected to MT5.")

        with self._lock:
            position = mt5.positions_get(ticket=ticket)
            if position is None or len(position) == 0:
                return OrderResult(success=False, comment=f"Position {ticket} not found.")

            pos = position[0]
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": ticket,
                "sl": sl if sl > 0 else pos.sl,
                "tp": tp if tp > 0 else pos.tp,
            }
            result = mt5.order_send(request)

        if result is None:
            return OrderResult(success=False, comment="modify order_send returned None.")
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(success=False, retcode=result.retcode, comment=result.comment)

        logger.info(f"Position {ticket} modified: SL={sl}, TP={tp}")
        return OrderResult(success=True, ticket=ticket, comment="SL/TP modified.")

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Get detailed symbol information (point, spread, lot sizes, etc.)."""
        if not self._connected or not MT5_AVAILABLE:
            return None
        with self._lock:
            info = mt5.symbol_info(symbol)
        if info is None:
            return None
        tick = self.get_tick(symbol)
        spread = (tick["ask"] - tick["bid"]) if tick else 0.0
        return {
            "symbol": info.name,
            "point": info.point,
            "digits": info.digits,
            "spread": round(spread / info.point) if info.point > 0 else 0,
            "spread_price": spread,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "trade_contract_size": info.trade_contract_size,
            "trade_tick_value": info.trade_tick_value,
            "trade_tick_size": info.trade_tick_size,
        }

    def get_positions_count(self) -> int:
        """Get total number of open positions."""
        if not self._connected or not MT5_AVAILABLE:
            return 0
        with self._lock:
            positions = mt5.positions_total()
        return positions if positions else 0

    def close_all_positions(self) -> list[OrderResult]:
        """Close all open positions (Emergency Stop)."""
        results = []
        positions = self.get_positions()
        for pos in positions:
            result = self.close_position(pos["ticket"], "EMERGENCY_CLOSE")
            results.append(result)
        return results
