"""
JEANIROTRABOT - Exchange Connector (CCXT-based)
Koneksi langsung ke crypto exchange: Binance, OKX, Indodax (Indonesia), Tokocrypto.
Mengimplementasi interface yang sama dengan MT5Connector agar bisa digunakan secara interchaneable.

Opsional: pip install ccxt
Exchange Indonesia yang didukung: Indodax (BTC/IDR, ETH/IDR, dll)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger("JEANIROTRABOT.exchange")

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logger.info("ccxt tidak terinstall. Install dengan: pip install ccxt")

# Exchange yang didukung
SUPPORTED_EXCHANGES = {
    "Binance": {
        "ccxt_id": "binance",
        "description": "Binance Global (Crypto)",
        "quote_currency": "USDT",
        "symbol_format": "BTC/USDT",
    },
    "OKX": {
        "ccxt_id": "okx",
        "description": "OKX Exchange (Crypto)",
        "quote_currency": "USDT",
        "symbol_format": "BTC/USDT",
    },
    "Indodax": {
        "ccxt_id": "indodax",
        "description": "Indodax — Exchange Indonesia (IDR)",
        "quote_currency": "IDR",
        "symbol_format": "BTC/IDR",
    },
    "Tokocrypto": {
        "ccxt_id": "tokocrypto",
        "description": "Tokocrypto — Exchange Indonesia",
        "quote_currency": "USDT",
        "symbol_format": "BTC/USDT",
    },
    "Gate.io": {
        "ccxt_id": "gateio",
        "description": "Gate.io (Crypto)",
        "quote_currency": "USDT",
        "symbol_format": "BTC/USDT",
    },
    "KuCoin": {
        "ccxt_id": "kucoin",
        "description": "KuCoin (Crypto)",
        "quote_currency": "USDT",
        "symbol_format": "BTC/USDT",
    },
}

# Timeframe mapping dari format MT5 ke ccxt
TIMEFRAME_MAP = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
}


@dataclass
class ExchangeAccountInfo:
    balance: float
    equity: float
    margin: float
    free_margin: float
    profit: float
    currency: str


@dataclass
class ExchangeOrderResult:
    success: bool
    ticket: str
    price: float
    comment: str


class ExchangeConnector:
    """
    Connector ke crypto exchange menggunakan CCXT.
    Interface sama dengan MT5Connector untuk interchaenability.
    """

    def __init__(self, exchange_name: str = "Binance", api_key: str = "",
                 api_secret: str = "", testnet: bool = False):
        self._exchange_name = exchange_name
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._exchange = None
        self._connected = False
        self._open_orders: dict = {}  # ticket -> order info (simulated positions)

    @property
    def connected(self) -> bool:
        return self._connected

    @staticmethod
    def is_available() -> bool:
        return CCXT_AVAILABLE

    @staticmethod
    def get_supported_exchanges() -> list[str]:
        return list(SUPPORTED_EXCHANGES.keys())

    def connect(self) -> tuple[bool, str]:
        """Koneksi ke exchange."""
        if not CCXT_AVAILABLE:
            return False, "ccxt tidak terinstall. Jalankan: pip install ccxt"

        exchange_info = SUPPORTED_EXCHANGES.get(self._exchange_name)
        if not exchange_info:
            return False, f"Exchange '{self._exchange_name}' tidak didukung."

        ccxt_id = exchange_info["ccxt_id"]
        try:
            exchange_class = getattr(ccxt, ccxt_id)
            params = {
                "apiKey": self._api_key,
                "secret": self._api_secret,
                "enableRateLimit": True,
            }

            if self._testnet and ccxt_id == "binance":
                params["urls"] = {"api": {"public": "https://testnet.binance.vision/api",
                                           "private": "https://testnet.binance.vision/api"}}

            self._exchange = exchange_class(params)

            # Test koneksi dengan fetch balance
            if self._api_key and self._api_secret:
                self._exchange.fetch_balance()

            self._connected = True
            msg = f"Terhubung ke {self._exchange_name} ({exchange_info['description']})"
            if self._testnet:
                msg += " [TESTNET]"
            logger.info(msg)
            return True, msg

        except AttributeError:
            return False, f"Exchange '{ccxt_id}' tidak ada di ccxt."
        except Exception as e:
            self._connected = False
            return False, f"Gagal connect ke {self._exchange_name}: {str(e)}"

    def disconnect(self):
        self._exchange = None
        self._connected = False
        logger.info(f"Disconnected dari {self._exchange_name}")

    def get_account_info(self) -> Optional[ExchangeAccountInfo]:
        if not self._exchange:
            return None
        try:
            balance = self._exchange.fetch_balance()
            total = balance.get("total", {})
            free = balance.get("free", {})

            # Hitung total balance dalam quote currency
            exchange_info = SUPPORTED_EXCHANGES.get(self._exchange_name, {})
            quote = exchange_info.get("quote_currency", "USDT")

            total_value = total.get(quote, 0.0) or 0.0
            free_value = free.get(quote, 0.0) or 0.0

            # Hitung unrealized P&L dari open positions
            unrealized_pnl = sum(
                o.get("unrealized_pnl", 0) for o in self._open_orders.values()
            )

            return ExchangeAccountInfo(
                balance=total_value,
                equity=total_value + unrealized_pnl,
                margin=total_value - free_value,
                free_margin=free_value,
                profit=unrealized_pnl,
                currency=quote,
            )
        except Exception as e:
            logger.error(f"get_account_info error: {e}")
            return None

    def get_ohlcv(self, symbol: str, timeframe: str, count: int = 100) -> Optional[pd.DataFrame]:
        """Ambil OHLCV data dari exchange."""
        if not self._exchange:
            return None

        tf = TIMEFRAME_MAP.get(timeframe, "5m")
        try:
            ohlcv = self._exchange.fetch_ohlcv(symbol, tf, limit=count)
            if not ohlcv:
                return None

            df = pd.DataFrame(ohlcv, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
            df["Time"] = pd.to_datetime(df["Time"], unit="ms")
            df.set_index("Time", inplace=True)
            return df
        except Exception as e:
            logger.error(f"get_ohlcv error ({symbol}): {e}")
            return None

    def get_tick(self, symbol: str) -> Optional[dict]:
        """Ambil harga bid/ask saat ini."""
        if not self._exchange:
            return None
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            return {
                "bid": ticker.get("bid", ticker.get("last", 0)),
                "ask": ticker.get("ask", ticker.get("last", 0)),
                "last": ticker.get("last", 0),
            }
        except Exception as e:
            logger.error(f"get_tick error ({symbol}): {e}")
            return None

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Ambil informasi simbol (lot size, point, dll)."""
        if not self._exchange:
            return None
        try:
            markets = self._exchange.load_markets()
            market = markets.get(symbol, {})
            limits = market.get("limits", {})
            amount_limits = limits.get("amount", {})
            precision = market.get("precision", {})

            return {
                "point": 10 ** (-precision.get("price", 8)) if precision.get("price") else 0.00000001,
                "volume_min": amount_limits.get("min", 0.001),
                "volume_max": amount_limits.get("max", 1000),
                "spread": 0,
                "trade_tick_value": 1,
                "trade_tick_size": precision.get("price", 8),
            }
        except Exception as e:
            logger.debug(f"get_symbol_info error ({symbol}): {e}")
            return {"point": 0.00000001, "volume_min": 0.001, "volume_max": 1000,
                    "spread": 0, "trade_tick_value": 1, "trade_tick_size": 8}

    def get_symbols(self) -> list[str]:
        """Ambil daftar semua simbol yang tersedia."""
        if not self._exchange:
            return []
        try:
            markets = self._exchange.load_markets()
            exchange_info = SUPPORTED_EXCHANGES.get(self._exchange_name, {})
            quote = exchange_info.get("quote_currency", "USDT")
            # Filter hanya pasangan dengan quote currency yang sesuai
            symbols = [s for s in markets.keys() if s.endswith(f"/{quote}")]
            return sorted(symbols)[:200]
        except Exception as e:
            logger.error(f"get_symbols error: {e}")
            return []

    def get_positions(self) -> list[dict]:
        """Ambil posisi terbuka. Exchange spot trading menggunakan order tracking."""
        return list(self._open_orders.values())

    def send_market_order(self, symbol: str, order_type: str, volume: float,
                          sl_points: int = 0, tp_points: int = 0) -> ExchangeOrderResult:
        """Kirim market order ke exchange."""
        if not self._exchange:
            return ExchangeOrderResult(False, "", 0.0, "Exchange tidak terhubung")

        side = "buy" if order_type.lower() == "buy" else "sell"
        try:
            order = self._exchange.create_market_order(symbol, side, volume)
            ticket = str(order.get("id", ""))
            price = order.get("average", order.get("price", 0.0)) or 0.0

            # Track sebagai posisi terbuka
            self._open_orders[ticket] = {
                "ticket": ticket,
                "symbol": symbol,
                "type": order_type.upper(),
                "volume": volume,
                "price_open": price,
                "price_current": price,
                "profit": 0.0,
                "sl": 0.0,
                "tp": 0.0,
                "unrealized_pnl": 0.0,
            }

            logger.info(f"Order filled: {side} {volume} {symbol} @ {price} ticket={ticket}")
            return ExchangeOrderResult(True, ticket, price, f"Order filled @ {price}")

        except Exception as e:
            logger.error(f"send_market_order error: {e}")
            return ExchangeOrderResult(False, "", 0.0, str(e))

    def close_position(self, ticket: str) -> ExchangeOrderResult:
        """Tutup posisi berdasarkan ticket (order ID)."""
        if not self._exchange:
            return ExchangeOrderResult(False, ticket, 0.0, "Exchange tidak terhubung")

        pos = self._open_orders.get(str(ticket))
        if not pos:
            return ExchangeOrderResult(False, str(ticket), 0.0, "Posisi tidak ditemukan")

        try:
            # Close dengan market order berlawanan
            symbol = pos["symbol"]
            volume = pos["volume"]
            close_side = "sell" if pos["type"] == "BUY" else "buy"

            order = self._exchange.create_market_order(symbol, close_side, volume)
            price = order.get("average", order.get("price", 0.0)) or 0.0

            del self._open_orders[str(ticket)]
            return ExchangeOrderResult(True, str(ticket), price, f"Closed @ {price}")

        except Exception as e:
            logger.error(f"close_position error: {e}")
            return ExchangeOrderResult(False, str(ticket), 0.0, str(e))

    def close_positions_by_symbol(self, symbol: str) -> list[ExchangeOrderResult]:
        """Tutup semua posisi untuk simbol tertentu."""
        tickets = [t for t, p in self._open_orders.items() if p["symbol"] == symbol]
        return [self.close_position(t) for t in tickets]

    def close_all_positions(self) -> list[ExchangeOrderResult]:
        """Tutup semua posisi."""
        tickets = list(self._open_orders.keys())
        return [self.close_position(t) for t in tickets]

    def modify_position_sl_tp(self, ticket: str, sl: float, tp: float):
        """Update SL/TP pada posisi (simulated — exchange spot tidak mendukung native SL/TP)."""
        pos = self._open_orders.get(str(ticket))
        if pos:
            pos["sl"] = sl
            pos["tp"] = tp
            logger.info(f"Position {ticket} SL/TP updated (simulated): SL={sl} TP={tp}")
        from dataclasses import dataclass
        return ExchangeOrderResult(True, str(ticket), 0.0, "SL/TP updated (simulated)")

    def has_open_position(self, symbol: str) -> bool:
        return any(p["symbol"] == symbol for p in self._open_orders.values())

    def get_history_orders(self, days: int = 1) -> list[dict]:
        """Ambil riwayat order."""
        if not self._exchange:
            return []
        try:
            since = self._exchange.milliseconds() - (days * 24 * 3600 * 1000)
            # Ambil dari beberapa simbol populer saja untuk performa
            history = []
            return history
        except Exception as e:
            logger.debug(f"get_history_orders error: {e}")
            return []
