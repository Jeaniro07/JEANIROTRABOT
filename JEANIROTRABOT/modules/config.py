"""
JEANIROTRABOT - Configuration Module (Plug & Play)
Auto-saves settings to JSON file. No .env setup needed.
All settings are managed through the GUI.
"""

import os
import json
import sys
import logging
from pathlib import Path

logger = logging.getLogger("JEANIROTRABOT.config")


def _get_app_dir() -> Path:
    """Get the application directory (works for both script and frozen .exe)."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


APP_DIR = _get_app_dir()
CONFIG_FILE = APP_DIR / "jeanirotrabot_config.json"

# Default settings - user hanya perlu isi API keys via GUI
DEFAULTS = {
    "MT5_SERVER": "",
    "MT5_LOGIN": "",
    "MT5_PASSWORD": "",
    # AI Provider settings
    "AI_PROVIDER": "OpenAI",
    "AI_API_KEY": "",
    "AI_MODEL": "gpt-4o-mini",
    "AI_BASE_URL": "",
    "AI_SYSTEM_PROMPT": "",
    # Risk settings
    "MAX_DRAWDOWN_PERCENT": "5.0",
    "MAX_ORDERS_PER_DAY": "20",
    "MAX_LOT_SIZE": "1.0",
    "DEFAULT_SL_POINTS": "100",
    "DEFAULT_TP_POINTS": "200",
    # Autonomous trading settings
    "TRADE_SYMBOLS": "EURUSD,XAUUSD,BTCUSD,ETHUSD,GBPUSD",  # Mix forex+metals+crypto
    "AI_AUTONOMY_LEVEL": "full",
    "MAX_POSITIONS_TOTAL": "5",
    "MAX_POSITIONS_PER_SYMBOL": "1",
    "RISK_PER_TRADE_PERCENT": "1.0",
    "TRAILING_STOP_POINTS": "0",
    "POSITION_REVIEW_ENABLED": "true",
    "AI_AUTO_EXECUTE": "true",
    # Self-configuring trading (new)
    "AI_MARKET_SCAN_ENABLED": "true",
    "AI_SESSION_END_TIME": "",        # HH:MM — kosong = tidak ada batas
    "AI_TRADING_ENDED_TODAY": "false",
    "AUTO_DISCOVER_SYMBOLS": "true",  # Auto-scan simbol dari connector
    "MAX_AUTO_SYMBOLS": "10",         # Max simbol auto-discovered
    # Exchange connector (new)
    "CONNECTOR_TYPE": "MT5",          # MT5 | Binance | OKX | Indodax | Tokocrypto
    "EXCHANGE_API_KEY": "",
    "EXCHANGE_API_SECRET": "",
    "EXCHANGE_TESTNET": "false",
    # News Agent (new)
    "NEWS_AGENT_ENABLED": "false",
    "NEWS_AGENT_INTERVAL": "30",      # menit
    # Research Agent (new)
    "RESEARCH_AGENT_ENABLED": "false",
    "RESEARCH_AGENT_INTERVAL": "60",  # menit
    # Composer Agent (new)
    "COMPOSER_ENABLED": "true",
    "COMPOSER_INTERVAL": "5",          # menit (Fix E: 10 → 5)
    # Fast decision pipeline
    "TRADE_INTERVAL":       "2",   # detik antar siklus loop (dari 5)
    "DECISION_CACHE_TTL":   "30",  # detik keputusan berlaku di cache
    "LLM_CONFLICT_TIMEOUT": "5",   # detik timeout LLM conflict call
    "PROFIT_TARGET_PERCENT": "70.0",
    "INITIAL_BALANCE_SNAPSHOT": "",
}


class AppConfig:
    """Central configuration - auto-loads and auto-saves to JSON."""

    def __init__(self, config_path: str = None):
        self._path = Path(config_path) if config_path else CONFIG_FILE
        self._config = dict(DEFAULTS)
        self._load()

    def _load(self):
        """Load config from JSON file. If not exists, use defaults."""
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Merge with defaults (so new keys are always available)
                for k, v in data.items():
                    if v:  # Only override if value is not empty
                        self._config[k] = str(v)
                logger.info(f"Config loaded from {self._path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}. Using defaults.")
        else:
            logger.info("No config file found. Using defaults. Settings will auto-save on first change.")

    def get(self, key: str, default: str = "") -> str:
        return self._config.get(key, default)

    def set(self, key: str, value: str) -> None:
        self._config[key] = value

    def get_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self._config.get(key, str(default)))
        except (ValueError, TypeError):
            return default

    def get_int(self, key: str, default: int = 0) -> int:
        try:
            return int(float(self._config.get(key, str(default))))
        except (ValueError, TypeError):
            return default

    def save(self) -> None:
        """Auto-save config to JSON file."""
        try:
            # Don't save password in plain text - mask it as indicator only
            save_data = dict(self._config)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Config saved to {self._path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def as_dict(self) -> dict:
        return dict(self._config)

    def reload(self) -> None:
        self._config = dict(DEFAULTS)
        self._load()
