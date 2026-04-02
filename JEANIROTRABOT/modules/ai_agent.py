"""
JEANIROTRABOT - Multi-Provider AI Trading Agent (Full Autonomous)
Supports: OpenAI, Anthropic Claude, Google Gemini, DeepSeek, Groq, xAI Grok,
          and any OpenAI-compatible API (Ollama, LM Studio, etc.)

The AI agent can: BUY, SELL, CLOSE, CLOSE_ALL, MODIFY_SL_TP, HOLD
All executions are automatic — no user confirmation needed.
"""

import logging
import json
from typing import Optional

logger = logging.getLogger("JEANIROTRABOT.ai")

try:
    from openai import OpenAI
    OPENAI_LIB = True
except ImportError:
    OPENAI_LIB = False
    logger.warning("openai library not installed. Install with: pip install openai")

# ──────────────────────────────────────────────
# Provider Registry
# ──────────────────────────────────────────────

PROVIDERS = {
    "OpenAI": {
        "base_url": None,
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "o3-mini"],
        "default_model": "gpt-4o-mini",
        "key_prefix": "sk-",
        "key_hint": "sk-xxxxxxxx",
    },
    "Anthropic (Claude)": {
        "base_url": "https://api.anthropic.com/v1/",
        "models": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"],
        "default_model": "claude-sonnet-4-20250514",
        "key_prefix": "sk-ant-",
        "key_hint": "sk-ant-xxxxxxxx",
        "extra_headers": {"anthropic-version": "2023-06-01"},
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro"],
        "default_model": "gemini-2.0-flash",
        "key_prefix": "AI",
        "key_hint": "AIzaxxxxxxxx",
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat",
        "key_prefix": "sk-",
        "key_hint": "sk-xxxxxxxx",
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
        "default_model": "llama-3.3-70b-versatile",
        "key_prefix": "gsk_",
        "key_hint": "gsk_xxxxxxxx",
    },
    "xAI (Grok)": {
        "base_url": "https://api.x.ai/v1",
        "models": ["grok-3-mini", "grok-3"],
        "default_model": "grok-3-mini",
        "key_prefix": "xai-",
        "key_hint": "xai-xxxxxxxx",
    },
    "Custom (OpenAI Compatible)": {
        "base_url": "http://localhost:11434/v1",
        "models": ["custom-model"],
        "default_model": "custom-model",
        "key_prefix": "",
        "key_hint": "any-key-or-empty",
    },
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a fully autonomous professional trading AI. You receive complete market data "
    "(symbol, timeframe, OHLCV, technical indicators, account info, open positions, risk status) "
    "and must respond with a JSON object containing your trading decision.\n\n"
    "You execute ALL decisions automatically — no human confirmation is needed.\n\n"
    "Response format (JSON only, no other text):\n"
    "{\n"
    '  "action": "BUY" | "SELL" | "HOLD" | "CLOSE" | "CLOSE_ALL" | "MODIFY_SL_TP",\n'
    '  "confidence": 0.0-1.0,\n'
    '  "reason": "brief explanation",\n'
    '  "sl_points": optional int (custom stop loss in points, omit to use default),\n'
    '  "tp_points": optional int (custom take profit in points, omit to use default),\n'
    '  "lot_size": optional float (custom lot size, omit for auto-calculated),\n'
    '  "ticket": optional int (required for CLOSE and MODIFY_SL_TP actions),\n'
    '  "new_sl": optional float (exact price, for MODIFY_SL_TP),\n'
    '  "new_tp": optional float (exact price, for MODIFY_SL_TP)\n'
    "}\n\n"
    "Actions:\n"
    "- BUY: Open a long position on the current symbol\n"
    "- SELL: Open a short position on the current symbol\n"
    "- HOLD: Do nothing, wait for better setup\n"
    "- CLOSE: Close a specific position (provide ticket number)\n"
    "- CLOSE_ALL: Close all positions on the current symbol\n"
    "- MODIFY_SL_TP: Modify stop loss / take profit on a position (provide ticket, new_sl/new_tp)\n\n"
    "Rules:\n"
    "- Only recommend BUY or SELL when confidence >= 0.6\n"
    "- Review open positions and close losing trades or take profit when appropriate\n"
    "- Consider trend (SMA crossover), momentum (RSI), volatility (spread), and price action\n"
    "- Use account equity and risk status to size positions appropriately\n"
    "- If data is insufficient or unclear, respond HOLD\n"
    "- Be conservative; capital preservation is priority\n"
    "- You may suggest trailing stops by using MODIFY_SL_TP to move SL closer to current price"
)


class AIAgent:
    """Multi-provider AI trading signal generator with full autonomous execution."""

    def __init__(self, provider: str = "OpenAI", api_key: str = "",
                 model: str = "", base_url: str = "", system_prompt: str = ""):
        self._provider = provider
        self._api_key = api_key
        self._model = model or self._get_default_model(provider)
        self._custom_base_url = base_url
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._client = None
        self._enabled = False

    # ── Properties ──

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str):
        self._system_prompt = value.strip() if value.strip() else DEFAULT_SYSTEM_PROMPT

    # ── Configuration ──

    def configure(self, provider: str, api_key: str, model: str = "",
                  base_url: str = ""):
        self._provider = provider
        self._api_key = api_key
        self._model = model or self._get_default_model(provider)
        self._custom_base_url = base_url
        self._client = None
        logger.info(f"AI configured: provider={provider}, model={self._model}")

    def set_api_key(self, key: str):
        self._api_key = key
        self._client = None

    def set_model(self, model: str):
        self._model = model
        self._client = None

    def set_provider(self, provider: str):
        self._provider = provider
        self._model = self._get_default_model(provider)
        self._client = None

    @staticmethod
    def get_providers() -> list[str]:
        return list(PROVIDERS.keys())

    @staticmethod
    def get_models(provider: str) -> list[str]:
        info = PROVIDERS.get(provider, {})
        return info.get("models", ["custom-model"])

    @staticmethod
    def get_key_hint(provider: str) -> str:
        info = PROVIDERS.get(provider, {})
        return info.get("key_hint", "api-key")

    @staticmethod
    def _get_default_model(provider: str) -> str:
        info = PROVIDERS.get(provider, {})
        return info.get("default_model", "gpt-4o-mini")

    # ── Client Management ──

    def _get_client(self) -> Optional[OpenAI]:
        if not OPENAI_LIB:
            return None
        if not self._api_key:
            return None
        if self._client is not None:
            return self._client

        provider_info = PROVIDERS.get(self._provider, {})
        base_url = self._custom_base_url or provider_info.get("base_url")
        extra_headers = provider_info.get("extra_headers")

        kwargs = {"api_key": self._api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if extra_headers:
            kwargs["default_headers"] = extra_headers

        try:
            self._client = OpenAI(**kwargs)
            return self._client
        except Exception as e:
            logger.error(f"Failed to create client: {e}")
            return None

    # ── Test & Analyze ──

    def test_connection(self) -> tuple[bool, str]:
        client = self._get_client()
        if client is None:
            if not OPENAI_LIB:
                return False, "openai library not installed."
            return False, "API key not set."

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=5,
            )
            content = response.choices[0].message.content.strip()
            return True, f"OK ({self._provider} / {self._model})"
        except Exception as e:
            self._client = None
            return False, f"{self._provider} error: {str(e)}"

    def analyze(
        self, symbol: str, timeframe: str, ohlcv_summary: str,
        sma20: float, sma50: float, rsi: float, bid: float, ask: float,
        account_equity: float = 0.0, account_balance: float = 0.0,
        free_margin: float = 0.0, open_positions: list = None,
        spread: float = 0.0, risk_status: dict = None,
        symbol_info: dict = None,
    ) -> dict:
        """Send full market context to AI and get autonomous trading decision."""
        default = {"action": "HOLD", "confidence": 0.0,
                   "reason": "AI unavailable", "raw_response": ""}

        if not self._enabled:
            default["reason"] = "AI agent is disabled."
            return default

        client = self._get_client()
        if client is None:
            default["reason"] = "No API client available."
            return default

        # Build comprehensive context
        user_prompt = (
            f"=== MARKET DATA ===\n"
            f"Symbol: {symbol}\n"
            f"Timeframe: {timeframe}\n"
            f"Current Bid: {bid}, Ask: {ask}, Spread: {spread:.1f} pts\n"
            f"SMA(20): {sma20:.5f}, SMA(50): {sma50:.5f}\n"
            f"RSI(14): {rsi:.2f}\n"
            f"Recent OHLCV (last 10 candles):\n{ohlcv_summary}\n\n"
        )

        # Account context
        if account_equity > 0:
            user_prompt += (
                f"=== ACCOUNT ===\n"
                f"Balance: {account_balance:.2f}, Equity: {account_equity:.2f}\n"
                f"Free Margin: {free_margin:.2f}\n"
            )

        # Risk context
        if risk_status:
            user_prompt += (
                f"=== RISK STATUS ===\n"
                f"Drawdown: {risk_status.get('drawdown_pct', 0):.2f}% "
                f"(max: {risk_status.get('max_drawdown_pct', 5)}%)\n"
                f"Orders today: {risk_status.get('orders_today', 0)}"
                f"/{risk_status.get('max_orders_per_day', 20)}\n"
                f"Default SL: {risk_status.get('sl_points', 100)} pts, "
                f"TP: {risk_status.get('tp_points', 200)} pts\n"
            )

        # Open positions context
        if open_positions:
            user_prompt += f"\n=== OPEN POSITIONS ({len(open_positions)}) ===\n"
            for p in open_positions:
                user_prompt += (
                    f"  Ticket #{p['ticket']}: {p['type']} {p['volume']} {p['symbol']} "
                    f"@ {p['price_open']:.5f} → {p['price_current']:.5f} "
                    f"P&L: {p['profit']:.2f} "
                    f"SL: {p['sl']:.5f} TP: {p['tp']:.5f}\n"
                )
        else:
            user_prompt += "\n=== OPEN POSITIONS: None ===\n"

        # Symbol info
        if symbol_info:
            user_prompt += (
                f"\n=== SYMBOL INFO ===\n"
                f"Point: {symbol_info.get('point', 0)}, "
                f"Min Lot: {symbol_info.get('volume_min', 0.01)}, "
                f"Tick Value: {symbol_info.get('trade_tick_value', 1)}\n"
            )

        user_prompt += (
            f"\nProvide your autonomous trading decision as JSON. "
            f"You may open new positions, close existing ones, modify SL/TP, or hold."
        )

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,
                temperature=0.3,
            )
            raw = response.choices[0].message.content.strip()
            logger.info(f"AI [{self._provider}/{self._model}]: {raw}")

            parsed = self._parse_response(raw)
            parsed["raw_response"] = raw
            return parsed

        except Exception as e:
            logger.error(f"AI analysis error ({self._provider}): {e}")
            self._client = None
            default["reason"] = f"AI error: {str(e)}"
            return default

    def _parse_response(self, raw: str) -> dict:
        """Parse AI response JSON with extended action support."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
            action = data.get("action", "HOLD").upper()
            valid_actions = ("BUY", "SELL", "HOLD", "CLOSE", "CLOSE_ALL", "MODIFY_SL_TP")
            if action not in valid_actions:
                action = "HOLD"
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            reason = data.get("reason", "No reason provided.")

            result = {"action": action, "confidence": confidence, "reason": reason}

            # Optional fields for autonomous execution
            if "sl_points" in data:
                result["sl_points"] = int(data["sl_points"])
            if "tp_points" in data:
                result["tp_points"] = int(data["tp_points"])
            if "lot_size" in data:
                result["lot_size"] = float(data["lot_size"])
            if "ticket" in data:
                result["ticket"] = int(data["ticket"])
            if "new_sl" in data:
                result["new_sl"] = float(data["new_sl"])
            if "new_tp" in data:
                result["new_tp"] = float(data["new_tp"])

            return result
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse AI response: {e}")
            return {"action": "HOLD", "confidence": 0.0,
                    "reason": f"Parse error: {raw[:100]}"}
