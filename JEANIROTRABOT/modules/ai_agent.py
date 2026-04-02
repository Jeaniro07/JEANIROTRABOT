"""
JEANIROTRABOT - Multi-Provider AI Trading Agent
Supports: OpenAI, Anthropic Claude, Google Gemini, DeepSeek, Groq, xAI Grok,
          and any OpenAI-compatible API (Ollama, LM Studio, etc.)
"""

import logging
import json
from typing import Optional

logger = logging.getLogger("JEANIROTRABOT.ai")

# Try importing the OpenAI library (used for OpenAI + all compatible APIs)
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
        "base_url": None,  # Default OpenAI
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
        "base_url": "http://localhost:11434/v1",  # Default Ollama
        "models": ["custom-model"],
        "default_model": "custom-model",
        "key_prefix": "",
        "key_hint": "any-key-or-empty",
    },
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a professional trading analyst AI. You receive market data "
    "(symbol, timeframe, recent OHLCV prices, technical indicators) and must "
    "respond with a JSON object containing your trading decision.\n\n"
    "Response format (JSON only, no other text):\n"
    '{"action": "BUY" | "SELL" | "HOLD", "confidence": 0.0-1.0, '
    '"reason": "brief explanation"}\n\n'
    "Rules:\n"
    "- Only recommend BUY or SELL when confidence >= 0.6\n"
    "- Consider trend (SMA crossover), momentum (RSI), and price action\n"
    "- If data is insufficient or unclear, respond HOLD\n"
    "- Be conservative; capital preservation is priority"
)


class AIAgent:
    """Multi-provider AI trading signal generator."""

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
        """Reconfigure the agent with new provider/key/model."""
        self._provider = provider
        self._api_key = api_key
        self._model = model or self._get_default_model(provider)
        self._custom_base_url = base_url
        self._client = None  # Reset client
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
        """Test if API key and provider work."""
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
        sma20: float, sma50: float, rsi: float, bid: float, ask: float
    ) -> dict:
        """Send market data to AI and get trading recommendation."""
        default = {"action": "HOLD", "confidence": 0.0,
                   "reason": "AI unavailable", "raw_response": ""}

        if not self._enabled:
            default["reason"] = "AI agent is disabled."
            return default

        client = self._get_client()
        if client is None:
            default["reason"] = "No API client available."
            return default

        user_prompt = (
            f"Symbol: {symbol}\n"
            f"Timeframe: {timeframe}\n"
            f"Current Bid: {bid}, Ask: {ask}\n"
            f"SMA(20): {sma20:.5f}, SMA(50): {sma50:.5f}\n"
            f"RSI(14): {rsi:.2f}\n"
            f"Recent OHLCV data (last 10 candles):\n{ohlcv_summary}\n\n"
            f"Provide your trading decision as JSON."
        )

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=300,
                temperature=0.3,
            )
            raw = response.choices[0].message.content.strip()
            logger.info(f"AI [{self._provider}/{self._model}]: {raw}")

            parsed = self._parse_response(raw)
            parsed["raw_response"] = raw
            return parsed

        except Exception as e:
            logger.error(f"AI analysis error ({self._provider}): {e}")
            self._client = None  # Reset on error
            default["reason"] = f"AI error: {str(e)}"
            return default

    def _parse_response(self, raw: str) -> dict:
        """Parse AI response JSON. Handles markdown code blocks."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
            action = data.get("action", "HOLD").upper()
            if action not in ("BUY", "SELL", "HOLD"):
                action = "HOLD"
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            reason = data.get("reason", "No reason provided.")
            return {"action": action, "confidence": confidence, "reason": reason}
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse AI response: {e}")
            return {"action": "HOLD", "confidence": 0.0,
                    "reason": f"Parse error: {raw[:100]}"}
