"""
JEANIROTRABOT - Multi-Provider AI Trading Agent (Full Autonomous)
Supports: OpenAI, Anthropic Claude, Google Gemini, DeepSeek, Groq, xAI Grok,
          OpenRouter, SiliconFlow, Azure OpenAI, and any OpenAI-compatible API.

The AI agent can:
  BUY, SELL, HOLD, CLOSE, CLOSE_ALL, MODIFY_SL_TP (original 6)
  SCAN_MARKET, SELECT_SYMBOLS, SET_SESSION_END, EARLY_TP, END_DAY, SCALE_IN, SCALE_OUT (new 7)

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
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "models": ["meta-llama/llama-3.1-8b-instruct:free", "google/gemini-2.0-flash-exp:free",
                   "openai/gpt-4o-mini", "anthropic/claude-3.5-haiku", "mistralai/mistral-7b-instruct:free"],
        "default_model": "meta-llama/llama-3.1-8b-instruct:free",
        "key_prefix": "sk-or-",
        "key_hint": "sk-or-xxxxxxxx",
        "extra_headers": {"HTTP-Referer": "https://jeanirotrabot.app", "X-Title": "JEANIROTRABOT"},
    },
    "SiliconFlow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "models": ["Qwen/Qwen2.5-7B-Instruct", "deepseek-ai/DeepSeek-V2.5",
                   "THUDM/glm-4-9b-chat", "Qwen/Qwen2.5-72B-Instruct"],
        "default_model": "Qwen/Qwen2.5-7B-Instruct",
        "key_prefix": "sf-",
        "key_hint": "sf-xxxxxxxx",
    },
    "Azure OpenAI": {
        "base_url": "",  # User wajib isi endpoint Azure
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-35-turbo"],
        "default_model": "gpt-4o",
        "key_prefix": "",
        "key_hint": "Azure API Key (dari Azure Portal)",
    },
}

DEFAULT_SYSTEM_PROMPT = (
    "Kamu adalah AI trading profesional yang sepenuhnya otonom. Kamu menerima data pasar lengkap "
    "(simbol, timeframe, OHLCV, indikator teknikal, info akun, posisi terbuka, status risiko) "
    "dan harus merespons dengan JSON berisi keputusan trading.\n\n"
    "Kamu menjalankan SEMUA keputusan secara otomatis — tidak ada konfirmasi manusia.\n\n"
    "Format respons (JSON only, tidak ada teks lain):\n"
    "{\n"
    '  "action": "BUY|SELL|HOLD|CLOSE|CLOSE_ALL|MODIFY_SL_TP|'
    'SCAN_MARKET|SELECT_SYMBOLS|SET_SESSION_END|EARLY_TP|END_DAY|SCALE_IN|SCALE_OUT",\n'
    '  "confidence": 0.0-1.0,\n'
    '  "reason": "penjelasan singkat",\n'
    '  "sl_points": int opsional,\n'
    '  "tp_points": int opsional,\n'
    '  "lot_size": float opsional,\n'
    '  "ticket": int opsional (wajib untuk CLOSE, MODIFY_SL_TP, SCALE_IN, SCALE_OUT, EARLY_TP),\n'
    '  "new_sl": float opsional (harga exact, untuk MODIFY_SL_TP),\n'
    '  "new_tp": float opsional (harga exact, untuk MODIFY_SL_TP),\n'
    '  "new_symbols": ["SYM1","SYM2"] opsional (untuk SELECT_SYMBOLS),\n'
    '  "session_end": "HH:MM" opsional (untuk SET_SESSION_END),\n'
    '  "close_percent": int 1-99 opsional (untuk SCALE_OUT, persentase volume ditutup),\n'
    '  "scale_volume": float opsional (untuk SCALE_IN, volume tambahan),\n'
    '  "all": bool opsional (untuk EARLY_TP, true = semua posisi)\n'
    "}\n\n"
    "AKSI ORIGINAL:\n"
    "- BUY: Buka posisi long pada simbol saat ini\n"
    "- SELL: Buka posisi short pada simbol saat ini\n"
    "- HOLD: Tunggu, tidak ada aksi\n"
    "- CLOSE: Tutup posisi tertentu (sertakan ticket)\n"
    "- CLOSE_ALL: Tutup semua posisi pada simbol ini\n"
    "- MODIFY_SL_TP: Modifikasi SL/TP posisi (sertakan ticket, new_sl/new_tp)\n\n"
    "AKSI SELF-CONFIGURE (baru):\n"
    "- SCAN_MARKET: Minta scan semua simbol tersedia untuk temukan peluang terbaik\n"
    "- SELECT_SYMBOLS: Ganti daftar simbol aktif trading (sertakan new_symbols)\n"
    "- SET_SESSION_END: Set waktu akhir trading hari ini (sertakan session_end HH:MM)\n"
    "- EARLY_TP: Ambil profit sekarang sebelum pasar berbalik (ticket atau all=true)\n"
    "- END_DAY: Tidak ada peluang hari ini — hentikan trading sampai besok\n"
    "- SCALE_IN: Tambah volume ke posisi existing (ticket + scale_volume)\n"
    "- SCALE_OUT: Tutup sebagian posisi untuk amankan profit (ticket + close_percent)\n\n"
    "ATURAN:\n"
    "- BUY/SELL hanya jika confidence >= threshold yang ditetapkan\n"
    "- Gunakan SCAN_MARKET jika tidak ada setup bagus di simbol saat ini\n"
    "- Gunakan END_DAY setelah scan jika tidak ada peluang di semua pasar\n"
    "- Gunakan SET_SESSION_END untuk batasi trading jika kondisi tidak ideal\n"
    "- Gunakan EARLY_TP jika indikator menunjukkan momentum akan berbalik\n"
    "- Gunakan SCALE_OUT untuk amankan profit parsial saat profit > 50% dari TP\n"
    "- Pertimbangkan tren (SMA crossover), momentum (RSI), volatilitas (spread)\n"
    "- Preservasi modal adalah prioritas utama\n"
    "- Jika data tidak cukup, respond HOLD"
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
        system_prompt_override: str = None,
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

        # Use override prompt if provided (thread-safe: no shared state mutation)
        effective_system_prompt = system_prompt_override or self._system_prompt

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
                    {"role": "system", "content": effective_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0.3,
                timeout=15,
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

    def analyze_raw(self, prompt: str, max_tokens: int = 800) -> dict:
        """
        Kirim prompt bebas ke AI dan parse hasilnya sebagai JSON.
        Digunakan oleh Composer, News Agent, dan Research Agent.
        """
        default = {"action": "HOLD", "confidence": 0.0, "reason": "AI unavailable"}
        if not self._enabled:
            return default

        client = self._get_client()
        if client is None:
            return default

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "Respond only in valid JSON format."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
                timeout=30,
            )
            raw = response.choices[0].message.content.strip()
            logger.debug(f"analyze_raw response: {raw[:200]}")
            return self._parse_response(raw)
        except Exception as e:
            logger.error(f"analyze_raw error: {e}")
            self._client = None
            return default

    def _parse_response(self, raw: str) -> dict:
        """Parse AI response JSON — mendukung 13 aksi total."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
            action = data.get("action", "HOLD").upper()
            valid_actions = (
                "BUY", "SELL", "HOLD", "CLOSE", "CLOSE_ALL", "MODIFY_SL_TP",
                "SCAN_MARKET", "SELECT_SYMBOLS", "SET_SESSION_END",
                "EARLY_TP", "END_DAY", "SCALE_IN", "SCALE_OUT",
            )
            if action not in valid_actions:
                action = "HOLD"
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            reason = data.get("reason", "No reason provided.")

            result = {"action": action, "confidence": confidence, "reason": reason}

            # Original fields
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

            # New self-configure fields
            if "new_symbols" in data:
                result["new_symbols"] = list(data["new_symbols"])
            if "session_end" in data:
                result["session_end"] = str(data["session_end"])
            if "close_percent" in data:
                result["close_percent"] = max(1, min(99, int(data["close_percent"])))
            if "scale_volume" in data:
                result["scale_volume"] = float(data["scale_volume"])
            if "all" in data:
                result["all"] = bool(data["all"])

            # Extra fields for Composer / News / Research
            for extra_key in ("sentiment", "key_events", "signal", "summary",
                              "fundamental_outlook", "key_risks", "recommendation",
                              "market_mode", "trading_focus", "adjusted_confidence",
                              "adjusted_lot_multiplier", "dynamic_system_prompt",
                              "agent_directives", "reasoning"):
                if extra_key in data:
                    result[extra_key] = data[extra_key]

            return result
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse AI response: {e}")
            return {"action": "HOLD", "confidence": 0.0,
                    "reason": f"Parse error: {raw[:100]}"}
