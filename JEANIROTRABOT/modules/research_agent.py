"""
JEANIROTRABOT - DeepResearch Agent
Menganalisis data fundamental pasar: economic calendar, fear & greed index,
dan data makroekonomi dari sumber gratis.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger("JEANIROTRABOT.research")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Sumber data fundamental gratis
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
ECONOMIC_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


class ResearchAgent:
    """DeepResearch Agent — analisis fundamental pasar untuk mendukung keputusan trading."""

    def __init__(self):
        self._last_run: float = 0
        self._interval_minutes: int = 60
        self._cached_report: dict = {}

    def set_interval(self, minutes: int):
        self._interval_minutes = max(15, minutes)

    def is_due(self) -> bool:
        elapsed = time.time() - self._last_run
        return elapsed >= (self._interval_minutes * 60)

    def get_fear_greed_index(self) -> dict:
        """Ambil Fear & Greed Index crypto dari alternative.me."""
        if not REQUESTS_AVAILABLE:
            return {"value": 50, "label": "Neutral", "available": False}

        try:
            resp = requests.get(FEAR_GREED_URL, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                item = data.get("data", [{}])[0]
                return {
                    "value": int(item.get("value", 50)),
                    "label": item.get("value_classification", "Neutral"),
                    "available": True,
                }
        except Exception as e:
            logger.debug(f"Fear & greed fetch error: {e}")
        return {"value": 50, "label": "Neutral", "available": False}

    def get_economic_calendar(self) -> list[dict]:
        """Ambil economic calendar minggu ini dari forexfactory."""
        if not REQUESTS_AVAILABLE:
            return []

        try:
            resp = requests.get(
                ECONOMIC_CALENDAR_URL,
                timeout=8,
                headers={"User-Agent": "JEANIROTRABOT/1.0"}
            )
            if resp.status_code == 200:
                events = resp.json()
                # Filter high-impact events only
                high_impact = [
                    {
                        "title": e.get("title", ""),
                        "country": e.get("country", ""),
                        "date": e.get("date", ""),
                        "impact": e.get("impact", ""),
                        "forecast": e.get("forecast", ""),
                        "previous": e.get("previous", ""),
                    }
                    for e in events
                    if e.get("impact", "").lower() in ("high", "medium")
                ]
                return high_impact[:20]
        except Exception as e:
            logger.debug(f"Economic calendar fetch error: {e}")
        return []

    def research_symbol(self, symbol: str, ai_agent, extra_context: str = "") -> dict:
        """
        Lakukan riset fundamental untuk simbol tertentu menggunakan AI.
        Returns: {"symbol", "fundamental_outlook", "key_risks", "recommendation", "confidence"}
        """
        default = {
            "symbol": symbol,
            "fundamental_outlook": "Data tidak tersedia",
            "key_risks": [],
            "recommendation": "NEUTRAL",
            "confidence": 0.5,
        }

        if not ai_agent or not ai_agent.enabled:
            return default

        # Kumpulkan data fundamental
        fg = self.get_fear_greed_index()
        calendar = self.get_economic_calendar()

        # Build research prompt
        research_prompt = f"=== ANALISIS FUNDAMENTAL: {symbol} ===\n\n"

        if fg.get("available"):
            research_prompt += (
                f"Fear & Greed Index (Crypto): {fg['value']}/100 — {fg['label']}\n"
            )

        if calendar:
            research_prompt += "\n=== ECONOMIC CALENDAR (High/Medium Impact) ===\n"
            for ev in calendar[:10]:
                research_prompt += (
                    f"• [{ev['country']}] {ev['title']} — {ev['date']} "
                    f"(Impact: {ev['impact']}, Forecast: {ev.get('forecast','?')}, "
                    f"Previous: {ev.get('previous','?')})\n"
                )

        if extra_context:
            research_prompt += f"\n=== KONTEKS TAMBAHAN ===\n{extra_context}\n"

        research_prompt += (
            f"\nBerikan analisis fundamental singkat untuk {symbol}:\n"
            "1. Outlook fundamental (bullish/bearish/neutral)\n"
            "2. Risiko utama yang perlu diwaspadai\n"
            "3. Rekomendasi berdasarkan fundamental\n\n"
            "Respond JSON:\n"
            '{"symbol": "...", "fundamental_outlook": "...", '
            '"key_risks": ["..."], "recommendation": "BULLISH|BEARISH|NEUTRAL", '
            '"confidence": 0.0-1.0}'
        )

        try:
            result = ai_agent.analyze_raw(research_prompt)
            if isinstance(result, dict) and "recommendation" in result:
                self._last_run = time.time()
                return result
        except Exception as e:
            logger.error(f"Research AI error: {e}")

        return default

    def run_full_research(self, symbols: list[str], ai_agent) -> dict:
        """
        Jalankan riset lengkap untuk semua simbol.
        Returns: {"fear_greed": {...}, "calendar": [...], "symbol_reports": {...}}
        """
        report = {
            "fear_greed": self.get_fear_greed_index(),
            "calendar": self.get_economic_calendar(),
            "symbol_reports": {},
            "timestamp": time.time(),
        }

        for symbol in symbols[:5]:  # Max 5 simbol untuk hemat token AI
            if ai_agent and ai_agent.enabled:
                report["symbol_reports"][symbol] = self.research_symbol(symbol, ai_agent)

        self._cached_report = report
        self._last_run = time.time()
        return report

    def get_cached_report(self) -> dict:
        return self._cached_report

    def get_market_summary_text(self) -> str:
        """Ringkasan teks dari laporan terakhir untuk dimasukkan ke Composer prompt."""
        if not self._cached_report:
            return "Belum ada data research."

        lines = []
        fg = self._cached_report.get("fear_greed", {})
        if fg.get("available"):
            lines.append(f"Fear & Greed Index: {fg['value']}/100 ({fg['label']})")

        calendar = self._cached_report.get("calendar", [])
        if calendar:
            lines.append(f"Economic events minggu ini: {len(calendar)} event medium/high impact")
            for ev in calendar[:3]:
                lines.append(f"  - [{ev['country']}] {ev['title']} ({ev['date']})")

        symbol_reports = self._cached_report.get("symbol_reports", {})
        for sym, rep in symbol_reports.items():
            lines.append(f"{sym}: {rep.get('recommendation','?')} — {rep.get('fundamental_outlook','')[:100]}")

        return "\n".join(lines) if lines else "Tidak ada data."
