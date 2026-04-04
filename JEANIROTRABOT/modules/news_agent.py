"""
JEANIROTRABOT - News Retrieval Agent
Mengambil berita forex/crypto dari RSS feeds dan menganalisis dengan AI.
Opsional: feedparser (pip install feedparser). Fallback ke requests jika tidak ada.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger("JEANIROTRABOT.news")

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logger.info("feedparser tidak terinstall. Menggunakan requests fallback untuk news.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# RSS feeds sumber berita
NEWS_FEEDS = {
    "Reuters Forex": "https://feeds.reuters.com/reuters/businessNews",
    "Investing.com": "https://www.investing.com/rss/news.rss",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Kontan (ID)": "https://rss.kontan.co.id/category/investasi",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
}

# Kata kunci pasar untuk filter berita
MARKET_KEYWORDS = {
    "forex": ["forex", "currency", "dollar", "euro", "yen", "pound", "usd", "eur", "gbp", "jpy",
              "fed", "interest rate", "central bank", "inflation", "gdp"],
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain", "binance", "defi",
               "altcoin", "usdt", "idr", "rupiah"],
    "commodities": ["gold", "oil", "silver", "xau", "crude", "brent"],
    "general": ["market", "trading", "economy", "recession", "rally", "bearish", "bullish"],
}


class NewsAgent:
    """News Retrieval Agent — mengambil dan menganalisis berita pasar."""

    def __init__(self):
        self._last_fetch: float = 0
        self._cached_news: list = []
        self._interval_minutes: int = 30

    def set_interval(self, minutes: int):
        self._interval_minutes = max(5, minutes)

    def is_due(self) -> bool:
        elapsed = time.time() - self._last_fetch
        return elapsed >= (self._interval_minutes * 60)

    def get_news(self, symbols: list[str] = None, max_items: int = 20) -> list[dict]:
        """
        Ambil berita dari RSS feeds.
        Returns list of: {"title", "summary", "source", "published", "relevance"}
        """
        if not FEEDPARSER_AVAILABLE and not REQUESTS_AVAILABLE:
            return [{"title": "Module tidak tersedia", "summary": "Install feedparser: pip install feedparser",
                     "source": "System", "published": "", "relevance": 0}]

        all_news = []
        for source_name, feed_url in NEWS_FEEDS.items():
            try:
                items = self._fetch_feed(feed_url, source_name)
                all_news.extend(items)
            except Exception as e:
                logger.debug(f"Feed error ({source_name}): {e}")

        # Filter dan score berdasarkan relevansi
        if symbols:
            all_news = self._filter_by_symbols(all_news, symbols)
        else:
            all_news = self._score_all(all_news)

        # Sort by relevance, ambil top N
        all_news.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        result = all_news[:max_items]

        self._last_fetch = time.time()
        self._cached_news = result
        return result

    def _fetch_feed(self, url: str, source: str) -> list[dict]:
        """Fetch RSS feed. Gunakan feedparser jika tersedia, fallback ke requests."""
        items = []

        if FEEDPARSER_AVAILABLE:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                items.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", entry.get("description", ""))[:300],
                    "source": source,
                    "published": entry.get("published", ""),
                    "relevance": 0,
                })
        elif REQUESTS_AVAILABLE:
            # Simple XML parse tanpa library
            try:
                resp = requests.get(url, timeout=5, headers={"User-Agent": "JEANIROTRABOT/1.0"})
                if resp.status_code == 200:
                    text = resp.text
                    # Extract <title> tags (basic parser)
                    import re
                    titles = re.findall(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", text, re.DOTALL)
                    descs = re.findall(r"<description>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</description>", text, re.DOTALL)
                    for i, title in enumerate(titles[1:11]):  # skip first (channel title)
                        items.append({
                            "title": title.strip(),
                            "summary": descs[i + 1].strip()[:300] if i + 1 < len(descs) else "",
                            "source": source,
                            "published": "",
                            "relevance": 0,
                        })
            except Exception as e:
                logger.debug(f"Requests fetch error: {e}")

        return items

    def _filter_by_symbols(self, news_list: list, symbols: list[str]) -> list[dict]:
        """Filter berita yang relevan dengan simbol aktif trading."""
        symbol_keywords = []
        for sym in symbols:
            sym_lower = sym.lower()
            # Ekstrak currency codes dari pair (EURUSD -> EUR, USD)
            if len(sym) >= 6:
                symbol_keywords.extend([sym_lower[:3], sym_lower[3:6], sym_lower])
            else:
                symbol_keywords.append(sym_lower)

        # Tambah keyword umum
        for category in MARKET_KEYWORDS.values():
            symbol_keywords.extend(category)

        scored = []
        for item in news_list:
            text = (item["title"] + " " + item["summary"]).lower()
            score = sum(1 for kw in symbol_keywords if kw in text)
            item["relevance"] = score
            if score > 0:
                scored.append(item)

        return scored if scored else news_list[:10]

    def _score_all(self, news_list: list) -> list[dict]:
        """Score semua berita tanpa filter simbol spesifik."""
        all_keywords = []
        for kws in MARKET_KEYWORDS.values():
            all_keywords.extend(kws)

        for item in news_list:
            text = (item["title"] + " " + item["summary"]).lower()
            item["relevance"] = sum(1 for kw in all_keywords if kw in text)

        return news_list

    def analyze_news_with_ai(self, news_items: list[dict], ai_agent) -> dict:
        """
        Kirim berita ke AI untuk analisis sentimen dan signal trading.
        Returns: {"sentiment": "BULLISH|BEARISH|NEUTRAL", "key_events": [], "signal": "BUY|SELL|HOLD", "summary": str}
        """
        if not news_items or not ai_agent or not ai_agent.enabled:
            return {"sentiment": "NEUTRAL", "key_events": [], "signal": "HOLD",
                    "summary": "Tidak ada berita atau AI tidak aktif."}

        # Build news context
        news_text = "=== BERITA PASAR TERKINI ===\n"
        for i, item in enumerate(news_items[:10], 1):
            news_text += f"{i}. [{item['source']}] {item['title']}\n"
            if item.get("summary"):
                news_text += f"   {item['summary'][:150]}\n"

        news_prompt = (
            f"{news_text}\n"
            "Analisis berita di atas dan berikan:\n"
            "1. Sentimen pasar keseluruhan (BULLISH/BEARISH/NEUTRAL)\n"
            "2. Event kunci yang perlu diperhatikan trader\n"
            "3. Rekomendasi trading berdasarkan sentimen berita\n\n"
            "Respond dalam JSON:\n"
            '{"sentiment": "BULLISH|BEARISH|NEUTRAL", '
            '"key_events": ["event1", "event2"], '
            '"signal": "BUY|SELL|HOLD", '
            '"summary": "ringkasan singkat"}'
        )

        try:
            result = ai_agent.analyze_raw(news_prompt)
            if isinstance(result, dict):
                return result
            return {"sentiment": "NEUTRAL", "key_events": [], "signal": "HOLD",
                    "summary": str(result)[:200]}
        except Exception as e:
            logger.error(f"News AI analysis error: {e}")
            return {"sentiment": "NEUTRAL", "key_events": [], "signal": "HOLD",
                    "summary": f"Error: {e}"}

    def get_cached_news(self) -> list[dict]:
        return self._cached_news
