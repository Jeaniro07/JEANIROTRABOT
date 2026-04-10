# Fast Decision Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Percepat pipeline keputusan BUY/SELL dari ~10-60 detik menjadi 2-8 detik dengan DecisionCache, LLM timeout 5s, dan interval loop 2s.

**Architecture:** Tiga perubahan inkremental tanpa mengubah arsitektur threading — tambah DecisionCache di TradingEngine untuk skip pipeline pada HOLD berulang, tambah `timeout` parameter ke `analyze_multi()` agar LLM tidak hang >5s, dan kurangi interval loop dari 5s ke 2s dengan config yang dapat diatur.

**Tech Stack:** Python 3.14, pytest, openai SDK (timeout parameter), threading (sudah ada).

---

## File Map

| File | Aksi | Perubahan |
|------|------|-----------|
| `JEANIROTRABOT/modules/config.py` | Modify | Tambah 3 key baru ke `DEFAULTS` dict |
| `JEANIROTRABOT/modules/ai_agent.py` | Modify | Tambah `timeout: float = 5.0` param ke `analyze_multi()` |
| `JEANIROTRABOT/modules/composer_agent.py` | Modify | Ubah `LLM_DEFAULT_TIMEOUT` 10→5, teruskan timeout ke `analyze_multi()` |
| `JEANIROTRABOT/modules/trading_engine.py` | Modify | Tambah `_decision_cache`, `_get_cached_decision()`, `_cache_decision()`, ubah `_process_symbol()`, baca interval+TTL dari config |
| `JEANIROTRABOT/tests/test_decision_cache.py` | Create | 5 unit tests untuk cache logic dan LLM timeout fallback |

---

## Task 1: Config Defaults

**Files:**
- Modify: `JEANIROTRABOT/modules/config.py:27-74` (DEFAULTS dict)

- [ ] **Step 1: Tambah 3 key ke DEFAULTS dict**

Buka `JEANIROTRABOT/modules/config.py`. Cari blok `DEFAULTS = {` (line 27). Di bagian akhir dict, setelah baris `"COMPOSER_INTERVAL": "5",` tambahkan:

```python
    # Fast decision pipeline
    "TRADE_INTERVAL":       "2",   # detik antar siklus loop (dari 5)
    "DECISION_CACHE_TTL":   "30",  # detik keputusan berlaku di cache
    "LLM_CONFLICT_TIMEOUT": "5",   # detik timeout LLM conflict call
```

- [ ] **Step 2: Verifikasi get_int() sudah ada**

`get_int()` ada di line 113. Tidak perlu diubah — sudah bisa baca key baru.

- [ ] **Step 3: Commit**

```bash
git add JEANIROTRABOT/modules/config.py
git commit -m "feat: add TRADE_INTERVAL, DECISION_CACHE_TTL, LLM_CONFLICT_TIMEOUT config defaults"
```

---

## Task 2: LLM Timeout di ai_agent.py

**Files:**
- Modify: `JEANIROTRABOT/modules/ai_agent.py:422-507`
- Test: `JEANIROTRABOT/tests/test_decision_cache.py`

- [ ] **Step 1: Tulis failing test**

Buat file `JEANIROTRABOT/tests/test_decision_cache.py`:

```python
"""
Tests untuk Fast Decision Pipeline:
  - DecisionCache logic (Tasks 3-4)
  - LLM timeout parameter (Task 2)
  - LLM timeout fallback ke MAJORITY_FALLBACK (Task 2)
"""
import time
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from modules.composer_agent import ComposerAgent, FinalDecision
from modules.specialist_agents import AgentSignal


# ──────────────────────────────────────────────
# Helper factories
# ──────────────────────────────────────────────

def _make_signal(name, action, score=0.6, conf=0.7, error=False):
    return AgentSignal(
        agent_name=name, action=action,
        score=score if action == "BUY" else (-score if action == "SELL" else 0.0),
        confidence=conf, error=error,
    )


def _conflict_signals():
    """2 BUY vs 2 SELL — guaranteed conflict."""
    return [
        _make_signal("TrendAgent",      "BUY",  0.6),
        _make_signal("MomentumAgent",   "BUY",  0.6),
        _make_signal("VolatilityAgent", "SELL", 0.6),
        _make_signal("VolumeAgent",     "SELL", 0.6),
    ]


# ──────────────────────────────────────────────
# Task 2 tests — LLM timeout
# ──────────────────────────────────────────────

class TestAnalyzeMultiTimeout:
    """analyze_multi() harus menerima dan meneruskan parameter timeout."""

    def test_analyze_multi_accepts_timeout_param(self):
        """analyze_multi() tidak boleh raise TypeError saat diberi timeout=5.0."""
        from modules.ai_agent import AIAgent
        agent = AIAgent(provider="OpenAI", api_key="fake-key")
        # Patch _get_client agar tidak butuh koneksi nyata
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"action":"HOLD","confidence":0.5,"reason":"test"}'))]
        mock_client.chat.completions.create.return_value = mock_response
        with patch.object(agent, '_get_client', return_value=mock_client):
            result = agent.analyze_multi(
                conflict_summary="test conflict",
                symbol="EURUSD",
                timeout=5.0,   # ← parameter yang harus diterima
            )
        assert isinstance(result, dict)

    def test_analyze_multi_passes_timeout_to_openai(self):
        """timeout=5.0 harus diteruskan ke client.chat.completions.create()."""
        from modules.ai_agent import AIAgent
        agent = AIAgent(provider="OpenAI", api_key="fake-key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"action":"HOLD","confidence":0.5,"reason":"ok"}'))]
        mock_client.chat.completions.create.return_value = mock_response
        with patch.object(agent, '_get_client', return_value=mock_client):
            agent.analyze_multi(conflict_summary="x", symbol="EURUSD", timeout=3.0)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("timeout") == 3.0, (
            f"timeout tidak diteruskan ke openai, call_kwargs={call_kwargs}"
        )


class TestLlmTimeoutFallback:
    """ComposerAgent._llm_decide() harus fallback ke MAJORITY_FALLBACK saat LLM timeout."""

    def test_llm_timeout_returns_majority_fallback(self):
        """Jika LLM raise Exception (misal Timeout), hasilnya MAJORITY_FALLBACK bukan crash."""
        composer = ComposerAgent()
        signals = _conflict_signals()

        mock_ai = MagicMock()
        mock_ai.enabled = True
        mock_ai.analyze_multi.side_effect = TimeoutError("LLM timed out")

        market_context = {"symbol": "EURUSD", "timeframe": "M15"}
        decision = composer.decide(signals, market_context, ai_agent=mock_ai)

        assert decision.action in ("BUY", "SELL", "HOLD"), (
            f"action harus valid, dapat: {decision.action}"
        )
        assert decision.mode == "MAJORITY_FALLBACK", (
            f"mode harus MAJORITY_FALLBACK, dapat: {decision.mode}"
        )

    def test_llm_network_error_returns_majority_fallback(self):
        """Exception apapun dari LLM harus fallback ke MAJORITY_FALLBACK."""
        composer = ComposerAgent()
        signals = _conflict_signals()

        mock_ai = MagicMock()
        mock_ai.enabled = True
        mock_ai.analyze_multi.side_effect = ConnectionError("network error")

        decision = composer.decide(signals, {"symbol": "EURUSD"}, ai_agent=mock_ai)
        assert decision.mode == "MAJORITY_FALLBACK"
```

- [ ] **Step 2: Jalankan test — harus FAIL**

```bash
cd JEANIROTRABOT
C:\Users\jeani\AppData\Local\Programs\Python\Python314\python.exe -m pytest tests/test_decision_cache.py::TestAnalyzeMultiTimeout -v
```

Expected: `FAILED` — `TypeError: analyze_multi() got an unexpected keyword argument 'timeout'`

- [ ] **Step 3: Tambah parameter `timeout` ke `analyze_multi()` di ai_agent.py**

Buka `JEANIROTRABOT/modules/ai_agent.py` line 422. Ubah signature `analyze_multi()`:

```python
def analyze_multi(
    self,
    conflict_summary: str,
    symbol: str = "",
    timeframe: str = "",
    ohlcv_summary: str = "",
    sma20: float = 0.0,
    sma50: float = 0.0,
    rsi: float = 50.0,
    bid: float = 0.0,
    ask: float = 0.0,
    account_equity: float = 0.0,
    account_balance: float = 0.0,
    free_margin: float = 0.0,
    open_positions: list = None,
    spread: float = 0.0,
    risk_status: dict = None,
    symbol_info: dict = None,
    timeout: float = 5.0,   # ← tambah baris ini
    **kwargs,
) -> dict:
```

Lalu cari `client.chat.completions.create(` di dalam `analyze_multi()` (sekitar line 499). Tambah `timeout=timeout`:

```python
response = self._get_client().chat.completions.create(
    model=self._model,
    messages=[
        {"role": "system", "content": self._system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    max_tokens=400,
    temperature=0.2,
    timeout=timeout,   # ← tambah baris ini
)
```

- [ ] **Step 4: Jalankan test — harus PASS**

```bash
C:\Users\jeani\AppData\Local\Programs\Python\Python314\python.exe -m pytest tests/test_decision_cache.py::TestAnalyzeMultiTimeout -v
```

Expected:
```
PASSED tests/test_decision_cache.py::TestAnalyzeMultiTimeout::test_analyze_multi_accepts_timeout_param
PASSED tests/test_decision_cache.py::TestAnalyzeMultiTimeout::test_analyze_multi_passes_timeout_to_openai
```

- [ ] **Step 5: Commit**

```bash
git add JEANIROTRABOT/modules/ai_agent.py JEANIROTRABOT/tests/test_decision_cache.py
git commit -m "feat: add timeout param to analyze_multi() — passed to openai call"
```

---

## Task 3: LLM Timeout di composer_agent.py

**Files:**
- Modify: `JEANIROTRABOT/modules/composer_agent.py:26,173-217`
- Test: `JEANIROTRABOT/tests/test_decision_cache.py::TestLlmTimeoutFallback`

- [ ] **Step 1: Jalankan test fallback — harus FAIL dulu**

```bash
C:\Users\jeani\AppData\Local\Programs\Python\Python314\python.exe -m pytest tests/test_decision_cache.py::TestLlmTimeoutFallback -v
```

Cek apakah test sudah PASS (berarti fallback sudah ada) atau FAIL. Jika sudah PASS, lanjut ke Step 3.

- [ ] **Step 2: Ubah `LLM_DEFAULT_TIMEOUT` dan teruskan timeout ke `analyze_multi()`**

Buka `JEANIROTRABOT/modules/composer_agent.py`.

**Baris 26** — ubah timeout:
```python
LLM_DEFAULT_TIMEOUT = 5.0        # detik (dari 10.0)
```

**Method `_llm_decide()`** (lines 173-217) — teruskan timeout ke `analyze_multi()`. Ubah baris `result = ai_agent.analyze_multi(...)`:

```python
    try:
        result = ai_agent.analyze_multi(
            conflict_summary=conflict_summary,
            timeout=LLM_DEFAULT_TIMEOUT,   # ← tambah baris ini
            **{k: v for k, v in market_context.items()},
        )
```

Bagian `except Exception` sudah ada dan sudah call `_majority_fallback` — tidak perlu diubah.

- [ ] **Step 3: Jalankan test fallback — harus PASS**

```bash
C:\Users\jeani\AppData\Local\Programs\Python\Python314\python.exe -m pytest tests/test_decision_cache.py::TestLlmTimeoutFallback -v
```

Expected:
```
PASSED tests/test_decision_cache.py::TestLlmTimeoutFallback::test_llm_timeout_returns_majority_fallback
PASSED tests/test_decision_cache.py::TestLlmTimeoutFallback::test_llm_network_error_returns_majority_fallback
```

- [ ] **Step 4: Pastikan 56 test lama masih passing**

```bash
C:\Users\jeani\AppData\Local\Programs\Python\Python314\python.exe -m pytest tests/ -v --ignore=tests/test_decision_cache.py
```

Expected: `56 passed`

- [ ] **Step 5: Commit**

```bash
git add JEANIROTRABOT/modules/composer_agent.py
git commit -m "feat: reduce LLM_DEFAULT_TIMEOUT 10s→5s, pass timeout to analyze_multi()"
```

---

## Task 4: DecisionCache di trading_engine.py

**Files:**
- Modify: `JEANIROTRABOT/modules/trading_engine.py:178-237,827-919`
- Test: `JEANIROTRABOT/tests/test_decision_cache.py`

- [ ] **Step 1: Tambah 3 test cache ke test_decision_cache.py**

Buka `JEANIROTRABOT/tests/test_decision_cache.py`. Tambahkan di bagian akhir file:

```python
# ──────────────────────────────────────────────
# Task 4 tests — DecisionCache
# ──────────────────────────────────────────────

class TestDecisionCache:
    """_get_cached_decision() dan _cache_decision() harus bekerja dengan benar."""

    def _make_engine(self):
        """Buat TradingEngine minimal dengan semua dependency di-mock."""
        from modules.trading_engine import TradingEngine
        from modules.config import AppConfig
        mock_connector = MagicMock()
        mock_risk      = MagicMock()
        mock_ai        = MagicMock()
        mock_config    = MagicMock(spec=AppConfig)
        mock_config.get.return_value = ""
        mock_config.get_int.side_effect = lambda key, default=0: default
        mock_config.get_float.side_effect = lambda key, default=0.0: default
        mock_config.get_bool.side_effect = lambda key, default=False: default
        return TradingEngine(mock_connector, mock_risk, mock_ai, mock_config)

    def _make_hold_decision(self):
        return FinalDecision(
            action="HOLD", confidence=0.5, score=0.0,
            reason="test", mode="CONSENSUS", vote_summary="4 HOLD",
        )

    def _make_buy_decision(self):
        return FinalDecision(
            action="BUY", confidence=0.8, score=0.7,
            reason="test", mode="CONSENSUS", vote_summary="3 BUY 1 HOLD",
        )

    def test_cache_miss_returns_none(self):
        """Symbol yang belum pernah diproses harus return None."""
        engine = self._make_engine()
        result = engine._get_cached_decision("EURUSD")
        assert result is None

    def test_cache_hit_returns_decision(self):
        """Setelah _cache_decision(), _get_cached_decision() harus return decision yang sama."""
        engine = self._make_engine()
        decision = self._make_hold_decision()
        engine._cache_decision("EURUSD", decision)
        result = engine._get_cached_decision("EURUSD")
        assert result is not None
        assert result.action == "HOLD"
        assert result.mode == "CONSENSUS"

    def test_cache_ttl_expires(self):
        """Setelah TTL habis, _get_cached_decision() harus return None."""
        engine = self._make_engine()
        engine._cache_ttl = 1  # 1 detik untuk test cepat
        decision = self._make_hold_decision()
        engine._cache_decision("EURUSD", decision)
        time.sleep(1.1)  # tunggu TTL habis
        result = engine._get_cached_decision("EURUSD")
        assert result is None, "Cache harus expired setelah TTL"

    def test_different_symbols_cached_independently(self):
        """Cache EURUSD tidak boleh mempengaruhi cache XAUUSD."""
        engine = self._make_engine()
        engine._cache_decision("EURUSD", self._make_hold_decision())
        engine._cache_decision("XAUUSD", self._make_buy_decision())
        assert engine._get_cached_decision("EURUSD").action == "HOLD"
        assert engine._get_cached_decision("XAUUSD").action == "BUY"
        assert engine._get_cached_decision("BTCUSD") is None

    def test_cache_overwrite(self):
        """_cache_decision() yang dipanggil dua kali harus overwrite entry lama."""
        engine = self._make_engine()
        engine._cache_decision("EURUSD", self._make_hold_decision())
        engine._cache_decision("EURUSD", self._make_buy_decision())
        result = engine._get_cached_decision("EURUSD")
        assert result.action == "BUY", "Entry lama harus tertimpa"
```

- [ ] **Step 2: Jalankan test cache — harus FAIL**

```bash
C:\Users\jeani\AppData\Local\Programs\Python\Python314\python.exe -m pytest tests/test_decision_cache.py::TestDecisionCache -v
```

Expected: `FAILED` — `AttributeError: 'TradingEngine' object has no attribute '_get_cached_decision'`

- [ ] **Step 3: Tambah cache ke `__init__()` di trading_engine.py**

Buka `JEANIROTRABOT/modules/trading_engine.py`. Cari `self._interval = 5` (line 196). Ganti dengan:

```python
self._interval  = config.get_int("TRADE_INTERVAL", 2)    # detik (dari 5)
self._cache_ttl = config.get_int("DECISION_CACHE_TTL", 30)  # detik
self._decision_cache: dict = {}  # {symbol: (FinalDecision, timestamp)}
```

- [ ] **Step 4: Tambah method `_get_cached_decision()` dan `_cache_decision()`**

Cari method pertama setelah `__init__()` di `trading_engine.py` (biasanya `set_log_callback` atau `start()`). Tambahkan dua method baru SEBELUM method tersebut:

```python
def _get_cached_decision(self, symbol: str):
    """Return cached FinalDecision jika masih valid, else None."""
    entry = self._decision_cache.get(symbol)
    if entry is None:
        return None
    decision, ts = entry
    if time.time() - ts > self._cache_ttl:
        del self._decision_cache[symbol]
        return None
    return decision

def _cache_decision(self, symbol: str, decision) -> None:
    """Simpan decision ke cache dengan timestamp sekarang."""
    self._decision_cache[symbol] = (decision, time.time())
```

Pastikan `import time` sudah ada di bagian atas file. Jika belum ada, tambahkan.

- [ ] **Step 5: Jalankan test cache — harus PASS**

```bash
C:\Users\jeani\AppData\Local\Programs\Python\Python314\python.exe -m pytest tests/test_decision_cache.py::TestDecisionCache -v
```

Expected:
```
PASSED test_cache_miss_returns_none
PASSED test_cache_hit_returns_decision
PASSED test_cache_ttl_expires
PASSED test_different_symbols_cached_independently
PASSED test_cache_overwrite
```

- [ ] **Step 6: Integrasikan cache ke `_process_symbol()`**

Buka `trading_engine.py`. Cari method `_process_symbol()` (line 827). Tambahkan cache check di awal method, SETELAH validasi `df` dan indikator (setelah "Core indicators NaN" check) tapi SEBELUM `self._pool.analyze(...)`:

Cari blok ini (sekitar line 843-851):
```python
    # 3. Get market data
    tick = self.connector.get_tick(symbol)
    if not tick:
        return

    symbol_info = self.connector.get_symbol_info(symbol)
    symbol_positions = [p for p in all_positions if p["symbol"] == symbol]

    # 4. Run specialist pool (MiroFish agent pool style)
    signals = self._pool.analyze(indicators, tick, symbol_info)
```

Ubah menjadi:

```python
    # 3. Get market data
    tick = self.connector.get_tick(symbol)
    if not tick:
        return

    symbol_info = self.connector.get_symbol_info(symbol)
    symbol_positions = [p for p in all_positions if p["symbol"] == symbol]

    # 3b. Decision cache — skip pipeline jika HOLD masih berlaku
    cached = self._get_cached_decision(symbol)
    if cached is not None and cached.action == "HOLD":
        self._log(f"[{symbol}] Cache HIT: HOLD (mode={cached.mode}) — skip pipeline")
        return  # HOLD berulang tidak perlu recompute agents

    # 4. Run specialist pool (MiroFish agent pool style)
    signals = self._pool.analyze(indicators, tick, symbol_info)
```

Lalu tambahkan `_cache_decision()` call setelah `ComposerAgent.decide()` (setelah line 877 `decision = self._mirofish_composer.decide(...)`):

Cari:
```python
    # 7. Log + notify GUI
    self._log_agent_signals(symbol, signals, decision)
```

Tambahkan SEBELUM baris itu:

```python
    # 6b. Simpan keputusan ke cache
    self._cache_decision(symbol, decision)

    # 7. Log + notify GUI
    self._log_agent_signals(symbol, signals, decision)
```

- [ ] **Step 7: Jalankan semua test**

```bash
C:\Users\jeani\AppData\Local\Programs\Python\Python314\python.exe -m pytest tests/ -v
```

Expected: `61 passed` (56 lama + 5 baru di test_decision_cache.py)

- [ ] **Step 8: Commit**

```bash
git add JEANIROTRABOT/modules/trading_engine.py JEANIROTRABOT/tests/test_decision_cache.py
git commit -m "feat: add DecisionCache to TradingEngine — skip HOLD re-computation, TTL 30s"
```

---

## Task 5: Interval Loop + Smoke Test Final

**Files:**
- Verify: `JEANIROTRABOT/modules/trading_engine.py` (interval sudah diubah di Task 4)

- [ ] **Step 1: Verifikasi interval sudah terbaca dari config**

Pastikan di `__init__()` tidak ada lagi `self._interval = 5` hardcoded — sudah diganti di Task 4 Step 3. Jika masih ada, ubah:

```python
self._interval = config.get_int("TRADE_INTERVAL", 2)
```

- [ ] **Step 2: Pipeline smoke test (tanpa MT5)**

```bash
C:\Users\jeani\AppData\Local\Programs\Python\Python314\python.exe -c "
import pandas as pd
import numpy as np
import time

from modules.trading_engine import compute_all_indicators, TradingEngine
from modules.specialist_agents import SpecialistPool
from modules.composer_agent import ComposerAgent

n = 100
close = pd.Series(1.08 + np.cumsum(np.random.randn(n) * 0.0005))
high  = close + 0.0003
low   = close - 0.0003
vol   = pd.Series(np.random.randint(100,1000,n), dtype=float)
df = pd.DataFrame({'Open':close,'High':high,'Low':low,'Close':close,'Volume':vol})

# Test pipeline speed (5 symbols x 5 cycles)
pool     = SpecialistPool()
composer = ComposerAgent()

t0 = time.perf_counter()
for sym in ['EURUSD','XAUUSD','BTCUSD','ETHUSD','GBPUSD']:
    inds = compute_all_indicators(df)
    sigs = pool.analyze(inds)
    dec  = composer.decide(sigs, {'symbol': sym})
    print(f'{sym}: {dec.action} ({dec.mode}) score={dec.score:+.2f}')
elapsed = time.perf_counter() - t0
print(f'5 symbols pipeline: {elapsed*1000:.1f}ms (target <500ms)')
assert elapsed < 0.5, f'Pipeline terlalu lambat: {elapsed:.2f}s'
print('SMOKE TEST PASSED')
" 2>&1
```

Expected output:
```
EURUSD: ... (CONSENSUS/MAJORITY_FALLBACK) score=...
...
5 symbols pipeline: XXXms (target <500ms)
SMOKE TEST PASSED
```

- [ ] **Step 3: Jalankan full test suite**

```bash
C:\Users\jeani\AppData\Local\Programs\Python\Python314\python.exe -m pytest tests/ -v
```

Expected: `61 passed`

- [ ] **Step 4: Commit final**

```bash
git add -A
git commit -m "feat: fast decision pipeline complete — cache+timeout+interval (Tasks 1-5)

- config.py: TRADE_INTERVAL=2, DECISION_CACHE_TTL=30, LLM_CONFLICT_TIMEOUT=5
- ai_agent.py: timeout param di analyze_multi(), diteruskan ke openai call
- composer_agent.py: LLM_DEFAULT_TIMEOUT 10s→5s, timeout diteruskan
- trading_engine.py: DecisionCache (_get_cached_decision, _cache_decision),
  HOLD cache hit skip pipeline, interval baca dari config (default 2s)
- tests/test_decision_cache.py: 5+4=9 tests baru, 61 total passing

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Ringkasan Perubahan

| Komponen | Sebelum | Sesudah |
|----------|---------|---------|
| Loop interval | 5s hardcoded | 2s (dari config) |
| LLM timeout | 10s (hang hingga 60s) | 5s + graceful fallback |
| HOLD berulang | Full pipeline setiap siklus | Skip langsung (0ms) |
| Cycle 5 symbol tanpa conflict | ~10s | ~2s |
| Cycle dengan 1 LLM conflict | ~22s | ~7s |
| Tests | 56 | 61 |
