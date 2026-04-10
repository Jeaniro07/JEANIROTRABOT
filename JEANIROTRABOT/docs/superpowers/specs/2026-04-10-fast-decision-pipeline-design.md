# Fast Decision Pipeline — Design Spec
**Date:** 2026-04-10
**Project:** JEANIROTRABOT
**Status:** Approved

---

## Problem Statement

Dengan 4-5+ symbol aktif, pipeline keputusan BUY/SELL saat ini terlalu lambat untuk
swing trading 10-15 detik. Penyebab utama:

1. **Sequential symbol processing** — 5 symbol diproses satu per satu; satu LLM call
   yang lambat memblokir semua symbol berikutnya.
2. **LLM conflict call blocking main loop** — timeout 10 detik, terjadi ~30-40% siklus
   di pasar mixed, menyebabkan siklus penuh bisa 12-60 detik.
3. **Interval loop 5 detik** — meski proses cepat, keputusan tidak bisa lebih sering
   dari 5 detik.
4. **Tidak ada caching** — HOLD berulang tetap menjalankan seluruh pipeline setiap
   siklus, membuang waktu komputasi.

**Target:** Keputusan efektif per symbol setiap 10-15 detik, worst-case (dengan 1 LLM
conflict call) tidak lebih dari 10 detik per siklus penuh.

---

## Approach: Decision Cache + Reduced Interval (Approach B)

Optimasi inkremental pada kode yang sudah ada — tanpa mengubah arsitektur threading,
tanpa rewrite async. Tiga perubahan terfokus:

1. `DecisionCache` — cache hasil keputusan per symbol
2. LLM timeout 5s + graceful fallback ke majority vote
3. Interval loop 2s + config parameter baru

---

## Architecture

### Cycle Flow (Sesudah)

```
Loop setiap 2 detik:
  untuk setiap symbol (sequential):
    ┌─ Cache HIT (< 30s) ─────────────────────────────────┐
    │  action=HOLD  → return langsung (0ms)               │
    │  action=BUY/SELL → verify indicators → execute      │
    └──────────────────────────────────────────────────────┘
    ┌─ Cache MISS / expired ──────────────────────────────┐
    │  1. Fetch OHLCV (MT5)                               │
    │  2. compute_all_indicators()                        │
    │  3. SpecialistPool.analyze() → 4 signals            │
    │  4. ComposerAgent.decide()                          │
    │     ├─ CONSENSUS (≥3-1, score≥0.30) → langsung      │
    │     └─ CONFLICT → LLM call (timeout=5s)             │
    │                   └─ Timeout/Error → MAJORITY_FALLBACK│
    │  5. Simpan ke DecisionCache (TTL=30s)               │
    │  6. Execute jika BUY/SELL                           │
    └──────────────────────────────────────────────────────┘
```

### Latency Projection (5 symbol)

| Kondisi | Sebelum | Sesudah |
|---------|---------|---------|
| Semua HOLD (cache hit) | ~10s | ~0.1s |
| Normal, tidak ada conflict | ~10s | ~2s |
| 1 symbol LLM conflict | ~22s | ~7s |
| Worst-case semua conflict | ~60s | ~27s |
| Keputusan per symbol (efektif) | 10s | 2-4s |

---

## Components

### 1. DecisionCache (`trading_engine.py`)

**Struktur data:**
```python
# Di TradingEngine.__init__()
self._decision_cache: dict[str, tuple[FinalDecision, float]] = {}
# key   = symbol string (mis. "EURUSD")
# value = (FinalDecision, timestamp_unix)
```

**Config baru di `config.py`:**
```python
DECISION_CACHE_TTL = 30   # detik — keputusan berlaku 30 detik
```

**Logic di `_process_symbol()`:**
```python
def _get_cached_decision(self, symbol) -> FinalDecision | None:
    entry = self._decision_cache.get(symbol)
    if entry is None:
        return None
    decision, ts = entry
    if time.time() - ts > self._cache_ttl:
        del self._decision_cache[symbol]
        return None
    return decision

def _cache_decision(self, symbol, decision):
    self._decision_cache[symbol] = (decision, time.time())
```

**Aturan cache hit:**
- `HOLD` → return langsung, tidak jalankan pipeline
- `BUY`/`SELL` → jalankan indicators + agents untuk re-verify; skip LLM (composer
  akan consensus karena sinyal seharusnya konsisten); execute jika masih sepakat

**Invalidasi cache:**
- TTL habis (30 detik)
- Action berubah dari BUY/SELL → HOLD (atau sebaliknya) dalam re-verify

---

### 2. LLM Timeout + Fallback (`composer_agent.py` + `ai_agent.py`)

**`composer_agent.py`:**
```python
# Turunkan dari 10.0 → 5.0
LLM_DEFAULT_TIMEOUT = 5.0

# _llm_decide() dibungkus exception handler
def _llm_decide(self, signals, market_context, ai_agent, vote_summary):
    try:
        result = ai_agent.analyze_multi(
            conflict_summary=...,
            timeout=LLM_DEFAULT_TIMEOUT,
        )
        # parse result → FinalDecision(mode="CONFLICT_LLM")
        return self._parse_llm_result(result, signals, vote_summary)
    except Exception:
        logger.warning("[Composer] LLM timeout/error → MAJORITY_FALLBACK")
        return self._majority_fallback(signals, vote_summary)
```

**`ai_agent.py`:**
```python
# Tambah parameter timeout ke analyze_multi()
def analyze_multi(self, conflict_summary, symbol, timeframe,
                  ..., timeout: float = 5.0) -> dict:
    response = self.client.chat.completions.create(
        model=self.model,
        messages=[...],
        max_tokens=400,
        temperature=0.2,
        timeout=timeout,   # ← tambah ini
    )
```

**Behavior:**
- LLM respons dalam < 5s → `CONFLICT_LLM` (akurat)
- LLM timeout/error → `MAJORITY_FALLBACK` (cepat, instan)
- Tidak ada silent hang, tidak ada 60 detik default library timeout

---

### 3. Interval + Config (`trading_engine.py` + `config.py`)

**`config.py` — tambah defaults:**
```python
TRADE_INTERVAL       = 2    # detik (loop interval, dari 5)
DECISION_CACHE_TTL   = 30   # detik (cache validity)
LLM_CONFLICT_TIMEOUT = 5    # detik (timeout LLM call)
```

**`trading_engine.py`:**
```python
# __init__() — baca dari config
self._interval   = config.get_int("TRADE_INTERVAL", 2)
self._cache_ttl  = config.get_int("DECISION_CACHE_TTL", 30)
```

---

## Data Flow Lengkap

```
TradingEngine._run_loop() [setiap 2s]
  │
  └─► for symbol in self._symbols:
        │
        ├─ _get_cached_decision(symbol)
        │    │
        │    ├─ HIT + HOLD → RETURN (skip semua)
        │    │
        │    └─ HIT + BUY/SELL → jalankan indicators+agents
        │                         skip LLM, re-verify saja
        │
        └─ MISS → pipeline penuh:
              │
              ├─ connector.get_ohlcv()          [~100-300ms]
              ├─ compute_all_indicators()        [~50-100ms]
              ├─ SpecialistPool.analyze()        [~20-50ms]
              ├─ ComposerAgent.decide()
              │    ├─ CONSENSUS → FinalDecision  [<5ms]
              │    └─ CONFLICT → ai_agent.analyze_multi(timeout=5s)
              │                   ├─ OK → CONFLICT_LLM  [<5s]
              │                   └─ Timeout → MAJORITY_FALLBACK [instan]
              │
              ├─ _cache_decision(symbol, decision)
              └─ _execute_decision(decision)
```

---

## Error Handling

| Skenario | Behavior |
|----------|----------|
| LLM timeout (>5s) | Fallback ke `MAJORITY_FALLBACK`, log warning |
| LLM error (API/network) | Sama seperti timeout |
| Cache stale saat market gap | TTL 30s cukup pendek, re-fetch otomatis |
| Semua agents error | `SAFE_HOLD` dari ComposerAgent (tidak berubah) |
| MT5 disconnected | Early return di `_process_symbol()` (tidak berubah) |

---

## Testing

Semua test yang ada (56 tests) harus tetap passing.

Test baru yang perlu ditambah:
1. `test_decision_cache_hold_skips_pipeline` — HOLD dari cache tidak memanggil SpecialistPool
2. `test_decision_cache_ttl_expires` — setelah TTL habis, pipeline dijalankan ulang
3. `test_decision_cache_buysell_reverifies` — BUY dari cache tetap re-verify indicators
4. `test_llm_timeout_fallback_to_majority` — timeout → MAJORITY_FALLBACK, bukan exception
5. `test_analyze_multi_timeout_param` — timeout parameter diteruskan ke OpenAI call

---

## Files Yang Diubah

| File | Perubahan |
|------|-----------|
| `modules/trading_engine.py` | Tambah `_decision_cache`, `_get_cached_decision()`, `_cache_decision()`, ubah interval default 5→2, baca config TTL |
| `modules/composer_agent.py` | Ubah `LLM_DEFAULT_TIMEOUT` 10→5, bungkus `_llm_decide()` dengan try/except+fallback |
| `modules/ai_agent.py` | Tambah parameter `timeout` ke `analyze_multi()` |
| `modules/config.py` | Tambah `TRADE_INTERVAL`, `DECISION_CACHE_TTL`, `LLM_CONFLICT_TIMEOUT` |
| `tests/test_decision_cache.py` | File test baru (5 test cases) |

**File yang TIDAK diubah:** `specialist_agents.py`, `chart.py`, `gui.py`, `mt5_connector.py`, `risk_manager.py`

---

## Success Criteria

- [ ] Full cycle 5 symbol tanpa conflict: ≤ 2 detik
- [ ] Full cycle dengan 1 LLM conflict call: ≤ 8 detik
- [ ] HOLD cache hit: ≤ 0.2 detik
- [ ] LLM timeout tidak pernah menggantung > 5 detik
- [ ] Semua 56 test lama + 5 test baru = 61 test passing
- [ ] Tidak ada perubahan pada antarmuka GUI
