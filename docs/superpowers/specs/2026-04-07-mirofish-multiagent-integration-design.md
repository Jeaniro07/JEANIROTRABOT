# Design: MiroFish Multi-Agent Integration ke JEANIROTRABOT

**Tanggal:** 2026-04-07
**Status:** Approved
**Referensi:** https://github.com/666ghj/MiroFish

---

## 1. Latar Belakang & Tujuan

JEANIROTRABOT saat ini menggunakan satu AI agent tunggal untuk memutuskan BUY/SELL/HOLD berdasarkan hanya SMA(20), SMA(50), dan RSI(14). Keputusan sepenuhnya bergantung pada satu panggilan LLM per cycle, yang lambat dan tidak memanfaatkan beragam perspektif indikator.

MiroFish (github.com/666ghj/MiroFish) adalah platform simulasi multi-agent berbasis OASIS (CAMEL-AI) yang menjalankan pool agent dengan kepribadian independen, lalu menggunakan ReportAgent ber-pola ReACT untuk merangkum dan memutuskan.

**Tujuan revisi ini:** Mengadaptasi pola multi-agent MiroFish langsung ke dalam JEANIROTRABOT agar keputusan trading jangka pendek lebih cepat, lebih akurat, dan lebih hemat biaya API.

---

## 2. Pendekatan yang Dipilih

**Pendekatan 2 — MiroFish Port (Full Architecture):**

- 2 module baru: `specialist_agents.py` (agent pool) dan `composer_agent.py` (ReACT-style Composer)
- Hybrid LLM: LLM hanya dipanggil saat sinyal conflict (2-2 split atau score tipis)
- Tidak ada dependency baru — semua pakai pandas/numpy yang sudah ada
- 1 aplikasi (tidak perlu service terpisah)

---

## 3. Arsitektur

### 3.1 Komponen Baru & Dimodifikasi

| File | Status | Fungsi |
|------|--------|--------|
| `modules/specialist_agents.py` | BARU | 4 specialist agents + SpecialistPool |
| `modules/composer_agent.py` | BARU | Conflict detection + Hybrid LLM decide |
| `modules/trading_engine.py` | MODIFIKASI | Integrasi pool + composer, tambah indikator |
| `modules/chart.py` | MODIFIKASI | Panel MACD + Bollinger Bands |
| `modules/gui.py` | MODIFIKASI | Panel "Agent Signals" real-time |
| `requirements.txt` | MODIFIKASI | Tidak ada dependency baru |

### 3.2 Data Flow

```
MT5Connector.get_ohlcv()
  → compute_all_indicators()       [SMA,EMA,RSI,MACD,Stoch,BB,ATR,OBV]
  → SpecialistPool.analyze()       [4 agents, sekuensial, ≤ 5ms total]
  → ComposerAgent.decide()
       IF consensus ≥3-1: keputusan langsung (tanpa LLM)
       IF conflict 2-2:   AIAgent.analyze_multi() → LLM call
  → RiskManager.validate()
  → MT5Connector.execute()
  → GUI.update_agent_panel()
```

---

## 4. Specialist Agents

### Interface Seragam

```python
@dataclass
class AgentSignal:
    agent_name: str
    action: str        # "BUY" | "SELL" | "HOLD"
    score: float       # -1.0 (kuat SELL) → +1.0 (kuat BUY)
    confidence: float  # 0.0 → 1.0
    reasons: list[str]
    error: bool = False

class BaseSpecialistAgent:
    def analyze(self, indicators: dict, tick: dict, symbol_info: dict) -> AgentSignal: ...
```

### TrendAgent

- **Indikator:** SMA20, SMA50, EMA9
- **Logika BUY:** EMA9 > SMA20 > SMA50 (full alignment up)
- **Logika SELL:** EMA9 < SMA20 < SMA50 (full alignment down)
- **Bonus score:** Golden cross / death cross baru (3 candle terakhir)
- **HOLD:** Sinyal bertentangan atau alignment tidak sempurna

### MomentumAgent

- **Indikator:** RSI(14), MACD(12,26,9), Stochastic(14,3)
- **RSI:** BUY jika < 40 dan naik, SELL jika > 60 dan turun
- **MACD:** BUY jika histogram positif dan cross signal line
- **Stochastic:** BUY jika %K < 20 cross %D, SELL jika %K > 80 cross %D
- **Score:** Weighted average (RSI 30%, MACD 40%, Stoch 30%)

### VolatilityAgent

- **Indikator:** Bollinger Bands(20,2), ATR(14)
- **BUY:** Harga bounce dari lower band + band width menyempit (squeeze)
- **SELL:** Harga bounce dari upper band
- **HOLD:** Harga di tengah band (BB_mid ± 30% width)
- **ATR filter:** ATR terlalu kecil (< 0.3× average ATR) → downgrade confidence 50%

### VolumeAgent

- **Indikator:** OBV, Volume SMA(20)
- **BUY:** OBV rising + harga rising = konfirmasi (boost confidence +0.2)
- **SELL:** OBV falling + harga falling = konfirmasi
- **Divergence warning:** OBV naik tapi harga turun (atau sebaliknya) → kurangi confidence 0.3
- **Volume spike:** Volume > 2× SMA → boost confidence +0.1

---

## 5. ComposerAgent (Hybrid Decision)

### Conflict Detection

```
votes = [agent.action for agent in signals]

CONSENSUS (tanpa LLM):
  count(BUY)  >= 3  → final BUY
  count(SELL) >= 3  → final SELL
  count(HOLD) >= 3  → final HOLD

CONFLICT (panggil LLM):
  count(BUY) == 2 AND count(SELL) == 2
  count(BUY) == 2 AND count(SELL) == 1 AND avg_score < 0.3
  (split tidak tegas + score rendah)
```

### Scoring (consensus path)

```
final_score = mean(score × confidence for agents agreeing with majority)
final_confidence = mean(confidence for agents agreeing with majority)
```

### LLM Prompt (conflict path)

Composer mengirim ke LLM konteks lengkap berformat:
```
=== AGENT SIGNALS (CONFLICT) ===
TrendAgent:     BUY  score=+0.78 conf=0.82 — "EMA9>SMA20>SMA50"
MomentumAgent:  SELL score=-0.61 conf=0.70 — "RSI overbought, MACD cross down"
VolatilityAgent: BUY score=+0.32 conf=0.45 — "BB lower bounce"
VolumeAgent:    SELL score=-0.55 conf=0.68 — "OBV diverging"

=== MARKET CONTEXT ===
[standard market data: bid, ask, spread, account, positions, risk]

Resolve this conflict and provide your trading decision as JSON.
```

### Fallback Cascade

```
Normal:    4 Agents → Consensus        → Execute         (~80% siklus)
Conflict:  4 Agents → Conflict → LLM  → Execute         (~15% siklus)
Degraded:  ≤2 Agents error → survivors decide           (~4% siklus)
Safe:      ≥3 Agents error → HOLD (tidak eksekusi)      (~1% siklus)
LLM fail:  timeout/error → majority vote tanpa LLM
```

---

## 6. Indikator Baru

| Indikator | Parameter | Fungsi |
|-----------|-----------|--------|
| EMA | 9 | Trend cepat (TrendAgent) |
| MACD | 12, 26, 9 | Momentum (MomentumAgent) |
| Stochastic | 14, 3 | Overbought/oversold (MomentumAgent) |
| Bollinger Bands | 20, 2 | Volatility + breakout (VolatilityAgent) |
| ATR | 14 | Volatility filter (VolatilityAgent) |
| OBV | — | Volume trend (VolumeAgent) |
| Volume SMA | 20 | Volume baseline (VolumeAgent) |

Semua dihitung menggunakan pandas/numpy (tidak ada library TA baru).

---

## 7. GUI Panel Baru

Panel "🤖 Agent Signals" ditambahkan di dashboard:

```
┌──────────────────────────────────────────┐
│  🤖 Agent Signals — EURUSD M5            │
├──────────────┬────────┬──────────────────┤
│ Agent        │ Signal │ Score            │
├──────────────┼────────┼──────────────────┤
│ 📈 Trend     │ BUY ✅  │ ████████░░ 0.78  │
│ ⚡ Momentum  │ BUY ✅  │ ██████░░░░ 0.61  │
│ 🌊 Volatility│ HOLD ⚠️│ ███░░░░░░░ 0.32  │
│ 📊 Volume    │ BUY ✅  │ ███████░░░ 0.70  │
├──────────────┼────────┼──────────────────┤
│ 🎯 COMPOSER  │ BUY 🚀 │ Consensus (3-1)  │
│              │        │ No LLM needed    │
└──────────────┴────────┴──────────────────┘
```

Status badge: `CONSENSUS` (hijau) | `CONFLICT → LLM` (kuning) | `SAFE MODE` (merah)

---

## 8. Chart Baru

- **MACD panel:** histogram + signal line di panel bawah (menggantikan atau berdampingan RSI)
- **Bollinger Bands overlay:** upper/lower band di atas candlestick
- Toggle per indikator tersedia di GUI

---

## 9. Error Handling

- Setiap specialist agent punya try/except mandiri — agent error return HOLD tanpa crash
- Indicator NaN > 20% → skip cycle
- LLM timeout (10 detik) → fallback ke majority vote
- LLM unavailable (no API key) → consensus-only mode otomatis
- GUI error tidak mempengaruhi trading engine (thread-safe callback)

---

## 10. Estimasi Waktu Per Cycle

| Kondisi | Estimasi |
|---------|---------|
| Consensus (tanpa LLM) | 150–350ms |
| Conflict (dengan LLM) | 1,200–3,500ms |
| Target timeframe | M1, M5, M15 |

---

## 11. Logging

Setiap cycle menghasilkan log terstruktur:
```
[TREND]      BUY  score=+0.78 conf=0.82 — "EMA9>SMA20>SMA50"
[MOMENTUM]   BUY  score=+0.61 conf=0.70 — "RSI rising, MACD positive"
[VOLATILITY] HOLD score=+0.32 conf=0.45 — "BB mid-range"
[VOLUME]     BUY  score=+0.70 conf=0.75 — "OBV rising"
[COMPOSER]   CONSENSUS 3-1 BUY — no LLM, score=+0.66 conf=0.76
[EXECUTE]    BUY 0.02 EURUSD SL=100 TP=200
```

---

## 12. Tidak Dalam Scope

- Integrasi Zep Cloud / GraphRAG (MiroFish full stack)
- Simulasi "debate rounds" iteratif (Pendekatan 3)
- Exchange Indonesia / Indodax (scope terpisah)
- Backtesting / paper trading mode
