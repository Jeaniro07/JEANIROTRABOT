# MiroFish Multi-Agent Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adaptasi pola multi-agent MiroFish ke JEANIROTRABOT — 4 specialist agents (Trend, Momentum, Volatility, Volume) + ComposerAgent Hybrid (LLM hanya saat conflict) untuk keputusan BUY/SELL/HOLD yang lebih cepat dan akurat.

**Architecture:** `SpecialistPool` menjalankan 4 agent indikator secara sekuensial → `ComposerAgent` deteksi consensus/conflict → jika consensus langsung eksekusi, jika conflict panggil LLM satu kali untuk resolve → `RiskManager` → MT5 execute.

**Tech Stack:** Python 3.11+, pandas, numpy, MetaTrader5, ttkbootstrap, matplotlib, mplfinance, openai SDK (sudah ada di requirements.txt — tidak ada dependency baru).

---

## File Map

| File | Status | Tanggung Jawab |
|------|--------|---------------|
| `JEANIROTRABOT/modules/specialist_agents.py` | CREATE | AgentSignal dataclass, BaseSpecialistAgent, TrendAgent, MomentumAgent, VolatilityAgent, VolumeAgent, SpecialistPool |
| `JEANIROTRABOT/modules/composer_agent.py` | CREATE | FinalDecision dataclass, ComposerAgent (consensus + conflict detection + LLM fallback) |
| `JEANIROTRABOT/modules/trading_engine.py` | MODIFY | Tambah compute_ema/macd/stoch/bb/atr/obv/compute_all_indicators; refactor _process_symbol() pakai pool+composer |
| `JEANIROTRABOT/modules/ai_agent.py` | MODIFY | Tambah analyze_multi() untuk conflict resolution |
| `JEANIROTRABOT/modules/chart.py` | MODIFY | Tambah toggle MACD + BB panels |
| `JEANIROTRABOT/modules/gui.py` | MODIFY | Tambah Agent Signals panel di right panel |
| `JEANIROTRABOT/tests/__init__.py` | CREATE | Empty (pytest discovery) |
| `JEANIROTRABOT/tests/test_indicators.py` | CREATE | Test semua fungsi compute_* baru |
| `JEANIROTRABOT/tests/test_specialist_agents.py` | CREATE | Test setiap agent + SpecialistPool |
| `JEANIROTRABOT/tests/test_composer_agent.py` | CREATE | Test consensus, conflict, fallback logic |

---

## Task 1: Test Infrastructure + Fungsi Indikator Baru

**Files:**
- Modify: `JEANIROTRABOT/modules/trading_engine.py`
- Create: `JEANIROTRABOT/tests/__init__.py`
- Create: `JEANIROTRABOT/tests/test_indicators.py`

- [ ] **Step 1.1 — Buat folder tests dan __init__.py**

```bash
cd JEANIROTRABOT
mkdir tests
type nul > tests\__init__.py
```

- [ ] **Step 1.2 — Tulis failing tests untuk fungsi indikator baru**

Buat file `JEANIROTRABOT/tests/test_indicators.py`:

```python
"""Tests untuk fungsi indikator baru di trading_engine.py"""
import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.trading_engine import (
    compute_ema, compute_macd, compute_stochastic,
    compute_bollinger_bands, compute_atr, compute_obv,
    compute_all_indicators,
)


def make_df(n=100, trend="up") -> pd.DataFrame:
    """Buat DataFrame OHLCV sintetis untuk testing."""
    np.random.seed(42)
    if trend == "up":
        close = pd.Series(np.linspace(1.08, 1.10, n) + np.random.randn(n) * 0.0002)
    elif trend == "down":
        close = pd.Series(np.linspace(1.10, 1.08, n) + np.random.randn(n) * 0.0002)
    else:
        close = pd.Series(np.ones(n) * 1.09 + np.random.randn(n) * 0.0002)

    high = close + np.abs(np.random.randn(n)) * 0.0005
    low = close - np.abs(np.random.randn(n)) * 0.0005
    volume = pd.Series(np.random.randint(100, 1000, n), dtype=float)

    return pd.DataFrame({
        "Open": close - np.random.randn(n) * 0.0001,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })


class TestComputeEma:
    def test_ema_length_matches_input(self):
        df = make_df(100)
        ema = compute_ema(df["Close"], 9)
        assert len(ema) == 100

    def test_ema_no_nan_after_warmup(self):
        df = make_df(100)
        ema = compute_ema(df["Close"], 9)
        # EMA(9) dengan adjust=False tidak menghasilkan NaN setelah 1 nilai
        assert ema.iloc[9:].isna().sum() == 0

    def test_ema_faster_than_sma_on_uptrend(self):
        """EMA harus lebih cepat responsif dari SMA pada uptrend."""
        from modules.trading_engine import compute_sma
        df = make_df(100, trend="up")
        ema9 = compute_ema(df["Close"], 9)
        sma9 = compute_sma(df["Close"], 9)
        # EMA akhir harus >= SMA akhir pada uptrend
        assert ema9.iloc[-1] >= sma9.iloc[-1] - 0.001


class TestComputeMacd:
    def test_macd_returns_three_series(self):
        df = make_df(100)
        macd, signal, hist = compute_macd(df["Close"])
        assert len(macd) == 100
        assert len(signal) == 100
        assert len(hist) == 100

    def test_hist_equals_macd_minus_signal(self):
        df = make_df(100)
        macd, signal, hist = compute_macd(df["Close"])
        valid = ~(macd.isna() | signal.isna())
        diff = (hist[valid] - (macd[valid] - signal[valid])).abs()
        assert diff.max() < 1e-10

    def test_macd_positive_on_uptrend(self):
        df = make_df(150, trend="up")
        macd, signal, hist = compute_macd(df["Close"])
        # Pada uptrend kuat, MACD akhir cenderung positif
        assert macd.iloc[-1] > 0


class TestComputeStochastic:
    def test_stoch_returns_two_series(self):
        df = make_df(100)
        k, d = compute_stochastic(df["High"], df["Low"], df["Close"])
        assert len(k) == 100
        assert len(d) == 100

    def test_stoch_range_0_to_100(self):
        df = make_df(100)
        k, d = compute_stochastic(df["High"], df["Low"], df["Close"])
        valid_k = k.dropna()
        assert valid_k.min() >= 0.0 - 1e-6
        assert valid_k.max() <= 100.0 + 1e-6

    def test_stoch_oversold_on_downtrend(self):
        df = make_df(100, trend="down")
        k, d = compute_stochastic(df["High"], df["Low"], df["Close"])
        # Downtrend kuat → %K akhir < 50
        assert k.iloc[-1] < 60


class TestComputeBollingerBands:
    def test_bb_returns_four_series(self):
        df = make_df(100)
        upper, mid, lower, width = compute_bollinger_bands(df["Close"])
        assert len(upper) == len(mid) == len(lower) == len(width) == 100

    def test_upper_above_mid_above_lower(self):
        df = make_df(100)
        upper, mid, lower, width = compute_bollinger_bands(df["Close"])
        valid = ~(upper.isna() | mid.isna() | lower.isna())
        assert (upper[valid] >= mid[valid]).all()
        assert (mid[valid] >= lower[valid]).all()

    def test_width_always_non_negative(self):
        df = make_df(100)
        _, _, _, width = compute_bollinger_bands(df["Close"])
        assert (width.dropna() >= 0).all()


class TestComputeAtr:
    def test_atr_length(self):
        df = make_df(100)
        atr = compute_atr(df["High"], df["Low"], df["Close"])
        assert len(atr) == 100

    def test_atr_non_negative(self):
        df = make_df(100)
        atr = compute_atr(df["High"], df["Low"], df["Close"])
        assert (atr.dropna() >= 0).all()


class TestComputeObv:
    def test_obv_length(self):
        df = make_df(100)
        obv = compute_obv(df["Close"], df["Volume"])
        assert len(obv) == 100

    def test_obv_rises_on_uptrend(self):
        df = make_df(100, trend="up")
        obv = compute_obv(df["Close"], df["Volume"])
        # OBV akhir harus lebih besar dari OBV awal pada uptrend
        assert obv.iloc[-1] > obv.iloc[0]


class TestComputeAllIndicators:
    def test_returns_dict_with_required_keys(self):
        df = make_df(150)
        result = compute_all_indicators(df)
        required = [
            "sma20", "sma50", "ema9", "rsi",
            "macd", "macd_signal", "macd_hist", "macd_hist_prev",
            "stoch_k", "stoch_d", "stoch_k_prev", "stoch_d_prev",
            "bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_width_prev",
            "atr", "atr_avg",
            "obv", "obv_prev",
            "volume", "volume_sma",
            "close", "close_prev",
            "sma20_prev", "sma50_prev", "ema9_prev",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_all_values_are_float_or_none(self):
        df = make_df(150)
        result = compute_all_indicators(df)
        for k, v in result.items():
            assert v is None or isinstance(v, float), f"{k}={v} is not float/None"

    def test_returns_none_for_insufficient_data(self):
        df = make_df(10)  # terlalu sedikit candle
        result = compute_all_indicators(df)
        # SMA50 harus None karena < 50 candle
        assert result["sma50"] is None
```

- [ ] **Step 1.3 — Jalankan tests, verifikasi FAIL**

```bash
cd JEANIROTRABOT
python -m pytest tests/test_indicators.py -v 2>&1 | head -40
```

Expected output: `ImportError: cannot import name 'compute_ema'` atau sejenisnya (FAIL karena fungsi belum ada).

- [ ] **Step 1.4 — Tambahkan fungsi indikator baru ke trading_engine.py**

Buka `JEANIROTRABOT/modules/trading_engine.py`. Setelah baris `def compute_rsi(...)` (sekitar baris 41), tambahkan fungsi-fungsi ini:

```python
def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, Signal line, Histogram."""
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K and %D."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    stoch_k = 100.0 * (close - lowest_low) / denom
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Upper band, Mid (SMA), Lower band, Width."""
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    width = upper - lower
    return upper, mid, lower, width


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


def _safe_last(series: pd.Series) -> Optional[float]:
    """Ambil nilai terakhir series, return None jika NaN."""
    val = series.iloc[-1]
    return float(val) if not pd.isna(val) else None


def _safe_prev(series: pd.Series, n: int = 1) -> Optional[float]:
    """Ambil nilai sebelumnya, return None jika NaN atau index OOB."""
    if len(series) <= n:
        return None
    val = series.iloc[-1 - n]
    return float(val) if not pd.isna(val) else None


def compute_all_indicators(df: pd.DataFrame) -> dict:
    """
    Hitung semua indikator dari DataFrame OHLCV.
    Return dict dengan semua nilai terakhir (float atau None jika NaN).
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    sma20 = compute_sma(close, 20)
    sma50 = compute_sma(close, 50)
    ema9 = compute_ema(close, 9)
    rsi = compute_rsi(close, 14)
    macd, macd_signal, macd_hist = compute_macd(close)
    stoch_k, stoch_d = compute_stochastic(high, low, close)
    bb_upper, bb_mid, bb_lower, bb_width = compute_bollinger_bands(close)
    atr = compute_atr(high, low, close)
    atr_ma = atr.rolling(window=20).mean()
    obv = compute_obv(close, volume)
    volume_sma = volume.rolling(window=20).mean()

    return {
        "sma20":        _safe_last(sma20),
        "sma50":        _safe_last(sma50),
        "ema9":         _safe_last(ema9),
        "rsi":          _safe_last(rsi),
        "macd":         _safe_last(macd),
        "macd_signal":  _safe_last(macd_signal),
        "macd_hist":    _safe_last(macd_hist),
        "macd_hist_prev": _safe_prev(macd_hist),
        "stoch_k":      _safe_last(stoch_k),
        "stoch_d":      _safe_last(stoch_d),
        "stoch_k_prev": _safe_prev(stoch_k),
        "stoch_d_prev": _safe_prev(stoch_d),
        "bb_upper":     _safe_last(bb_upper),
        "bb_mid":       _safe_last(bb_mid),
        "bb_lower":     _safe_last(bb_lower),
        "bb_width":     _safe_last(bb_width),
        "bb_width_prev": _safe_prev(bb_width),
        "atr":          _safe_last(atr),
        "atr_avg":      _safe_last(atr_ma),
        "obv":          _safe_last(obv),
        "obv_prev":     _safe_prev(obv),
        "volume":       _safe_last(volume),
        "volume_sma":   _safe_last(volume_sma),
        "close":        _safe_last(close),
        "close_prev":   _safe_prev(close),
        "sma20_prev":   _safe_prev(sma20),
        "sma50_prev":   _safe_prev(sma50),
        "ema9_prev":    _safe_prev(ema9),
    }
```

Tambahkan juga `from typing import Optional` di bagian import atas `trading_engine.py` jika belum ada (sudah ada di file asli).

- [ ] **Step 1.5 — Jalankan tests, verifikasi PASS**

```bash
cd JEANIROTRABOT
python -m pytest tests/test_indicators.py -v
```

Expected:
```
tests/test_indicators.py::TestComputeEma::test_ema_length_matches_input PASSED
tests/test_indicators.py::TestComputeEma::test_ema_no_nan_after_warmup PASSED
...
tests/test_indicators.py::TestComputeAllIndicators::test_returns_none_for_insufficient_data PASSED
15 passed in X.XXs
```

- [ ] **Step 1.6 — Commit**

```bash
git add JEANIROTRABOT/modules/trading_engine.py JEANIROTRABOT/tests/__init__.py JEANIROTRABOT/tests/test_indicators.py
git commit -m "feat: add indicator functions (EMA, MACD, Stoch, BB, ATR, OBV, compute_all_indicators)"
```

---

## Task 2: AgentSignal + BaseSpecialistAgent + TrendAgent

**Files:**
- Create: `JEANIROTRABOT/modules/specialist_agents.py`
- Create: `JEANIROTRABOT/tests/test_specialist_agents.py`

- [ ] **Step 2.1 — Tulis failing tests untuk AgentSignal dan TrendAgent**

Buat file `JEANIROTRABOT/tests/test_specialist_agents.py`:

```python
"""Tests untuk specialist_agents.py"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.specialist_agents import (
    AgentSignal, TrendAgent, MomentumAgent,
    VolatilityAgent, VolumeAgent, SpecialistPool,
)


def make_indicators(
    sma20=1.0820, sma50=1.0800, ema9=1.0830,
    sma20_prev=1.0815, sma50_prev=1.0802, ema9_prev=1.0825,
    rsi=55.0, macd=0.0005, macd_signal=0.0003, macd_hist=0.0002,
    macd_hist_prev=-0.0001,
    stoch_k=65.0, stoch_d=60.0, stoch_k_prev=58.0, stoch_d_prev=62.0,
    bb_upper=1.0850, bb_mid=1.0820, bb_lower=1.0790, bb_width=0.0060,
    bb_width_prev=0.0065,
    atr=0.0010, atr_avg=0.0012,
    obv=50000.0, obv_prev=48000.0,
    volume=500.0, volume_sma=400.0,
    close=1.0825, close_prev=1.0815,
) -> dict:
    return {k: v for k, v in locals().items()}


# ── AgentSignal ──

class TestAgentSignal:
    def test_creation(self):
        sig = AgentSignal("TestAgent", "BUY", 0.8, 0.9, ["reason1"])
        assert sig.agent_name == "TestAgent"
        assert sig.action == "BUY"
        assert sig.score == 0.8
        assert sig.confidence == 0.9
        assert sig.reasons == ["reason1"]
        assert sig.error is False

    def test_error_signal(self):
        sig = AgentSignal("TestAgent", "HOLD", 0.0, 0.0, [], error=True, error_msg="oops")
        assert sig.error is True
        assert sig.error_msg == "oops"


# ── TrendAgent ──

class TestTrendAgent:
    def setup_method(self):
        self.agent = TrendAgent()

    def test_buy_on_full_uptrend_alignment(self):
        ind = make_indicators(ema9=1.0835, sma20=1.0820, sma50=1.0800)
        sig = self.agent.analyze(ind)
        assert sig.action == "BUY"
        assert sig.score > 0
        assert sig.confidence > 0.5

    def test_sell_on_full_downtrend_alignment(self):
        ind = make_indicators(ema9=1.0790, sma20=1.0810, sma50=1.0825,
                               ema9_prev=1.0795, sma20_prev=1.0815, sma50_prev=1.0822)
        sig = self.agent.analyze(ind)
        assert sig.action == "SELL"
        assert sig.score < 0

    def test_hold_on_mixed_alignment(self):
        # ema9 > sma20 tapi sma20 < sma50
        ind = make_indicators(ema9=1.0825, sma20=1.0820, sma50=1.0830)
        sig = self.agent.analyze(ind)
        assert sig.action == "HOLD"

    def test_golden_cross_boosts_score(self):
        # Sebelum: sma20 < sma50, sekarang: sma20 > sma50
        ind_no_cross = make_indicators(
            ema9=1.0825, sma20=1.0820, sma50=1.0810,
            sma20_prev=1.0815, sma50_prev=1.0812,  # sudah cross sebelumnya
        )
        ind_cross = make_indicators(
            ema9=1.0825, sma20=1.0820, sma50=1.0810,
            sma20_prev=1.0808, sma50_prev=1.0812,  # cross baru terjadi
        )
        sig_no_cross = self.agent.analyze(ind_no_cross)
        sig_cross = self.agent.analyze(ind_cross)
        assert sig_cross.score > sig_no_cross.score

    def test_error_on_none_indicators(self):
        ind = make_indicators(sma20=None)
        sig = self.agent.analyze(ind)
        assert sig.action == "HOLD"
        assert sig.error is True

    def test_score_clamped_between_minus1_and_1(self):
        ind = make_indicators(ema9=1.0835, sma20=1.0820, sma50=1.0800,
                               sma20_prev=1.0795, sma50_prev=1.0812)
        sig = self.agent.analyze(ind)
        assert -1.0 <= sig.score <= 1.0
        assert 0.0 <= sig.confidence <= 1.0
```

- [ ] **Step 2.2 — Jalankan tests, verifikasi FAIL**

```bash
cd JEANIROTRABOT
python -m pytest tests/test_specialist_agents.py::TestAgentSignal tests/test_specialist_agents.py::TestTrendAgent -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'modules.specialist_agents'`

- [ ] **Step 2.3 — Buat specialist_agents.py dengan AgentSignal + BaseSpecialistAgent + TrendAgent**

Buat file baru `JEANIROTRABOT/modules/specialist_agents.py`:

```python
"""
JEANIROTRABOT - Specialist Agent Pool (MiroFish OASIS-style)
4 specialist agents dengan kepribadian trading berbeda.
Setiap agent menganalisis indikator spesifik → AgentSignal.
SpecialistPool menjalankan semua agent dan mengumpulkan sinyal.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("JEANIROTRABOT.specialists")


# ─────────────────────────────────────────────
# Data Contract
# ─────────────────────────────────────────────

@dataclass
class AgentSignal:
    """Output standar setiap specialist agent."""
    agent_name: str
    action: str          # "BUY" | "SELL" | "HOLD"
    score: float         # -1.0 (kuat SELL) → +1.0 (kuat BUY)
    confidence: float    # 0.0 → 1.0
    reasons: list[str] = field(default_factory=list)
    error: bool = False
    error_msg: str = ""


# ─────────────────────────────────────────────
# Base Class
# ─────────────────────────────────────────────

class BaseSpecialistAgent:
    """Interface seragam untuk semua specialist agent."""
    NAME = "BaseAgent"

    def analyze(
        self,
        indicators: dict,
        tick: Optional[dict] = None,
        symbol_info: Optional[dict] = None,
    ) -> AgentSignal:
        raise NotImplementedError


# ─────────────────────────────────────────────
# TrendAgent — SMA / EMA crossover
# ─────────────────────────────────────────────

class TrendAgent(BaseSpecialistAgent):
    """
    Menganalisis arah tren melalui alignment EMA9 / SMA20 / SMA50.
    BUY  jika EMA9 > SMA20 > SMA50 (full uptrend alignment).
    SELL jika EMA9 < SMA20 < SMA50 (full downtrend alignment).
    Bonus score untuk golden/death cross yang baru terjadi.
    """
    NAME = "TrendAgent"

    def analyze(self, indicators: dict, tick=None, symbol_info=None) -> AgentSignal:
        try:
            sma20     = indicators.get("sma20")
            sma50     = indicators.get("sma50")
            ema9      = indicators.get("ema9")
            sma20_prev = indicators.get("sma20_prev")
            sma50_prev = indicators.get("sma50_prev")

            if any(v is None for v in [sma20, sma50, ema9, sma20_prev, sma50_prev]):
                return AgentSignal(self.NAME, "HOLD", 0.0, 0.0,
                                   ["Insufficient data for trend analysis"], error=True)

            score = 0.0
            confidence = 0.3
            reasons = []

            # Full alignment check
            if ema9 > sma20 > sma50:
                score = 0.65
                confidence = 0.72
                action = "BUY"
                reasons.append(f"EMA9({ema9:.5f})>SMA20({sma20:.5f})>SMA50({sma50:.5f}) uptrend")
            elif ema9 < sma20 < sma50:
                score = -0.65
                confidence = 0.72
                action = "SELL"
                reasons.append(f"EMA9({ema9:.5f})<SMA20({sma20:.5f})<SMA50({sma50:.5f}) downtrend")
            else:
                score = 0.0
                confidence = 0.28
                action = "HOLD"
                reasons.append("No clear EMA/SMA alignment")

            # Golden cross bonus (sma20 baru saja cross above sma50)
            if sma20_prev <= sma50_prev and sma20 > sma50:
                score = min(1.0, score + 0.20)
                confidence = min(1.0, confidence + 0.10)
                reasons.append("Golden cross detected")
                action = "BUY"
            # Death cross bonus
            elif sma20_prev >= sma50_prev and sma20 < sma50:
                score = max(-1.0, score - 0.20)
                confidence = min(1.0, confidence + 0.10)
                reasons.append("Death cross detected")
                action = "SELL"

            score = max(-1.0, min(1.0, score))
            confidence = max(0.0, min(1.0, confidence))

            logger.debug(f"[TrendAgent] {action} score={score:+.2f} conf={confidence:.2f}")
            return AgentSignal(self.NAME, action, score, confidence, reasons)

        except Exception as e:
            logger.warning(f"TrendAgent error: {e}")
            return AgentSignal(self.NAME, "HOLD", 0.0, 0.0,
                               [f"Error: {e}"], error=True, error_msg=str(e))
```

- [ ] **Step 2.4 — Jalankan tests TrendAgent, verifikasi PASS**

```bash
cd JEANIROTRABOT
python -m pytest tests/test_specialist_agents.py::TestAgentSignal tests/test_specialist_agents.py::TestTrendAgent -v
```

Expected: `7 passed`

- [ ] **Step 2.5 — Commit**

```bash
git add JEANIROTRABOT/modules/specialist_agents.py JEANIROTRABOT/tests/test_specialist_agents.py
git commit -m "feat: add AgentSignal + TrendAgent (EMA/SMA crossover specialist)"
```

---

## Task 3: MomentumAgent

**Files:**
- Modify: `JEANIROTRABOT/modules/specialist_agents.py`
- Modify: `JEANIROTRABOT/tests/test_specialist_agents.py`

- [ ] **Step 3.1 — Tambahkan tests MomentumAgent ke test file**

Tambahkan class berikut di akhir `JEANIROTRABOT/tests/test_specialist_agents.py`:

```python
class TestMomentumAgent:
    def setup_method(self):
        self.agent = MomentumAgent()

    def test_buy_on_oversold_rsi_positive_macd(self):
        ind = make_indicators(
            rsi=32.0,
            macd=0.0005, macd_signal=0.0002, macd_hist=0.0003, macd_hist_prev=-0.0001,
            stoch_k=18.0, stoch_d=22.0, stoch_k_prev=15.0, stoch_d_prev=20.0,
        )
        sig = self.agent.analyze(ind)
        assert sig.action == "BUY"
        assert sig.score > 0.2

    def test_sell_on_overbought_rsi_negative_macd(self):
        ind = make_indicators(
            rsi=72.0,
            macd=-0.0005, macd_signal=-0.0002, macd_hist=-0.0003, macd_hist_prev=0.0001,
            stoch_k=85.0, stoch_d=80.0, stoch_k_prev=88.0, stoch_d_prev=82.0,
        )
        sig = self.agent.analyze(ind)
        assert sig.action == "SELL"
        assert sig.score < -0.2

    def test_hold_on_neutral_indicators(self):
        ind = make_indicators(
            rsi=50.0,
            macd=0.00001, macd_signal=0.00001, macd_hist=0.0, macd_hist_prev=0.0,
            stoch_k=50.0, stoch_d=50.0, stoch_k_prev=50.0, stoch_d_prev=50.0,
        )
        sig = self.agent.analyze(ind)
        assert sig.action == "HOLD"

    def test_score_clamped(self):
        ind = make_indicators(rsi=5.0, macd_hist=1.0, macd_hist_prev=-0.01,
                               stoch_k=5.0, stoch_d=10.0, stoch_k_prev=3.0, stoch_d_prev=8.0)
        sig = self.agent.analyze(ind)
        assert -1.0 <= sig.score <= 1.0
        assert 0.0 <= sig.confidence <= 1.0

    def test_error_on_none_rsi(self):
        ind = make_indicators(rsi=None)
        sig = self.agent.analyze(ind)
        assert sig.action == "HOLD"
        assert sig.error is True
```

- [ ] **Step 3.2 — Jalankan, verifikasi FAIL**

```bash
python -m pytest tests/test_specialist_agents.py::TestMomentumAgent -v 2>&1 | head -10
```

Expected: `FAIL` (MomentumAgent belum ada)

- [ ] **Step 3.3 — Tambahkan MomentumAgent ke specialist_agents.py**

Tambahkan class berikut setelah `TrendAgent` di `specialist_agents.py`:

```python
class MomentumAgent(BaseSpecialistAgent):
    """
    Menganalisis kecepatan & kekuatan pergerakan harga.
    Indikator: RSI(14) bobot 30%, MACD(12,26,9) bobot 40%, Stochastic(14,3) bobot 30%.
    BUY jika RSI oversold + MACD positif + Stoch oversold-cross.
    SELL jika RSI overbought + MACD negatif + Stoch overbought-cross.
    """
    NAME = "MomentumAgent"

    def analyze(self, indicators: dict, tick=None, symbol_info=None) -> AgentSignal:
        try:
            rsi           = indicators.get("rsi")
            macd_hist     = indicators.get("macd_hist")
            macd_hist_prev = indicators.get("macd_hist_prev")
            macd          = indicators.get("macd")
            stoch_k       = indicators.get("stoch_k")
            stoch_d       = indicators.get("stoch_d")
            stoch_k_prev  = indicators.get("stoch_k_prev")
            stoch_d_prev  = indicators.get("stoch_d_prev")

            if any(v is None for v in [rsi, macd_hist, stoch_k, stoch_d]):
                return AgentSignal(self.NAME, "HOLD", 0.0, 0.0,
                                   ["Insufficient momentum data"], error=True)

            reasons = []

            # ── RSI signal (30% weight) ──
            if rsi < 40:
                rsi_score = (40 - rsi) / 40.0          # 0→1 saat RSI 40→0
                reasons.append(f"RSI={rsi:.1f} oversold")
            elif rsi > 60:
                rsi_score = -((rsi - 60) / 40.0)       # 0→-1 saat RSI 60→100
                reasons.append(f"RSI={rsi:.1f} overbought")
            else:
                rsi_score = 0.0
                reasons.append(f"RSI={rsi:.1f} neutral")

            # ── MACD signal (40% weight) ──
            macd_ref = abs(macd) if macd else 1e-5
            if macd_hist > 0:
                macd_score = min(1.0, macd_hist / macd_ref)
                reasons.append("MACD histogram positive")
                if macd_hist_prev is not None and macd_hist_prev <= 0:
                    macd_score = min(1.0, macd_score + 0.30)
                    reasons.append("MACD bullish crossover")
            elif macd_hist < 0:
                macd_score = max(-1.0, -min(1.0, abs(macd_hist) / macd_ref))
                reasons.append("MACD histogram negative")
                if macd_hist_prev is not None and macd_hist_prev >= 0:
                    macd_score = max(-1.0, macd_score - 0.30)
                    reasons.append("MACD bearish crossover")
            else:
                macd_score = 0.0

            # ── Stochastic signal (30% weight) ──
            if stoch_k < 20:
                stoch_score = 0.60
                reasons.append(f"Stoch %K={stoch_k:.1f} oversold")
                if stoch_k_prev is not None and stoch_d_prev is not None:
                    if stoch_k_prev <= stoch_d_prev and stoch_k > stoch_d:
                        stoch_score = 0.90
                        reasons.append("Stoch bullish cross")
            elif stoch_k > 80:
                stoch_score = -0.60
                reasons.append(f"Stoch %K={stoch_k:.1f} overbought")
                if stoch_k_prev is not None and stoch_d_prev is not None:
                    if stoch_k_prev >= stoch_d_prev and stoch_k < stoch_d:
                        stoch_score = -0.90
                        reasons.append("Stoch bearish cross")
            else:
                stoch_score = 0.0

            # ── Weighted final score ──
            final_score = (rsi_score * 0.30) + (macd_score * 0.40) + (stoch_score * 0.30)
            final_score = max(-1.0, min(1.0, final_score))
            confidence = min(1.0, abs(final_score) + 0.30)

            if final_score >= 0.20:
                action = "BUY"
            elif final_score <= -0.20:
                action = "SELL"
            else:
                action = "HOLD"

            logger.debug(f"[MomentumAgent] {action} score={final_score:+.2f}")
            return AgentSignal(self.NAME, action, final_score, confidence, reasons)

        except Exception as e:
            logger.warning(f"MomentumAgent error: {e}")
            return AgentSignal(self.NAME, "HOLD", 0.0, 0.0,
                               [f"Error: {e}"], error=True, error_msg=str(e))
```

- [ ] **Step 3.4 — Jalankan tests MomentumAgent, verifikasi PASS**

```bash
python -m pytest tests/test_specialist_agents.py::TestMomentumAgent -v
```

Expected: `5 passed`

- [ ] **Step 3.5 — Commit**

```bash
git add JEANIROTRABOT/modules/specialist_agents.py JEANIROTRABOT/tests/test_specialist_agents.py
git commit -m "feat: add MomentumAgent (RSI/MACD/Stochastic weighted specialist)"
```

---

## Task 4: VolatilityAgent

**Files:**
- Modify: `JEANIROTRABOT/modules/specialist_agents.py`
- Modify: `JEANIROTRABOT/tests/test_specialist_agents.py`

- [ ] **Step 4.1 — Tambahkan tests VolatilityAgent**

Tambahkan di akhir `tests/test_specialist_agents.py`:

```python
class TestVolatilityAgent:
    def setup_method(self):
        self.agent = VolatilityAgent()

    def test_buy_near_lower_band(self):
        # close dekat bb_lower
        ind = make_indicators(
            close=1.0792, bb_upper=1.0850, bb_mid=1.0820, bb_lower=1.0790,
            bb_width=0.0060, bb_width_prev=0.0070, atr=0.0010, atr_avg=0.0012,
        )
        sig = self.agent.analyze(ind)
        assert sig.action == "BUY"
        assert sig.score > 0

    def test_sell_near_upper_band(self):
        ind = make_indicators(
            close=1.0848, bb_upper=1.0850, bb_mid=1.0820, bb_lower=1.0790,
            bb_width=0.0060, bb_width_prev=0.0065, atr=0.0010, atr_avg=0.0012,
        )
        sig = self.agent.analyze(ind)
        assert sig.action == "SELL"
        assert sig.score < 0

    def test_hold_in_middle_zone(self):
        ind = make_indicators(
            close=1.0820, bb_upper=1.0850, bb_mid=1.0820, bb_lower=1.0790,
            bb_width=0.0060, bb_width_prev=0.0065, atr=0.0010, atr_avg=0.0012,
        )
        sig = self.agent.analyze(ind)
        assert sig.action == "HOLD"

    def test_low_atr_reduces_confidence(self):
        # Normal ATR
        ind_normal = make_indicators(
            close=1.0792, bb_upper=1.0850, bb_mid=1.0820, bb_lower=1.0790,
            atr=0.0010, atr_avg=0.0012,
        )
        # Sangat rendah ATR (flat market)
        ind_flat = make_indicators(
            close=1.0792, bb_upper=1.0850, bb_mid=1.0820, bb_lower=1.0790,
            atr=0.0002, atr_avg=0.0012,  # atr < 30% dari avg
        )
        sig_normal = self.agent.analyze(ind_normal)
        sig_flat = self.agent.analyze(ind_flat)
        assert sig_flat.confidence < sig_normal.confidence

    def test_error_on_none_bb(self):
        ind = make_indicators(bb_upper=None)
        sig = self.agent.analyze(ind)
        assert sig.action == "HOLD"
        assert sig.error is True
```

- [ ] **Step 4.2 — Jalankan, verifikasi FAIL**

```bash
python -m pytest tests/test_specialist_agents.py::TestVolatilityAgent -v 2>&1 | head -10
```

- [ ] **Step 4.3 — Tambahkan VolatilityAgent ke specialist_agents.py**

Tambahkan setelah `MomentumAgent`:

```python
class VolatilityAgent(BaseSpecialistAgent):
    """
    Menganalisis kondisi pasar dan potensi breakout via Bollinger Bands + ATR.
    BUY  jika harga dekat lower band (price_position ≤ 0.15) dengan BB menyempit.
    SELL jika harga dekat upper band (price_position ≥ 0.85).
    ATR filter: pasar terlalu flat → turunkan confidence 50%.
    """
    NAME = "VolatilityAgent"

    def analyze(self, indicators: dict, tick=None, symbol_info=None) -> AgentSignal:
        try:
            bb_upper    = indicators.get("bb_upper")
            bb_lower    = indicators.get("bb_lower")
            bb_width    = indicators.get("bb_width")
            bb_width_prev = indicators.get("bb_width_prev")
            atr         = indicators.get("atr")
            atr_avg     = indicators.get("atr_avg")
            close       = indicators.get("close")

            if any(v is None for v in [bb_upper, bb_lower, bb_width, close]):
                return AgentSignal(self.NAME, "HOLD", 0.0, 0.0,
                                   ["Insufficient volatility data"], error=True)

            reasons = []
            bb_range = bb_upper - bb_lower

            if bb_range <= 0:
                return AgentSignal(self.NAME, "HOLD", 0.0, 0.25,
                                   ["BB range is zero"])

            price_position = (close - bb_lower) / bb_range  # 0=lower, 1=upper

            if price_position <= 0.15:
                score = 0.60
                confidence = 0.65
                action = "BUY"
                reasons.append(f"Price near BB lower ({price_position:.2f}) — potential bounce")
            elif price_position >= 0.85:
                score = -0.60
                confidence = 0.65
                action = "SELL"
                reasons.append(f"Price near BB upper ({price_position:.2f}) — potential reversal")
            else:
                score = 0.0
                confidence = 0.30
                action = "HOLD"
                reasons.append(f"Price in BB middle zone ({price_position:.2f})")

            # BB squeeze bonus (band menyempit → momentum building)
            if bb_width_prev is not None and action != "HOLD":
                if bb_width < bb_width_prev:
                    score = max(-1.0, min(1.0, score * 1.15))
                    confidence = min(1.0, confidence + 0.08)
                    reasons.append("BB squeezing")

            # ATR filter: pasar terlalu flat
            if atr is not None and atr_avg is not None and atr_avg > 0:
                if atr < 0.30 * atr_avg:
                    confidence *= 0.50
                    reasons.append(f"Low ATR ({atr:.5f} < 30% avg) — flat market")

            score = max(-1.0, min(1.0, score))
            confidence = max(0.0, min(1.0, confidence))

            logger.debug(f"[VolatilityAgent] {action} score={score:+.2f}")
            return AgentSignal(self.NAME, action, score, confidence, reasons)

        except Exception as e:
            logger.warning(f"VolatilityAgent error: {e}")
            return AgentSignal(self.NAME, "HOLD", 0.0, 0.0,
                               [f"Error: {e}"], error=True, error_msg=str(e))
```

- [ ] **Step 4.4 — Jalankan tests, verifikasi PASS**

```bash
python -m pytest tests/test_specialist_agents.py::TestVolatilityAgent -v
```

Expected: `5 passed`

- [ ] **Step 4.5 — Commit**

```bash
git add JEANIROTRABOT/modules/specialist_agents.py JEANIROTRABOT/tests/test_specialist_agents.py
git commit -m "feat: add VolatilityAgent (Bollinger Bands + ATR specialist)"
```

---

## Task 5: VolumeAgent + SpecialistPool

**Files:**
- Modify: `JEANIROTRABOT/modules/specialist_agents.py`
- Modify: `JEANIROTRABOT/tests/test_specialist_agents.py`

- [ ] **Step 5.1 — Tambahkan tests VolumeAgent + SpecialistPool**

Tambahkan di akhir `tests/test_specialist_agents.py`:

```python
class TestVolumeAgent:
    def setup_method(self):
        self.agent = VolumeAgent()

    def test_buy_on_obv_and_price_rising(self):
        ind = make_indicators(obv=50000.0, obv_prev=48000.0,
                               close=1.0825, close_prev=1.0815,
                               volume=500.0, volume_sma=400.0)
        sig = self.agent.analyze(ind)
        assert sig.action == "BUY"
        assert sig.score > 0

    def test_sell_on_obv_and_price_falling(self):
        ind = make_indicators(obv=45000.0, obv_prev=47000.0,
                               close=1.0815, close_prev=1.0825,
                               volume=500.0, volume_sma=400.0)
        sig = self.agent.analyze(ind)
        assert sig.action == "SELL"
        assert sig.score < 0

    def test_volume_spike_boosts_confidence(self):
        ind_normal = make_indicators(volume=500.0, volume_sma=400.0,
                                      obv=50000.0, obv_prev=48000.0)
        ind_spike  = make_indicators(volume=900.0, volume_sma=400.0,
                                      obv=50000.0, obv_prev=48000.0)
        sig_normal = self.agent.analyze(ind_normal)
        sig_spike  = self.agent.analyze(ind_spike)
        assert sig_spike.confidence >= sig_normal.confidence

    def test_error_on_none_obv(self):
        ind = make_indicators(obv=None)
        sig = self.agent.analyze(ind)
        assert sig.error is True


class TestSpecialistPool:
    def setup_method(self):
        self.pool = SpecialistPool()

    def test_returns_four_signals(self):
        ind = make_indicators()
        signals = self.pool.analyze(ind)
        assert len(signals) == 4

    def test_all_signals_have_correct_agent_names(self):
        ind = make_indicators()
        signals = self.pool.analyze(ind)
        names = [s.agent_name for s in signals]
        assert "TrendAgent" in names
        assert "MomentumAgent" in names
        assert "VolatilityAgent" in names
        assert "VolumeAgent" in names

    def test_pool_survives_single_agent_error(self):
        """Pool harus tetap return 4 signals meski 1 agent error."""
        ind = make_indicators(sma20=None)  # TrendAgent akan error
        signals = self.pool.analyze(ind)
        assert len(signals) == 4
        trend_sig = next(s for s in signals if s.agent_name == "TrendAgent")
        assert trend_sig.error is True
        # Agent lain tetap berjalan
        momentum_sig = next(s for s in signals if s.agent_name == "MomentumAgent")
        assert momentum_sig.error is False
```

- [ ] **Step 5.2 — Jalankan, verifikasi FAIL**

```bash
python -m pytest tests/test_specialist_agents.py::TestVolumeAgent tests/test_specialist_agents.py::TestSpecialistPool -v 2>&1 | head -15
```

- [ ] **Step 5.3 — Tambahkan VolumeAgent + SpecialistPool ke specialist_agents.py**

Tambahkan setelah `VolatilityAgent`:

```python
class VolumeAgent(BaseSpecialistAgent):
    """
    Mengkonfirmasi sinyal harga dengan analisis volume.
    OBV rising + harga rising → konfirmasi bullish.
    OBV falling + harga falling → konfirmasi bearish.
    Divergence OBV vs harga → sinyal lemah.
    Volume spike → boost confidence; volume lesu → kurangi confidence.
    """
    NAME = "VolumeAgent"

    def analyze(self, indicators: dict, tick=None, symbol_info=None) -> AgentSignal:
        try:
            obv       = indicators.get("obv")
            obv_prev  = indicators.get("obv_prev")
            volume    = indicators.get("volume")
            volume_sma = indicators.get("volume_sma")
            close     = indicators.get("close")
            close_prev = indicators.get("close_prev")

            if any(v is None for v in [obv, obv_prev, close, close_prev]):
                return AgentSignal(self.NAME, "HOLD", 0.0, 0.0,
                                   ["Insufficient volume data"], error=True)

            reasons = []
            obv_rising   = obv > obv_prev
            price_rising = close > close_prev

            if obv_rising and price_rising:
                score = 0.60
                confidence = 0.70
                action = "BUY"
                reasons.append("OBV rising with price — bullish confirmation")
            elif not obv_rising and not price_rising:
                score = -0.60
                confidence = 0.70
                action = "SELL"
                reasons.append("OBV falling with price — bearish confirmation")
            elif obv_rising and not price_rising:
                score = 0.20
                confidence = 0.40
                action = "BUY"
                reasons.append("OBV rising, price falling — bullish divergence")
            else:
                score = -0.20
                confidence = 0.35
                action = "HOLD"
                reasons.append("OBV falling, price rising — bearish divergence warning")

            # Volume spike / lesu adjustment
            if volume is not None and volume_sma is not None and volume_sma > 0:
                if volume > 2.0 * volume_sma:
                    confidence = min(1.0, confidence + 0.10)
                    reasons.append(f"Volume spike ({volume:.0f} > 2× SMA)")
                elif volume < 0.5 * volume_sma:
                    confidence = max(0.0, confidence - 0.15)
                    reasons.append(f"Low volume ({volume:.0f} < 0.5× SMA)")

            score = max(-1.0, min(1.0, score))
            confidence = max(0.0, min(1.0, confidence))

            logger.debug(f"[VolumeAgent] {action} score={score:+.2f}")
            return AgentSignal(self.NAME, action, score, confidence, reasons)

        except Exception as e:
            logger.warning(f"VolumeAgent error: {e}")
            return AgentSignal(self.NAME, "HOLD", 0.0, 0.0,
                               [f"Error: {e}"], error=True, error_msg=str(e))


# ─────────────────────────────────────────────
# Specialist Pool
# ─────────────────────────────────────────────

class SpecialistPool:
    """
    Menjalankan semua specialist agents secara sekuensial.
    Agent yang error return HOLD — tidak menghentikan agent lain.
    """

    def __init__(self):
        self._agents: list[BaseSpecialistAgent] = [
            TrendAgent(),
            MomentumAgent(),
            VolatilityAgent(),
            VolumeAgent(),
        ]

    def analyze(
        self,
        indicators: dict,
        tick: Optional[dict] = None,
        symbol_info: Optional[dict] = None,
    ) -> list[AgentSignal]:
        """Return list AgentSignal dari semua 4 agent."""
        signals = []
        for agent in self._agents:
            try:
                signal = agent.analyze(indicators, tick, symbol_info)
            except Exception as e:
                logger.error(f"Uncaught error in {agent.NAME}: {e}")
                signal = AgentSignal(agent.NAME, "HOLD", 0.0, 0.0,
                                     [f"Uncaught: {e}"], error=True, error_msg=str(e))
            signals.append(signal)
        return signals
```

- [ ] **Step 5.4 — Jalankan semua tests specialist, verifikasi PASS**

```bash
python -m pytest tests/test_specialist_agents.py -v
```

Expected: `22 passed` (semua tests dari Task 2, 3, 4, 5)

- [ ] **Step 5.5 — Commit**

```bash
git add JEANIROTRABOT/modules/specialist_agents.py JEANIROTRABOT/tests/test_specialist_agents.py
git commit -m "feat: add VolumeAgent + SpecialistPool (OBV/Volume confirmation)"
```

---

## Task 6: ComposerAgent — FinalDecision + Consensus Logic

**Files:**
- Create: `JEANIROTRABOT/modules/composer_agent.py`
- Create: `JEANIROTRABOT/tests/test_composer_agent.py`

- [ ] **Step 6.1 — Tulis failing tests untuk consensus logic**

Buat `JEANIROTRABOT/tests/test_composer_agent.py`:

```python
"""Tests untuk composer_agent.py"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.specialist_agents import AgentSignal
from modules.composer_agent import ComposerAgent, FinalDecision


def make_signal(name="Agent", action="BUY", score=0.7, confidence=0.8) -> AgentSignal:
    return AgentSignal(name, action, score, confidence, [f"{name} reason"])


def make_error_signal(name="Agent") -> AgentSignal:
    return AgentSignal(name, "HOLD", 0.0, 0.0, ["error"], error=True)


class TestFinalDecision:
    def test_creation(self):
        fd = FinalDecision("BUY", 0.8, 0.7, "reason", "CONSENSUS", "3 BUY, 1 HOLD")
        assert fd.action == "BUY"
        assert fd.mode == "CONSENSUS"

    def test_optional_fields_default(self):
        fd = FinalDecision("HOLD", 0.0, 0.0, "reason", "SAFE_HOLD", "")
        assert fd.sl_points == 0
        assert fd.tp_points == 0
        assert fd.lot_size == 0.0


class TestComposerAgentConsensus:
    def setup_method(self):
        self.composer = ComposerAgent()

    def test_4_buy_votes_returns_buy_consensus(self):
        signals = [make_signal(f"A{i}", "BUY", 0.7, 0.8) for i in range(4)]
        result = self.composer.decide(signals, {})
        assert result.action == "BUY"
        assert result.mode == "CONSENSUS"

    def test_4_sell_votes_returns_sell_consensus(self):
        signals = [make_signal(f"A{i}", "SELL", -0.7, 0.8) for i in range(4)]
        result = self.composer.decide(signals, {})
        assert result.action == "SELL"
        assert result.mode == "CONSENSUS"

    def test_3_buy_1_hold_returns_buy_consensus(self):
        signals = [
            make_signal("A1", "BUY", 0.7, 0.8),
            make_signal("A2", "BUY", 0.6, 0.75),
            make_signal("A3", "BUY", 0.65, 0.72),
            make_signal("A4", "HOLD", 0.1, 0.3),
        ]
        result = self.composer.decide(signals, {})
        assert result.action == "BUY"
        assert result.mode == "CONSENSUS"

    def test_3_hold_returns_hold_consensus(self):
        signals = [
            make_signal("A1", "HOLD", 0.05, 0.3),
            make_signal("A2", "HOLD", -0.05, 0.3),
            make_signal("A3", "HOLD", 0.0, 0.25),
            make_signal("A4", "BUY", 0.4, 0.5),
        ]
        result = self.composer.decide(signals, {})
        assert result.action == "HOLD"
        assert result.mode == "CONSENSUS"

    def test_confidence_is_avg_of_agreeing_agents(self):
        signals = [
            make_signal("A1", "BUY", 0.8, 0.9),
            make_signal("A2", "BUY", 0.6, 0.7),
            make_signal("A3", "BUY", 0.7, 0.8),
            make_signal("A4", "HOLD", 0.1, 0.3),
        ]
        result = self.composer.decide(signals, {})
        expected_conf = (0.9 + 0.7 + 0.8) / 3
        assert abs(result.confidence - expected_conf) < 0.01

    def test_safe_hold_when_3_or_more_errors(self):
        signals = [
            make_error_signal("A1"),
            make_error_signal("A2"),
            make_error_signal("A3"),
            make_signal("A4", "BUY", 0.8, 0.9),
        ]
        result = self.composer.decide(signals, {})
        assert result.action == "HOLD"
        assert result.mode == "SAFE_HOLD"

    def test_2_2_conflict_without_llm_uses_majority_fallback(self):
        """Tanpa AI agent, 2-2 split harus jatuh ke majority_fallback (HOLD karena tie)."""
        signals = [
            make_signal("A1", "BUY", 0.7, 0.8),
            make_signal("A2", "BUY", 0.6, 0.7),
            make_signal("A3", "SELL", -0.7, 0.8),
            make_signal("A4", "SELL", -0.6, 0.7),
        ]
        result = self.composer.decide(signals, {}, ai_agent=None)
        assert result.mode == "MAJORITY_FALLBACK"
        assert result.action == "HOLD"  # true tie → HOLD

    def test_3_1_split_with_low_score_triggers_conflict(self):
        """3-1 split dengan score rata-rata < 0.3 harus jatuh ke conflict mode."""
        signals = [
            make_signal("A1", "BUY", 0.15, 0.4),
            make_signal("A2", "BUY", 0.10, 0.35),
            make_signal("A3", "BUY", 0.12, 0.38),
            make_signal("A4", "SELL", -0.6, 0.7),
        ]
        # Tanpa LLM → majority fallback
        result = self.composer.decide(signals, {}, ai_agent=None)
        # Harus bukan consensus murni (score terlalu rendah)
        assert result.mode in ("MAJORITY_FALLBACK", "CONFLICT_LLM")
```

- [ ] **Step 6.2 — Jalankan, verifikasi FAIL**

```bash
python -m pytest tests/test_composer_agent.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'modules.composer_agent'`

- [ ] **Step 6.3 — Buat composer_agent.py dengan FinalDecision + consensus logic**

Buat `JEANIROTRABOT/modules/composer_agent.py`:

```python
"""
JEANIROTRABOT - Composer Agent (MiroFish ReportAgent-style)
Mengumpulkan sinyal dari semua specialist agents dan membuat keputusan final.

Hybrid mode:
  - Consensus (≥3-1 dengan score kuat): keputusan langsung, tanpa LLM
  - Conflict (2-2 atau 3-1 lemah):      LLM call sekali untuk resolve
  - Fallback (LLM error/unavailable):   majority vote
  - Safe (≥3 agents error):             HOLD tanpa eksekusi
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from modules.specialist_agents import AgentSignal

logger = logging.getLogger("JEANIROTRABOT.composer")

CONFLICT_SCORE_THRESHOLD = 0.30   # avg score minimum untuk consensus tanpa LLM
LLM_DEFAULT_TIMEOUT = 10.0        # detik


# ─────────────────────────────────────────────
# Output Contract
# ─────────────────────────────────────────────

@dataclass
class FinalDecision:
    """Output ComposerAgent yang diteruskan ke TradingEngine untuk eksekusi."""
    action: str          # "BUY" | "SELL" | "HOLD"
    confidence: float
    score: float
    reason: str
    mode: str            # "CONSENSUS" | "CONFLICT_LLM" | "MAJORITY_FALLBACK" | "SAFE_HOLD"
    vote_summary: str    # e.g. "3 BUY, 1 HOLD"
    sl_points: int = 0
    tp_points: int = 0
    lot_size: float = 0.0
    ticket: int = 0
    new_sl: float = 0.0
    new_tp: float = 0.0


# ─────────────────────────────────────────────
# ComposerAgent
# ─────────────────────────────────────────────

class ComposerAgent:
    """
    Menggabungkan sinyal specialist agents → keputusan trading final.
    Dipanggil dengan ai_agent=None untuk consensus-only mode.
    """

    def __init__(self, llm_timeout: float = LLM_DEFAULT_TIMEOUT):
        self._llm_timeout = llm_timeout

    def decide(
        self,
        signals: list[AgentSignal],
        market_context: dict,
        ai_agent=None,
    ) -> FinalDecision:
        """
        Entrypoint utama. Evaluasi sinyal → consensus atau conflict → FinalDecision.
        market_context: dict yang sama yang dikirim ke AIAgent.analyze_multi()
        ai_agent: instance AIAgent (atau None untuk no-LLM mode)
        """
        valid   = [s for s in signals if not s.error]
        errored = [s for s in signals if s.error]

        vote_summary = self._build_vote_summary(signals)

        # Safe mode: terlalu banyak agent error
        if len(errored) >= 3:
            logger.warning(f"[Composer] SAFE_HOLD — {len(errored)} agents errored")
            return FinalDecision(
                action="HOLD", confidence=0.0, score=0.0,
                reason=f"Safe mode: {len(errored)}/4 agents errored",
                mode="SAFE_HOLD", vote_summary=vote_summary,
            )

        if not valid:
            return FinalDecision(
                action="HOLD", confidence=0.0, score=0.0,
                reason="No valid agent signals",
                mode="SAFE_HOLD", vote_summary="",
            )

        buy_sigs  = [s for s in valid if s.action == "BUY"]
        sell_sigs = [s for s in valid if s.action == "SELL"]
        hold_sigs = [s for s in valid if s.action == "HOLD"]
        n = len(valid)

        # ── Consensus check ──
        # 4-0 atau 3-1 dengan score kuat
        if len(buy_sigs) >= 3:
            avg_score = sum(abs(s.score) for s in buy_sigs) / len(buy_sigs)
            if len(buy_sigs) == n or avg_score >= CONFLICT_SCORE_THRESHOLD:
                return self._build_consensus("BUY", buy_sigs, vote_summary)

        if len(sell_sigs) >= 3:
            avg_score = sum(abs(s.score) for s in sell_sigs) / len(sell_sigs)
            if len(sell_sigs) == n or avg_score >= CONFLICT_SCORE_THRESHOLD:
                return self._build_consensus("SELL", sell_sigs, vote_summary)

        if len(hold_sigs) >= 3:
            return self._build_consensus("HOLD", hold_sigs, vote_summary)

        # ── Conflict — gunakan LLM jika tersedia ──
        if ai_agent is not None and ai_agent.enabled:
            logger.info(f"[Composer] CONFLICT ({vote_summary}) → calling LLM")
            return self._llm_decide(signals, market_context, ai_agent, vote_summary)

        # ── Fallback: majority vote tanpa LLM ──
        logger.info(f"[Composer] CONFLICT ({vote_summary}) → majority fallback (no LLM)")
        return self._majority_fallback(valid, vote_summary)

    # ── Internal helpers ──

    def _build_consensus(
        self, action: str, agreeing: list[AgentSignal], vote_summary: str
    ) -> FinalDecision:
        avg_score = sum(s.score for s in agreeing) / len(agreeing)
        avg_conf  = sum(s.confidence for s in agreeing) / len(agreeing)
        top_reasons = [r for s in agreeing for r in s.reasons][:4]
        reason_str = "; ".join(top_reasons)
        logger.info(f"[Composer] CONSENSUS {action} ({vote_summary}) score={avg_score:+.2f}")
        return FinalDecision(
            action=action,
            confidence=round(avg_conf, 3),
            score=round(avg_score, 3),
            reason=f"Consensus ({vote_summary}): {reason_str}",
            mode="CONSENSUS",
            vote_summary=vote_summary,
        )

    def _majority_fallback(
        self, valid: list[AgentSignal], vote_summary: str
    ) -> FinalDecision:
        buy_sigs  = [s for s in valid if s.action == "BUY"]
        sell_sigs = [s for s in valid if s.action == "SELL"]

        if len(buy_sigs) > len(sell_sigs):
            majority = buy_sigs
            action = "BUY"
        elif len(sell_sigs) > len(buy_sigs):
            majority = sell_sigs
            action = "SELL"
        else:
            return FinalDecision(
                action="HOLD", confidence=0.30, score=0.0,
                reason=f"True tie — HOLD ({vote_summary})",
                mode="MAJORITY_FALLBACK", vote_summary=vote_summary,
            )

        avg_score = sum(s.score for s in majority) / len(majority)
        avg_conf  = sum(s.confidence for s in majority) / len(majority) * 0.80
        return FinalDecision(
            action=action,
            confidence=round(avg_conf, 3),
            score=round(avg_score, 3),
            reason=f"Majority fallback ({vote_summary})",
            mode="MAJORITY_FALLBACK",
            vote_summary=vote_summary,
        )

    def _llm_decide(
        self,
        signals: list[AgentSignal],
        market_context: dict,
        ai_agent,
        vote_summary: str,
    ) -> FinalDecision:
        agent_lines = "\n".join(
            f"  {s.agent_name}: {s.action} score={s.score:+.2f} conf={s.confidence:.2f}"
            f" — {'; '.join(s.reasons[:2])}"
            for s in signals if not s.error
        )
        conflict_summary = (
            f"=== AGENT SIGNALS (CONFLICT: {vote_summary}) ===\n"
            f"{agent_lines}\n"
            f"The specialist agents disagree. Resolve this conflict and decide."
        )

        try:
            result = ai_agent.analyze_multi(
                conflict_summary=conflict_summary,
                **{k: v for k, v in market_context.items()},
            )
            action     = result.get("action", "HOLD")
            confidence = float(result.get("confidence", 0.5))
            reason     = result.get("reason", "LLM resolved")

            logger.info(f"[Composer] LLM resolved: {action} conf={confidence:.2f}")
            return FinalDecision(
                action=action,
                confidence=confidence,
                score=confidence if action == "BUY" else (-confidence if action == "SELL" else 0.0),
                reason=f"LLM resolved conflict ({vote_summary}): {reason}",
                mode="CONFLICT_LLM",
                vote_summary=vote_summary,
                sl_points=result.get("sl_points", 0),
                tp_points=result.get("tp_points", 0),
                lot_size=result.get("lot_size", 0.0),
                ticket=result.get("ticket", 0),
                new_sl=result.get("new_sl", 0.0),
                new_tp=result.get("new_tp", 0.0),
            )
        except Exception as e:
            logger.warning(f"[Composer] LLM error: {e} — fallback to majority vote")
            return self._majority_fallback([s for s in signals if not s.error], vote_summary)

    @staticmethod
    def _build_vote_summary(signals: list[AgentSignal]) -> str:
        buy  = sum(1 for s in signals if s.action == "BUY")
        sell = sum(1 for s in signals if s.action == "SELL")
        hold = sum(1 for s in signals if s.action == "HOLD")
        return f"{buy} BUY, {sell} SELL, {hold} HOLD"
```

- [ ] **Step 6.4 — Jalankan tests composer, verifikasi PASS**

```bash
python -m pytest tests/test_composer_agent.py -v
```

Expected: `11 passed`

- [ ] **Step 6.5 — Commit**

```bash
git add JEANIROTRABOT/modules/composer_agent.py JEANIROTRABOT/tests/test_composer_agent.py
git commit -m "feat: add ComposerAgent with consensus detection and majority fallback"
```

---

## Task 7: AIAgent.analyze_multi() — LLM Conflict Resolution

**Files:**
- Modify: `JEANIROTRABOT/modules/ai_agent.py`

- [ ] **Step 7.1 — Tambahkan method analyze_multi() ke AIAgent**

Buka `JEANIROTRABOT/modules/ai_agent.py`. Setelah akhir method `analyze()` (sekitar baris 345), tambahkan method baru:

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
        **kwargs,
    ) -> dict:
        """
        Conflict resolution: kirim rangkuman sinyal agent yang conflict +
        market context ke LLM, minta keputusan final.
        Dipanggil oleh ComposerAgent hanya saat terjadi conflict.
        """
        default = {"action": "HOLD", "confidence": 0.0,
                   "reason": "LLM unavailable for conflict resolution", "raw_response": ""}

        if not self._enabled:
            default["reason"] = "AI agent is disabled."
            return default

        client = self._get_client()
        if client is None:
            default["reason"] = "No API client for conflict resolution."
            return default

        # Build prompt: conflict summary dulu, lalu market context
        user_prompt = conflict_summary + "\n\n"
        user_prompt += f"=== MARKET DATA ===\n"
        if symbol:
            user_prompt += f"Symbol: {symbol}, Timeframe: {timeframe}\n"
        if bid:
            user_prompt += f"Bid: {bid}, Ask: {ask}, Spread: {spread:.1f} pts\n"
        if sma20:
            user_prompt += f"SMA(20): {sma20:.5f}, SMA(50): {sma50:.5f}, RSI(14): {rsi:.2f}\n"
        if ohlcv_summary:
            user_prompt += f"Recent OHLCV:\n{ohlcv_summary}\n"

        if account_equity > 0:
            user_prompt += (
                f"\n=== ACCOUNT ===\n"
                f"Balance: {account_balance:.2f}, Equity: {account_equity:.2f}\n"
                f"Free Margin: {free_margin:.2f}\n"
            )

        if risk_status:
            user_prompt += (
                f"\n=== RISK ===\n"
                f"Drawdown: {risk_status.get('drawdown_pct', 0):.2f}%  "
                f"Orders today: {risk_status.get('orders_today', 0)}/"
                f"{risk_status.get('max_orders_per_day', 20)}\n"
            )

        if open_positions:
            user_prompt += f"\n=== OPEN POSITIONS ({len(open_positions)}) ===\n"
            for p in open_positions:
                user_prompt += (
                    f"  #{p['ticket']}: {p['type']} {p['volume']} {p['symbol']} "
                    f"P&L: {p['profit']:.2f}\n"
                )

        user_prompt += (
            "\nResolve the agent conflict above. Provide your final trading decision as JSON. "
            "Be decisive — pick the most supported action given all evidence."
        )

        try:
            response = self._get_client().chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=400,
                temperature=0.2,   # lebih rendah dari analyze() untuk konsistensi
            )
            raw = response.choices[0].message.content.strip()
            logger.info(f"AI [CONFLICT/{self._provider}/{self._model}]: {raw}")
            parsed = self._parse_response(raw)
            parsed["raw_response"] = raw
            return parsed

        except Exception as e:
            logger.error(f"analyze_multi error ({self._provider}): {e}")
            self._client = None
            default["reason"] = f"LLM conflict error: {str(e)}"
            return default
```

- [ ] **Step 7.2 — Verifikasi import di composer_agent.py tidak circular**

```bash
cd JEANIROTRABOT
python -c "from modules.composer_agent import ComposerAgent; print('OK')"
python -c "from modules.ai_agent import AIAgent; print('OK')"
python -c "from modules.specialist_agents import SpecialistPool; print('OK')"
```

Expected: `OK` tiga kali.

- [ ] **Step 7.3 — Jalankan semua tests**

```bash
python -m pytest tests/ -v
```

Expected: semua tests pass (tidak ada test baru di task ini, tapi pastikan tidak ada regresi).

- [ ] **Step 7.4 — Commit**

```bash
git add JEANIROTRABOT/modules/ai_agent.py
git commit -m "feat: add AIAgent.analyze_multi() for conflict resolution LLM call"
```

---

## Task 8: TradingEngine Integration

**Files:**
- Modify: `JEANIROTRABOT/modules/trading_engine.py`

Ini adalah task terbesar — refactor `_process_symbol()` dan `__init__()` TradingEngine.

- [ ] **Step 8.1 — Tambahkan imports baru di trading_engine.py**

Di bagian atas `trading_engine.py`, setelah import yang sudah ada, tambahkan:

```python
from modules.specialist_agents import SpecialistPool, AgentSignal
from modules.composer_agent import ComposerAgent, FinalDecision
```

- [ ] **Step 8.2 — Update TradingEngine.__init__() untuk inisialisasi pool + composer**

Di dalam method `__init__()` TradingEngine (sekitar baris 47–65), tambahkan 2 baris setelah `self._interval = 5`:

```python
        self._pool = SpecialistPool()
        self._composer = ComposerAgent()
        self._last_signals: list[AgentSignal] = []   # untuk GUI update
        self._last_decision: Optional[FinalDecision] = None
        self._signal_callback: Optional[Callable[[list, object], None]] = None
```

- [ ] **Step 8.3 — Tambahkan method set_signal_callback()**

Tambahkan method baru setelah `set_log_callback()`:

```python
    def set_signal_callback(self, cb: Callable[[list, object], None]):
        """Callback dipanggil setiap cycle dengan (signals, decision) untuk GUI update."""
        self._signal_callback = cb
```

- [ ] **Step 8.4 — Tambahkan method _log_agent_signals()**

Tambahkan method baru setelah method `_log()`:

```python
    def _log_agent_signals(self, symbol: str, signals: list[AgentSignal], decision: FinalDecision):
        """Log sinyal semua agent + keputusan composer ke log callback dan update GUI."""
        for sig in signals:
            status = "ERROR" if sig.error else sig.action
            self._log(
                f"[{sig.agent_name}] {status} score={sig.score:+.2f} "
                f"conf={sig.confidence:.2f} — {'; '.join(sig.reasons[:2])}"
            )
        self._log(
            f"[COMPOSER] {decision.mode} → {decision.action} "
            f"score={decision.score:+.2f} conf={decision.confidence:.2f} "
            f"({decision.vote_summary})"
        )
        self._last_signals = signals
        self._last_decision = decision
        if self._signal_callback:
            try:
                self._signal_callback(signals, decision)
            except Exception:
                pass
```

- [ ] **Step 8.5 — Tambahkan method _execute_decision() yang menggantikan _execute_ai_decision()**

Tambahkan method baru (bisa sebelum `_execute_ai_decision` yang lama, nanti yang lama tetap ada untuk fallback basic strategy):

```python
    def _execute_decision(
        self,
        decision: FinalDecision,
        symbol: str,
        symbol_info: Optional[dict],
        total_positions: int,
        symbol_pos_count: int,
    ):
        """Eksekusi FinalDecision dari ComposerAgent — mapping ke MT5 orders."""
        action = decision.action
        confidence = decision.confidence

        if action == "HOLD":
            return

        if action in ("BUY", "SELL"):
            if confidence < 0.6:
                self._log(f"Skipping {action}: confidence {confidence:.2f} < 0.6")
                return

            ok, msg = self.risk.can_open_position(total_positions, symbol_pos_count)
            if not ok:
                self._log(f"Skipping {action}: {msg}")
                return

            # Lot size: decision custom > dynamic > default
            if decision.lot_size > 0:
                lot = self.risk.validate_lot_size(decision.lot_size)
            elif symbol_info:
                sl_pts = decision.sl_points if decision.sl_points > 0 else self.risk.default_sl_points
                lot = self.risk.calculate_lot_size(
                    equity=0,
                    sl_points=sl_pts,
                    tick_value=symbol_info.get("trade_tick_value", 1),
                    tick_size=symbol_info.get("trade_tick_size", 1),
                )
            else:
                lot = self.risk.validate_lot_size(self._lot_size)

            sl = decision.sl_points if decision.sl_points > 0 else self.risk.default_sl_points
            tp = decision.tp_points if decision.tp_points > 0 else self.risk.default_tp_points

            self._log(f"AUTO-EXECUTE [{decision.mode}]: {action} {lot} {symbol} SL={sl} TP={tp}")
            order_result = self.connector.send_market_order(
                symbol=symbol, order_type=action.lower(),
                volume=lot, sl_points=sl, tp_points=tp,
            )
            if order_result.success:
                self.risk.record_order()
                self._log(f"Order filled: ticket={order_result.ticket} @ {order_result.price}")
            else:
                self._log(f"Order failed: {order_result.comment}")

        elif action == "CLOSE":
            ticket = decision.ticket
            if not ticket:
                self._log("CLOSE action missing ticket.")
                return
            self._log(f"AUTO-CLOSE: ticket #{ticket}")
            close_result = self.connector.close_position(ticket)
            if close_result.success:
                self._log(f"Closed #{ticket} @ {close_result.price}")
            else:
                self._log(f"Close failed: {close_result.comment}")

        elif action == "CLOSE_ALL":
            self._log(f"AUTO-CLOSE_ALL: {symbol}")
            results = self.connector.close_positions_by_symbol(symbol)
            for r in results:
                status = f"Closed #{r.ticket} @ {r.price}" if r.success else f"Failed: {r.comment}"
                self._log(f"  {status}")

        elif action == "MODIFY_SL_TP":
            ticket = decision.ticket
            if not ticket:
                self._log("MODIFY_SL_TP missing ticket.")
                return
            self._log(f"AUTO-MODIFY: #{ticket} SL={decision.new_sl} TP={decision.new_tp}")
            mod_result = self.connector.modify_position_sl_tp(ticket, decision.new_sl, decision.new_tp)
            if mod_result.success:
                self._log(f"Position #{ticket} SL/TP modified.")
            else:
                self._log(f"Modify failed: {mod_result.comment}")
```

- [ ] **Step 8.6 — Refactor _process_symbol() untuk gunakan pool + composer**

Ganti seluruh method `_process_symbol()` yang ada dengan versi baru ini:

```python
    def _process_symbol(self, symbol: str, account, all_positions: list,
                        risk_status: dict, total_positions: int):
        """Process satu symbol: fetch data → indicators → specialist pool → composer → execute."""
        # 1. Fetch OHLCV (naikkan minimum ke 60 untuk semua indikator)
        df = self.connector.get_ohlcv(symbol, self._timeframe, count=150)
        if df is None or len(df) < 60:
            self._log(f"[{symbol}] Insufficient data ({len(df) if df is not None else 0} candles)")
            return

        # 2. Compute all indicators
        indicators = compute_all_indicators(df)

        # Validasi indikator kritis
        if indicators["sma20"] is None or indicators["rsi"] is None:
            self._log(f"[{symbol}] Core indicators NaN, skipping cycle")
            return

        # 3. Get market data
        tick = self.connector.get_tick(symbol)
        if not tick:
            return

        symbol_info = self.connector.get_symbol_info(symbol)
        symbol_positions = [p for p in all_positions if p["symbol"] == symbol]

        # 4. Run specialist pool (MiroFish agent pool style)
        signals = self._pool.analyze(indicators, tick, symbol_info)

        # 5. Build market context untuk LLM conflict resolution
        ohlcv_summary = df.tail(10).to_string()
        market_context = {
            "symbol":          symbol,
            "timeframe":       self._timeframe,
            "ohlcv_summary":   ohlcv_summary,
            "sma20":           indicators["sma20"],
            "sma50":           indicators["sma50"] or 0.0,
            "rsi":             indicators["rsi"],
            "bid":             tick["bid"],
            "ask":             tick["ask"],
            "account_equity":  account.equity,
            "account_balance": account.balance,
            "free_margin":     account.free_margin,
            "open_positions":  symbol_positions,
            "spread":          symbol_info.get("spread", 0) if symbol_info else 0,
            "risk_status":     risk_status,
            "symbol_info":     symbol_info,
        }

        # 6. Composer decide (MiroFish ReportAgent style)
        ai_agent = self.ai if self.ai.enabled else None
        decision = self._composer.decide(signals, market_context, ai_agent)

        # 7. Log + notify GUI
        self._log_agent_signals(symbol, signals, decision)

        # 8. Execute decision
        self._execute_decision(decision, symbol, symbol_info,
                                total_positions, len(symbol_positions))
```

- [ ] **Step 8.7 — Verifikasi import dan basic syntax**

```bash
cd JEANIROTRABOT
python -c "from modules.trading_engine import TradingEngine, compute_all_indicators; print('OK')"
```

Expected: `OK`

- [ ] **Step 8.8 — Jalankan semua tests**

```bash
python -m pytest tests/ -v
```

Expected: semua tests pass.

- [ ] **Step 8.9 — Commit**

```bash
git add JEANIROTRABOT/modules/trading_engine.py
git commit -m "feat: integrate SpecialistPool + ComposerAgent into TradingEngine._process_symbol()"
```

---

## Task 9: Chart — Panel MACD + Bollinger Bands

**Files:**
- Modify: `JEANIROTRABOT/modules/chart.py`

- [ ] **Step 9.1 — Tambahkan toggle MACD + BB ke ChartManager**

Buka `JEANIROTRABOT/modules/chart.py`. Di dalam `__init__()` method `ChartManager`, tambahkan 2 variabel baru setelah `self._show_rsi = True`:

```python
        self._show_macd = True
        self._show_bb = True
```

- [ ] **Step 9.2 — Tambahkan methods toggle_macd() dan toggle_bb()**

Setelah method `toggle_rsi()` tambahkan:

```python
    def toggle_macd(self, value: bool):
        self._show_macd = value

    def toggle_bb(self, value: bool):
        self._show_bb = value
```

- [ ] **Step 9.3 — Update method render() untuk MACD + BB**

Cari method `render()` di `chart.py`. Tambahkan import baru di atas file jika belum ada:
```python
from modules.trading_engine import compute_macd, compute_bollinger_bands
```

Di dalam method `render()`, setelah blok `if self._show_rsi:` dan sebelum baris `title = ...`, tambahkan:

```python
        if self._show_bb:
            from modules.trading_engine import compute_bollinger_bands as _cbb
            bb_upper, bb_mid, bb_lower, _ = _cbb(df["Close"])
            addplots.append(mpf.make_addplot(
                bb_upper, color="#ff880055", width=0.8, linestyle="--", label="BB Upper"
            ))
            addplots.append(mpf.make_addplot(
                bb_lower, color="#ff880055", width=0.8, linestyle="--", label="BB Lower"
            ))

        if self._show_macd:
            from modules.trading_engine import compute_macd as _cmacd
            macd_line, signal_line, histogram = _cmacd(df["Close"])
            macd_panel = 3 if self._show_rsi else 2
            addplots.append(mpf.make_addplot(
                macd_line, panel=macd_panel, color="#00bfff", width=1.0, label="MACD"
            ))
            addplots.append(mpf.make_addplot(
                signal_line, panel=macd_panel, color="#ff8800", width=1.0, label="Signal"
            ))
            addplots.append(mpf.make_addplot(
                histogram, panel=macd_panel, type="bar",
                color=["#00ff8855" if v >= 0 else "#ff444455"
                       for v in histogram.fillna(0)],
                label="Hist"
            ))
```

Update juga `panel_ratios` di `render()` untuk memperhitungkan panel MACD:

Ganti baris:
```python
        panel_ratios = (4, 1, 2) if self._show_rsi else (4, 1)
```
Dengan:
```python
        if self._show_rsi and self._show_macd:
            panel_ratios = (4, 1, 2, 2)
        elif self._show_rsi:
            panel_ratios = (4, 1, 2)
        elif self._show_macd:
            panel_ratios = (4, 1, 2)
        else:
            panel_ratios = (4, 1)
```

- [ ] **Step 9.4 — Verifikasi chart module load tanpa error**

```bash
cd JEANIROTRABOT
python -c "from modules.chart import ChartManager; cm = ChartManager(); print('OK')"
```

Expected: `OK`

- [ ] **Step 9.5 — Commit**

```bash
git add JEANIROTRABOT/modules/chart.py
git commit -m "feat: add MACD + Bollinger Bands panels to ChartManager"
```

---

## Task 10: GUI — Panel Agent Signals

**Files:**
- Modify: `JEANIROTRABOT/modules/gui.py`

- [ ] **Step 10.1 — Tambahkan import di gui.py**

Di `gui.py`, tambahkan import di bagian atas (setelah import `from modules.chart import ChartManager`):

```python
from modules.specialist_agents import AgentSignal
from modules.composer_agent import FinalDecision
```

- [ ] **Step 10.2 — Update JeaniroTrabotApp.__init__() untuk set signal callback**

Di `__init__()`, setelah baris `self.engine.set_log_callback(self._append_log_threadsafe)`, tambahkan:

```python
        self.engine.set_signal_callback(self._update_agent_panel_threadsafe)
```

- [ ] **Step 10.3 — Tambahkan toggle MACD dan BB ke chart controls**

Di dalam method `_build_center_panel()`, setelah baris Checkbutton RSI, tambahkan:

```python
        self.var_macd = tk.BooleanVar(value=True)
        self.var_bb = tk.BooleanVar(value=True)

        ttkb.Checkbutton(ctrl, text="MACD", variable=self.var_macd,
                         bootstyle="primary-round-toggle",
                         command=self._refresh_chart).pack(side=tk.LEFT, padx=5)
        ttkb.Checkbutton(ctrl, text="BB", variable=self.var_bb,
                         bootstyle="danger-round-toggle",
                         command=self._refresh_chart).pack(side=tk.LEFT, padx=5)
```

- [ ] **Step 10.4 — Update _refresh_chart() untuk pass toggle MACD + BB ke chart_mgr**

Cari method `_refresh_chart()` di gui.py. Tambahkan 2 baris setelah `self.chart_mgr.toggle_rsi(...)`:

```python
            self.chart_mgr.toggle_macd(self.var_macd.get())
            self.chart_mgr.toggle_bb(self.var_bb.get())
```

- [ ] **Step 10.5 — Tambahkan method _build_agent_panel() ke gui.py**

Tambahkan method baru setelah `_build_right_panel()`:

```python
    def _build_agent_panel(self, parent):
        """Panel Agent Signals — menampilkan vote + score tiap specialist agent real-time."""
        lf = ttkb.Labelframe(parent, text="🤖 Agent Signals", bootstyle="primary")
        lf.pack(fill=tk.X, padx=5, pady=5)

        # Header
        hdr = ttkb.Frame(lf)
        hdr.pack(fill=tk.X, padx=3, pady=(3, 0))
        ttkb.Label(hdr, text="Agent", width=12, anchor="w", font=("Courier", 8, "bold")).pack(side=tk.LEFT)
        ttkb.Label(hdr, text="Signal", width=6, anchor="center", font=("Courier", 8, "bold")).pack(side=tk.LEFT)
        ttkb.Label(hdr, text="Score", width=12, anchor="w", font=("Courier", 8, "bold")).pack(side=tk.LEFT)

        # Agent rows
        self._agent_rows: dict[str, dict] = {}
        agent_specs = [
            ("TrendAgent",      "📈"),
            ("MomentumAgent",   "⚡"),
            ("VolatilityAgent", "🌊"),
            ("VolumeAgent",     "📊"),
        ]
        for name, icon in agent_specs:
            row = ttkb.Frame(lf)
            row.pack(fill=tk.X, padx=3, pady=1)
            lbl_name = ttkb.Label(row, text=f"{icon} {name[:8]}", width=12, anchor="w",
                                   font=("Courier", 8))
            lbl_name.pack(side=tk.LEFT)
            lbl_action = ttkb.Label(row, text="--", width=6, anchor="center",
                                     font=("Courier", 8, "bold"))
            lbl_action.pack(side=tk.LEFT)
            lbl_score = ttkb.Label(row, text="--", width=12, anchor="w",
                                    font=("Courier", 8))
            lbl_score.pack(side=tk.LEFT)
            self._agent_rows[name] = {"action": lbl_action, "score": lbl_score, "name": lbl_name}

        # Separator
        ttkb.Separator(lf).pack(fill=tk.X, padx=3, pady=2)

        # Composer result
        comp_row = ttkb.Frame(lf)
        comp_row.pack(fill=tk.X, padx=3, pady=2)
        ttkb.Label(comp_row, text="🎯 COMPOSER", width=12, anchor="w",
                   font=("Courier", 8, "bold")).pack(side=tk.LEFT)
        self.lbl_composer_action = ttkb.Label(comp_row, text="--", width=6,
                                               anchor="center", font=("Courier", 9, "bold"))
        self.lbl_composer_action.pack(side=tk.LEFT)
        self.lbl_composer_mode = ttkb.Label(comp_row, text="", anchor="w",
                                             font=("Courier", 8), foreground="#888888")
        self.lbl_composer_mode.pack(side=tk.LEFT, padx=2)

        return lf

    def _update_agent_panel_threadsafe(self, signals: list, decision):
        """Thread-safe update panel dari trading engine thread."""
        try:
            self.root.after(0, lambda: self._update_agent_panel(signals, decision))
        except Exception:
            pass

    def _update_agent_panel(self, signals: list, decision):
        """Update label setiap agent dan composer di panel."""
        ACTION_COLORS = {"BUY": ACCENT_GREEN, "SELL": ACCENT_RED, "HOLD": "#aaaaaa"}

        for sig in signals:
            row = self._agent_rows.get(sig.agent_name)
            if not row:
                continue
            action = sig.action if not sig.error else "ERR"
            color = ACTION_COLORS.get(action, "#888888")
            row["action"].configure(text=action, foreground=color)

            score_bar = self._make_score_bar(sig.score)
            score_text = f"{score_bar} {sig.score:+.2f}"
            row["score"].configure(text=score_text, foreground=color)

        if decision:
            action = decision.action
            color = ACTION_COLORS.get(action, "#aaaaaa")
            self.lbl_composer_action.configure(text=action, foreground=color)

            mode_color = {"CONSENSUS": ACCENT_GREEN,
                          "CONFLICT_LLM": "#ffaa00",
                          "MAJORITY_FALLBACK": "#aaaaaa",
                          "SAFE_HOLD": ACCENT_RED}.get(decision.mode, "#888888")
            self.lbl_composer_mode.configure(
                text=f"{decision.mode} ({decision.vote_summary})",
                foreground=mode_color,
            )

    @staticmethod
    def _make_score_bar(score: float, width: int = 8) -> str:
        """Buat bar ASCII untuk visualisasi score -1.0 → +1.0."""
        filled = int(abs(score) * width)
        filled = min(filled, width)
        bar = "█" * filled + "░" * (width - filled)
        return bar
```

- [ ] **Step 10.6 — Panggil _build_agent_panel() di _build_right_panel()**

Di dalam method `_build_right_panel()`, di akhir method (sebelum penutup), tambahkan:

```python
        # Agent Signals Panel
        self._build_agent_panel(parent)
```

- [ ] **Step 10.7 — Verifikasi gui.py import bersih**

```bash
cd JEANIROTRABOT
python -c "
import tkinter as tk
import sys
sys.path.insert(0, '.')
# Cek import saja tanpa buka window
from modules.gui import JeaniroTrabotApp
print('GUI imports OK')
"
```

Expected: `GUI imports OK`

- [ ] **Step 10.8 — Commit**

```bash
git add JEANIROTRABOT/modules/gui.py
git commit -m "feat: add Agent Signals panel to GUI with real-time specialist vote display"
```

---

## Task 11: Smoke Test + Final Integration Verify

**Files:**
- Run tests dan manual verify

- [ ] **Step 11.1 — Jalankan full test suite**

```bash
cd JEANIROTRABOT
python -m pytest tests/ -v --tb=short
```

Expected output (minimal):
```
tests/test_indicators.py::TestComputeEma::test_ema_length_matches_input PASSED
...
tests/test_specialist_agents.py::TestSpecialistPool::test_pool_survives_single_agent_error PASSED
tests/test_composer_agent.py::TestComposerAgentConsensus::test_4_buy_votes_returns_buy_consensus PASSED
...
XX passed in X.XXs
```

Semua tests harus PASS. Jika ada fail, debug dulu sebelum lanjut.

- [ ] **Step 11.2 — Verifikasi full application import chain**

```bash
cd JEANIROTRABOT
python -c "
import sys
sys.path.insert(0, '.')
from modules.trading_engine import TradingEngine, compute_all_indicators
from modules.specialist_agents import SpecialistPool, AgentSignal
from modules.composer_agent import ComposerAgent, FinalDecision
from modules.ai_agent import AIAgent
from modules.chart import ChartManager
print('All modules import OK')

# Smoke test: pool + composer pipeline tanpa MT5
import pandas as pd
import numpy as np

np.random.seed(0)
close = pd.Series(np.linspace(1.08, 1.10, 150) + np.random.randn(150) * 0.0002)
df = pd.DataFrame({
    'Open': close - 0.0001, 'High': close + 0.0005,
    'Low': close - 0.0005, 'Close': close,
    'Volume': pd.Series(np.random.randint(100, 1000, 150), dtype=float),
})

indicators = compute_all_indicators(df)
pool = SpecialistPool()
signals = pool.analyze(indicators)
composer = ComposerAgent()
decision = composer.decide(signals, {})

print(f'Pipeline OK: {len(signals)} signals → {decision.action} ({decision.mode})')
for s in signals:
    print(f'  {s.agent_name}: {s.action} score={s.score:+.2f}')
print(f'  COMPOSER: {decision.action} conf={decision.confidence:.2f} [{decision.vote_summary}]')
"
```

Expected output:
```
All modules import OK
Pipeline OK: 4 signals → BUY (CONSENSUS)
  TrendAgent: BUY score=+0.65
  MomentumAgent: BUY score=+0.XX
  VolatilityAgent: HOLD score=+0.XX
  VolumeAgent: BUY score=+0.60
  COMPOSER: BUY conf=0.XX [3 BUY, 1 HOLD]
```

(Action dan scores bisa berbeda, yang penting tidak ada error)

- [ ] **Step 11.3 — Commit final**

```bash
git add -A
git commit -m "feat: MiroFish multi-agent integration complete — SpecialistPool + ComposerAgent hybrid decision engine"
```

---

## Self-Review Checklist

### Spec Coverage

| Spec Section | Task yang Mengimplementasikan |
|---|---|
| TrendAgent (SMA/EMA crossover) | Task 2 |
| MomentumAgent (RSI/MACD/Stoch) | Task 3 |
| VolatilityAgent (BB/ATR) | Task 4 |
| VolumeAgent (OBV/Volume SMA) | Task 5 |
| SpecialistPool | Task 5 |
| ComposerAgent consensus (≥3-1) | Task 6 |
| ComposerAgent conflict detection (2-2) | Task 6 |
| ComposerAgent LLM conflict resolution | Task 7 |
| AIAgent.analyze_multi() | Task 7 |
| compute_all_indicators() (EMA,MACD,BB,ATR,OBV) | Task 1 |
| TradingEngine integration | Task 8 |
| GUI Agent Signals panel | Task 10 |
| Chart MACD + BB panels | Task 9 |
| Fallback cascade (safe_hold, majority_fallback) | Task 6 |
| Error handling per agent | Task 2–5 (try/except) |
| LLM timeout / unavailable fallback | Task 6 (_llm_decide fallback) |
| Logging terstruktur per agent | Task 8 (_log_agent_signals) |

### Type Consistency Check

- `AgentSignal` defined Task 2, used in Task 5 (SpecialistPool), Task 6 (ComposerAgent.decide), Task 8 (trading_engine), Task 10 (GUI) ✅
- `FinalDecision` defined Task 6, used in Task 8 (_execute_decision), Task 10 (GUI update) ✅
- `compute_all_indicators()` defined Task 1, returns `dict`, used in Task 8 ✅
- `SpecialistPool.analyze()` returns `list[AgentSignal]`, consumed by `ComposerAgent.decide()` ✅
- `ComposerAgent.decide(signals, market_context, ai_agent)` → `FinalDecision` ✅
- `AIAgent.analyze_multi(conflict_summary, **market_context)` → `dict` ✅
- `set_signal_callback(cb: Callable[[list, object], None])` defined Task 8, set in Task 10 ✅
