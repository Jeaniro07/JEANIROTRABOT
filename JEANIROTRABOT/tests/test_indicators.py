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
    low  = close - np.abs(np.random.randn(n)) * 0.0005
    volume = pd.Series(np.random.randint(100, 1000, n), dtype=float)

    return pd.DataFrame({
        "Open":   close - np.random.randn(n) * 0.0001,
        "High":   high,
        "Low":    low,
        "Close":  close,
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
        assert ema.iloc[9:].isna().sum() == 0

    def test_ema_faster_than_sma_on_uptrend(self):
        """EMA harus lebih cepat responsif dari SMA pada uptrend."""
        from modules.trading_engine import compute_sma
        df = make_df(100, trend="up")
        ema9 = compute_ema(df["Close"], 9)
        sma9 = compute_sma(df["Close"], 9)
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
        df = make_df(10)
        result = compute_all_indicators(df)
        assert result["sma50"] is None
