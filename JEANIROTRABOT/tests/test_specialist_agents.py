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
        # ema9 > sma20 tapi sma20 < sma50, dan tidak ada golden/death cross
        ind = make_indicators(
            ema9=1.0825, sma20=1.0820, sma50=1.0830,
            sma20_prev=1.0818, sma50_prev=1.0832,  # sma20 sudah di bawah sma50 sejak sebelumnya (no cross)
        )
        sig = self.agent.analyze(ind)
        assert sig.action == "HOLD"

    def test_golden_cross_boosts_score(self):
        ind_no_cross = make_indicators(
            ema9=1.0825, sma20=1.0820, sma50=1.0810,
            sma20_prev=1.0815, sma50_prev=1.0812,
        )
        ind_cross = make_indicators(
            ema9=1.0825, sma20=1.0820, sma50=1.0810,
            sma20_prev=1.0808, sma50_prev=1.0812,
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


# ── MomentumAgent ──

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


# ── VolatilityAgent ──

class TestVolatilityAgent:
    def setup_method(self):
        self.agent = VolatilityAgent()

    def test_buy_near_lower_band(self):
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
        ind_normal = make_indicators(
            close=1.0792, bb_upper=1.0850, bb_mid=1.0820, bb_lower=1.0790,
            atr=0.0010, atr_avg=0.0012,
        )
        ind_flat = make_indicators(
            close=1.0792, bb_upper=1.0850, bb_mid=1.0820, bb_lower=1.0790,
            atr=0.0002, atr_avg=0.0012,
        )
        sig_normal = self.agent.analyze(ind_normal)
        sig_flat   = self.agent.analyze(ind_flat)
        assert sig_flat.confidence < sig_normal.confidence

    def test_error_on_none_bb(self):
        ind = make_indicators(bb_upper=None)
        sig = self.agent.analyze(ind)
        assert sig.action == "HOLD"
        assert sig.error is True


# ── VolumeAgent ──

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


# ── SpecialistPool ──

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
        momentum_sig = next(s for s in signals if s.agent_name == "MomentumAgent")
        assert momentum_sig.error is False
