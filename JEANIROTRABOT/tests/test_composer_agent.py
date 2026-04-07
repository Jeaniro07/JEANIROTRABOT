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
        assert result.action == "HOLD"  # true tie -> HOLD

    def test_3_1_split_with_low_score_triggers_conflict_fallback(self):
        """3-1 split dengan score rata-rata < 0.3 harus jatuh ke majority fallback (no LLM)."""
        signals = [
            make_signal("A1", "BUY", 0.15, 0.4),
            make_signal("A2", "BUY", 0.10, 0.35),
            make_signal("A3", "BUY", 0.12, 0.38),
            make_signal("A4", "SELL", -0.6, 0.7),
        ]
        result = self.composer.decide(signals, {}, ai_agent=None)
        assert result.mode in ("MAJORITY_FALLBACK", "CONFLICT_LLM")

    def test_vote_summary_correct_format(self):
        signals = [
            make_signal("A1", "BUY", 0.7, 0.8),
            make_signal("A2", "BUY", 0.6, 0.75),
            make_signal("A3", "SELL", -0.5, 0.6),
            make_signal("A4", "HOLD", 0.0, 0.3),
        ]
        result = self.composer.decide(signals, {}, ai_agent=None)
        assert "BUY" in result.vote_summary
        assert "SELL" in result.vote_summary
        assert "HOLD" in result.vote_summary

    def test_all_errors_returns_safe_hold(self):
        signals = [make_error_signal(f"A{i}") for i in range(4)]
        result = self.composer.decide(signals, {})
        assert result.action == "HOLD"
        assert result.mode == "SAFE_HOLD"
