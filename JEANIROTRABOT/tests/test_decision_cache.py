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
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"action":"HOLD","confidence":0.5,"reason":"test"}'))]
        mock_client.chat.completions.create.return_value = mock_response
        with patch.object(agent, '_get_client', return_value=mock_client):
            result = agent.analyze_multi(
                conflict_summary="test conflict",
                symbol="EURUSD",
                timeout=5.0,
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
