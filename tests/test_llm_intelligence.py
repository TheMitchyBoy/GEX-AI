"""Tests for AI intelligence upgrade: RAG, tools, two-pass, rich context."""

from __future__ import annotations

import json
from unittest.mock import patch

from models.llm_agent import chat_with_agent
from models.llm_context import build_context_bundle
from models.llm_rag import retrieve_similar_sessions
from models.llm_tools import AgentToolRunner, TOOL_SCHEMAS
from tests.synthetic_data import generate_synthetic_history


def test_rich_context_bundle_fields():
    history = generate_synthetic_history(n_snapshots=40)
    bundle = build_context_bundle(history, lookback_days=30, rich=True, slim=False)
    assert bundle["ticker"] == "SPX"
    assert "quant_synthesis" in bundle
    assert "atm_strike_band" in bundle
    assert "forecast_track_record" in bundle
    assert "similar_sessions" in bundle
    assert "last_move_attribution" in bundle


def test_retrieve_similar_sessions():
    history = generate_synthetic_history(n_snapshots=60)
    sessions = retrieve_similar_sessions(history, top_n=2)
    assert isinstance(sessions, list)
    if sessions:
        assert "market_date" in sessions[0]
        assert "similarity" in sessions[0]


def test_tool_schemas_defined():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {
        "get_forecast",
        "get_similar_setups",
        "get_strikes_near_atm",
        "get_backtest_summary",
        "compare_knn_llm",
    }


def test_agent_tool_runner_forecast():
    history = generate_synthetic_history(n_snapshots=30)
    runner = AgentToolRunner(history, lookback_days=30)
    raw = runner.execute("get_forecast", "{}")
    data = json.loads(raw)
    assert "knn" in data
    assert "horizons" in data


def test_agent_tool_runner_strikes():
    history = generate_synthetic_history(n_snapshots=30)
    runner = AgentToolRunner(history, lookback_days=30)
    raw = runner.execute("get_strikes_near_atm", {})
    data = json.loads(raw)
    assert "spot" in data
    assert "strikes_near_atm" in data


def test_chat_two_pass_mock():
    history = generate_synthetic_history(n_snapshots=40)
    facts = {"spot": 5000, "regime": "LONG gamma", "gamma_flip": 4980}
    with patch("models.llm_agent.is_llm_configured", return_value=True):
        with patch("models.llm_agent.openai_chat_json", return_value=(facts, None)):
            with patch("models.llm_agent.openai_chat", return_value=("LONG gamma above flip.", None)):
                with patch("models.llm_agent.openai_chat_with_tools", return_value=(None, "skip", [])):
                    result = chat_with_agent(
                        history,
                        [{"role": "user", "content": "What is the regime?"}],
                        two_pass=True,
                        use_tools=False,
                        mode="deep",
                    )
    assert result["reply"] is not None
    assert result["two_pass"] is True
    assert result["intelligence"]["rich_context"] is True


def test_chat_with_tools_mock():
    history = generate_synthetic_history(n_snapshots=40)
    tool_log = [{"tool": "get_forecast", "arguments": "{}", "result_preview": "{}"}]
    with patch("models.llm_agent.is_llm_configured", return_value=True):
        with patch("models.llm_agent.get_cached", return_value=None):
            with patch("models.llm_agent.openai_chat_json", return_value=(None, None)):
                with patch(
                    "models.llm_agent.openai_chat_with_tools",
                    return_value=("Forecast from tools: short gamma.", None, tool_log),
                ):
                    result = chat_with_agent(
                        history,
                        [{"role": "user", "content": "Give me the forecast"}],
                        two_pass=False,
                        use_tools=True,
                        mode="deep",
                    )
    assert "gamma" in (result["reply"] or "").lower()
    assert "get_forecast" in result["tools_used"]
