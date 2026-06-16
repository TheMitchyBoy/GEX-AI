"""Tests for GEX conversational agent."""

from __future__ import annotations

from unittest.mock import patch

from models.llm_agent import SUGGESTED_PROMPTS, build_agent_context, chat_with_agent
from tests.synthetic_data import generate_synthetic_history


def test_suggested_prompts_nonempty():
    assert len(SUGGESTED_PROMPTS) >= 3


def test_build_agent_context():
    history = generate_synthetic_history(n_snapshots=30)
    ctx = build_agent_context(history, lookback_days=30)
    assert ctx["bundle"]["ticker"] == "SPX"
    assert ctx["estimated_tokens"] > 0


def test_chat_without_api_key():
    history = generate_synthetic_history(n_snapshots=30)
    with patch("models.llm_agent.is_llm_configured", return_value=False):
        result = chat_with_agent(history, [{"role": "user", "content": "What is the regime?"}], mode="fast")
    assert result["reply"] is not None
    assert "OPENAI" in (result.get("error") or "")


def test_chat_with_mock_openai():
    history = generate_synthetic_history(n_snapshots=40)
    with patch("models.llm_agent.is_llm_configured", return_value=True):
        with patch("models.llm_agent.get_cached", return_value=None):
            with patch("models.llm_agent.openai_chat", return_value=("Short gamma near flip — expect volatility.", None)):
                result = chat_with_agent(history, [{"role": "user", "content": "Summarize the regime"}], mode="fast")
    assert "gamma" in result["reply"].lower() or "Short" in result["reply"]
    assert result["error"] is None
