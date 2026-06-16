"""Tests for comprehensive AI improvements."""

from __future__ import annotations

from unittest.mock import patch

from models.agreement import compute_agreement
from models.context_compress import compress_bundle
from models.ensemble import blend_delta, learn_weights_from_backtest
from models.event_prompts import event_prompt_snippets
from models.llm_agent import chat_with_agent
from models.llm_rag import retrieve_regime_matched_sessions
from models.quant_fallback import quant_only_reply
from tests.synthetic_data import generate_synthetic_history


def test_compress_bundle_trims_strikes():
    bundle = {
        "top_strikes": [{"strike": i, "gex_bn_per_pct": 1.0} for i in range(20)],
        "intraday_timeline": [{"ts": str(i)} for i in range(30)],
    }
    out = compress_bundle(bundle, max_strikes=5, max_timeline=8)
    assert len(out["top_strikes"]) == 5
    assert len(out["intraday_timeline"]) == 8
    assert out["_compressed"] is True


def test_agreement_score_high_when_aligned():
    knn = {"predicted_delta_gex": 0.05, "predicted_regime": "LONG gamma"}
    score = compute_agreement(knn=knn, gboost_delta=0.04, online_delta=0.05)
    assert score["score"] >= 0.7


def test_ensemble_blend():
    out = blend_delta(knn_delta=0.1, gboost_delta=0.05, online_delta=0.08, ticker="SPX")
    assert out["ensemble_delta_gex"] is not None
    assert "knn" in out["weights_used"]


def test_learn_weights_from_backtest():
    w = learn_weights_from_backtest({"mae_delta_gex": 0.03, "regime_accuracy": 0.6}, "TEST_TICKER_X")
    assert abs(sum(w.values()) - 1.0) < 0.01


def test_event_prompts_0dte():
    snippets = event_prompt_snippets({"zero_dte_ratio": 0.5})
    assert any("0DTE" in s for s in snippets)


def test_quant_only_reply():
    bundle = {
        "summary": {"spot": 5000, "net_gamma_regime": "LONG gamma", "total_gex_bn_per_pct": 1.2,
                    "gamma_flip": 4980, "flip_distance_pct": 0.002, "call_wall": 5050, "put_wall": 4950},
        "knn_forecast": {"predicted_delta_gex_bn": 0.01, "predicted_regime": "LONG gamma", "confidence": 0.7, "spot_bias": "neutral"},
    }
    text = quant_only_reply(bundle)
    assert "5000" in text
    assert "LONG" in text


def test_chat_quant_mode():
    history = generate_synthetic_history(n_snapshots=30)
    result = chat_with_agent(history, [{"role": "user", "content": "regime?"}], mode="quant")
    assert result["reply"]
    assert result["mode"] == "quant"


def test_chat_fast_mode_mock():
    history = generate_synthetic_history(n_snapshots=40)
    with patch("models.llm_agent.is_llm_configured", return_value=True):
        with patch("models.llm_agent.get_cached", return_value=None):
            with patch("models.llm_agent.openai_chat", return_value=("LONG gamma above flip.", None)):
                result = chat_with_agent(history, [{"role": "user", "content": "Regime?"}], mode="fast")
    assert "gamma" in (result["reply"] or "").lower()
    assert result["mode"] == "fast"


def test_regime_matched_sessions():
    history = generate_synthetic_history(n_snapshots=80)
    sessions = retrieve_regime_matched_sessions(history, top_n=2)
    assert isinstance(sessions, list)
