"""Tests for LLM forecast layer."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from models.llm_client import parse_prediction_json
from models.llm_context import build_context_bundle
from models.llm_predict import generate_llm_forecast
from tests.synthetic_data import generate_synthetic_history


def test_parse_prediction_json_with_fence():
    raw = '```json\n{"predicted_regime": "LONG gamma", "confidence": 0.8}\n```'
    parsed = parse_prediction_json(raw)
    assert parsed["predicted_regime"] == "LONG gamma"
    assert parsed["confidence"] == 0.8


def test_build_context_bundle():
    history = generate_synthetic_history(n_snapshots=30)
    bundle = build_context_bundle(history, lookback_days=30)
    assert bundle["ticker"] == "SPX"
    assert "summary" in bundle
    assert len(bundle["intraday_timeline"]) > 0
    assert bundle["knn_forecast"] is not None


def test_llm_forecast_fallback_without_api_key():
    history = generate_synthetic_history(n_snapshots=40)
    with patch("models.llm_predict.is_llm_configured", return_value=False):
        result = generate_llm_forecast(history, lookback_days=30, persist=False)
    assert result["prediction_source"] == "knn_fallback"
    assert result["llm_enhanced"] is False
    assert "predicted_delta_gex_bn" in result
    assert result["context_summary"]["llm_configured"] is False


def test_llm_forecast_with_mock_openai():
    history = generate_synthetic_history(n_snapshots=40)
    mock_response = {
        "predicted_regime": "SHORT gamma",
        "predicted_delta_gex_bn": -0.05,
        "predicted_total_gex_bn": -0.25,
        "spot_bias": "bearish",
        "confidence": 0.72,
        "gamma_flip": 5480.0,
        "key_levels": {"support": [5450], "resistance": [5520], "pin": 5500},
        "scenarios": [],
        "predictions": ["Expect mean reversion toward flip"],
        "reasoning": "Dealers short gamma with flip below spot.",
    }

    with patch("models.llm_predict.is_llm_configured", return_value=True):
        with patch("models.llm_predict.openai_chat_json", return_value=(mock_response, None)):
            result = generate_llm_forecast(history, lookback_days=30, persist=False)

    assert result["llm_enhanced"] is True
    assert result["prediction_source"] == "llm"
    assert result["spot_bias"] == "bearish"
    assert result["confidence"] == pytest.approx(0.72)
