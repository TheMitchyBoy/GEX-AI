"""Extended tests for improvements."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from api.alerts import evaluate_alerts
from db.json_fallbacks import resolve_strike_series, strike_series_from_surface_json
from db.reconciliation import _outcome_metrics
from models.calibration import calibrate_confidence
from models.gboost import build_training_matrix, train_gboost
from models.llm_cache import clear_cache, get_cached, set_cached
from models.llm_context import estimate_token_count
from models.multi_horizon import predict_multi_horizon
from tests.synthetic_data import generate_synthetic_history


def test_bulk_strike_fallback_surface_json():
    rows = [{"strike": 5500, "GEX": 0.05}, {"strike": 5510, "GEX": -0.02}]
    series = strike_series_from_surface_json(rows)
    assert len(series) == 2
    strike, _ = resolve_strike_series(None, {"surface_json": rows})
    assert not strike.empty


def test_calibration_dampens_without_samples():
    assert calibrate_confidence(0.9, None, 0) < 0.9


def test_llm_cache():
    clear_cache()
    set_cached("SPX", "2026-06-01_120000", {"x": 1})
    assert get_cached("SPX", "2026-06-01_120000")["x"] == 1


def test_token_estimate():
    bundle = {"summary": {"spot": 5500}, "intraday_timeline": [{"ts": "t"}]}
    assert estimate_token_count(bundle) > 0


def test_multi_horizon():
    history = generate_synthetic_history(n_snapshots=50)
    horizons = predict_multi_horizon(history, horizons=(1, 3), lookback_days=30)
    assert 1 in horizons


def test_gboost_train():
    history = generate_synthetic_history(n_snapshots=60)
    matrix = build_training_matrix(history)
    assert matrix is not None
    with patch("models.gboost.joblib.dump"):
        result = train_gboost(history, "SPX")
    assert result is not None


def test_alerts_trigger():
    forecast = {"regime_flip_probability": 0.5, "predicted_delta_gex": 0.1}
    enriched = {"spot": 5500, "gamma_flip": 5498}
    alerts = evaluate_alerts(forecast, enriched)
    assert any(a["type"] == "regime_flip" for a in alerts)


def test_reconciliation_outcome():
    outcome = _outcome_metrics(
        {"predicted_delta_gex_bn": 0.05, "spot_bias": "bullish", "predicted_regime": "LONG gamma", "confidence": 0.7},
        {"delta_gex_bn": 0.04, "spot_before": 5500, "spot_after": 5510, "regime": "LONG gamma"},
    )
    assert outcome["sign_hit"] is True
