"""Unit tests for feature engineering and KNN forecaster."""

from __future__ import annotations

import numpy as np

from db.features import enrich_snapshot_metrics, extract_surface_vector, estimate_gamma_flip
from models.backtest import run_backtest
from models.predict import predict_next_snapshot, similar_setups
from tests.synthetic_data import generate_synthetic_history
import pandas as pd


def test_surface_vector_normalized():
    strike = pd.Series([0.01, -0.02, 0.03], index=[100.0, 105.0, 110.0])
    vec = extract_surface_vector(strike, spot=105.0, n_bins=8)
    assert vec.shape == (8,)
    assert np.linalg.norm(vec) <= 1.0 + 1e-9 or np.allclose(vec, 0)


def test_gamma_flip_zero_crossing():
    cumulative = pd.Series([-1.0, -0.2, 0.3, 0.8], index=[100.0, 105.0, 110.0, 115.0])
    flip = estimate_gamma_flip(cumulative)
    assert flip is not None
    assert 105.0 < flip < 110.0


def test_enrich_snapshot_metrics():
    history = generate_synthetic_history(n_snapshots=5)
    enriched = enrich_snapshot_metrics(history[-1].copy())
    assert enriched["gamma_flip"] is not None
    assert enriched["surface_vector"].shape == (32,)
    assert "wall_spread" in enriched


def test_predict_next_snapshot():
    history = generate_synthetic_history(n_snapshots=40)
    forecast = predict_next_snapshot(history, lookback_days=30)
    assert forecast is not None
    assert "predicted_delta_gex" in forecast
    assert "confidence" in forecast
    assert 0.0 <= forecast["confidence"] <= 1.0
    assert forecast["spot_bias"] in ("up", "down", "neutral")


def test_insufficient_data_returns_none():
    history = generate_synthetic_history(n_snapshots=2)
    assert predict_next_snapshot(history) is None


def test_similar_setups():
    history = generate_synthetic_history(n_snapshots=30)
    setups = similar_setups(history, top_n=3)
    assert len(setups) <= 3
    assert setups[0]["similarity"] > 0


def test_backtest_produces_metrics():
    history = generate_synthetic_history(n_snapshots=60)
    report = run_backtest(history, lookback_days=30)
    assert report.n_forecasts > 0
    assert report.mae_delta_gex >= 0
    assert 0 <= report.regime_accuracy <= 1
