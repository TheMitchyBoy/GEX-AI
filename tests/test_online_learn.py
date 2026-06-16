"""Tests for River online learning integration."""

from __future__ import annotations

import config
from models.online_learn import (
    learn_pair,
    maybe_learn_latest,
    model_path,
    predict_online_delta,
    warm_start,
)
from tests.synthetic_data import generate_synthetic_history


def _clean_online_artifacts(ticker: str = "SPX") -> None:
    for suffix in (".river.pkl", ".json"):
        p = config.MODELS_DIR / f"{ticker.upper()}_online_gex{suffix}"
        if p.exists():
            p.unlink()


def test_warm_start_and_predict():
    _clean_online_artifacts()
    history = generate_synthetic_history(n_snapshots=50)
    result = warm_start(history, "SPX")
    assert result["ok"] is True
    assert result["n_updates"] >= 20
    assert model_path("SPX").exists()

    pred = predict_online_delta(history, "SPX")
    assert pred is not None
    assert isinstance(pred, float)
    _clean_online_artifacts()


def test_incremental_learn_latest():
    _clean_online_artifacts()
    history = generate_synthetic_history(n_snapshots=40)
    warm_start(history[:30], "SPX")
    full = history[:32]
    n = maybe_learn_latest(full, "SPX")
    assert n == 1
    n2 = maybe_learn_latest(full, "SPX")
    assert n2 == 0
    _clean_online_artifacts()


def test_learn_pair():
    _clean_online_artifacts()
    history = generate_synthetic_history(n_snapshots=10)
    assert learn_pair(history[0], history[1], "SPX") is True
    _clean_online_artifacts()


def test_predict_requires_min_updates(monkeypatch):
    _clean_online_artifacts()
    monkeypatch.setattr(config, "ONLINE_MIN_UPDATES", 100)
    history = generate_synthetic_history(n_snapshots=30)
    warm_start(history, "SPX")
    pred = predict_online_delta(history, "SPX")
    assert pred is None
    _clean_online_artifacts()
