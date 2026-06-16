"""Ensemble ΔGEX forecast blending KNN, GBoost, and River online learner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config
from db.features import safe_float


def default_weights() -> dict[str, float]:
    return {
        "knn": float(getattr(config, "ENSEMBLE_WEIGHT_KNN", 0.5)),
        "gboost": float(getattr(config, "ENSEMBLE_WEIGHT_GBOOST", 0.25)),
        "online": float(getattr(config, "ENSEMBLE_WEIGHT_ONLINE", 0.25)),
    }


def weights_path(ticker: str) -> Path:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return config.MODELS_DIR / f"{ticker.upper()}_ensemble_weights.json"


def load_weights(ticker: str) -> dict[str, float]:
    path = weights_path(ticker)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return {k: float(v) for k, v in data.items() if k in ("knn", "gboost", "online")}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return default_weights()


def save_weights(ticker: str, weights: dict[str, float]) -> None:
    weights_path(ticker).write_text(json.dumps({"ticker": ticker.upper(), **weights}, indent=2))


def blend_delta(
    *,
    knn_delta: float | None,
    gboost_delta: float | None,
    online_delta: float | None,
    ticker: str,
) -> dict[str, Any]:
    """Weighted ensemble of available model deltas."""
    weights = load_weights(ticker)
    parts: list[tuple[str, float, float]] = []
    if knn_delta is not None:
        parts.append(("knn", weights.get("knn", 0.5), float(knn_delta)))
    if gboost_delta is not None:
        parts.append(("gboost", weights.get("gboost", 0.25), float(gboost_delta)))
    if online_delta is not None:
        parts.append(("online", weights.get("online", 0.25), float(online_delta)))
    if not parts:
        return {"ensemble_delta_gex": None, "weights_used": {}, "components": {}}
    w_sum = sum(w for _, w, _ in parts)
    if w_sum <= 0:
        w_sum = len(parts)
        parts = [(n, 1.0, v) for n, _, v in parts]
    delta = sum((w / w_sum) * v for _, w, v in parts)
    return {
        "ensemble_delta_gex": delta,
        "weights_used": {n: round(w / w_sum, 3) for n, w, _ in parts},
        "components": {n: v for n, _, v in parts},
    }


def learn_weights_from_backtest(report: dict[str, Any], ticker: str) -> dict[str, float]:
    """Heuristic weight update from walk-forward backtest MAE."""
    mae = safe_float(report.get("mae_delta_gex"), 1.0)
    regime_acc = safe_float(report.get("regime_accuracy"), 0.5)
    weights = load_weights(ticker)
    if mae < 0.05 and regime_acc > 0.55:
        weights["knn"] = min(0.65, weights.get("knn", 0.5) + 0.05)
    elif mae > 0.1:
        weights["gboost"] = min(0.4, weights.get("gboost", 0.25) + 0.05)
        weights["online"] = min(0.35, weights.get("online", 0.25) + 0.05)
    total = sum(weights.values())
    weights = {k: round(v / total, 3) for k, v in weights.items()}
    save_weights(ticker, weights)
    return weights
