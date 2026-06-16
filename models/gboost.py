"""Gradient boosting overlay for ΔGEX prediction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import config
from db.features import enrich_snapshot_metrics, safe_float

logger = logging.getLogger(__name__)
LAG_FEATURES = [
    "total_gex", "pos_gex", "neg_gex", "gex_std", "near_term_ratio", "gamma_flip",
    "flip_distance_pct", "wall_spread", "zero_dte_ratio", "term_curvature",
]


def _row_features(enriched: dict[str, Any]) -> dict[str, float]:
    return {k: safe_float(enriched.get(k)) for k in LAG_FEATURES}


def build_training_matrix(history: list[dict[str, Any]], lag: int = 4) -> tuple[pd.DataFrame, pd.Series] | None:
    enriched = [enrich_snapshot_metrics(h.copy()) for h in history]
    rows, targets = [], []
    for i in range(lag, len(enriched) - 1):
        flat: dict[str, float] = {}
        for j in range(lag):
            feats = _row_features(enriched[i - lag + 1 + j])
            for k, v in feats.items():
                flat[f"{k}_lag{j}"] = v
        rows.append(flat)
        targets.append(enriched[i + 1]["total_gex"] - enriched[i]["total_gex"])
    if len(rows) < 8:
        return None
    return pd.DataFrame(rows), pd.Series(targets)


def model_path(ticker: str) -> Path:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return config.MODELS_DIR / f"{ticker.upper()}_gex_delta.joblib"


def train_gboost(history: list[dict[str, Any]], ticker: str) -> dict[str, Any] | None:
    data = build_training_matrix(history)
    if data is None:
        return None
    X, y = data
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score

        model = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=42)
        scores = cross_val_score(model, X, y, cv=min(5, len(X) // 3), scoring="neg_mean_absolute_error")
        model.fit(X, y)
        bundle = {"model": model, "features": list(X.columns), "n_train": len(X), "cv_mae": float(-scores.mean())}
        joblib.dump(bundle, model_path(ticker))
        meta = model_path(ticker).with_suffix(".json")
        meta.write_text(json.dumps({"ticker": ticker, "n_train": len(X), "cv_mae": bundle["cv_mae"]}))
        return bundle
    except Exception:
        logger.exception("GBoost training failed for %s", ticker)
        return None


def predict_gboost_delta(history: list[dict[str, Any]], ticker: str, lag: int = 4) -> float | None:
    path = model_path(ticker)
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
        enriched = [enrich_snapshot_metrics(h.copy()) for h in history]
        if len(enriched) < lag:
            return None
        flat: dict[str, float] = {}
        for j in range(lag):
            feats = _row_features(enriched[-lag + j])
            for k, v in feats.items():
                flat[f"{k}_lag{j}"] = v
        row = pd.DataFrame([flat])
        for col in bundle["features"]:
            if col not in row.columns:
                row[col] = 0.0
        return float(bundle["model"].predict(row[bundle["features"]])[0])
    except Exception:
        logger.warning("GBoost inference failed for %s", ticker, exc_info=True)
        return None
