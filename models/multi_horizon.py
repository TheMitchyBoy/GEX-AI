"""Multi-horizon GEX forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np

import config
from db.features import attach_market_features, enrich_snapshot_metrics, snapshot_feature_vector
from models.predict import MIN_KNN_SNAPSHOTS, _weighted_knn_predict, prepare_training_rows, select_recent_history


def predict_multi_horizon(
    history: list[dict[str, Any]],
    horizons: tuple[int, ...] | None = None,
    k: int = 4,
    lookback_days: int | None = None,
) -> dict[int, dict[str, Any]]:
    from models.predict import predict_next_snapshot

    horizons = horizons or config.MULTI_HORIZONS
    results: dict[int, dict[str, Any]] = {}
    base = predict_next_snapshot(history, k=k, lookback_days=lookback_days)
    if base is None:
        return results
    if 1 in horizons:
        results[1] = base

    lookback_days = lookback_days if lookback_days is not None else config.LOOKBACK_DAYS
    windowed = select_recent_history(history, lookback_days=lookback_days, min_snapshots=MIN_KNN_SNAPSHOTS)
    enriched = [enrich_snapshot_metrics(h.copy()) for h in windowed]
    attach_market_features(enriched)
    current = enriched[-1]
    x_now = snapshot_feature_vector(current)

    for horizon in horizons:
        if horizon <= 1:
            continue
        rows = []
        for i in range(len(enriched) - horizon):
            cur, nxt = enriched[i], enriched[i + horizon]
            rows.append((snapshot_feature_vector(cur), nxt["total_gex"] - cur["total_gex"]))
        if len(rows) < 3:
            continue
        x_train = np.vstack([r[0] for r in rows])
        targets = {"delta_gex": np.array([r[1] for r in rows])}
        preds, _, _, confidence, intervals = _weighted_knn_predict(x_train, targets, x_now, k=min(k, len(rows)))
        low, high = intervals["delta_gex"]
        results[horizon] = {
            "horizon": horizon,
            "horizon_minutes": horizon * 10,
            "predicted_delta_gex": preds["delta_gex"],
            "predicted_total_gex": current["total_gex"] + preds["delta_gex"],
            "predicted_delta_gex_low": low,
            "predicted_delta_gex_high": high,
            "confidence": confidence,
        }
    return results
