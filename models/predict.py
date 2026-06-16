"""Weighted KNN baseline forecaster for next GEX snapshot."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

import config
from db.features import (
    attach_market_features,
    compute_spot_bias,
    enrich_snapshot_metrics,
    parse_timestamp,
    safe_float,
    snapshot_feature_vector,
)

MIN_KNN_SNAPSHOTS = config.MIN_KNN_SNAPSHOTS
RECENCY_DECAY = config.RECENCY_DECAY
INTERVAL_Z = config.INTERVAL_Z


def _zscore_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (matrix - mean) / std, mean, std


def _weighted_knn_predict(
    train_features: np.ndarray,
    train_targets: dict[str, np.ndarray],
    query: np.ndarray,
    k: int = 4,
    surface_vectors: list[np.ndarray] | None = None,
    query_surface: np.ndarray | None = None,
    surface_weight: float = 0.35,
    recency_weights: np.ndarray | None = None,
) -> tuple[dict[str, float], list[int], np.ndarray, float, dict[str, tuple[float, float]]]:
    z_train, mean, std = _zscore_matrix(train_features)
    z_query = (query - mean) / std
    distances = np.linalg.norm(z_train - z_query, axis=1)

    if surface_vectors and query_surface is not None and len(surface_vectors) == len(distances):
        for i, sv in enumerate(surface_vectors):
            denom = max(np.linalg.norm(sv) * np.linalg.norm(query_surface), 1e-12)
            cos_dist = 1.0 - float(np.dot(sv, query_surface) / denom)
            distances[i] = (1 - surface_weight) * distances[i] + surface_weight * cos_dist

    k = min(k, len(distances))
    nn_idx = np.argsort(distances)[:k]
    nn_dist = distances[nn_idx]
    weights = 1.0 / (nn_dist + 1e-6)
    if recency_weights is not None and len(recency_weights) == len(distances):
        weights = weights * recency_weights[nn_idx]
    weights = weights / weights.sum()

    predictions: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    for key, targets in train_targets.items():
        point = float(np.sum(weights * targets[nn_idx]))
        predictions[key] = point
        var = float(np.sum(weights * (targets[nn_idx] - point) ** 2))
        std_t = var**0.5
        intervals[key] = (point - INTERVAL_Z * std_t, point + INTERVAL_Z * std_t)

    avg_dist = float(nn_dist.mean())
    confidence = max(0.0, min(1.0, 1.0 / (1.0 + avg_dist)))
    return predictions, list(nn_idx), weights, confidence, intervals


def _calibrate_confidence(raw_confidence: float, train_count: int) -> tuple[float, dict[str, Any]]:
    sample_factor = max(0.0, min(1.0, (train_count - 3) / 25.0))
    calibrated = raw_confidence * (0.45 + 0.40 * sample_factor)
    calibrated = max(0.0, min(1.0, calibrated))
    return calibrated, {
        "raw_neighbor_confidence": raw_confidence,
        "sample_factor": sample_factor,
        "training_rows": train_count,
        "method": "neighbor_distance_sample_damped",
    }


def select_recent_history(
    history: list[dict[str, Any]],
    lookback_days: int | None = None,
    min_snapshots: int = 0,
) -> list[dict[str, Any]]:
    lookback_days = lookback_days if lookback_days is not None else config.LOOKBACK_DAYS
    ordered = sorted(history, key=lambda row: row["ts"])
    if not ordered or not lookback_days or lookback_days <= 0:
        return ordered
    latest = parse_timestamp(ordered[-1]["ts"])
    cutoff = latest - timedelta(days=lookback_days)
    windowed = [row for row in ordered if parse_timestamp(row["ts"]) >= cutoff]
    if min_snapshots and len(windowed) < min_snapshots:
        return ordered[-min_snapshots:]
    return windowed


def prepare_training_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for i in range(len(history) - 1):
        cur = enrich_snapshot_metrics(history[i].copy())
        nxt = history[i + 1]
        prev = enrich_snapshot_metrics(history[i - 1].copy()) if i > 0 else None
        if prev:
            cur["total_gex_momentum"] = cur["total_gex"] - prev["total_gex"]
            cur["flip_velocity"] = safe_float(cur["gamma_flip"]) - safe_float(prev.get("gamma_flip"), 0.0)
        else:
            cur["total_gex_momentum"] = 0.0
            cur["flip_velocity"] = 0.0

        rows.append(
            {
                "ts": cur["ts"],
                "features": snapshot_feature_vector(cur),
                "surface_vector": cur.get("surface_vector", np.zeros(config.SURFACE_BINS)),
                "target_total_gex": nxt["total_gex"],
                "target_delta_gex": nxt["total_gex"] - cur["total_gex"],
                "target_flip": safe_float(nxt.get("gamma_flip"), safe_float(cur.get("gamma_flip"), 0.0)),
                "target_near_term_ratio": safe_float(nxt.get("near_term_ratio"), 0.0),
                "target_zero_dte_ratio": safe_float(nxt.get("zero_dte_ratio"), 0.0),
                "target_term_curvature": safe_float(nxt.get("term_curvature"), 0.0),
                "target_strike": nxt.get("strike"),
                "next_ts": nxt["ts"],
            }
        )
    return rows


def _regime_flip_probability(train: list[dict], nn_idx: list[int], current_total: float) -> float:
    flips = 0
    for i in nn_idx:
        before = train[i]["target_total_gex"] - train[i]["target_delta_gex"]
        after = train[i]["target_total_gex"]
        if (before >= 0) != (after >= 0):
            flips += 1
    base_prob = flips / max(len(nn_idx), 1)
    if current_total != 0 and abs(
        current_total + np.mean([train[i]["target_delta_gex"] for i in nn_idx])
    ) < abs(current_total) * 0.1:
        base_prob = min(1.0, base_prob + 0.15)
    return float(base_prob)


def attribute_last_move(enriched: list[dict[str, Any]]) -> dict[str, Any]:
    if len(enriched) < 2:
        return {"message": "insufficient history"}
    prev = enriched[-2]
    cur = enriched[-1]
    delta_gex = safe_float(cur.get("total_gex")) - safe_float(prev.get("total_gex"))

    prev_strike = prev.get("strike", pd.Series(dtype=float))
    cur_strike = cur.get("strike", pd.Series(dtype=float))
    if isinstance(prev_strike, pd.Series) and isinstance(cur_strike, pd.Series) and not cur_strike.empty:
        aligned = cur_strike.reindex(cur_strike.index.union(prev_strike.index), fill_value=0.0)
        prev_aligned = prev_strike.reindex(aligned.index, fill_value=0.0)
        strike_delta = aligned - prev_aligned
        top = strike_delta.abs().sort_values(ascending=False).head(5)
        drivers = [
            {"strike": float(k), "delta_gex_bn": float(v)}
            for k, v in top.items()
            if abs(v) > 1e-6
        ]
    else:
        drivers = []

    prev_exp = prev.get("summary", {}).get("expiration_json") or prev.get("expiration_json")
    cur_exp = cur.get("summary", {}).get("expiration_json") or cur.get("expiration_json")
    exp_drivers: list[dict[str, Any]] = []
    if isinstance(prev_exp, dict) and isinstance(cur_exp, dict):
        for key in set(prev_exp) | set(cur_exp):
            d = safe_float(cur_exp.get(key)) - safe_float(prev_exp.get(key))
            if abs(d) > 1e-6:
                exp_drivers.append({"expiration": key, "delta_gex_bn": d})
        exp_drivers.sort(key=lambda x: abs(x["delta_gex_bn"]), reverse=True)

    return {
        "delta_gex_bn": delta_gex,
        "top_strike_drivers": drivers[:5],
        "top_expiration_drivers": exp_drivers[:5],
        "from_ts": prev["ts"],
        "to_ts": cur["ts"],
    }


def predict_next_snapshot(
    history: list[dict[str, Any]],
    k: int = 4,
    lookback_days: int | None = None,
) -> dict[str, Any] | None:
    lookback_days = lookback_days if lookback_days is not None else config.LOOKBACK_DAYS
    windowed = select_recent_history(history, lookback_days=lookback_days)
    if len(windowed) < MIN_KNN_SNAPSHOTS:
        windowed = select_recent_history(history, lookback_days=lookback_days, min_snapshots=MIN_KNN_SNAPSHOTS)
    if len(windowed) < MIN_KNN_SNAPSHOTS:
        return None

    enriched = [enrich_snapshot_metrics(h.copy()) for h in windowed]
    attach_market_features(enriched)
    train = prepare_training_rows(enriched)
    if len(train) < 3:
        return None

    knn_train = train[:-1] if len(train) > 4 else train
    current = enriched[-1]
    prev = enriched[-2] if len(enriched) > 1 else None
    if prev:
        current["total_gex_momentum"] = current["total_gex"] - prev["total_gex"]
        current["flip_velocity"] = safe_float(current["gamma_flip"]) - safe_float(prev.get("gamma_flip"), 0.0)

    x_train = np.vstack([row["features"] for row in knn_train])
    x_now = snapshot_feature_vector(current)
    surface_vectors = [row["surface_vector"] for row in knn_train]
    query_surface = current.get("surface_vector", np.zeros(config.SURFACE_BINS))

    n_train = len(knn_train)
    recency_weights = np.array([RECENCY_DECAY ** (n_train - 1 - i) for i in range(n_train)], dtype=float)

    targets = {
        "total_gex": np.array([row["target_total_gex"] for row in knn_train]),
        "delta_gex": np.array([row["target_delta_gex"] for row in knn_train]),
        "flip": np.array([row["target_flip"] for row in knn_train]),
        "near_term_ratio": np.array([row["target_near_term_ratio"] for row in knn_train]),
        "zero_dte_ratio": np.array([row["target_zero_dte_ratio"] for row in knn_train]),
        "term_curvature": np.array([row["target_term_curvature"] for row in knn_train]),
    }

    preds, nn_idx, nn_weights, confidence, intervals = _weighted_knn_predict(
        x_train,
        targets,
        x_now,
        k=k,
        surface_vectors=surface_vectors,
        query_surface=query_surface,
        recency_weights=recency_weights,
    )
    preds["total_gex"] = current["total_gex"] + preds["delta_gex"]
    confidence, confidence_breakdown = _calibrate_confidence(confidence, len(train))

    neighbors = []
    for rank, i in enumerate(nn_idx, start=1):
        src = next(row for row in enriched if row["ts"] == knn_train[i]["ts"])
        neighbors.append(
            {
                "rank": rank,
                "snapshot": src["ts_label"],
                "next_snapshot": parse_timestamp(knn_train[i]["next_ts"]).strftime("%Y-%m-%d %H:%M:%S"),
                "distance": float(np.linalg.norm(x_train[i] - x_now)),
                "next_total_gex": float(knn_train[i]["target_total_gex"]),
                "next_delta_gex": float(knn_train[i]["target_delta_gex"]),
            }
        )

    regime_flip_prob = _regime_flip_probability(knn_train, nn_idx, current["total_gex"])
    predicted_regime = "LONG gamma" if preds["total_gex"] >= 0 else "SHORT gamma"
    delta_low, delta_high = intervals.get("delta_gex", (preds["delta_gex"], preds["delta_gex"]))
    neighbor_deltas = targets["delta_gex"][nn_idx]
    neighbor_mae = float(np.mean(np.abs(neighbor_deltas - preds["delta_gex"])))
    attribution = attribute_last_move(enriched)

    spot_bias = compute_spot_bias(
        spot=safe_float(current.get("spot")),
        predicted_flip=preds["flip"],
        call_wall=safe_float(current.get("call_wall")),
        put_wall=safe_float(current.get("put_wall")),
        magnet=safe_float(current.get("max_positive_magnet")),
        predicted_regime=predicted_regime,
    )

    return {
        "predicted_total_gex": preds["total_gex"],
        "predicted_delta_gex": preds["delta_gex"],
        "predicted_regime": predicted_regime,
        "predicted_flip": preds["flip"],
        "spot_bias": spot_bias,
        "confidence": confidence,
        "confidence_breakdown": confidence_breakdown,
        "predicted_delta_gex_low": delta_low,
        "predicted_delta_gex_high": delta_high,
        "predicted_total_gex_low": current["total_gex"] + delta_low,
        "predicted_total_gex_high": current["total_gex"] + delta_high,
        "prediction_interval": {"low": delta_low, "high": delta_high},
        "regime_flip_probability": regime_flip_prob,
        "neighbor_typical_abs_error": neighbor_mae,
        "last_move_attribution": attribution,
        "neighbors": neighbors,
        "current_snapshot_ts": current["ts"],
        "current_spot": current["spot"],
        "current_total_gex": current["total_gex"],
        "current_regime": current.get("regime"),
        "current_gamma_flip": current.get("gamma_flip"),
        "current_call_wall": current.get("call_wall"),
        "current_put_wall": current.get("put_wall"),
        "training_snapshot_count": len(enriched),
        "training_window_days": lookback_days,
        "forecast_horizon": "next_snapshot",
        "model": "weighted_knn",
    }


def similar_setups(
    history: list[dict[str, Any]],
    top_n: int = 5,
    lookback_days: int | None = None,
) -> list[dict[str, Any]]:
    lookback_days = lookback_days if lookback_days is not None else config.LOOKBACK_DAYS
    history = select_recent_history(history, lookback_days=lookback_days)
    if len(history) < 3:
        return []

    enriched = [enrich_snapshot_metrics(h.copy()) for h in history]
    current = enriched[-1]
    rows = []
    for i in range(len(enriched) - 1):
        row = enriched[i]
        rows.append((i, row, snapshot_feature_vector(row), row.get("surface_vector", np.zeros(config.SURFACE_BINS))))

    current_feat = snapshot_feature_vector(current)
    current_surface = current.get("surface_vector", np.zeros(config.SURFACE_BINS))

    matrix = np.vstack([feat for _, _, feat, _ in rows])
    z_matrix, mean, std = _zscore_matrix(matrix)
    z_current = (current_feat - mean) / std
    distances = np.linalg.norm(z_matrix - z_current, axis=1)

    for i, (_, _, _, sv) in enumerate(rows):
        denom = max(np.linalg.norm(sv) * np.linalg.norm(current_surface), 1e-12)
        cos_dist = 1.0 - float(np.dot(sv, current_surface) / denom)
        distances[i] = 0.65 * distances[i] + 0.35 * cos_dist

    idx_sorted = np.argsort(distances)[: min(top_n, len(distances))]
    results = []
    for idx in idx_sorted:
        hist_idx, snap, _, _ = rows[idx]
        next_snap = enriched[hist_idx + 1] if hist_idx + 1 < len(enriched) else None
        delta = (next_snap["total_gex"] - snap["total_gex"]) if next_snap else None
        results.append(
            {
                "snapshot": snap["ts_label"],
                "distance": float(distances[idx]),
                "similarity": 1.0 / (1.0 + float(distances[idx])),
                "regime": snap.get("regime"),
                "total_gex": snap["total_gex"],
                "next_snapshot": next_snap["ts_label"] if next_snap else None,
                "next_total_gex": next_snap["total_gex"] if next_snap else None,
                "next_delta_gex": delta,
                "next_regime": next_snap.get("regime") if next_snap else None,
                "ts": snap["ts"],
            }
        )
    return results
