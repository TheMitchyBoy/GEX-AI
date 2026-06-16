"""Session-level RAG: retrieve similar trading days from snapshot history."""

from __future__ import annotations

from typing import Any

import numpy as np

from db.features import enrich_snapshot_metrics, safe_float


def _session_summary(day_snaps: list[dict[str, Any]]) -> dict[str, Any]:
    enriched = [enrich_snapshot_metrics(s.copy()) for s in day_snaps]
    first, last = enriched[0], enriched[-1]
    spots = [safe_float(e.get("spot")) for e in enriched]
    gex = [safe_float(e.get("total_gex")) for e in enriched]
    return {
        "market_date": last.get("market_date"),
        "snapshot_count": len(enriched),
        "open_spot": spots[0],
        "close_spot": spots[-1],
        "spot_move_pct": ((spots[-1] - spots[0]) / spots[0] * 100) if spots[0] else 0.0,
        "avg_gex_bn": float(np.mean(gex)) if gex else 0.0,
        "end_regime": last.get("regime"),
        "end_total_gex": last.get("total_gex"),
        "end_gamma_flip": safe_float(last.get("gamma_flip")),
        "avg_flip_distance_pct": float(np.mean([safe_float(e.get("flip_distance_pct")) for e in enriched])),
        "is_fomc_week": (last.get("summary") or {}).get("is_fomc_week"),
    }


def _session_vector(summary: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            summary.get("avg_gex_bn", 0.0),
            summary.get("spot_move_pct", 0.0),
            summary.get("avg_flip_distance_pct", 0.0),
            1.0 if "LONG" in str(summary.get("end_regime", "")).upper() else -1.0,
            safe_float(summary.get("is_fomc_week")),
        ],
        dtype=float,
    )


def retrieve_similar_sessions(
    history: list[dict[str, Any]],
    *,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Group snapshots by market_date and rank past sessions vs today."""
    if len(history) < 5:
        return []

    by_date: dict[str, list[dict[str, Any]]] = {}
    for snap in history:
        md = snap.get("market_date") or snap.get("ts", "")[:10]
        by_date.setdefault(md, []).append(snap)

    sessions = {md: _session_summary(snaps) for md, snaps in by_date.items() if len(snaps) >= 3}
    if len(sessions) < 2:
        return []

    current_date = history[-1].get("market_date") or history[-1]["ts"][:10]
    current = sessions.get(current_date)
    if not current:
        return []

    cur_vec = _session_vector(current)
    scored: list[tuple[float, dict[str, Any]]] = []
    for md, summary in sessions.items():
        if md == current_date:
            continue
        vec = _session_vector(summary)
        dist = float(np.linalg.norm(vec - cur_vec))
        scored.append((dist, {**summary, "similarity": 1.0 / (1.0 + dist)}))

    scored.sort(key=lambda x: x[0])
    return [s for _, s in scored[:top_n]]


def retrieve_regime_matched_sessions(
    history: list[dict[str, Any]],
    *,
    top_n: int = 3,
    flip_window_pct: float = 0.005,
) -> list[dict[str, Any]]:
    """Sessions where regime AND near-flip distance match today."""
    if len(history) < 5:
        return []
    current = enrich_snapshot_metrics(history[-1].copy())
    cur_regime = str(current.get("regime") or "").upper()
    cur_flip_dist = abs(safe_float(current.get("flip_distance_pct")))

    by_date: dict[str, list[dict[str, Any]]] = {}
    for snap in history:
        md = snap.get("market_date") or snap.get("ts", "")[:10]
        by_date.setdefault(md, []).append(snap)

    current_date = history[-1].get("market_date") or history[-1]["ts"][:10]
    matches: list[dict[str, Any]] = []
    for md, snaps in by_date.items():
        if md == current_date or len(snaps) < 3:
            continue
        summary = _session_summary(snaps)
        regime = str(summary.get("end_regime") or "").upper()
        flip_dist = abs(safe_float(summary.get("avg_flip_distance_pct")))
        if ("LONG" in cur_regime) != ("LONG" in regime):
            continue
        if abs(flip_dist - cur_flip_dist) > flip_window_pct:
            continue
        matches.append({**summary, "match_type": "regime_and_flip_proximity"})
    matches.sort(key=lambda s: abs(s.get("spot_move_pct", 0)))
    return matches[:top_n]


def retrieve_outcome_matched_sessions(
    history: list[dict[str, Any]],
    knn_delta: float | None,
    *,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Past sessions whose next-day move sign matched today's KNN ΔGEX sign."""
    if knn_delta is None or len(history) < 10:
        return []
    target_sign = 1 if knn_delta > 0 else (-1 if knn_delta < 0 else 0)
    if target_sign == 0:
        return []

    by_date: dict[str, list[dict[str, Any]]] = {}
    for snap in history:
        md = snap.get("market_date") or snap.get("ts", "")[:10]
        by_date.setdefault(md, []).append(snap)

    dates = sorted(by_date.keys())
    current_date = history[-1].get("market_date") or history[-1]["ts"][:10]
    matches: list[dict[str, Any]] = []
    for i, md in enumerate(dates[:-1]):
        if md == current_date:
            continue
        snaps = by_date[md]
        if len(snaps) < 3:
            continue
        summary = _session_summary(snaps)
        move_sign = 1 if summary.get("spot_move_pct", 0) > 0 else (-1 if summary.get("spot_move_pct", 0) < 0 else 0)
        if move_sign == target_sign:
            matches.append({**summary, "match_type": "outcome_sign_aligned_with_knn"})
    matches.sort(key=lambda s: -abs(s.get("spot_move_pct", 0)))
    return matches[:top_n]
