"""Compress LLM context bundles to reduce tokens and latency."""

from __future__ import annotations

import copy
from typing import Any


def compress_bundle(bundle: dict[str, Any], *, max_strikes: int = 5, max_timeline: int = 8) -> dict[str, Any]:
    """Trim non-essential fields while preserving key trading signals."""
    if not bundle:
        return bundle
    b = copy.deepcopy(bundle)

    if isinstance(b.get("top_strikes"), list):
        b["top_strikes"] = b["top_strikes"][:max_strikes]
    if isinstance(b.get("atm_strike_band"), list):
        b["atm_strike_band"] = b["atm_strike_band"][:max_strikes]
    if isinstance(b.get("cumulative_gex_near_spot"), list):
        b["cumulative_gex_near_spot"] = b["cumulative_gex_near_spot"][:max_strikes]
    if isinstance(b.get("intraday_timeline"), list):
        b["intraday_timeline"] = b["intraday_timeline"][-max_timeline:]
    if isinstance(b.get("similar_setups"), list):
        b["similar_setups"] = b["similar_setups"][:3]
    if isinstance(b.get("similar_sessions"), list):
        b["similar_sessions"] = b["similar_sessions"][:2]
    if isinstance(b.get("knn_forecast"), dict) and isinstance(b["knn_forecast"].get("neighbors"), list):
        b["knn_forecast"]["neighbors"] = b["knn_forecast"]["neighbors"][:2]

    track = b.get("forecast_track_record") or {}
    if isinstance(track.get("recent_outcomes"), list):
        track["recent_outcomes"] = track["recent_outcomes"][:4]
        b["forecast_track_record"] = track

    exp = b.get("gex_by_expiration")
    if isinstance(exp, dict) and len(exp) > 6:
        top = sorted(exp.items(), key=lambda kv: abs(float(kv[1]) if kv[1] is not None else 0), reverse=True)[:6]
        b["gex_by_expiration"] = dict(top)

    b["_compressed"] = True
    return b
