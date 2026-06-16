"""Build LLM context bundles from Postgres snapshot history."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from db.features import enrich_snapshot_metrics, safe_float
from models.predict import predict_next_snapshot, similar_setups


def _top_strikes(strike: pd.Series, n: int = 8) -> list[dict[str, float]]:
    if strike is None or strike.empty:
        return []
    ranked = strike.abs().sort_values(ascending=False).head(n)
    return [{"strike": float(k), "gex_bn_per_pct": float(strike.loc[k])} for k in ranked.index]


def build_context_bundle(
    history: list[dict[str, Any]],
    *,
    lookback_days: int,
    intraday_points: int = 24,
) -> dict[str, Any]:
    if not history:
        return {}

    enriched = [enrich_snapshot_metrics(h.copy()) for h in history]
    current = enriched[-1]
    summary = current.get("summary") or {}

    knn = predict_next_snapshot(history, lookback_days=lookback_days)
    analogs = similar_setups(history, top_n=5, lookback_days=lookback_days)

    recent = enriched[-intraday_points:]
    timeline = [
        {
            "ts": row["ts"],
            "spot": row["spot"],
            "total_gex_bn": row["total_gex"],
            "regime": row.get("regime"),
            "gamma_flip": safe_float(row.get("gamma_flip")),
        }
        for row in recent
    ]

    exp_json = current.get("expiration_json") or summary.get("expiration_json")
    if isinstance(exp_json, str):
        try:
            exp_json = json.loads(exp_json)
        except json.JSONDecodeError:
            exp_json = {}

    return {
        "ticker": current.get("ticker"),
        "market_date": current.get("market_date"),
        "snapshot_ts": current["ts"],
        "summary": {
            "spot": current["spot"],
            "total_gex_bn_per_pct": current["total_gex"],
            "net_gamma_regime": current.get("regime"),
            "gamma_flip": safe_float(current.get("gamma_flip")),
            "call_wall": safe_float(current.get("call_wall")),
            "put_wall": safe_float(current.get("put_wall")),
            "wall_spread": safe_float(current.get("wall_spread")),
            "flip_distance_pct": safe_float(current.get("flip_distance_pct")),
            "near_term_ratio": safe_float(current.get("near_term_ratio")),
            "zero_dte_ratio": safe_float(current.get("zero_dte_ratio")),
            "term_curvature": safe_float(current.get("term_curvature")),
            "flow_event_count": safe_float(summary.get("flow_event_count")),
            "flow_buy_ratio": safe_float(summary.get("flow_buy_ratio")),
            "event_risk_score": safe_float(summary.get("event_risk_score")),
            "is_fomc_week": summary.get("is_fomc_week"),
            "net_charm_bn": summary.get("net_charm_bn"),
            "net_vanna_bn": summary.get("net_vanna_bn"),
            "net_delta_bn": summary.get("net_delta_bn"),
            "vix_level": summary.get("vix_level"),
            "spy_return": summary.get("spy_return"),
        },
        "top_strikes": _top_strikes(current.get("strike", pd.Series(dtype=float))),
        "gex_by_expiration": exp_json if isinstance(exp_json, dict) else {},
        "intraday_timeline": timeline,
        "knn_forecast": _slim_knn(knn),
        "similar_setups": analogs,
    }


def bundle_to_prompt_json(bundle: dict[str, Any]) -> str:
    return json.dumps(bundle, indent=2, default=str)


def _slim_knn(knn: dict[str, Any] | None) -> dict[str, Any] | None:
    if not knn:
        return None
    return {
        "predicted_delta_gex_bn": knn.get("predicted_delta_gex"),
        "predicted_total_gex_bn": knn.get("predicted_total_gex"),
        "predicted_regime": knn.get("predicted_regime"),
        "predicted_flip": knn.get("predicted_flip"),
        "spot_bias": knn.get("spot_bias"),
        "confidence": knn.get("confidence"),
        "prediction_interval": knn.get("prediction_interval"),
        "regime_flip_probability": knn.get("regime_flip_probability"),
        "neighbors": knn.get("neighbors", [])[:3],
        "last_move_attribution": knn.get("last_move_attribution"),
    }
