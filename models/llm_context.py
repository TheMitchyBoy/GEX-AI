"""Build LLM context bundles from Postgres snapshot history."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

import config
from db.connection import get_connection
from db.features import enrich_snapshot_metrics, safe_float, select_atm_strike_series
from db.prediction_history import fetch_recent_resolved_outcomes
from db.queries import fetch_calibration_stats
from models.gboost import predict_gboost_delta
from models.llm_rag import retrieve_outcome_matched_sessions, retrieve_regime_matched_sessions, retrieve_similar_sessions
from models.online_learn import predict_online_delta
from models.ensemble import blend_delta
from models.agreement import compute_agreement
from models.context_compress import compress_bundle
from models.predict import attribute_last_move, predict_next_snapshot, similar_setups


def _top_strikes(strike: pd.Series, n: int = 8) -> list[dict[str, float]]:
    if strike is None or strike.empty:
        return []
    ranked = strike.abs().sort_values(ascending=False).head(n)
    return [{"strike": float(k), "gex_bn_per_pct": float(strike.loc[k])} for k in ranked.index]


def _atm_strike_band(strike: pd.Series, spot: float, *, max_strikes: int = 20) -> list[dict[str, float]]:
    if strike is None or strike.empty or spot <= 0:
        return []
    near = select_atm_strike_series(strike, spot, window_pct=0.03, min_strikes=8)
    rows = [{"strike": float(k), "gex_bn_per_pct": float(near.loc[k])} for k in near.index]
    return rows[:max_strikes]


def _cumulative_near_spot(cumulative: pd.Series, spot: float, n: int = 12) -> list[dict[str, float]]:
    if cumulative is None or cumulative.empty or spot <= 0:
        return []
    near = select_atm_strike_series(cumulative, spot, window_pct=0.03, min_strikes=5)
    return [{"strike": float(k), "cumulative_gex_bn": float(near.loc[k])} for k in near.index][:n]


def _forecast_track_record(ticker: str) -> dict[str, Any]:
    try:
        with get_connection() as conn:
            stats = fetch_calibration_stats(conn, ticker, limit=200)
            recent = fetch_recent_resolved_outcomes(conn, ticker, limit=8)
        return {"calibration": stats, "recent_outcomes": recent}
    except Exception:
        return {"calibration": {"n": 0}, "recent_outcomes": []}


def build_context_bundle(
    history: list[dict[str, Any]],
    *,
    lookback_days: int,
    intraday_points: int = 12,
    slim: bool = True,
    rich: bool = False,
) -> dict[str, Any]:
    if not history:
        return {}

    enriched = [enrich_snapshot_metrics(h.copy()) for h in history]
    current = enriched[-1]
    summary = current.get("summary") or {}
    ticker = current.get("ticker", config.DEFAULT_TICKER)

    knn = predict_next_snapshot(history, lookback_days=lookback_days)
    analogs = similar_setups(history, top_n=5 if rich else 3, lookback_days=lookback_days)
    timeline_n = 24 if rich else 8
    recent = enriched[-timeline_n:]

    exp_json = current.get("expiration_json") or summary.get("expiration_json")
    if isinstance(exp_json, str):
        try:
            exp_json = json.loads(exp_json)
        except json.JSONDecodeError:
            exp_json = {}

    strike = current.get("strike", pd.Series(dtype=float))
    cumulative = current.get("cumulative", pd.Series(dtype=float))
    spot = safe_float(current.get("spot"))

    bundle: dict[str, Any] = {
        "ticker": ticker,
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
            "flow_aggressiveness": safe_float(summary.get("flow_aggressiveness")),
            "event_risk_score": safe_float(summary.get("event_risk_score")),
            "is_fomc_week": summary.get("is_fomc_week"),
            "is_opex_week": summary.get("is_opex_week"),
            "net_charm_bn": summary.get("net_charm_bn"),
            "net_vanna_bn": summary.get("net_vanna_bn"),
            "net_delta_bn": summary.get("net_delta_bn"),
            "gamma_oi_bn": summary.get("gamma_oi_bn"),
            "gamma_vol_bn": summary.get("gamma_vol_bn"),
            "vix_level": summary.get("vix_level"),
            "spy_return": summary.get("spy_return"),
        },
        "top_strikes": _top_strikes(strike, n=10 if rich else 6),
        "gex_by_expiration": exp_json if isinstance(exp_json, dict) else {},
        "intraday_timeline": [
            {
                "ts": row["ts"],
                "spot": row["spot"],
                "total_gex_bn": row["total_gex"],
                "regime": row.get("regime"),
                "gamma_flip": safe_float(row.get("gamma_flip")),
            }
            for row in recent
        ],
        "knn_forecast": _slim_knn(knn),
        "similar_setups": analogs,
    }

    if rich:
        gboost = predict_gboost_delta(history, str(ticker))
        online = predict_online_delta(history, str(ticker)) if config.ONLINE_LEARNING_ENABLED else None
        knn_delta = knn.get("predicted_delta_gex") if knn else None
        ensemble = blend_delta(
            knn_delta=knn_delta,
            gboost_delta=gboost,
            online_delta=online,
            ticker=str(ticker),
        ) if config.ENSEMBLE_ENABLED else {}
        bundle["quant_synthesis"] = {
            "knn_delta_gex_bn": knn_delta,
            "knn_confidence": knn.get("confidence") if knn else None,
            "knn_regime_flip_probability": knn.get("regime_flip_probability") if knn else None,
            "gboost_delta_gex_bn": gboost,
            "online_delta_gex_bn": online,
            "ensemble_delta_gex_bn": ensemble.get("ensemble_delta_gex"),
            "ensemble_weights": ensemble.get("weights_used"),
            "blend_note": "Synthesize quant outputs; cite specific strike/flow facts if disagreeing with KNN.",
        }
        bundle["model_agreement"] = compute_agreement(
            knn=knn,
            gboost_delta=gboost,
            online_delta=online,
            ensemble=ensemble,
        )
        bundle["atm_strike_band"] = _atm_strike_band(strike, spot, max_strikes=8)
        bundle["cumulative_gex_near_spot"] = _cumulative_near_spot(cumulative, spot, n=8)
        bundle["last_move_attribution"] = attribute_last_move(enriched)
        bundle["similar_sessions"] = retrieve_similar_sessions(history, top_n=3)
        bundle["regime_matched_sessions"] = retrieve_regime_matched_sessions(history, top_n=2)
        bundle["outcome_matched_sessions"] = retrieve_outcome_matched_sessions(history, knn_delta, top_n=2)
        bundle["forecast_track_record"] = _forecast_track_record(str(ticker))
        if isinstance(bundle["gex_by_expiration"], dict) and slim:
            pass  # keep full expirations in rich mode

    if config.LLM_CONTEXT_COMPRESS:
        bundle = compress_bundle(bundle)

    return bundle


def bundle_to_prompt_json(bundle: dict[str, Any]) -> str:
    return json.dumps(bundle, indent=2, default=str)


def estimate_token_count(bundle: dict[str, Any]) -> int:
    return max(1, len(bundle_to_prompt_json(bundle)) // 4)


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
        "gboost_delta_gex_bn": knn.get("gboost_delta_gex"),
    }
