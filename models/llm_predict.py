"""LLM-enhanced GEX predictions with KNN/rule-based fallback."""

from __future__ import annotations

import json
import logging
from typing import Any

import config
from db.connection import get_connection
from db.features import safe_float
from db.queries import insert_prediction
from models.llm_client import is_llm_configured, openai_chat_json, parse_prediction_json
from models.llm_context import build_context_bundle, bundle_to_prompt_json
from models.predict import predict_next_snapshot

logger = logging.getLogger(__name__)

PREDICTION_SCHEMA = {
    "predicted_regime": "LONG gamma | SHORT gamma",
    "predicted_delta_gex_bn": "float — expected change in total GEX next snapshot",
    "predicted_total_gex_bn": "float — expected total GEX next snapshot",
    "spot_bias": "bullish | bearish | neutral",
    "confidence": "0.0–1.0",
    "gamma_flip": "float strike or null",
    "key_levels": {
        "support": ["float strikes"],
        "resistance": ["float strikes"],
        "pin": "float strike or null",
    },
    "scenarios": [{"label": "str", "probability": "0.0–1.0", "description": "str"}],
    "predictions": ["list of 3–5 actionable prediction strings"],
    "reasoning": "2–3 sentence explanation citing specific data points",
}

SYSTEM_PROMPT = (
    "You are an expert options market analyst specializing in dealer gamma exposure (GEX). "
    "You receive a JSON bundle from a Postgres GEX snapshot pipeline: current regime metrics, "
    "strike concentrations, expiration term structure, intraday timeline, weighted KNN forecast, "
    "and similar historical setups.\n\n"
    "Analyze ALL provided data holistically. Weight net GEX regime, gamma flip proximity, "
    "call/put walls, term structure, flow/event flags, KNN forecast, and historical analogs.\n\n"
    "Respond with ONLY valid JSON matching this schema:\n"
    f"{json.dumps(PREDICTION_SCHEMA, indent=2)}"
)


def _normalize_prediction(parsed: dict[str, Any], *, source: str) -> dict[str, Any]:
    spot_bias = str(parsed.get("spot_bias", parsed.get("predicted_spot_bias", "neutral"))).lower()
    if spot_bias in ("up", "long"):
        spot_bias = "bullish"
    elif spot_bias in ("down", "short"):
        spot_bias = "bearish"

    return {
        "predicted_regime": str(parsed.get("predicted_regime", "neutral")),
        "predicted_delta_gex_bn": safe_float(parsed.get("predicted_delta_gex_bn")),
        "predicted_total_gex_bn": safe_float(parsed.get("predicted_total_gex_bn")),
        "spot_bias": spot_bias,
        "confidence": min(1.0, max(0.0, safe_float(parsed.get("confidence"), 0.5))),
        "gamma_flip": safe_float(parsed.get("gamma_flip")) or None,
        "key_levels": parsed.get("key_levels") or {"support": [], "resistance": [], "pin": None},
        "scenarios": parsed.get("scenarios") or [],
        "predictions": parsed.get("predictions") or [],
        "reasoning": str(parsed.get("reasoning", "")),
        "llm_enhanced": source == "llm",
        "prediction_source": source,
    }


def _rule_based_from_knn(knn: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    summary = bundle.get("summary") or {}
    spot_bias_map = {"up": "bullish", "down": "bearish", "neutral": "neutral"}
    knn_bias = spot_bias_map.get(str(knn.get("spot_bias", "neutral")).lower(), "neutral")

    return _normalize_prediction(
        {
            "predicted_regime": knn.get("predicted_regime"),
            "predicted_delta_gex_bn": knn.get("predicted_delta_gex"),
            "predicted_total_gex_bn": knn.get("predicted_total_gex"),
            "spot_bias": knn_bias,
            "confidence": knn.get("confidence"),
            "gamma_flip": knn.get("predicted_flip") or summary.get("gamma_flip"),
            "key_levels": {
                "support": [summary.get("put_wall")] if summary.get("put_wall") else [],
                "resistance": [summary.get("call_wall")] if summary.get("call_wall") else [],
                "pin": summary.get("call_wall"),
            },
            "scenarios": [
                {
                    "label": "KNN baseline",
                    "probability": knn.get("confidence", 0.5),
                    "description": (
                        f"Weighted KNN expects ΔGEX {knn.get('predicted_delta_gex'):.3f} "
                        f"with regime flip P={knn.get('regime_flip_probability', 0):.0%}"
                    ),
                }
            ],
            "predictions": [
                f"Next snapshot regime: {knn.get('predicted_regime')}",
                f"Spot bias toward magnets/flip: {knn.get('spot_bias')}",
            ],
            "reasoning": (
                f"Rule-based fallback using KNN on {knn.get('training_snapshot_count', 0)} snapshots. "
                f"Current {summary.get('net_gamma_regime')} at spot {summary.get('spot')}."
            ),
        },
        source="knn_fallback",
    )


def generate_llm_forecast(
    history: list[dict[str, Any]],
    *,
    lookback_days: int | None = None,
    persist: bool | None = None,
    extra_instructions: str | None = None,
) -> dict[str, Any]:
    lookback_days = lookback_days if lookback_days is not None else config.LOOKBACK_DAYS
    persist = config.WRITE_PREDICTIONS if persist is None else persist

    bundle = build_context_bundle(history, lookback_days=lookback_days)
    knn = predict_next_snapshot(history, lookback_days=lookback_days)

    result: dict[str, Any] | None = None
    llm_error: str | None = None

    if is_llm_configured():
        user_content = (
            "Using every data point in this GEX context bundle, produce structured market "
            "predictions for the next ~10 minute snapshot interval.\n\n"
            f"{bundle_to_prompt_json(bundle)}"
        )
        if extra_instructions:
            user_content += f"\n\nAdditional instructions:\n{extra_instructions}"
        parsed, llm_error = openai_chat_json(SYSTEM_PROMPT, user_content)
        if parsed:
            result = _normalize_prediction(parsed, source="llm")

    if result is None:
        if knn:
            result = _rule_based_from_knn(knn, bundle)
        else:
            result = _normalize_prediction(
                {
                    "predicted_regime": bundle.get("summary", {}).get("net_gamma_regime", "unknown"),
                    "predicted_delta_gex_bn": 0.0,
                    "predicted_total_gex_bn": safe_float(bundle.get("summary", {}).get("total_gex_bn_per_pct")),
                    "spot_bias": "neutral",
                    "confidence": 0.2,
                    "gamma_flip": bundle.get("summary", {}).get("gamma_flip"),
                    "predictions": ["Insufficient history for quantitative forecast"],
                    "reasoning": "Not enough snapshots for KNN; no LLM available.",
                },
                source="insufficient_data",
            )

    out = {
        **result,
        "ticker": bundle.get("ticker"),
        "snapshot_ts": bundle.get("snapshot_ts"),
        "market_date": bundle.get("market_date"),
        "knn_forecast": knn,
        "context_summary": {
            "timeline_points": len(bundle.get("intraday_timeline", [])),
            "top_strike_count": len(bundle.get("top_strikes", [])),
            "has_knn_forecast": knn is not None,
            "similar_setup_count": len(bundle.get("similar_setups", [])),
            "llm_configured": is_llm_configured(),
            "llm_error": llm_error,
        },
    }

    if persist and bundle.get("snapshot_ts"):
        _persist_prediction(out)

    return out


def _persist_prediction(prediction: dict[str, Any]) -> None:
    payload = {
        "predicted_regime": prediction.get("predicted_regime"),
        "predicted_delta_gex_bn": prediction.get("predicted_delta_gex_bn"),
        "predicted_total_gex_bn": prediction.get("predicted_total_gex_bn"),
        "spot_bias": prediction.get("spot_bias"),
        "confidence": prediction.get("confidence"),
        "gamma_flip": prediction.get("gamma_flip"),
        "key_levels": prediction.get("key_levels"),
        "scenarios": prediction.get("scenarios"),
        "predictions": prediction.get("predictions"),
        "reasoning": prediction.get("reasoning"),
        "llm_enhanced": prediction.get("llm_enhanced"),
        "prediction_source": prediction.get("prediction_source"),
        "knn_forecast": prediction.get("knn_forecast"),
    }
    try:
        with get_connection() as conn:
            insert_prediction(
                conn,
                ticker=str(prediction.get("ticker", config.DEFAULT_TICKER)),
                snapshot_ts=str(prediction["snapshot_ts"]),
                market_date=str(prediction.get("market_date") or prediction["snapshot_ts"][:10]),
                payload=payload,
                source=config.LLM_PREDICTION_SOURCE if prediction.get("llm_enhanced") else config.PREDICTION_SOURCE,
            )
    except Exception:
        logger.exception("Failed to persist LLM prediction")
