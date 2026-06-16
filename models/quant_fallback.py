"""Quant-only agent reply when OpenAI is unavailable."""

from __future__ import annotations

from typing import Any

from db.features import safe_float


def quant_only_reply(bundle: dict[str, Any], *, agreement: dict[str, Any] | None = None) -> str:
    """Conversational quant summary when LLM is off."""
    summary = bundle.get("summary") or {}
    knn = bundle.get("knn_forecast") or {}
    quant = bundle.get("quant_synthesis") or {}

    spot = safe_float(summary.get("spot"))
    flip = safe_float(summary.get("gamma_flip"))
    gex = safe_float(summary.get("total_gex_bn_per_pct"))
    regime = summary.get("net_gamma_regime") or "unknown"
    flip_dist = safe_float(summary.get("flip_distance_pct"))
    call_wall = safe_float(summary.get("call_wall"))
    put_wall = safe_float(summary.get("put_wall"))

    parts = [
        f"Quick read from the quant models (LLM is off): we're in **{regime}** with spot at **{spot:.2f}** "
        f"and total GEX around **{gex:.3f} Bn$/1%**.",
    ]
    if flip > 0:
        parts.append(
            f"Gamma flip sits at **{flip:.0f}** — that's about **{abs(flip_dist)*100:.2f}%** "
            f"{'below' if flip_dist < 0 else 'above'} spot, which matters for how dealers hedge into moves."
        )
    if call_wall > 0 or put_wall > 0:
        parts.append(f"Walls to watch: call **{call_wall:.0f}**, put **{put_wall:.0f}**.")
    if knn:
        parts.append(
            f"KNN sees ΔGEX of **{safe_float(knn.get('predicted_delta_gex_bn')):.4f}** next snapshot, "
            f"bias **{knn.get('spot_bias', 'neutral')}**, confidence **{safe_float(knn.get('confidence')):.0%}**."
        )
    if quant.get("gboost_delta_gex_bn") is not None:
        parts.append(f"GBoost lines up at ΔGEX **{safe_float(quant.get('gboost_delta_gex_bn')):.4f}**.")
    if agreement and agreement.get("score") is not None:
        parts.append(f"Model agreement is **{agreement['score']:.0%}** — {agreement.get('notes', '')}")
    parts.append("\n*(Educational only — not financial advice.)*")
    return " ".join(parts)
