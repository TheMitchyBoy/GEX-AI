"""Quant-only agent reply when OpenAI is unavailable."""

from __future__ import annotations

from typing import Any

from db.features import safe_float


def quant_only_reply(bundle: dict[str, Any], *, agreement: dict[str, Any] | None = None) -> str:
    """Generate a rule-based markdown summary from KNN + context."""
    summary = bundle.get("summary") or {}
    knn = bundle.get("knn_forecast") or {}
    quant = bundle.get("quant_synthesis") or {}
    attr = bundle.get("last_move_attribution") or {}

    spot = safe_float(summary.get("spot"))
    flip = safe_float(summary.get("gamma_flip"))
    gex = safe_float(summary.get("total_gex_bn_per_pct"))
    regime = summary.get("net_gamma_regime") or "unknown"
    flip_dist = safe_float(summary.get("flip_distance_pct"))

    lines = [
        "**Current state** (quant-only mode — OpenAI unavailable)",
        f"- Spot: {spot:.2f} | Regime: {regime} | Total GEX: {gex:.3f} Bn$/1%",
        f"- Gamma flip: {flip:.2f} | Distance: {flip_dist*100:.2f}%",
        f"- Call wall: {safe_float(summary.get('call_wall')):.0f} | Put wall: {safe_float(summary.get('put_wall')):.0f}",
        "",
        "**KNN forecast**",
        f"- ΔGEX: {safe_float(knn.get('predicted_delta_gex_bn')):.4f} Bn$/1%",
        f"- Predicted regime: {knn.get('predicted_regime', 'n/a')}",
        f"- Confidence: {safe_float(knn.get('confidence')):.0%}",
        f"- Spot bias: {knn.get('spot_bias', 'neutral')}",
    ]
    if quant.get("gboost_delta_gex_bn") is not None:
        lines.append(f"- GBoost ΔGEX: {safe_float(quant.get('gboost_delta_gex_bn')):.4f}")
    if attr:
        lines.extend(["", "**Last move attribution**", f"- {attr}"])
    if agreement:
        lines.extend([
            "",
            f"**Model agreement score:** {agreement.get('score', 0):.0%}",
            f"- {agreement.get('notes', '')}",
        ])
    lines.append("\n*Educational analysis only — not financial advice.*")
    return "\n".join(lines)
