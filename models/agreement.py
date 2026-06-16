"""Quant model agreement scoring for agent confidence calibration."""

from __future__ import annotations

from typing import Any

from db.features import safe_float


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def compute_agreement(
    *,
    knn: dict[str, Any] | None,
    llm: dict[str, Any] | None = None,
    gboost_delta: float | None = None,
    online_delta: float | None = None,
    ensemble: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score 0–1 how aligned quant outputs are."""
    if not knn:
        return {"score": 0.0, "regime_agreement": None, "delta_spread": None, "notes": "no KNN forecast"}

    knn_regime = str(knn.get("predicted_regime") or "").upper()
    knn_delta = safe_float(knn.get("predicted_delta_gex"))

    deltas = [("knn", knn_delta)]
    if gboost_delta is not None:
        deltas.append(("gboost", float(gboost_delta)))
    if online_delta is not None:
        deltas.append(("online", float(online_delta)))
    if ensemble and ensemble.get("ensemble_delta_gex") is not None:
        deltas.append(("ensemble", float(ensemble["ensemble_delta_gex"])))

    signs = [_sign(v) for _, v in deltas if v != 0]
    sign_agree = len(set(signs)) <= 1 if signs else True
    spread = max(v for _, v in deltas) - min(v for _, v in deltas) if len(deltas) > 1 else 0.0

    regime_agree = None
    if llm:
        llm_regime = str(llm.get("predicted_regime") or "").upper()
        if knn_regime and llm_regime:
            regime_agree = ("LONG" in knn_regime) == ("LONG" in llm_regime)
        llm_delta = safe_float(llm.get("predicted_delta_gex_bn") or llm.get("predicted_delta_gex"))
        if llm_delta != 0:
            deltas.append(("llm", llm_delta))
            spread = max(v for _, v in deltas) - min(v for _, v in deltas)

    score = 1.0
    if not sign_agree:
        score -= 0.35
    if spread > 0.08:
        score -= 0.25
    elif spread > 0.04:
        score -= 0.1
    if regime_agree is False:
        score -= 0.25
    score = max(0.0, min(1.0, score))

    notes = []
    if score >= 0.75:
        notes.append("Models largely agree — cite ensemble view confidently.")
    elif score >= 0.5:
        notes.append("Mixed signals — explain disagreement and cite strike-level facts.")
    else:
        notes.append("Low agreement — be cautious; flag uncertainty explicitly.")

    return {
        "score": round(score, 2),
        "regime_agreement": regime_agree,
        "delta_sign_agreement": sign_agree,
        "delta_spread_bn": round(spread, 4),
        "components": {n: v for n, v in deltas},
        "notes": " ".join(notes),
    }
