"""Evaluate GEX agent grounding, agreement, and quant alignment."""

from __future__ import annotations

from typing import Any

from db.features import enrich_snapshot_metrics, safe_float
from models.agreement import compute_agreement
from models.llm_agent import chat_with_agent
from models.predict import predict_next_snapshot


def evaluate_agent_grounding(
    history: list[dict[str, Any]],
    *,
    lookback_days: int = 30,
) -> dict[str, Any]:
    """Run probes and check citations vs DB ground truth."""
    if len(history) < 4:
        return {"error": "insufficient history", "scores": {}}

    current = enrich_snapshot_metrics(history[-1].copy())
    knn = predict_next_snapshot(history, lookback_days=lookback_days)
    ground_truth = {
        "spot": safe_float(current.get("spot")),
        "regime": str(current.get("regime") or ""),
        "gamma_flip": safe_float(current.get("gamma_flip")),
        "total_gex": safe_float(current.get("total_gex")),
        "call_wall": safe_float(current.get("call_wall")),
        "put_wall": safe_float(current.get("put_wall")),
        "flip_distance_pct": safe_float(current.get("flip_distance_pct")),
    }
    agreement = compute_agreement(knn=knn)

    probes = [
        ("regime", "What is the current gamma regime? Reply with the regime name only."),
        ("spot", f"What is the current spot level? The answer should be near {ground_truth['spot']:.0f}."),
        ("flip", f"What is the gamma flip level? It should be near {ground_truth['gamma_flip']:.0f}."),
        ("walls", "What are the call wall and put wall strikes? Cite numbers."),
    ]

    results = []
    for name, probe in probes:
        resp = chat_with_agent(
            history,
            [{"role": "user", "content": probe}],
            lookback_days=lookback_days,
            refresh_context=True,
            mode="fast",
        )
        results.append({"probe": name, "reply": resp.get("reply"), "error": resp.get("error")})

    def _contains(text: str | None, needle: str) -> bool:
        return needle.lower() in (text or "").lower()

    regime_ok = _contains(results[0].get("reply"), ground_truth["regime"].split()[0] if ground_truth["regime"] else "")
    spot_ok = str(int(ground_truth["spot"])) in (results[1].get("reply") or "").replace(",", "")
    flip_ok = str(int(ground_truth["gamma_flip"])) in (results[1].get("reply") or "").replace(",", "") if ground_truth["gamma_flip"] else False
    if not flip_ok and len(results) > 2:
        flip_ok = str(int(ground_truth["gamma_flip"])) in (results[2].get("reply") or "").replace(",", "")
    walls_ok = False
    if len(results) > 3:
        reply = results[3].get("reply") or ""
        walls_ok = str(int(ground_truth["call_wall"])) in reply or str(int(ground_truth["put_wall"])) in reply

    checks = [regime_ok, spot_ok, flip_ok, walls_ok]
    knn_sign = knn.get("predicted_delta_gex") if knn else None
    quant_aligned = None
    if knn_sign is not None and results[0].get("reply"):
        quant_aligned = ("long" in results[0]["reply"].lower()) == (knn_sign >= 0 if "LONG" in str(knn.get("predicted_regime", "")).upper() else knn_sign < 0)

    return {
        "ground_truth": ground_truth,
        "model_agreement": agreement,
        "probes": results,
        "scores": {
            "regime_citation": regime_ok,
            "spot_citation": spot_ok,
            "flip_citation": flip_ok,
            "walls_citation": walls_ok,
            "grounding_rate": sum(checks) / len(checks),
            "quant_regime_aligned": quant_aligned,
        },
    }
