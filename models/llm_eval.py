"""Evaluate GEX agent grounding and forecast alignment."""

from __future__ import annotations

from typing import Any

from db.features import enrich_snapshot_metrics, safe_float
from models.llm_agent import chat_with_agent


def evaluate_agent_grounding(
    history: list[dict[str, Any]],
    *,
    lookback_days: int = 30,
) -> dict[str, Any]:
    """Run standard Q&A probes and check citations vs DB ground truth."""
    if len(history) < 4:
        return {"error": "insufficient history", "scores": {}}

    current = enrich_snapshot_metrics(history[-1].copy())
    ground_truth = {
        "spot": safe_float(current.get("spot")),
        "regime": str(current.get("regime") or ""),
        "gamma_flip": safe_float(current.get("gamma_flip")),
        "total_gex": safe_float(current.get("total_gex")),
        "call_wall": safe_float(current.get("call_wall")),
        "put_wall": safe_float(current.get("put_wall")),
    }

    probes = [
        "What is the current gamma regime? Reply with the regime name only.",
        f"What is the current spot level? The answer should be near {ground_truth['spot']:.0f}.",
    ]

    results = []
    for probe in probes:
        resp = chat_with_agent(
            history,
            [{"role": "user", "content": probe}],
            lookback_days=lookback_days,
            refresh_context=True,
            use_tools=False,
        )
        reply = (resp.get("reply") or "").lower()
        results.append({"probe": probe, "reply": resp.get("reply"), "error": resp.get("error")})

    regime_ok = ground_truth["regime"].lower() in (results[0].get("reply") or "").lower() if results else False
    spot_text = results[1].get("reply") or "" if len(results) > 1 else ""
    spot_ok = str(int(ground_truth["spot"])) in spot_text.replace(",", "")

    return {
        "ground_truth": ground_truth,
        "probes": results,
        "scores": {
            "regime_citation": regime_ok,
            "spot_citation": spot_ok,
            "grounding_rate": sum([regime_ok, spot_ok]) / 2.0,
        },
    }
