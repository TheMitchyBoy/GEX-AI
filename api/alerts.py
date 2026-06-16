"""Alert evaluation and webhook delivery."""

from __future__ import annotations

import logging
from typing import Any

import httpx

import config

logger = logging.getLogger(__name__)


def evaluate_alerts(forecast: dict[str, Any], enriched: dict[str, Any]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    flip_prob = float(forecast.get("regime_flip_probability") or 0)
    if flip_prob >= config.ALERT_REGIME_FLIP_THRESHOLD:
        alerts.append({"type": "regime_flip", "severity": "high", "value": flip_prob, "message": f"Regime flip P={flip_prob:.0%}"})

    spot = float(enriched.get("spot") or forecast.get("current_spot") or 0)
    flip = float(enriched.get("gamma_flip") or forecast.get("current_gamma_flip") or 0)
    if spot > 0 and flip > 0:
        dist = abs(flip - spot) / spot
        if dist <= config.ALERT_FLIP_DISTANCE_PCT:
            alerts.append({"type": "near_flip", "severity": "medium", "value": dist, "message": f"Spot within {dist:.2%} of gamma flip"})

    delta = abs(float(forecast.get("predicted_delta_gex") or forecast.get("predicted_delta_gex_bn") or 0))
    if delta >= config.ALERT_DELTA_GEX_THRESHOLD:
        alerts.append({"type": "large_delta_gex", "severity": "medium", "value": delta, "message": f"Large ΔGEX forecast: {delta:.3f}"})
    return alerts


def send_alerts(alerts: list[dict[str, Any]], *, ticker: str, ts: str) -> bool:
    if not alerts or not config.ALERT_WEBHOOK_URL:
        return False
    payload = {"ticker": ticker, "snapshot_ts": ts, "alerts": alerts}
    try:
        httpx.post(config.ALERT_WEBHOOK_URL, json=payload, timeout=10.0)
        logger.info("Sent %s alerts for %s", len(alerts), ticker)
        return True
    except Exception:
        logger.exception("Alert webhook failed")
        return False
