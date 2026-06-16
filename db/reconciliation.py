"""Reconcile logged predictions against realized next snapshots."""

from __future__ import annotations

import json
import logging
from typing import Any

from db.connection import get_connection
from db.features import safe_float
from db.queries import (
    fetch_next_ts_after,
    fetch_snapshot_at_ts,
    fetch_unresolved_predictions,
    resolve_prediction,
)

logger = logging.getLogger(__name__)


def _outcome_metrics(predicted: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    pred_delta = safe_float(
        predicted.get("predicted_delta_gex_bn") or predicted.get("predicted_delta_gex"), 0.0
    )
    actual_delta = safe_float(actual.get("delta_gex_bn"), 0.0)
    sign_hit = (pred_delta >= 0) == (actual_delta >= 0) if pred_delta != 0 or actual_delta != 0 else True

    spot0 = safe_float(actual.get("spot_before"), 0.0)
    spot1 = safe_float(actual.get("spot_after"), 0.0)
    spot_move = (spot1 - spot0) / spot0 if spot0 > 0 and spot1 > 0 else 0.0
    bias = str(predicted.get("spot_bias") or predicted.get("bias") or "neutral").lower()
    bias_hit = None
    if bias in {"bullish", "long", "up"} and spot_move != 0:
        bias_hit = spot_move > 0
    elif bias in {"bearish", "short", "down"} and spot_move != 0:
        bias_hit = spot_move < 0
    elif bias in {"neutral", "mean_reversion"}:
        bias_hit = abs(spot_move) < 0.004

    pred_regime = str(predicted.get("predicted_regime") or "").upper()
    actual_regime = str(actual.get("regime") or "").upper()
    regime_hit = ("LONG" in pred_regime) == ("LONG" in actual_regime) if pred_regime and actual_regime else None

    return {
        "delta_mae": abs(pred_delta - actual_delta),
        "sign_hit": sign_hit,
        "bias_hit": bias_hit,
        "regime_hit": regime_hit,
        "confidence": safe_float(predicted.get("confidence"), 0.0),
        "spot_move_pct": round(spot_move * 100, 4),
    }


def reconcile_predictions(ticker: str) -> int:
    """Resolve open predictions once the next snapshot after anchor is available."""
    ticker = ticker.upper()
    resolved = 0
    with get_connection() as conn:
        rows = fetch_unresolved_predictions(conn, ticker)
        for row in rows:
            anchor_ts = row["snapshot_ts"]
            next_ts = fetch_next_ts_after(conn, ticker, anchor_ts)
            if not next_ts:
                continue
            before = fetch_snapshot_at_ts(conn, ticker, anchor_ts)
            after = fetch_snapshot_at_ts(conn, ticker, next_ts)
            if not before or not after:
                continue
            try:
                predicted = row["payload_json"]
                if isinstance(predicted, str):
                    predicted = json.loads(predicted)
            except json.JSONDecodeError:
                continue

            actual = {
                "snapshot_ts": next_ts,
                "spot_before": before.get("spot"),
                "spot_after": after.get("spot"),
                "regime": after.get("regime"),
                "total_gex_bn": after.get("total_gex"),
                "delta_gex_bn": safe_float(after.get("total_gex"), 0.0)
                - safe_float(before.get("total_gex"), 0.0),
            }
            outcome = _outcome_metrics(predicted, actual)
            resolve_prediction(conn, int(row["id"]), actual, outcome)
            resolved += 1
    if resolved:
        logger.info("Reconciled %s predictions for %s", resolved, ticker)
    return resolved
