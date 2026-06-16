"""Confidence calibration from backtest and resolved predictions."""

from __future__ import annotations

from typing import Any

import config
from db.connection import get_connection
from db.queries import fetch_calibration_stats


def calibrate_confidence(raw: float, empirical_accuracy: float | None, sample_count: int) -> float:
    if empirical_accuracy is None or sample_count < 5:
        return max(0.0, min(1.0, raw * 0.85))
    gap = abs(raw - empirical_accuracy)
    adjusted = raw * (1.0 - 0.5 * gap)
    sample_factor = min(1.0, sample_count / 30.0)
    return max(0.0, min(1.0, adjusted * (0.7 + 0.3 * sample_factor) + empirical_accuracy * 0.1))


def get_empirical_accuracy(ticker: str, *, source: str | None = None) -> dict[str, Any]:
    try:
        with get_connection() as conn:
            return fetch_calibration_stats(conn, ticker.upper(), source=source)
    except Exception:
        return {"n": 0}


def apply_calibration(raw_confidence: float, ticker: str, *, source: str | None = None) -> tuple[float, dict[str, Any]]:
    stats = get_empirical_accuracy(ticker, source=source)
    calibrated = calibrate_confidence(raw_confidence, stats.get("sign_accuracy"), stats.get("n", 0))
    return calibrated, {"raw": raw_confidence, "calibrated": calibrated, "empirical_stats": stats}
