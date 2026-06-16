"""Walk-forward backtesting for GEX forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import config
from db.features import enrich_snapshot_metrics, safe_float
from models.predict import MIN_KNN_SNAPSHOTS, predict_next_snapshot


@dataclass
class BacktestReport:
    ticker: str
    lookback_days: int
    n_forecasts: int
    mae_delta_gex: float
    regime_accuracy: float
    spot_bias_hit_rate: float
    interval_coverage: float
    rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "lookback_days": self.lookback_days,
            "n_forecasts": self.n_forecasts,
            "mae_delta_gex": self.mae_delta_gex,
            "regime_accuracy": self.regime_accuracy,
            "spot_bias_hit_rate": self.spot_bias_hit_rate,
            "interval_coverage": self.interval_coverage,
        }


def _spot_direction(spot_now: float, spot_next: float) -> str:
    if spot_next > spot_now * 1.0001:
        return "up"
    if spot_next < spot_now * 0.9999:
        return "down"
    return "neutral"


def run_backtest(
    history: list[dict[str, Any]],
    *,
    lookback_days: int | None = None,
    min_train: int = MIN_KNN_SNAPSHOTS,
) -> BacktestReport:
    lookback_days = lookback_days if lookback_days is not None else config.LOOKBACK_DAYS
    ticker = history[0].get("ticker", config.DEFAULT_TICKER) if history else config.DEFAULT_TICKER
    rows: list[dict[str, Any]] = []

    enriched_all = [enrich_snapshot_metrics(h.copy()) for h in history]

    for i in range(min_train, len(history)):
        window = history[:i]
        actual = enriched_all[i]
        prev = enriched_all[i - 1]
        forecast = predict_next_snapshot(window, lookback_days=lookback_days)
        if not forecast:
            continue

        actual_delta = safe_float(actual.get("total_gex")) - safe_float(prev.get("total_gex"))
        pred_delta = safe_float(forecast.get("predicted_delta_gex"))
        low = safe_float(forecast.get("predicted_delta_gex_low"))
        high = safe_float(forecast.get("predicted_delta_gex_high"))
        actual_regime = "LONG gamma" if safe_float(actual.get("total_gex")) >= 0 else "SHORT gamma"
        pred_regime = forecast.get("predicted_regime")
        spot_now = safe_float(prev.get("spot"))
        spot_next = safe_float(actual.get("spot"))
        actual_dir = _spot_direction(spot_now, spot_next)
        pred_bias = forecast.get("spot_bias", "neutral")

        rows.append(
            {
                "ts": actual["ts"],
                "actual_delta_gex": actual_delta,
                "predicted_delta_gex": pred_delta,
                "abs_error": abs(actual_delta - pred_delta),
                "in_interval": low <= actual_delta <= high,
                "regime_correct": actual_regime == pred_regime,
                "spot_bias": pred_bias,
                "spot_direction": actual_dir,
                "spot_bias_hit": pred_bias == "neutral" or pred_bias == actual_dir,
                "confidence": forecast.get("confidence"),
            }
        )

    if not rows:
        return BacktestReport(
            ticker=ticker,
            lookback_days=lookback_days,
            n_forecasts=0,
            mae_delta_gex=0.0,
            regime_accuracy=0.0,
            spot_bias_hit_rate=0.0,
            interval_coverage=0.0,
            rows=[],
        )

    return BacktestReport(
        ticker=ticker,
        lookback_days=lookback_days,
        n_forecasts=len(rows),
        mae_delta_gex=float(np.mean([r["abs_error"] for r in rows])),
        regime_accuracy=float(np.mean([1.0 if r["regime_correct"] else 0.0 for r in rows])),
        spot_bias_hit_rate=float(np.mean([1.0 if r["spot_bias_hit"] else 0.0 for r in rows])),
        interval_coverage=float(np.mean([1.0 if r["in_interval"] else 0.0 for r in rows])),
        rows=rows,
    )
