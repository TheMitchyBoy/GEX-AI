"""Poll for new snapshots and optionally write forecasts to llm_predictions."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import config
from db.connection import get_connection
from db.loader import load_snapshot_history
from db.queries import get_latest_ts, insert_prediction
from models.predict import predict_next_snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_once(ticker: str | None = None) -> dict[str, Any] | None:
    ticker = ticker or config.DEFAULT_TICKER
    with get_connection() as conn:
        latest_ts = get_latest_ts(conn, ticker)
    if not latest_ts:
        logger.warning("No snapshots for %s", ticker)
        return None

    history = load_snapshot_history(ticker, lookback_days=config.LOOKBACK_DAYS)
    forecast = predict_next_snapshot(history, lookback_days=config.LOOKBACK_DAYS)
    if not forecast:
        logger.warning("Insufficient history to forecast for %s", ticker)
        return None

    payload = {
        "predicted_delta_gex_bn": forecast["predicted_delta_gex"],
        "predicted_total_gex_bn": forecast["predicted_total_gex"],
        "predicted_regime": forecast["predicted_regime"],
        "predicted_flip": forecast["predicted_flip"],
        "spot_bias": forecast["spot_bias"],
        "confidence": forecast["confidence"],
        "prediction_interval": forecast["prediction_interval"],
        "regime_flip_probability": forecast["regime_flip_probability"],
        "model": forecast["model"],
    }

    if config.WRITE_PREDICTIONS:
        market_date = history[-1].get("market_date", latest_ts[:10])
        with get_connection() as conn:
            insert_prediction(
                conn,
                ticker=ticker,
                snapshot_ts=latest_ts,
                market_date=market_date,
                payload=payload,
                source=config.PREDICTION_SOURCE,
            )
        logger.info("Wrote prediction for %s @ %s", ticker, latest_ts)
    else:
        logger.info("Forecast for %s @ %s: %s", ticker, latest_ts, json.dumps(payload, default=str))

    return {"latest_ts": latest_ts, "forecast": forecast}


def poll_loop() -> None:
    ticker = config.DEFAULT_TICKER
    last_ts: str | None = None
    logger.info("Starting forecast poller for %s (interval=%ss)", ticker, config.FORECAST_POLL_SEC)

    while True:
        try:
            with get_connection() as conn:
                current_ts = get_latest_ts(conn, ticker)
            if current_ts and current_ts != last_ts:
                run_once(ticker)
                last_ts = current_ts
        except Exception:
            logger.exception("Poll iteration failed")
        time.sleep(config.FORECAST_POLL_SEC)


if __name__ == "__main__":
    poll_loop()
