"""Poll for new snapshots: KNN + LLM forecasts, reconciliation, alerts."""

from __future__ import annotations

import json
import logging
import select
import time
from typing import Any

import config
from api.alerts import evaluate_alerts, send_alerts
from db.connection import get_connection
from db.features import enrich_snapshot_metrics
from db.loader import load_snapshot_history, materialize_features_for_history
from db.queries import ensure_extensions, get_latest_ts, insert_prediction_deduped
from db.reconciliation import reconcile_predictions
from models.llm_predict import generate_llm_forecast
from models.predict import predict_next_snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _persist_knn(ticker: str, latest_ts: str, history: list[dict], forecast: dict) -> None:
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
    market_date = history[-1].get("market_date", latest_ts[:10])
    with get_connection() as conn:
        insert_prediction_deduped(
            conn, ticker=ticker, snapshot_ts=latest_ts, market_date=market_date,
            payload=payload, source=config.PREDICTION_SOURCE,
        )


def run_once(ticker: str | None = None) -> dict[str, Any] | None:
    ticker = ticker or config.DEFAULT_TICKER
    with get_connection() as conn:
        ensure_extensions(conn)
        latest_ts = get_latest_ts(conn, ticker)
    if not latest_ts:
        logger.warning("No snapshots for %s", ticker)
        return None

    reconcile_predictions(ticker)
    history = load_snapshot_history(ticker, lookback_days=config.LOOKBACK_DAYS)

    if config.ONLINE_LEARNING_ENABLED and history:
        try:
            from models.online_learn import ensure_bootstrapped, maybe_learn_latest

            ensure_bootstrapped(history, ticker)
            learned = maybe_learn_latest(history, ticker)
            if learned:
                logger.info("Online model learned from latest snapshot pair for %s", ticker)
        except Exception:
            logger.debug("Online learning step skipped", exc_info=True)

    if config.MATERIALIZE_FEATURES and history:
        try:
            materialize_features_for_history(history[-50:])
        except Exception:
            logger.debug("Feature materialization skipped", exc_info=True)

    forecast = predict_next_snapshot(history, lookback_days=config.LOOKBACK_DAYS)
    if not forecast:
        logger.warning("Insufficient history to forecast for %s", ticker)
        return None

    llm_result = None
    if config.RUN_LLM_ON_POLL:
        try:
            llm_result = generate_llm_forecast(history, lookback_days=config.LOOKBACK_DAYS, persist=config.WRITE_PREDICTIONS)
        except Exception:
            logger.exception("LLM forecast failed for %s", ticker)

    if config.WRITE_PREDICTIONS:
        _persist_knn(ticker, latest_ts, history, forecast)

    enriched = enrich_snapshot_metrics(history[-1].copy())
    alerts = evaluate_alerts(forecast, enriched)
    if alerts:
        send_alerts(alerts, ticker=ticker, ts=latest_ts)

    out = {"latest_ts": latest_ts, "forecast": forecast, "llm": llm_result, "alerts": alerts}
    logger.info("Processed %s @ %s", ticker, latest_ts)
    return out


def _listen_once(ticker: str, timeout: float = 55.0) -> bool:
    """Wait for NOTIFY on new snapshot; returns True if notified."""
    try:
        with get_connection() as conn:
            conn.execute(f"LISTEN {config.PG_NOTIFY_CHANNEL}")
            conn.commit()
            while conn.polling():
                conn.poll()
            if select.select([conn], [], [], timeout) == ([], [], []):
                return False
            conn.poll()
            while conn.notifies:
                notify = conn.notifies.pop(0)
                if notify.payload and ticker.upper() in (notify.payload or ""):
                    return True
            return True
    except Exception:
        logger.debug("LISTEN/NOTIFY unavailable, falling back to poll", exc_info=True)
        return False


def poll_loop() -> None:
    ticker = config.DEFAULT_TICKER
    last_ts: str | None = None
    logger.info("Starting forecast poller for %s", ticker)

    while True:
        try:
            if config.USE_LISTEN_NOTIFY:
                _listen_once(ticker, timeout=float(config.FORECAST_POLL_SEC))
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
