#!/usr/bin/env python3
"""Generate daily GEX insights (LLM or rule-based) into daily_insights table."""

from __future__ import annotations

import json
import logging
import sys

import config
from db.connection import get_connection, require_database_url
from db.loader import load_snapshot_history
from db.queries import ensure_extensions, upsert_daily_insight
from models.backtest import run_backtest
from models.ensemble import learn_weights_from_backtest
from models.gboost import train_gboost
from models.llm_client import is_llm_configured, openai_chat_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_daily_insight(ticker: str, market_date: str) -> dict:
    history = [h for h in load_snapshot_history(ticker, lookback_days=30) if h.get("market_date") == market_date]
    if not history:
        history = load_snapshot_history(ticker, lookback_days=7)
    backtest = run_backtest(history, lookback_days=min(30, len(history)))
    summary = {
        "market_date": market_date,
        "snapshot_count": len(history),
        "backtest": backtest.to_dict(),
        "last_regime": history[-1].get("regime") if history else None,
        "last_total_gex": history[-1].get("total_gex") if history else None,
    }
    if is_llm_configured():
        parsed, err = openai_chat_json(
            "You are a GEX market analyst. Respond with JSON: {lessons: [], key_levels: {}, strategy_notes: string}",
            f"Summarize this trading day GEX data:\n{json.dumps(summary, default=str)}",
        )
        if parsed:
            summary["llm_insights"] = parsed
        elif err:
            summary["llm_error"] = err
    if config.AUTO_TRAIN_GBOOST and len(history) >= 20:
        try:
            gboost = train_gboost(history, ticker)
            if gboost:
                summary["gboost_retrained"] = {"n_train": gboost["n_train"], "cv_mae": gboost["cv_mae"]}
            weights = learn_weights_from_backtest(backtest.to_dict(), ticker)
            summary["ensemble_weights"] = weights
        except Exception:
            logger.exception("Auto-train failed for %s", ticker)
    return summary


def main() -> int:
    try:
        require_database_url()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    ticker = config.DEFAULT_TICKER
    market_date = sys.argv[1] if len(sys.argv) > 1 else None
    if not market_date and load_snapshot_history(ticker, lookback_days=1):
        market_date = load_snapshot_history(ticker, lookback_days=1)[-1].get("market_date")
    if not market_date:
        print("No market_date", file=sys.stderr)
        return 2
    payload = generate_daily_insight(ticker, market_date)
    with get_connection() as conn:
        ensure_extensions(conn)
        upsert_daily_insight(conn, ticker=ticker, market_date=market_date, kind="daily_summary", payload=payload)
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
