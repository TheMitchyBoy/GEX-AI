"""FastAPI service for GEX forecasts."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import config
from db.connection import get_connection, require_database_url
from db.loader import load_snapshot_history
from db.queries import fetch_intraday_timeline, fetch_latest_snapshot, fetch_snapshot_strikes, get_latest_ts, get_row_counts
from models.backtest import run_backtest
from models.predict import predict_next_snapshot, similar_setups

app = FastAPI(title="GEX Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    payload: dict[str, Any] = {"status": "ok", "database_configured": bool(config.DATABASE_URL)}
    if config.DATABASE_URL:
        try:
            with get_connection() as conn:
                payload["row_counts"] = get_row_counts(conn)
                payload["latest_ts"] = {
                    config.DEFAULT_TICKER: get_latest_ts(conn, config.DEFAULT_TICKER)
                }
        except Exception as exc:
            payload["status"] = "degraded"
            payload["database_error"] = str(exc)
    if config.PROCESSOR_HEALTH_URL:
        try:
            resp = httpx.get(config.PROCESSOR_HEALTH_URL, timeout=5.0)
            payload["processor"] = resp.json()
        except Exception as exc:
            payload["processor_error"] = str(exc)
    return payload


@app.get("/forecast/{ticker}")
def forecast(
    ticker: str,
    lookback_days: int = Query(default=config.LOOKBACK_DAYS, ge=1, le=365),
) -> dict[str, Any]:
    require_database_url()
    history = load_snapshot_history(ticker.upper(), lookback_days=lookback_days)
    if len(history) < config.MIN_KNN_SNAPSHOTS:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient data: need at least {config.MIN_KNN_SNAPSHOTS} snapshots, got {len(history)}",
        )
    result = predict_next_snapshot(history, lookback_days=lookback_days)
    if not result:
        raise HTTPException(status_code=422, detail="Could not produce forecast")
    result["similar_setups"] = similar_setups(history, lookback_days=lookback_days)
    return result


@app.get("/history/{ticker}")
def history(
    ticker: str,
    market_date: str | None = None,
    lookback_days: int = Query(default=config.LOOKBACK_DAYS, ge=1, le=365),
) -> dict[str, Any]:
    require_database_url()
    with get_connection() as conn:
        if market_date:
            df = fetch_intraday_timeline(conn, ticker.upper(), market_date)
            rows = df.to_dict(orient="records")
        else:
            rows = load_snapshot_history(ticker.upper(), lookback_days=lookback_days, include_strikes=False)
            rows = [
                {
                    "ts": r["ts"],
                    "market_date": r.get("market_date"),
                    "spot": r.get("spot"),
                    "total_gex": r.get("total_gex"),
                    "regime": r.get("regime"),
                }
                for r in rows
            ]
    return {"ticker": ticker.upper(), "count": len(rows), "rows": rows}


@app.get("/similar/{ticker}")
def similar(
    ticker: str,
    top_n: int = Query(default=5, ge=1, le=20),
    lookback_days: int = Query(default=config.LOOKBACK_DAYS, ge=1, le=365),
) -> dict[str, Any]:
    require_database_url()
    history = load_snapshot_history(ticker.upper(), lookback_days=lookback_days, include_strikes=False)
    setups = similar_setups(history, top_n=top_n, lookback_days=lookback_days)
    return {"ticker": ticker.upper(), "similar_setups": setups}


@app.get("/strikes/{ticker}")
def strikes(ticker: str, ts: str | None = None) -> dict[str, Any]:
    require_database_url()
    with get_connection() as conn:
        if not ts:
            latest = fetch_latest_snapshot(conn, ticker.upper())
            if not latest:
                raise HTTPException(status_code=404, detail="No snapshots found")
            ts = latest["ts"]
        df = fetch_snapshot_strikes(conn, ticker.upper(), ts)
    return {"ticker": ticker.upper(), "ts": ts, "strikes": df.to_dict(orient="records")}


@app.get("/backtest/{ticker}")
def backtest(
    ticker: str,
    lookback_days: int = Query(default=30, ge=7, le=365),
) -> dict[str, Any]:
    require_database_url()
    history = load_snapshot_history(ticker.upper(), lookback_days=lookback_days, include_strikes=True)
    report = run_backtest(history, lookback_days=lookback_days)
    out = report.to_dict()
    out["recent_rows"] = report.rows[-10:]
    return out
