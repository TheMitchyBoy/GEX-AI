"""FastAPI service for GEX forecasts."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import config
from api.agent_ui import router as agent_ui_router
from api.alerts import evaluate_alerts
from api.llm_routes import router as llm_router
from api.middleware import SecurityMiddleware, get_metrics
from db.connection import get_connection, require_database_url
from db.features import enrich_snapshot_metrics
from db.loader import load_snapshot_history
from db.queries import (
    fetch_calibration_stats,
    fetch_daily_insights,
    fetch_intraday_timeline,
    fetch_latest_snapshot,
    fetch_snapshot_strikes,
    get_latest_ts,
    get_row_counts,
)
from models.backtest import run_backtest
from models.llm_client import is_llm_configured
from models.llm_predict import generate_llm_forecast
from models.multi_horizon import predict_multi_horizon
from models.predict import predict_next_snapshot, similar_setups

app = FastAPI(title="GEX Prediction API", version="2.0.0")
app.include_router(agent_ui_router)
app.include_router(llm_router)
app.add_middleware(SecurityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    from fastapi.responses import JSONResponse

    return JSONResponse(
        {
            "status": "ok",
            "service": "gex-ai-api",
            "agent_ui": "/agent",
            "docs": "/docs",
            "health": "/health",
        }
    )


@app.get("/health")
def health() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "ok",
        "database_configured": bool(config.DATABASE_URL),
        "llm_configured": is_llm_configured(),
        "supported_tickers": config.SUPPORTED_TICKERS,
    }
    if config.DATABASE_URL:
        try:
            with get_connection() as conn:
                payload["row_counts"] = get_row_counts(conn)
                payload["latest_ts"] = {t: get_latest_ts(conn, t) for t in config.SUPPORTED_TICKERS[:3]}
        except Exception as exc:
            payload["status"] = "degraded"
            payload["database_error"] = str(exc)
    if config.PROCESSOR_HEALTH_URL:
        try:
            payload["processor"] = httpx.get(config.PROCESSOR_HEALTH_URL, timeout=5.0).json()
        except Exception as exc:
            payload["processor_error"] = str(exc)
    return payload


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return get_metrics()


@app.get("/forecast/{ticker}")
def forecast(
    ticker: str,
    lookback_days: int = Query(default=config.LOOKBACK_DAYS, ge=1, le=365),
    multi_horizon: bool = Query(default=True),
) -> dict[str, Any]:
    require_database_url()
    history = load_snapshot_history(ticker.upper(), lookback_days=lookback_days)
    if len(history) < config.MIN_KNN_SNAPSHOTS:
        raise HTTPException(status_code=422, detail=f"Insufficient data: {len(history)} snapshots")
    result = predict_next_snapshot(history, lookback_days=lookback_days)
    if not result:
        raise HTTPException(status_code=422, detail="Could not produce forecast")
    result["similar_setups"] = similar_setups(history, lookback_days=lookback_days)
    if multi_horizon:
        result["horizons"] = predict_multi_horizon(history, lookback_days=lookback_days)
    enriched = enrich_snapshot_metrics(history[-1].copy())
    result["alerts"] = evaluate_alerts(result, enriched)
    return result


@app.get("/compare/{ticker}")
def compare_forecasts(
    ticker: str,
    lookback_days: int = Query(default=config.LOOKBACK_DAYS, ge=1, le=365),
) -> dict[str, Any]:
    require_database_url()
    history = load_snapshot_history(ticker.upper(), lookback_days=lookback_days)
    if len(history) < config.MIN_KNN_SNAPSHOTS:
        raise HTTPException(status_code=422, detail="Insufficient data")
    knn = predict_next_snapshot(history, lookback_days=lookback_days)
    llm = generate_llm_forecast(history, lookback_days=lookback_days, persist=False)
    agree_regime = knn and llm and knn.get("predicted_regime") == llm.get("predicted_regime")
    return {
        "ticker": ticker.upper(),
        "knn": knn,
        "llm": llm,
        "agreement": {
            "regime": agree_regime,
            "delta_gex_diff": abs(float(knn.get("predicted_delta_gex", 0)) - float(llm.get("predicted_delta_gex_bn", 0))) if knn and llm else None,
        },
    }


@app.get("/calibration/{ticker}")
def calibration(ticker: str, source: str | None = None) -> dict[str, Any]:
    require_database_url()
    with get_connection() as conn:
        stats = fetch_calibration_stats(conn, ticker.upper(), source=source)
    return {"ticker": ticker.upper(), **stats}


@app.get("/insights/{ticker}")
def daily_insights(ticker: str, market_date: str | None = None, limit: int = 10) -> dict[str, Any]:
    require_database_url()
    with get_connection() as conn:
        rows = fetch_daily_insights(conn, ticker.upper(), market_date=market_date, limit=limit)
    return {"ticker": ticker.upper(), "insights": rows}


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
            rows = [{"ts": r["ts"], "market_date": r.get("market_date"), "spot": r.get("spot"), "total_gex": r.get("total_gex"), "regime": r.get("regime")} for r in rows]
    return {"ticker": ticker.upper(), "count": len(rows), "rows": rows}


@app.get("/similar/{ticker}")
def similar(ticker: str, top_n: int = Query(default=5, ge=1, le=20), lookback_days: int = Query(default=config.LOOKBACK_DAYS, ge=1, le=365)) -> dict[str, Any]:
    require_database_url()
    history = load_snapshot_history(ticker.upper(), lookback_days=lookback_days, include_strikes=False)
    return {"ticker": ticker.upper(), "similar_setups": similar_setups(history, top_n=top_n, lookback_days=lookback_days)}


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
def backtest(ticker: str, lookback_days: int = Query(default=30, ge=7, le=365)) -> dict[str, Any]:
    require_database_url()
    history = load_snapshot_history(ticker.upper(), lookback_days=lookback_days, include_strikes=True)
    report = run_backtest(history, lookback_days=lookback_days)
    out = report.to_dict()
    out["recent_rows"] = report.rows[-10:]
    return out
