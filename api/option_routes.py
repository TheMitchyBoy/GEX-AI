"""API routes for option price ingest, learning, and forecasts."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

import config
from db.connection import require_database_url
from integrations.uw_client import is_configured
from models.option_learn import model_status
from services.option_pipeline import ingest_uw_quotes, learn_from_db, predict_option_moves, run_option_cycle

router = APIRouter(prefix="/options", tags=["options"])


@router.get("/status")
def options_status() -> dict[str, Any]:
    return {
        "uw_configured": is_configured(),
        "learn_enabled": config.OPTION_LEARN_ENABLED,
        "poll_on_forecast": config.OPTION_LEARN_ON_POLL,
        "min_updates": config.OPTION_MIN_UPDATES,
        "supported_tickers": config.SUPPORTED_TICKERS,
    }


@router.get("/learn/{ticker}/status")
def learn_status(ticker: str) -> dict[str, Any]:
    return {
        "ticker": ticker.upper(),
        "atm_call": model_status(ticker, "atm_call"),
        "atm_put": model_status(ticker, "atm_put"),
    }


@router.post("/ingest/{ticker}")
def ingest_options(ticker: str) -> dict[str, Any]:
    if not is_configured():
        raise HTTPException(status_code=503, detail="UW_API_KEY is not set")
    require_database_url()
    result = ingest_uw_quotes(ticker.upper())
    if not result.get("ok"):
        raise HTTPException(status_code=422, detail=result.get("error", "ingest failed"))
    return result


@router.post("/learn/{ticker}")
def learn_options(ticker: str, slot: str = "atm_call") -> dict[str, Any]:
    require_database_url()
    result = learn_from_db(ticker.upper(), slot=slot)
    if not result.get("ok"):
        raise HTTPException(status_code=422, detail=result.get("error", "learn failed"))
    return result


@router.get("/forecast/{ticker}")
def forecast_options(ticker: str) -> dict[str, Any]:
    require_database_url()
    result = predict_option_moves(ticker.upper())
    if not result.get("ok"):
        raise HTTPException(status_code=422, detail=result.get("error", "forecast failed"))
    return result


@router.post("/cycle/{ticker}")
def full_cycle(ticker: str) -> dict[str, Any]:
    if not is_configured():
        raise HTTPException(status_code=503, detail="UW_API_KEY is not set")
    require_database_url()
    return run_option_cycle(ticker.upper())
