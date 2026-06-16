"""LLM forecast API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

import config
from db.connection import get_connection, require_database_url
from db.loader import load_snapshot_history
from db.queries import fetch_llm_predictions
from models.llm_client import is_llm_configured
from models.llm_predict import generate_llm_forecast

router = APIRouter(prefix="/llm", tags=["llm"])


class LLMForecastRequest(BaseModel):
    lookback_days: int = Field(default=config.LOOKBACK_DAYS, ge=1, le=365)
    persist: bool | None = None
    extra_instructions: str | None = Field(default=None, max_length=2000)


@router.get("/status")
def llm_status() -> dict[str, Any]:
    return {
        "llm_configured": is_llm_configured(),
        "model": config.LLM_MODEL if is_llm_configured() else None,
        "write_predictions": config.WRITE_PREDICTIONS,
        "prediction_source": config.LLM_PREDICTION_SOURCE,
    }


@router.get("/forecast/{ticker}")
def llm_forecast_get(
    ticker: str,
    lookback_days: int = Query(default=config.LOOKBACK_DAYS, ge=1, le=365),
    persist: bool = Query(default=False),
) -> dict[str, Any]:
    return _llm_forecast(ticker, lookback_days=lookback_days, persist=persist)


@router.post("/forecast/{ticker}")
def llm_forecast_post(ticker: str, body: LLMForecastRequest) -> dict[str, Any]:
    return _llm_forecast(
        ticker,
        lookback_days=body.lookback_days,
        persist=body.persist,
        extra_instructions=body.extra_instructions,
    )


def _llm_forecast(
    ticker: str,
    *,
    lookback_days: int,
    persist: bool | None = None,
    extra_instructions: str | None = None,
) -> dict[str, Any]:
    require_database_url()
    history = load_snapshot_history(ticker.upper(), lookback_days=lookback_days)
    if len(history) < config.MIN_KNN_SNAPSHOTS:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient data: need at least {config.MIN_KNN_SNAPSHOTS} snapshots, got {len(history)}",
        )
    return generate_llm_forecast(
        history,
        lookback_days=lookback_days,
        persist=persist,
        extra_instructions=extra_instructions,
    )


@router.get("/predictions/{ticker}")
def llm_predictions(
    ticker: str,
    limit: int = Query(default=20, ge=1, le=200),
    source: str | None = None,
    unresolved_only: bool = False,
) -> dict[str, Any]:
    require_database_url()
    with get_connection() as conn:
        rows = fetch_llm_predictions(
            conn,
            ticker.upper(),
            limit=limit,
            source=source,
            unresolved_only=unresolved_only,
        )
    return {"ticker": ticker.upper(), "count": len(rows), "predictions": rows}
