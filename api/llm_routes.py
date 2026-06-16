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
from models.llm_agent import SUGGESTED_PROMPTS, chat_with_agent

router = APIRouter(prefix="/llm", tags=["llm"])


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=8000)


class AgentChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)
    lookback_days: int = Field(default=config.LOOKBACK_DAYS, ge=1, le=365)
    refresh_context: bool = True


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
        "cache_enabled": config.LLM_CACHE_ENABLED,
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


@router.get("/prompts")
def agent_prompts() -> dict[str, Any]:
    return {"suggested_prompts": SUGGESTED_PROMPTS}


@router.post("/chat/{ticker}")
def agent_chat(ticker: str, body: AgentChatRequest) -> dict[str, Any]:
    require_database_url()
    history = load_snapshot_history(ticker.upper(), lookback_days=body.lookback_days)
    if len(history) < config.MIN_KNN_SNAPSHOTS:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient data: need at least {config.MIN_KNN_SNAPSHOTS} snapshots",
        )
    messages = [{"role": m.role, "content": m.content} for m in body.messages]
    result = chat_with_agent(
        history,
        messages,
        lookback_days=body.lookback_days,
        refresh_context=body.refresh_context,
    )
    if result.get("error") and not result.get("reply"):
        raise HTTPException(status_code=503, detail=result["error"])
    return {
        "ticker": ticker.upper(),
        "reply": result.get("reply"),
        "model": result.get("model"),
        "context_summary": {
            "snapshot_ts": (result.get("context") or {}).get("bundle", {}).get("snapshot_ts"),
            "estimated_tokens": (result.get("context") or {}).get("estimated_tokens"),
        },
    }
