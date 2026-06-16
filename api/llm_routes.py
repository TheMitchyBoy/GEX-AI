"""LLM forecast API routes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
from db.connection import get_connection, require_database_url
from db.loader import load_snapshot_history
from db.queries import fetch_llm_predictions
from models.llm_client import is_llm_configured
from models.llm_predict import generate_llm_forecast
from models.llm_agent import SUGGESTED_PROMPTS, chat_with_agent, stream_agent_reply

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["llm"])


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=8000)


class AgentChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)
    lookback_days: int = Field(default=config.LOOKBACK_DAYS, ge=1, le=365)
    refresh_context: bool = True
    use_tools: bool | None = None
    two_pass: bool | None = None
    mode: str = Field(default="fast", pattern="^(fast|deep|quant)$")
    session_id: str | None = Field(default=None, max_length=64)
    stream: bool = False


class FeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    rating: int = Field(ge=-1, le=1)
    message: str | None = Field(default=None, max_length=4000)
    reply: str | None = Field(default=None, max_length=8000)
    snapshot_ts: str | None = None


class LLMForecastRequest(BaseModel):
    lookback_days: int = Field(default=config.LOOKBACK_DAYS, ge=1, le=365)
    persist: bool | None = None
    extra_instructions: str | None = Field(default=None, max_length=2000)


@router.get("/status")
def llm_status() -> dict[str, Any]:
    return {
        "llm_configured": is_llm_configured(),
        "model": config.LLM_MODEL if is_llm_configured() else None,
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
        "write_predictions": config.WRITE_PREDICTIONS,
        "prediction_source": config.LLM_PREDICTION_SOURCE,
        "cache_enabled": config.LLM_CACHE_ENABLED,
        "two_pass": config.LLM_TWO_PASS,
        "use_tools": config.LLM_USE_TOOLS,
        "rich_context": config.LLM_RICH_CONTEXT,
        "max_tool_rounds": config.LLM_MAX_TOOL_ROUNDS,
        "agent_fast_mode": config.LLM_AGENT_FAST,
        "structured_output": config.LLM_STRUCTURED_OUTPUT,
        "context_compress": config.LLM_CONTEXT_COMPRESS,
        "ensemble_enabled": config.ENSEMBLE_ENABLED,
        "modes": ["fast", "deep", "quant"],
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
    try:
        history = load_snapshot_history(ticker.upper(), lookback_days=body.lookback_days)
    except Exception as exc:
        logger.exception("Failed to load history for agent chat")
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc
    if len(history) < config.MIN_KNN_SNAPSHOTS:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient data: need at least {config.MIN_KNN_SNAPSHOTS} snapshots",
        )
    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    if body.stream and body.mode != "deep":
        def _gen():
            for chunk in stream_agent_reply(
                history, messages,
                lookback_days=body.lookback_days,
                mode=body.mode,
                session_id=body.session_id,
            ):
                yield chunk

        return StreamingResponse(_gen(), media_type="text/plain")

    try:
        result = chat_with_agent(
            history,
            messages,
            lookback_days=body.lookback_days,
            refresh_context=body.refresh_context,
            use_tools=body.use_tools,
            two_pass=body.two_pass,
            mode=body.mode,
            session_id=body.session_id,
        )
    except Exception as exc:
        logger.exception("Agent chat failed for %s", ticker)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc
    if result.get("error") and not result.get("reply"):
        raise HTTPException(status_code=503, detail=result["error"])
    return {
        "ticker": ticker.upper(),
        "reply": result.get("reply"),
        "model": result.get("model"),
        "mode": result.get("mode"),
        "agreement": result.get("agreement"),
        "latency": result.get("latency"),
        "two_pass": result.get("two_pass"),
        "tools_used": result.get("tools_used") or [],
        "intelligence": result.get("intelligence"),
        "from_cache": result.get("from_cache", False),
        "context_summary": {
            "snapshot_ts": (result.get("context") or {}).get("bundle", {}).get("snapshot_ts"),
            "estimated_tokens": (result.get("context") or {}).get("estimated_tokens"),
        },
    }


@router.get("/eval/{ticker}")
def agent_eval(
    ticker: str,
    lookback_days: int = Query(default=30, ge=1, le=365),
) -> dict[str, Any]:
    """Run grounding evaluation probes (requires OpenAI + DB)."""
    if not is_llm_configured():
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
    require_database_url()
    from models.llm_eval import evaluate_agent_grounding

    history = load_snapshot_history(ticker.upper(), lookback_days=lookback_days)
    if len(history) < config.MIN_KNN_SNAPSHOTS:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient data: need at least {config.MIN_KNN_SNAPSHOTS} snapshots",
        )
    try:
        return evaluate_agent_grounding(history, lookback_days=lookback_days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/feedback/{ticker}")
def agent_feedback(ticker: str, body: FeedbackRequest) -> dict[str, Any]:
    from db.agent_store import save_feedback

    row = save_feedback(
        ticker=ticker.upper(),
        session_id=body.session_id,
        rating=body.rating,
        message=body.message,
        reply=body.reply,
        snapshot_ts=body.snapshot_ts,
    )
    return {"ok": True, "feedback": row}
