"""Conversational GEX agent with live Postgres context."""

from __future__ import annotations

from typing import Any

import config
from models.llm_client import is_llm_configured, openai_chat
from models.llm_context import build_context_bundle, bundle_to_prompt_json, estimate_token_count
from models.llm_predict import generate_llm_forecast
from models.multi_horizon import predict_multi_horizon

AGENT_SYSTEM_PROMPT = """You are the GEX Market Agent — an expert options analyst specializing in dealer gamma exposure (GEX).

You have access to live snapshot data from a PostgreSQL GEX pipeline: spot, net GEX, gamma regime, flip level, call/put walls, term structure, flow metrics, KNN quant forecast, similar historical setups, and multi-horizon projections.

Guidelines:
- Ground every answer in the provided context JSON. Cite specific numbers (spot, flip, GEX, walls).
- Explain dealer positioning and how gamma affects expected spot behavior.
- Distinguish facts from forecasts; state confidence when uncertain.
- Be concise but actionable. Use bullet points for levels and scenarios.
- If asked for trade ideas, frame as educational analysis — not financial advice.
- If data is missing from context, say so instead of inventing values.

Current market context is injected with each user message under [GEX_CONTEXT]."""


SUGGESTED_PROMPTS = [
    "What's the current gamma regime and what does it imply for spot?",
    "Where are the key support/resistance levels from GEX walls and flip?",
    "Summarize the KNN forecast and your view for the next 30 minutes.",
    "What similar historical setups suggest about the next move?",
    "Is there pin risk near spot? Which strikes matter most?",
    "Explain the last GEX move — what strikes drove it?",
]


def build_agent_context(history: list[dict[str, Any]], *, lookback_days: int | None = None) -> dict[str, Any]:
    lookback_days = lookback_days if lookback_days is not None else config.LOOKBACK_DAYS
    bundle = build_context_bundle(history, lookback_days=lookback_days, slim=True)
    llm_forecast = None
    horizons = {}
    try:
        llm_forecast = generate_llm_forecast(history, lookback_days=lookback_days, persist=False)
    except Exception:
        pass
    try:
        horizons = predict_multi_horizon(history, lookback_days=lookback_days)
    except Exception:
        pass
    return {
        "bundle": bundle,
        "llm_forecast": llm_forecast,
        "horizons": {str(k): v for k, v in horizons.items()},
        "estimated_tokens": estimate_token_count(bundle),
    }


def _format_context_block(ctx: dict[str, Any]) -> str:
    parts = [f"[GEX_CONTEXT]\n{bundle_to_prompt_json(ctx.get('bundle', {}))}"]
    if ctx.get("llm_forecast"):
        parts.append(f"[LLM_FORECAST]\n{bundle_to_prompt_json(ctx['llm_forecast'])}")
    if ctx.get("horizons"):
        parts.append(f"[MULTI_HORIZON]\n{bundle_to_prompt_json(ctx['horizons'])}")
    return "\n\n".join(parts)


def chat_with_agent(
    history: list[dict[str, Any]],
    messages: list[dict[str, str]],
    *,
    lookback_days: int | None = None,
    refresh_context: bool = True,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send conversation to GEX agent. messages: [{role: user|assistant, content: str}, ...]."""
    lookback_days = lookback_days if lookback_days is not None else config.LOOKBACK_DAYS

    if not is_llm_configured():
        return {
            "reply": None,
            "error": "OPENAI_API_KEY is not configured. Set it in .env to use the GEX agent.",
            "context": None,
        }

    if len(history) < config.MIN_KNN_SNAPSHOTS:
        return {
            "reply": None,
            "error": f"Need at least {config.MIN_KNN_SNAPSHOTS} snapshots for agent context.",
            "context": None,
        }

    ctx = context if context and not refresh_context else build_agent_context(history, lookback_days=lookback_days)
    context_block = _format_context_block(ctx)

    # Attach fresh context to the latest user turn only
    api_messages: list[dict[str, str]] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role not in ("user", "assistant"):
            continue
        if role == "user" and i == len(messages) - 1:
            content = f"{context_block}\n\n[USER_QUESTION]\n{content}"
        api_messages.append({"role": role, "content": content})

    reply, error = openai_chat(AGENT_SYSTEM_PROMPT, api_messages, max_tokens=config.LLM_MAX_TOKENS)

    return {
        "reply": reply,
        "error": error,
        "context": ctx,
        "model": config.LLM_MODEL,
        "message_count": len(api_messages),
    }
