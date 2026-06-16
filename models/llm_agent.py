"""Conversational GEX agent with rich context, tools, and two-pass reasoning."""

from __future__ import annotations

import json
from typing import Any

import config
from models.llm_cache import get_cached, set_cached
from models.llm_client import is_llm_configured, openai_chat, openai_chat_json, openai_chat_with_tools
from models.llm_context import build_context_bundle, bundle_to_prompt_json, estimate_token_count
from models.llm_predict import generate_llm_forecast
from models.llm_tools import TOOL_SCHEMAS, AgentToolRunner
from models.multi_horizon import predict_multi_horizon

AGENT_SYSTEM_PROMPT = """You are the GEX Market Agent — an expert options analyst specializing in dealer gamma exposure (GEX).

You receive live Postgres GEX data: spot, net GEX, regime, gamma flip, call/put walls, term structure, flow/greeks, KNN/GBoost quant forecasts, similar setups, session analogs, and your own forecast track record.

## Domain playbooks
- **LONG gamma / positive GEX:** dealers hedge against moves → mean reversion, dampened trends, pin toward positive-gamma magnets.
- **SHORT gamma / negative GEX:** dealers amplify moves → trend extension, breaks accelerate, higher realized vol near flip.
- **Spot below flip (short gamma territory):** upside moves can accelerate; flip acts as resistance magnet.
- **Spot above flip (long gamma territory):** pullbacks toward flip are common; chop/pin behavior increases.
- **0DTE-heavy (high zero_dte_ratio):** pin risk into close; charm matters more into final hour.
- **FOMC/event weeks (is_fomc_week, event_risk_score):** widen uncertainty; cite event flags; reduce confidence.

## Rules
1. Ground every claim in provided JSON or tool results. Cite numbers: spot, flip, GEX (Bn$/1%), walls, confidence.
2. Synthesize quant outputs (KNN, GBoost, horizons) — do not contradict them without citing a specific strike/flow fact.
3. Use forecast_track_record to calibrate confidence. If sign_accuracy < 0.55 or n < 5, be more cautious.
4. Distinguish facts vs forecasts. State confidence 0–1.
5. Educational analysis only — not financial advice.

## Response structure
1. **Current state** — regime, spot vs flip, total GEX, key walls (with numbers)
2. **Dealer positioning** — what hedging flow implies
3. **Base case (10–30 min)** — direction, magnets, ΔGEX view from quant
4. **Alternate scenario** — trigger level that flips the view
5. **Confidence** — 0–1 and what would change your mind"""


FACTS_EXTRACTION_PROMPT = """Extract structured facts from the GEX context. Respond with ONLY valid JSON:
{
  "spot": number,
  "gamma_flip": number,
  "regime": "LONG gamma|SHORT gamma",
  "total_gex_bn": number,
  "flip_distance_pct": number,
  "call_wall": number,
  "put_wall": number,
  "knn_delta_gex_bn": number|null,
  "knn_confidence": number|null,
  "regime_flip_probability": number|null,
  "top_risk": "one sentence",
  "calibration_note": "one sentence on forecast track record if available"
}"""


SUGGESTED_PROMPTS = [
    "What's the current gamma regime and what does it imply for spot?",
    "Where are the key support/resistance levels from GEX walls and flip?",
    "Summarize the KNN and GBoost forecasts for the next 30 minutes.",
    "What similar historical setups and sessions suggest about the next move?",
    "Is there pin risk near spot? Which strikes matter most?",
    "Explain the last GEX move — what strikes drove it?",
    "How accurate have our recent forecasts been? Should we trust today's view?",
]


def build_agent_context(history: list[dict[str, Any]], *, lookback_days: int | None = None) -> dict[str, Any]:
    lookback_days = lookback_days if lookback_days is not None else config.LOOKBACK_DAYS
    bundle = build_context_bundle(
        history,
        lookback_days=lookback_days,
        slim=not config.LLM_RICH_CONTEXT,
        rich=config.LLM_RICH_CONTEXT,
    )
    llm_forecast = None
    horizons: dict[str, Any] = {}
    try:
        llm_forecast = generate_llm_forecast(history, lookback_days=lookback_days, persist=False)
    except Exception:
        pass
    try:
        horizons = {str(k): v for k, v in predict_multi_horizon(history, lookback_days=lookback_days).items()}
    except Exception:
        pass
    return {
        "bundle": bundle,
        "llm_forecast": llm_forecast,
        "horizons": horizons,
        "estimated_tokens": estimate_token_count(bundle),
    }


def _format_context_block(ctx: dict[str, Any]) -> str:
    parts = [f"[GEX_CONTEXT]\n{bundle_to_prompt_json(ctx.get('bundle', {}))}"]
    if ctx.get("llm_forecast"):
        parts.append(f"[LLM_FORECAST]\n{bundle_to_prompt_json(ctx['llm_forecast'])}")
    if ctx.get("horizons"):
        parts.append(f"[MULTI_HORIZON]\n{bundle_to_prompt_json(ctx['horizons'])}")
    if ctx.get("extracted_facts"):
        parts.append(f"[EXTRACTED_FACTS]\n{bundle_to_prompt_json(ctx['extracted_facts'])}")
    if ctx.get("tool_results"):
        parts.append(f"[TOOL_RESULTS]\n{bundle_to_prompt_json(ctx['tool_results'])}")
    return "\n\n".join(parts)


def _extract_facts(context_block: str) -> dict[str, Any] | None:
    parsed, err = openai_chat_json(
        FACTS_EXTRACTION_PROMPT,
        context_block,
        max_tokens=600,
        temperature=0.1,
    )
    return parsed if not err else None


def _calibration_guidance(ctx: dict[str, Any]) -> str:
    track = (ctx.get("bundle") or {}).get("forecast_track_record") or {}
    cal = track.get("calibration") or {}
    n = cal.get("n", 0)
    sign_acc = cal.get("sign_accuracy")
    if not n:
        return "No resolved forecast history — use moderate confidence."
    if sign_acc is not None and sign_acc < 0.55:
        return f"Historical sign accuracy {sign_acc:.0%} on {n} forecasts — be cautious."
    if sign_acc is not None:
        return f"Historical sign accuracy {sign_acc:.0%} on {n} forecasts — calibrate confidence accordingly."
    return f"Forecast history n={n}."


def chat_with_agent(
    history: list[dict[str, Any]],
    messages: list[dict[str, str]],
    *,
    lookback_days: int | None = None,
    refresh_context: bool = True,
    context: dict[str, Any] | None = None,
    use_tools: bool | None = None,
    two_pass: bool | None = None,
) -> dict[str, Any]:
    lookback_days = lookback_days if lookback_days is not None else config.LOOKBACK_DAYS
    use_tools = config.LLM_USE_TOOLS if use_tools is None else use_tools
    two_pass = config.LLM_TWO_PASS if two_pass is None else two_pass

    if not is_llm_configured():
        return {"reply": None, "error": "OPENAI_API_KEY is not configured.", "context": None}

    if len(history) < config.MIN_KNN_SNAPSHOTS:
        return {
            "reply": None,
            "error": f"Need at least {config.MIN_KNN_SNAPSHOTS} snapshots for agent context.",
            "context": None,
        }

    bundle = build_context_bundle(history, lookback_days=lookback_days, rich=config.LLM_RICH_CONTEXT)
    ts = bundle.get("snapshot_ts")
    ticker = bundle.get("ticker", config.DEFAULT_TICKER)

    if ts and refresh_context:
        cached = get_cached(str(ticker), str(ts))
        if cached and cached.get("message_count") == len(messages):
            cached["from_cache"] = True
            return cached

    ctx = context if context and not refresh_context else build_agent_context(history, lookback_days=lookback_days)
    context_block = _format_context_block(ctx)
    tool_results: list[dict[str, Any]] = []

    user_question = messages[-1]["content"] if messages else ""
    calibration_note = _calibration_guidance(ctx)

    # Pass 1: structured fact extraction
    if two_pass:
        facts = _extract_facts(context_block)
        if facts:
            ctx["extracted_facts"] = facts
            context_block = _format_context_block(ctx)

    # Tool loop (optional)
    if use_tools:
        runner = AgentToolRunner(history, lookback_days=lookback_days)
        tool_messages = [
            {
                "role": "user",
                "content": (
                    f"{context_block}\n\n[CALIBRATION]\n{calibration_note}\n\n"
                    f"[USER_QUESTION]\n{user_question}\n\n"
                    "Call tools if you need fresher forecast, strikes, backtest, or KNN vs LLM comparison."
                ),
            }
        ]
        tool_reply, tool_err, tool_log = openai_chat_with_tools(
            AGENT_SYSTEM_PROMPT,
            tool_messages,
            TOOL_SCHEMAS,
            runner.execute,
        )
        if tool_log:
            tool_results = tool_log
            ctx["tool_results"] = tool_results
        if tool_reply and not tool_err:
            out = _build_response(tool_reply, ctx, messages, ticker, ts, tool_results=tool_results)
            if ts:
                set_cached(str(ticker), str(ts), out)
            return out

    # Pass 2: final answer
    api_messages: list[dict[str, str]] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role not in ("user", "assistant"):
            continue
        if role == "user" and i == len(messages) - 1:
            content = (
                f"{context_block}\n\n[CALIBRATION]\n{calibration_note}\n\n[USER_QUESTION]\n{content}"
            )
        api_messages.append({"role": role, "content": content})

    reply, error = openai_chat(AGENT_SYSTEM_PROMPT, api_messages, max_tokens=config.LLM_MAX_TOKENS)
    out = _build_response(reply, ctx, messages, ticker, ts, error=error, tool_results=tool_results)
    if ts and reply:
        set_cached(str(ticker), str(ts), out)
    return out


def _build_response(
    reply: str | None,
    ctx: dict[str, Any],
    messages: list[dict[str, str]],
    ticker: str,
    ts: str | None,
    *,
    error: str | None = None,
    tool_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "reply": reply,
        "error": error,
        "context": ctx,
        "model": config.LLM_MODEL,
        "message_count": len(messages),
        "ticker": ticker,
        "snapshot_ts": ts,
        "two_pass": config.LLM_TWO_PASS,
        "tools_used": [t.get("tool") for t in (tool_results or [])],
        "intelligence": {
            "rich_context": config.LLM_RICH_CONTEXT,
            "two_pass": config.LLM_TWO_PASS,
            "tools_enabled": config.LLM_USE_TOOLS,
            "estimated_tokens": ctx.get("estimated_tokens"),
            "has_track_record": bool((ctx.get("bundle") or {}).get("forecast_track_record")),
            "has_similar_sessions": bool((ctx.get("bundle") or {}).get("similar_sessions")),
        },
    }
