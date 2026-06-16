"""Conversational GEX agent with modes, structured output, and intelligence pipeline."""

from __future__ import annotations

import json
from typing import Any, Iterator

import config
from db.agent_store import append_session_memory, load_session_memory, recent_feedback_summary
from models.event_prompts import build_event_system_addon
from models.llm_cache import get_cached, set_cached
from models.llm_client import (
    is_llm_configured,
    model_for_task,
    openai_chat,
    openai_chat_json,
    openai_chat_stream,
    openai_chat_with_tools,
)
from models.llm_context import build_context_bundle, bundle_to_prompt_json, estimate_token_count
from models.llm_metrics import get_llm_metrics, timed_stage
from models.llm_predict import generate_llm_forecast
from models.llm_tools import TOOL_SCHEMAS, AgentToolRunner
from models.multi_horizon import predict_multi_horizon
from models.quant_fallback import quant_only_reply

AGENT_MODES = ("fast", "deep", "quant")

AGENT_SYSTEM_PROMPT = """You are the GEX Market Agent — an expert options analyst specializing in dealer gamma exposure (GEX).

You receive live Postgres GEX data: spot, net GEX, regime, gamma flip, call/put walls, term structure, flow/greeks, KNN/GBoost/ensemble quant forecasts, similar setups, session analogs, and forecast track record.

## Domain playbooks
- **LONG gamma / positive GEX:** dealers hedge against moves → mean reversion, dampened trends, pin toward positive-gamma magnets.
- **SHORT gamma / negative GEX:** dealers amplify moves → trend extension, breaks accelerate, higher realized vol near flip.
- **Spot below flip (short gamma territory):** upside moves can accelerate; flip acts as resistance magnet.
- **Spot above flip (long gamma territory):** pullbacks toward flip are common; chop/pin behavior increases.
- **0DTE-heavy (high zero_dte_ratio):** pin risk into close; charm matters more into final hour.
- **FOMC/event weeks (is_fomc_week, event_risk_score):** widen uncertainty; cite event flags; reduce confidence.

## Rules
1. Ground every claim in provided JSON or tool results. Cite numbers: spot, flip, GEX (Bn$/1%), walls, confidence.
2. Use model_agreement.score — if < 0.5, flag uncertainty; if >= 0.75, synthesize confidently.
3. Use forecast_track_record to calibrate confidence. If sign_accuracy < 0.55 or n < 5, be more cautious.
4. Distinguish facts vs forecasts. State confidence 0–1.
5. Educational analysis only — not financial advice."""

STRUCTURED_OUTPUT_ADDON = """
## Output format
Respond with markdown using these sections:
1. **Current state** — regime, spot vs flip, total GEX, key walls (with numbers)
2. **Dealer positioning** — what hedging flow implies
3. **Base case (10–30 min)** — direction, magnets, ΔGEX from quant/ensemble
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
  "ensemble_delta_gex_bn": number|null,
  "agreement_score": number|null,
  "top_risk": "one sentence",
  "calibration_note": "one sentence on forecast track record if available"
}"""

SUGGESTED_PROMPTS = [
    "What's the current gamma regime and what does it imply for spot?",
    "Where are the key support/resistance levels from GEX walls and flip?",
    "Summarize the KNN and ensemble forecasts for the next 30 minutes.",
    "What similar historical setups and sessions suggest about the next move?",
    "Is there pin risk near spot? Which strikes matter most?",
    "Explain the last GEX move — what strikes drove it?",
    "How accurate have our recent forecasts been? Should we trust today's view?",
]


def _resolve_mode_settings(mode: str | None, use_tools: bool | None, two_pass: bool | None) -> tuple[str, bool, bool]:
    mode = (mode or "fast").lower()
    if mode not in AGENT_MODES:
        mode = "fast"
    if mode == "quant":
        return mode, False, False
    if mode == "deep":
        return mode, True if use_tools is None else use_tools, True if two_pass is None else two_pass
    # fast
    if use_tools is None:
        use_tools = False if config.LLM_AGENT_FAST else config.LLM_USE_TOOLS
    if two_pass is None:
        two_pass = False if config.LLM_AGENT_FAST else config.LLM_TWO_PASS
    return mode, use_tools, two_pass


def build_agent_context(
    history: list[dict[str, Any]],
    *,
    lookback_days: int | None = None,
    for_chat: bool = False,
) -> dict[str, Any]:
    lookback_days = lookback_days if lookback_days is not None else config.LOOKBACK_DAYS
    with timed_stage("build_context_bundle"):
        bundle = build_context_bundle(
            history,
            lookback_days=lookback_days,
            slim=not config.LLM_RICH_CONTEXT,
            rich=config.LLM_RICH_CONTEXT,
        )
    llm_forecast = None
    horizons: dict[str, Any] = {}
    if not for_chat:
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


def _system_prompt(bundle: dict[str, Any]) -> str:
    summary = bundle.get("summary") or {}
    parts = [AGENT_SYSTEM_PROMPT]
    event_addon = build_event_system_addon(summary)
    if event_addon:
        parts.append(event_addon)
    if config.LLM_STRUCTURED_OUTPUT:
        parts.append(STRUCTURED_OUTPUT_ADDON)
    agreement = bundle.get("model_agreement") or {}
    if agreement.get("notes"):
        parts.append(f"## Model agreement\n{agreement['notes']} (score={agreement.get('score')})")
    return "\n\n".join(parts)


def _format_context_block(ctx: dict[str, Any], *, session_id: str | None = None, ticker: str | None = None) -> str:
    parts = [f"[GEX_CONTEXT]\n{bundle_to_prompt_json(ctx.get('bundle', {}))}"]
    if ctx.get("llm_forecast"):
        parts.append(f"[LLM_FORECAST]\n{bundle_to_prompt_json(ctx['llm_forecast'])}")
    if ctx.get("horizons"):
        parts.append(f"[MULTI_HORIZON]\n{bundle_to_prompt_json(ctx['horizons'])}")
    if ctx.get("extracted_facts"):
        parts.append(f"[EXTRACTED_FACTS]\n{bundle_to_prompt_json(ctx['extracted_facts'])}")
    if ctx.get("tool_results"):
        parts.append(f"[TOOL_RESULTS]\n{bundle_to_prompt_json(ctx['tool_results'])}")
    if session_id and ticker and config.AGENT_MEMORY_ENABLED:
        mem = load_session_memory(ticker, session_id, limit=3)
        if mem:
            parts.append(f"[SESSION_MEMORY]\n{bundle_to_prompt_json(mem)}")
    return "\n\n".join(parts)


def _extract_facts(context_block: str) -> dict[str, Any] | None:
    with timed_stage("fact_extraction"):
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
    fb = recent_feedback_summary(str((ctx.get("bundle") or {}).get("ticker", config.DEFAULT_TICKER)))
    parts = []
    if not n:
        parts.append("No resolved forecast history — use moderate confidence.")
    elif sign_acc is not None and sign_acc < 0.55:
        parts.append(f"Historical sign accuracy {sign_acc:.0%} on {n} forecasts — be cautious.")
    elif sign_acc is not None:
        parts.append(f"Historical sign accuracy {sign_acc:.0%} on {n} forecasts.")
    if fb.get("n", 0) >= 5 and fb.get("positive_rate") is not None:
        parts.append(f"User feedback positive rate {fb['positive_rate']:.0%} on last {fb['n']} ratings.")
    return " ".join(parts) if parts else "Calibrate confidence to data quality."


def chat_with_agent(
    history: list[dict[str, Any]],
    messages: list[dict[str, str]],
    *,
    lookback_days: int | None = None,
    refresh_context: bool = True,
    context: dict[str, Any] | None = None,
    use_tools: bool | None = None,
    two_pass: bool | None = None,
    mode: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    lookback_days = lookback_days if lookback_days is not None else config.LOOKBACK_DAYS
    mode, use_tools, two_pass = _resolve_mode_settings(mode, use_tools, two_pass)

    if len(history) < config.MIN_KNN_SNAPSHOTS:
        return {
            "reply": None,
            "error": f"Need at least {config.MIN_KNN_SNAPSHOTS} snapshots for agent context.",
            "context": None,
        }

    ctx = context if context and not refresh_context else build_agent_context(
        history, lookback_days=lookback_days, for_chat=True
    )
    bundle = ctx.get("bundle") or {}
    ts = bundle.get("snapshot_ts")
    ticker = str(bundle.get("ticker", config.DEFAULT_TICKER))

    if mode == "quant" or not is_llm_configured():
        agreement = bundle.get("model_agreement")
        reply = quant_only_reply(bundle, agreement=agreement)
        return _build_response(
            reply, ctx, messages, ticker, ts,
            mode=mode, two_pass=False, use_tools=False,
            error=None if is_llm_configured() else "OPENAI_API_KEY is not configured — quant-only mode.",
        )

    if ts and refresh_context:
        cached = get_cached(
            ticker, str(ts), messages=messages, two_pass=two_pass, use_tools=use_tools, mode=mode
        )
        if cached:
            cached["from_cache"] = True
            return cached

    system = _system_prompt(bundle)
    context_block = _format_context_block(ctx, session_id=session_id, ticker=ticker)
    tool_results: list[dict[str, Any]] = []
    user_question = messages[-1]["content"] if messages else ""
    calibration_note = _calibration_guidance(ctx)

    if two_pass:
        facts = _extract_facts(context_block)
        if facts:
            ctx["extracted_facts"] = facts
            context_block = _format_context_block(ctx, session_id=session_id, ticker=ticker)

    if use_tools:
        runner = AgentToolRunner(history, lookback_days=lookback_days)
        tool_messages = [{
            "role": "user",
            "content": (
                f"{context_block}\n\n[CALIBRATION]\n{calibration_note}\n\n"
                f"[USER_QUESTION]\n{user_question}\n\n"
                "Call tools if you need fresher forecast, strikes, backtest, or KNN vs LLM comparison."
            ),
        }]
        with timed_stage("tool_loop"):
            tool_reply, tool_err, tool_log = openai_chat_with_tools(
                system, tool_messages, TOOL_SCHEMAS, runner.execute,
                model=model_for_task("tools"),
            )
        if tool_log:
            tool_results = tool_log
            ctx["tool_results"] = tool_results
        if tool_reply and not tool_err:
            out = _build_response(
                tool_reply, ctx, messages, ticker, ts,
                tool_results=tool_results, mode=mode, two_pass=two_pass, use_tools=use_tools,
            )
            _finalize_session(ticker, session_id, user_question, tool_reply, ts)
            if ts:
                set_cached(ticker, str(ts), out, messages=messages, two_pass=two_pass, use_tools=use_tools, mode=mode)
            return out

    api_messages: list[dict[str, str]] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role not in ("user", "assistant"):
            continue
        if role == "user" and i == len(messages) - 1:
            content = f"{context_block}\n\n[CALIBRATION]\n{calibration_note}\n\n[USER_QUESTION]\n{content}"
        api_messages.append({"role": role, "content": content})

    with timed_stage("final_answer"):
        reply, error = openai_chat(
            system, api_messages, max_tokens=config.LLM_MAX_TOKENS, model=model_for_task("answer")
        )
    out = _build_response(
        reply, ctx, messages, ticker, ts,
        error=error, tool_results=tool_results, mode=mode, two_pass=two_pass, use_tools=use_tools,
    )
    if reply:
        _finalize_session(ticker, session_id, user_question, reply, ts)
    if ts and reply:
        set_cached(ticker, str(ts), out, messages=messages, two_pass=two_pass, use_tools=use_tools, mode=mode)
    return out


def stream_agent_reply(
    history: list[dict[str, Any]],
    messages: list[dict[str, str]],
    *,
    lookback_days: int | None = None,
    mode: str | None = None,
    session_id: str | None = None,
    ctx: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Stream final answer tokens (fast mode, no tools)."""
    mode, _, _ = _resolve_mode_settings(mode, False, False)
    lookback_days = lookback_days if lookback_days is not None else config.LOOKBACK_DAYS
    try:
        if mode == "quant" or not is_llm_configured():
            bundle = (ctx or {}).get("bundle") or build_context_bundle(
                history, lookback_days=lookback_days, rich=config.LLM_RICH_CONTEXT
            )
            agreement = bundle.get("model_agreement")
            yield quant_only_reply(bundle, agreement=agreement)
            return
        if ctx is None:
            with timed_stage("build_agent_context"):
                ctx = build_agent_context(history, lookback_days=lookback_days, for_chat=True)
        bundle = ctx.get("bundle") or {}
        system = _system_prompt(bundle)
        context_block = _format_context_block(ctx, session_id=session_id, ticker=bundle.get("ticker"))
        calibration_note = _calibration_guidance(ctx)
        user_question = messages[-1]["content"] if messages else ""
        api_messages = [{
            "role": "user",
            "content": f"{context_block}\n\n[CALIBRATION]\n{calibration_note}\n\n[USER_QUESTION]\n{user_question}",
        }]
        with timed_stage("stream_answer"):
            for token in openai_chat_stream(system, api_messages, model=model_for_task("answer")):
                if token:
                    yield token
    except Exception as exc:
        yield f"\n\nSorry — the agent hit an error: {exc}"


def _finalize_session(ticker: str, session_id: str | None, question: str, reply: str, ts: str | None) -> None:
    if not config.AGENT_MEMORY_ENABLED or not session_id:
        return
    append_session_memory(ticker, session_id, {
        "question": question[:500],
        "reply_preview": reply[:500],
        "snapshot_ts": ts,
    })


def _build_response(
    reply: str | None,
    ctx: dict[str, Any],
    messages: list[dict[str, str]],
    ticker: str,
    ts: str | None,
    *,
    error: str | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    mode: str = "fast",
    two_pass: bool = False,
    use_tools: bool = False,
) -> dict[str, Any]:
    bundle = ctx.get("bundle") or {}
    return {
        "reply": reply,
        "error": error,
        "context": ctx,
        "model": config.LLM_MODEL,
        "mode": mode,
        "message_count": len(messages),
        "ticker": ticker,
        "snapshot_ts": ts,
        "two_pass": two_pass,
        "use_tools": use_tools,
        "tools_used": [t.get("tool") for t in (tool_results or [])],
        "agreement": bundle.get("model_agreement"),
        "latency": get_llm_metrics(),
        "intelligence": {
            "rich_context": config.LLM_RICH_CONTEXT,
            "compressed": bundle.get("_compressed", False),
            "two_pass": two_pass,
            "tools_enabled": use_tools,
            "agent_fast_mode": config.LLM_AGENT_FAST,
            "structured_output": config.LLM_STRUCTURED_OUTPUT,
            "estimated_tokens": ctx.get("estimated_tokens"),
            "has_track_record": bool(bundle.get("forecast_track_record")),
            "has_similar_sessions": bool(bundle.get("similar_sessions")),
            "agreement_score": (bundle.get("model_agreement") or {}).get("score"),
        },
    }
