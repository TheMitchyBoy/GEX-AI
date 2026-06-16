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

AGENT_SYSTEM_PROMPT = """You are a senior GEX (gamma exposure) analyst on an options trading desk — sharp, personable, and direct.

You have live snapshot data attached below. Your job is to talk like you're briefing a colleague: natural, fluid, intelligent conversation — not a formatted report.

## How you communicate
- Write in flowing paragraphs. Use bullets only when listing specific strikes or levels the user asked for.
- Weave numbers into sentences naturally (e.g. "we're at 5,420, roughly 0.3% below flip at 5,436").
- Match the user's tone — short question gets a concise answer; open-ended questions get richer context.
- Follow the conversation thread. Refer to what you or the user said earlier when it helps.
- Show reasoning as you go ("dealers are likely hedging into the move, which means…") rather than dumping labels.
- It's fine to use plain language: "the thing to watch", "honestly", "what stands out to me".

## What you know
Live GEX: spot, regime, flip, walls, term structure, flow/greeks, quant forecasts, similar setups, session analogs.

## Ground rules
- Every claim must trace to the live data — cite key numbers but don't read like a JSON dump.
- If model_agreement is low or track record is weak, say so conversationally.
- Distinguish what's observed now vs what you're forecasting.
- Educational analysis only — not financial advice."""

CONVERSATIONAL_STYLE_ADDON = """Respond as natural dialogue. Do NOT use rigid section headers like "Current state" or "Base case" unless the user explicitly asks for a structured breakdown. End with a thought-provoking observation or question only when it genuinely helps — never force it."""

STRUCTURED_OUTPUT_ADDON = """
## Output format (only when structured mode is enabled)
Use these sections:
1. **Current state** — regime, spot vs flip, total GEX, key walls
2. **Dealer positioning** — hedging flow implications
3. **Base case (10–30 min)** — direction, magnets, ΔGEX view
4. **Alternate scenario** — what flips the view
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
    "What's the vibe right now — are dealers long or short gamma?",
    "Walk me through where spot sits relative to flip and what that means.",
    "Anything interesting in the quant forecasts for the next half hour?",
    "Does today's setup remind you of any recent sessions?",
    "Where's pin risk — which strikes matter most around here?",
    "What drove the last GEX move?",
    "How much should I trust our forecasts today?",
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


def _agent_temperature() -> float:
    return config.LLM_AGENT_TEMPERATURE


def _system_prompt(bundle: dict[str, Any]) -> str:
    summary = bundle.get("summary") or {}
    parts = [AGENT_SYSTEM_PROMPT]
    event_addon = build_event_system_addon(summary)
    if event_addon:
        parts.append(event_addon)
    if config.LLM_STRUCTURED_OUTPUT:
        parts.append(STRUCTURED_OUTPUT_ADDON)
    elif config.LLM_CONVERSATIONAL:
        parts.append(CONVERSATIONAL_STYLE_ADDON)
    agreement = bundle.get("model_agreement") or {}
    if agreement.get("notes"):
        parts.append(f"Quant models: {agreement['notes']} (agreement={agreement.get('score')})")
    return "\n\n".join(parts)


def _system_with_live_data(
    bundle: dict[str, Any],
    context_block: str,
    calibration_note: str,
) -> str:
    return (
        f"{_system_prompt(bundle)}\n\n"
        f"[LIVE DATA — ground every claim in this; do not dump it verbatim]\n"
        f"{context_block}\n\n"
        f"[CALIBRATION NOTE]\n{calibration_note}"
    )


def _build_api_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Natural multi-turn history — live data stays in system prompt, not user messages."""
    out: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out


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

    context_block = _format_context_block(ctx, session_id=session_id, ticker=ticker)
    tool_results: list[dict[str, Any]] = []
    user_question = messages[-1]["content"] if messages else ""
    calibration_note = _calibration_guidance(ctx)

    if two_pass:
        facts = _extract_facts(context_block)
        if facts:
            ctx["extracted_facts"] = facts
            context_block = _format_context_block(ctx, session_id=session_id, ticker=ticker)

    system = _system_with_live_data(bundle, context_block, calibration_note)

    if use_tools:
        runner = AgentToolRunner(history, lookback_days=lookback_days)
        tool_messages = [
            *_build_api_messages(messages[:-1]),
            {
                "role": "user",
                "content": (
                    f"{user_question}\n\n"
                    "(Use tools if you need fresher forecast, strikes, backtest, or KNN vs LLM comparison.)"
                ),
            },
        ]
        with timed_stage("tool_loop"):
            tool_reply, tool_err, tool_log = openai_chat_with_tools(
                system,
                tool_messages,
                TOOL_SCHEMAS,
                runner.execute,
                model=model_for_task("tools"),
                temperature=_agent_temperature(),
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

    api_messages = _build_api_messages(messages)

    with timed_stage("final_answer"):
        reply, error = openai_chat(
            system,
            api_messages,
            max_tokens=config.LLM_MAX_TOKENS,
            model=model_for_task("answer"),
            temperature=_agent_temperature(),
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
        context_block = _format_context_block(ctx, session_id=session_id, ticker=bundle.get("ticker"))
        calibration_note = _calibration_guidance(ctx)
        system = _system_with_live_data(bundle, context_block, calibration_note)
        api_messages = _build_api_messages(messages)
        with timed_stage("stream_answer"):
            for token in openai_chat_stream(
                system,
                api_messages,
                model=model_for_task("answer"),
                temperature=_agent_temperature(),
            ):
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
            "conversational": config.LLM_CONVERSATIONAL,
            "estimated_tokens": ctx.get("estimated_tokens"),
            "has_track_record": bool(bundle.get("forecast_track_record")),
            "has_similar_sessions": bool(bundle.get("similar_sessions")),
            "agreement_score": (bundle.get("model_agreement") or {}).get("score"),
        },
    }
