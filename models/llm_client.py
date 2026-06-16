"""OpenAI client helpers for GEX LLM forecasts."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import config

logger = logging.getLogger(__name__)


def resolve_openai_config() -> tuple[str, str] | None:
    key = config.OPENAI_API_KEY.strip()
    if not key:
        return None
    model = config.LLM_MODEL.strip() or "gpt-4o-mini"
    return key, model


def is_llm_configured() -> bool:
    return resolve_openai_config() is not None


def _classify_llm_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "insufficient_quota" in text or ("429" in text and "quota" in text):
        return "OpenAI quota exceeded — add billing credits at platform.openai.com"
    if "invalid_api_key" in text or "incorrect api key" in text or "401" in text:
        return "OpenAI API key is invalid — check OPENAI_API_KEY"
    if "rate_limit" in text or ("429" in text and "quota" not in text):
        return "OpenAI rate limit hit — try again shortly"
    if "model" in text and ("not found" in text or "does not exist" in text):
        return "OpenAI model unavailable — check LLM_MODEL"
    return "LLM request failed — see server logs for details"


def openai_chat(
    system: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    json_mode: bool = False,
) -> tuple[str | None, str | None]:
    """Multi-turn chat. messages: [{role, content}, ...] excluding system."""
    cfg = resolve_openai_config()
    if not cfg:
        return None, "OPENAI_API_KEY is not configured"
    api_key, model = cfg
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        payload: list[dict[str, str]] = [{"role": "system", "content": system}]
        payload.extend(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": payload,
            "max_tokens": max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS,
            "temperature": temperature if temperature is not None else config.LLM_TEMPERATURE,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        return (content.strip() if content else None), None
    except Exception as exc:
        logger.warning("OpenAI chat failed: %s", exc)
        return None, _classify_llm_error(exc)


def openai_chat_with_tools(
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_executor: Any,
    *,
    max_rounds: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    """Run tool-calling loop; tool_executor(name, arguments_str) -> str."""
    cfg = resolve_openai_config()
    if not cfg:
        return None, "OPENAI_API_KEY is not configured", []
    api_key, model = cfg
    max_rounds = max_rounds if max_rounds is not None else config.LLM_MAX_TOOL_ROUNDS
    tool_log: list[dict[str, Any]] = []

    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        payload: list[dict[str, Any]] = [{"role": "system", "content": system}]
        payload.extend(messages)

        for _ in range(max_rounds + 1):
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": payload,
                "max_tokens": max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS,
                "temperature": temperature if temperature is not None else config.LLM_TEMPERATURE,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message

            if not msg.tool_calls:
                content = (msg.content or "").strip()
                return content or None, None, tool_log

            payload.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                result = tool_executor(tc.function.name, tc.function.arguments)
                tool_log.append({"tool": tc.function.name, "arguments": tc.function.arguments, "result_preview": result[:500]})
                payload.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        return None, "Tool loop exceeded max rounds", tool_log
    except Exception as exc:
        logger.warning("OpenAI tool chat failed: %s", exc)
        return None, _classify_llm_error(exc), tool_log


def openai_chat_json(
    system: str,
    user_message: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return (parsed_json, user_error)."""
    cfg = resolve_openai_config()
    if not cfg:
        return None, "OPENAI_API_KEY is not configured"
    api_key, model = cfg
    try:
        parsed, err = openai_chat(
            system,
            [{"role": "user", "content": user_message}],
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=True,
        )
        if err:
            return None, err
        parsed_dict = parse_prediction_json(parsed)
        if parsed_dict is None:
            return None, "LLM returned invalid JSON"
        return parsed_dict, None
    except Exception as exc:
        logger.warning("OpenAI chat failed: %s", exc)
        return None, _classify_llm_error(exc)


def parse_prediction_json(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None
