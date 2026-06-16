"""In-memory LLM cache keyed by ticker+snapshot+question hash."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import config

_CACHE: dict[str, dict[str, Any]] = {}


def _question_hash(messages: list[dict[str, str]] | None = None, **flags: Any) -> str:
    payload = {
        "last": (messages[-1]["content"] if messages else ""),
        "count": len(messages or []),
        **{k: v for k, v in flags.items() if v is not None},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _key(ticker: str, ts: str, qhash: str = "") -> str:
    return f"{ticker.upper()}:{ts}:{qhash}"


def get_cached(
    ticker: str,
    ts: str,
    *,
    messages: list[dict[str, str]] | None = None,
    two_pass: bool | None = None,
    use_tools: bool | None = None,
    mode: str | None = None,
) -> dict[str, Any] | None:
    if not config.LLM_CACHE_ENABLED:
        return None
    qh = _question_hash(messages, two_pass=two_pass, use_tools=use_tools, mode=mode)
    return _CACHE.get(_key(ticker, ts, qh))


def set_cached(
    ticker: str,
    ts: str,
    payload: dict[str, Any],
    *,
    messages: list[dict[str, str]] | None = None,
    two_pass: bool | None = None,
    use_tools: bool | None = None,
    mode: str | None = None,
) -> None:
    if not config.LLM_CACHE_ENABLED:
        return
    qh = _question_hash(messages, two_pass=two_pass, use_tools=use_tools, mode=mode)
    _CACHE[_key(ticker, ts, qh)] = payload
    if len(_CACHE) > 300:
        oldest = next(iter(_CACHE))
        _CACHE.pop(oldest, None)


def clear_cache() -> None:
    _CACHE.clear()
