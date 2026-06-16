"""In-memory LLM forecast cache keyed by ticker+ts."""

from __future__ import annotations

from typing import Any

import config

_CACHE: dict[str, dict[str, Any]] = {}


def _key(ticker: str, ts: str) -> str:
    return f"{ticker.upper()}:{ts}"


def get_cached(ticker: str, ts: str) -> dict[str, Any] | None:
    if not config.LLM_CACHE_ENABLED:
        return None
    return _CACHE.get(_key(ticker, ts))


def set_cached(ticker: str, ts: str, payload: dict[str, Any]) -> None:
    if not config.LLM_CACHE_ENABLED:
        return
    _CACHE[_key(ticker, ts)] = payload
    if len(_CACHE) > 200:
        oldest = next(iter(_CACHE))
        _CACHE.pop(oldest, None)


def clear_cache() -> None:
    _CACHE.clear()
