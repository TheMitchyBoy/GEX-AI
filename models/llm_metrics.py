"""LLM latency and stage metrics."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Generator

_metrics: dict[str, list[float]] = defaultdict(list)


@contextmanager
def timed_stage(name: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        _metrics[name].append(elapsed_ms)
        if len(_metrics[name]) > 500:
            _metrics[name] = _metrics[name][-500:]


def get_llm_metrics() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, samples in _metrics.items():
        if not samples:
            continue
        out[name] = {
            "count": len(samples),
            "avg_ms": round(sum(samples) / len(samples), 1),
            "last_ms": round(samples[-1], 1),
        }
    return out


def clear_llm_metrics() -> None:
    _metrics.clear()
