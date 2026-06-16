"""API middleware: auth, rate limiting, metrics, cache headers."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

import config

_metrics: dict[str, float | int] = defaultdict(float)
_rate_buckets: dict[str, list[float]] = defaultdict(list)


def get_metrics() -> dict[str, float | int]:
    return dict(_metrics)


def record_metric(name: str, value: float = 1.0) -> None:
    _metrics[name] = float(_metrics.get(name, 0)) + value


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in ("/", "/agent", "/chat", "/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        # Public agent UI calls these without API key
        if request.url.path.startswith("/llm/chat") or request.url.path in ("/llm/prompts", "/llm/status"):
            return await call_next(request)
        if request.url.path.startswith("/llm/feedback"):
            return await call_next(request)

        if config.API_KEY:
            key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
            if key != config.API_KEY:
                return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})

        if config.RATE_LIMIT_PER_MIN > 0:
            client = request.client.host if request.client else "unknown"
            now = time.time()
            bucket = _rate_buckets[client]
            bucket[:] = [t for t in bucket if now - t < 60]
            if len(bucket) >= config.RATE_LIMIT_PER_MIN:
                record_metric("rate_limited")
                return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
            bucket.append(now)

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        record_metric("requests")
        record_metric("latency_ms_total", elapsed_ms)

        if request.url.path.startswith(("/forecast", "/history", "/similar", "/backtest")):
            response.headers["Cache-Control"] = f"public, max-age={config.CACHE_MAX_AGE_SEC}"
        return response
