"""Serve embedded GEX hub (dashboard + agent) from the API."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(tags=["agent-ui"])

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_INDEX = _STATIC_DIR / "index.html"
_AGENT = _STATIC_DIR / "agent.html"


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def home_page() -> HTMLResponse:
    return HTMLResponse(_INDEX.read_text(encoding="utf-8"))


@router.get("/agent", include_in_schema=False)
def agent_redirect():
    return RedirectResponse(url="/", status_code=302)


@router.get("/agent/legacy", response_class=HTMLResponse, include_in_schema=False)
def agent_legacy_page() -> HTMLResponse:
    """Chat-only UI kept for bookmarks and tests."""
    return HTMLResponse(_AGENT.read_text(encoding="utf-8"))


@router.get("/chat", include_in_schema=False)
def chat_redirect():
    return RedirectResponse(url="/", status_code=302)
