"""Serve embedded GEX agent chat UI from the API."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(tags=["agent-ui"])

_STATIC = Path(__file__).resolve().parent / "static" / "agent.html"


@router.get("/agent", response_class=HTMLResponse, include_in_schema=False)
def agent_page() -> HTMLResponse:
    return HTMLResponse(_STATIC.read_text(encoding="utf-8"))


@router.get("/chat", include_in_schema=False)
def chat_redirect():
    return RedirectResponse(url="/agent", status_code=302)
