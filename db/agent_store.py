"""Agent feedback and session memory (Postgres with JSON file fallback)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)

_STORE_DIR = Path(config.MODELS_DIR).parent / "agent_data"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_dir() -> Path:
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    return _STORE_DIR


def save_feedback(
    *,
    ticker: str,
    session_id: str,
    rating: int,
    message: str | None = None,
    reply: str | None = None,
    snapshot_ts: str | None = None,
) -> dict[str, Any]:
    row = {
        "ticker": ticker.upper(),
        "session_id": session_id,
        "rating": rating,
        "message": message,
        "reply_preview": (reply or "")[:500],
        "snapshot_ts": snapshot_ts,
        "created_at": _now(),
    }
    path = _ensure_dir() / "feedback.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    try:
        from db.connection import get_connection
        from db.queries import ensure_extensions

        with get_connection() as conn:
            ensure_extensions(conn)
            conn.execute(
                """
                INSERT INTO agent_feedback (ticker, session_id, rating, message, reply_preview, snapshot_ts, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (row["ticker"], session_id, rating, message, row["reply_preview"], snapshot_ts, row["created_at"]),
            )
            conn.commit()
    except Exception:
        logger.debug("agent_feedback table unavailable, using file store only", exc_info=True)
    return row


def load_session_memory(ticker: str, session_id: str, *, limit: int = 5) -> list[dict[str, Any]]:
    path = _ensure_dir() / f"memory_{ticker.upper()}_{session_id}.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data[-limit:] if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def append_session_memory(ticker: str, session_id: str, entry: dict[str, Any]) -> None:
    path = _ensure_dir() / f"memory_{ticker.upper()}_{session_id}.json"
    entries = load_session_memory(ticker, session_id, limit=100)
    entry["ts"] = _now()
    entries.append(entry)
    entries = entries[-50:]
    path.write_text(json.dumps(entries, indent=2))


def recent_feedback_summary(ticker: str, *, limit: int = 20) -> dict[str, Any]:
    path = _ensure_dir() / "feedback.jsonl"
    if not path.exists():
        return {"n": 0, "positive_rate": None}
    rows = []
    try:
        for line in path.read_text().splitlines()[-limit:]:
            row = json.loads(line)
            if row.get("ticker") == ticker.upper():
                rows.append(row)
    except (json.JSONDecodeError, OSError):
        return {"n": 0, "positive_rate": None}
    if not rows:
        return {"n": 0, "positive_rate": None}
    positive = sum(1 for r in rows if int(r.get("rating", 0)) > 0)
    return {"n": len(rows), "positive_rate": round(positive / len(rows), 2)}
