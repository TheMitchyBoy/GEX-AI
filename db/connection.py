"""PostgreSQL connection helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg

import config


def require_database_url() -> str:
    if not config.DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and configure Railway Postgres."
        )
    return config.DATABASE_URL


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """Yield a psycopg connection; caller manages transactions."""
    url = require_database_url()
    conn = psycopg.connect(url)
    try:
        yield conn
    finally:
        conn.close()
