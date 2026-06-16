"""Read-only SQL queries against GEX processor tables."""

from __future__ import annotations

from typing import Any

import pandas as pd
import psycopg
from psycopg.rows import dict_row

SNAPSHOT_COLUMNS = """
    ticker, ts, market_date, spot, total_gex, regime,
    summary_json, expiration_json, surface_json, greek_exposure_json, indexed_at
"""


def get_row_counts(conn: psycopg.Connection) -> dict[str, int]:
    tables = ["snapshots", "snapshot_strikes", "llm_predictions", "trades"]
    counts: dict[str, int] = {}
    for table in tables:
        try:
            row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            counts[table] = int(row[0]) if row else 0
        except psycopg.Error:
            counts[table] = -1
    return counts


def get_latest_ts(conn: psycopg.Connection, ticker: str) -> str | None:
    row = conn.execute(
        "SELECT ts FROM snapshots WHERE ticker = %s ORDER BY ts DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    return row[0] if row else None


def fetch_latest_snapshot(conn: psycopg.Connection, ticker: str) -> dict[str, Any] | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT {SNAPSHOT_COLUMNS}
            FROM snapshots
            WHERE ticker = %s
            ORDER BY ts DESC
            LIMIT 1
            """,
            (ticker,),
        )
        return cur.fetchone()


def fetch_snapshots(
    conn: psycopg.Connection,
    ticker: str,
    *,
    lookback_days: int | None = None,
    market_date: str | None = None,
) -> list[dict[str, Any]]:
    clauses = ["ticker = %s"]
    params: list[Any] = [ticker]

    if market_date:
        clauses.append("market_date = %s")
        params.append(market_date)

    if lookback_days and lookback_days > 0:
        clauses.append(
            """
            ts >= (
                SELECT to_char(
                    to_timestamp(
                        replace((SELECT MAX(ts) FROM snapshots WHERE ticker = %s), '_', ' '),
                        'YYYY-MM-DD HH24MISS'
                    ) - (%s || ' days')::interval,
                    'YYYY-MM-DD_HH24MISS'
                )
            )
            """
        )
        params.extend([ticker, lookback_days])

    where = " AND ".join(clauses)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT {SNAPSHOT_COLUMNS}
            FROM snapshots
            WHERE {where}
            ORDER BY ts ASC
            """,
            params,
        )
        return list(cur.fetchall())


def fetch_intraday_timeline(
    conn: psycopg.Connection,
    ticker: str,
    market_date: str,
) -> pd.DataFrame:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT ts, spot, total_gex, regime, summary_json
            FROM snapshots
            WHERE ticker = %s AND market_date = %s
            ORDER BY ts ASC
            """,
            (ticker, market_date),
        )
        rows = cur.fetchall()
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_snapshot_strikes(
    conn: psycopg.Connection,
    ticker: str,
    ts: str,
) -> pd.DataFrame:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT strike, gex_bn_per_pct, cumulative_gex_bn_per_pct
            FROM snapshot_strikes
            WHERE ticker = %s AND ts = %s
            ORDER BY strike ASC
            """,
            (ticker, ts),
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["strike", "gex_bn_per_pct", "cumulative_gex_bn_per_pct"])
    return pd.DataFrame(rows)


def insert_prediction(
    conn: psycopg.Connection,
    *,
    ticker: str,
    snapshot_ts: str,
    market_date: str,
    payload: dict[str, Any],
    source: str,
) -> None:
    import json
    from datetime import datetime, timezone

    created_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO llm_predictions (ticker, source, snapshot_ts, market_date, created_at, payload_json)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT DO NOTHING
        """,
        (ticker, source, snapshot_ts, market_date, created_at, json.dumps(payload)),
    )
    conn.commit()
