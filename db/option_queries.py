"""PostgreSQL queries for option quote ingest and learning."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

_OPTION_SCHEMA = (Path(__file__).resolve().parent / "option_schema.sql").read_text()


def ensure_option_schema(conn: psycopg.Connection) -> None:
    """Create option_quotes tables only (isolated from full schema_extensions)."""
    for stmt in _OPTION_SCHEMA.split(";"):
        stmt = stmt.strip()
        if stmt and not stmt.startswith("--"):
            try:
                conn.execute(stmt)
                conn.commit()
            except psycopg.Error:
                conn.rollback()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def upsert_option_quote(conn: psycopg.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO option_quotes (
            ticker, snapshot_ts, quote_ts, slot, uw_ticker, option_symbol, option_type,
            expiry, strike, spot, mid_price, last_price, nbbo_bid, nbbo_ask,
            implied_volatility, volume, open_interest, moneyness, dte,
            gex_at_strike, total_gex, gamma_flip, flow_features, source
        ) VALUES (
            %(ticker)s, %(snapshot_ts)s, %(quote_ts)s, %(slot)s, %(uw_ticker)s,
            %(option_symbol)s, %(option_type)s, %(expiry)s, %(strike)s, %(spot)s,
            %(mid_price)s, %(last_price)s, %(nbbo_bid)s, %(nbbo_ask)s,
            %(implied_volatility)s, %(volume)s, %(open_interest)s, %(moneyness)s, %(dte)s,
            %(gex_at_strike)s, %(total_gex)s, %(gamma_flip)s,
            %(flow_features)s::jsonb, %(source)s
        )
        ON CONFLICT (ticker, quote_ts, slot) DO UPDATE SET
            snapshot_ts = EXCLUDED.snapshot_ts,
            mid_price = EXCLUDED.mid_price,
            last_price = EXCLUDED.last_price,
            nbbo_bid = EXCLUDED.nbbo_bid,
            nbbo_ask = EXCLUDED.nbbo_ask,
            implied_volatility = EXCLUDED.implied_volatility,
            volume = EXCLUDED.volume,
            open_interest = EXCLUDED.open_interest,
            gex_at_strike = EXCLUDED.gex_at_strike,
            total_gex = EXCLUDED.total_gex,
            gamma_flip = EXCLUDED.gamma_flip,
            flow_features = EXCLUDED.flow_features
        """,
        {
            **row,
            "flow_features": json.dumps(row.get("flow_features") or {}),
        },
    )


def fetch_option_quotes(
    conn: psycopg.Connection,
    ticker: str,
    *,
    slot: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        if slot:
            cur.execute(
                """
                SELECT * FROM option_quotes
                WHERE ticker = %s AND slot = %s
                ORDER BY quote_ts ASC
                LIMIT %s
                """,
                (ticker.upper(), slot, limit),
            )
        else:
            cur.execute(
                """
                SELECT * FROM option_quotes
                WHERE ticker = %s
                ORDER BY quote_ts ASC
                LIMIT %s
                """,
                (ticker.upper(), limit),
            )
        return list(cur.fetchall())


def fetch_latest_option_quotes(conn: psycopg.Connection, ticker: str) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (slot) *
            FROM option_quotes
            WHERE ticker = %s
            ORDER BY slot, quote_ts DESC
            """,
            (ticker.upper(),),
        )
        return list(cur.fetchall())


def insert_option_prediction(conn: psycopg.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO option_price_predictions (
            ticker, snapshot_ts, slot, option_symbol,
            predicted_delta_mid, predicted_pct_change, confidence,
            model, features_json, created_at
        ) VALUES (
            %(ticker)s, %(snapshot_ts)s, %(slot)s, %(option_symbol)s,
            %(predicted_delta_mid)s, %(predicted_pct_change)s, %(confidence)s,
            %(model)s, %(features_json)s::jsonb, %(created_at)s
        )
        ON CONFLICT (ticker, snapshot_ts, slot) DO UPDATE SET
            option_symbol = EXCLUDED.option_symbol,
            predicted_delta_mid = EXCLUDED.predicted_delta_mid,
            predicted_pct_change = EXCLUDED.predicted_pct_change,
            confidence = EXCLUDED.confidence,
            model = EXCLUDED.model,
            features_json = EXCLUDED.features_json,
            created_at = EXCLUDED.created_at
        """,
        {
            **row,
            "features_json": json.dumps(row.get("features_json") or {}),
        },
    )


def gex_at_strike(conn: psycopg.Connection, ticker: str, ts: str, strike: float) -> float | None:
    row = conn.execute(
        """
        SELECT gex_bn_per_pct FROM snapshot_strikes
        WHERE ticker = %s AND ts = %s AND strike = %s
        LIMIT 1
        """,
        (ticker.upper(), ts, strike),
    ).fetchone()
    if row:
        return float(row[0])
    row = conn.execute(
        """
        SELECT gex_bn_per_pct FROM snapshot_strikes
        WHERE ticker = %s AND ts = %s
        ORDER BY ABS(strike - %s) ASC
        LIMIT 1
        """,
        (ticker.upper(), ts, strike),
    ).fetchone()
    return float(row[0]) if row else None
