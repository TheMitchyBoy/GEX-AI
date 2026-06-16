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
        """,
        (ticker, source, snapshot_ts, market_date, created_at, json.dumps(payload)),
    )
    conn.commit()


def fetch_llm_predictions(
    conn: psycopg.Connection,
    ticker: str,
    *,
    limit: int = 50,
    source: str | None = None,
    unresolved_only: bool = False,
) -> list[dict[str, Any]]:
    clauses = ["ticker = %s"]
    params: list[Any] = [ticker]
    if source:
        clauses.append("source = %s")
        params.append(source)
    if unresolved_only:
        clauses.append("resolved_at IS NULL")
    where = " AND ".join(clauses)
    params.append(limit)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT id, ticker, source, snapshot_ts, market_date, created_at,
                   resolved_at, payload_json, actual_json, outcome_json
            FROM llm_predictions
            WHERE {where}
            ORDER BY created_at DESC
            LIMIT %s
            """,
            params,
        )
        return list(cur.fetchall())


def fetch_snapshot_strikes_bulk(
    conn: psycopg.Connection,
    ticker: str,
    ts_list: list[str],
) -> dict[str, pd.DataFrame]:
    if not ts_list:
        return {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT ts, strike, gex_bn_per_pct, cumulative_gex_bn_per_pct
            FROM snapshot_strikes WHERE ticker = %s AND ts = ANY(%s)
            ORDER BY ts ASC, strike ASC
            """,
            (ticker, ts_list),
        )
        rows = cur.fetchall()
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    return {ts: grp.drop(columns=["ts"]).reset_index(drop=True) for ts, grp in df.groupby("ts")}


def insert_prediction_deduped(conn, *, ticker, snapshot_ts, market_date, payload, source) -> bool:
    import json
    from datetime import datetime, timezone

    if conn.execute(
        "SELECT 1 FROM llm_predictions WHERE ticker=%s AND snapshot_ts=%s AND source=%s LIMIT 1",
        (ticker, snapshot_ts, source),
    ).fetchone():
        return False
    conn.execute(
        "INSERT INTO llm_predictions (ticker,source,snapshot_ts,market_date,created_at,payload_json) VALUES (%s,%s,%s,%s,%s,%s::jsonb)",
        (ticker, source, snapshot_ts, market_date, datetime.now(timezone.utc).isoformat(), json.dumps(payload)),
    )
    conn.commit()
    return True


def fetch_unresolved_predictions(conn, ticker):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id,snapshot_ts,payload_json,source FROM llm_predictions WHERE ticker=%s AND resolved_at IS NULL AND snapshot_ts IS NOT NULL ORDER BY created_at",
            (ticker,),
        )
        return list(cur.fetchall())


def resolve_prediction(conn, prediction_id, actual, outcome):
    import json
    from datetime import datetime, timezone

    conn.execute(
        "UPDATE llm_predictions SET resolved_at=%s,actual_json=%s::jsonb,outcome_json=%s::jsonb WHERE id=%s",
        (datetime.now(timezone.utc).isoformat(), json.dumps(actual), json.dumps(outcome), prediction_id),
    )
    conn.commit()


def fetch_snapshot_at_ts(conn, ticker, ts):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT {SNAPSHOT_COLUMNS} FROM snapshots WHERE ticker=%s AND ts=%s", (ticker, ts))
        return cur.fetchone()


def fetch_next_ts_after(conn, ticker, anchor_ts):
    row = conn.execute(
        "SELECT ts FROM snapshots WHERE ticker=%s AND ts>%s ORDER BY ts ASC LIMIT 1", (ticker, anchor_ts)
    ).fetchone()
    return row[0] if row else None


def fetch_calibration_stats(conn, ticker, *, source=None, limit=500):
    import json

    clauses, params = ["ticker=%s", "resolved_at IS NOT NULL"], [ticker]
    if source:
        clauses.append("source=%s")
        params.append(source)
    params.append(limit)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT outcome_json,payload_json FROM llm_predictions WHERE {' AND '.join(clauses)} ORDER BY resolved_at DESC LIMIT %s",
            params,
        )
        rows = cur.fetchall()
    sign_hits, bias_hits, regime_hits, confidences = [], [], [], []
    for row in rows:
        try:
            outcome = row["outcome_json"] if isinstance(row["outcome_json"], dict) else json.loads(row["outcome_json"])
            payload = row["payload_json"] if isinstance(row["payload_json"], dict) else json.loads(row["payload_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        if outcome.get("sign_hit") is not None:
            sign_hits.append(bool(outcome["sign_hit"]))
        if outcome.get("bias_hit") is not None:
            bias_hits.append(bool(outcome["bias_hit"]))
        if outcome.get("regime_hit") is not None:
            regime_hits.append(bool(outcome["regime_hit"]))
        confidences.append(float(payload.get("confidence") or 0))
    n = len(sign_hits)
    return {
        "n": n,
        "sign_accuracy": sum(sign_hits) / n if n else None,
        "bias_accuracy": sum(bias_hits) / len(bias_hits) if bias_hits else None,
        "regime_accuracy": sum(regime_hits) / len(regime_hits) if regime_hits else None,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else None,
    }


def upsert_snapshot_features(conn, *, ticker, ts, feature_json, surface_vector):
    import json
    from datetime import datetime, timezone

    conn.execute(
        """
        INSERT INTO snapshot_features (ticker,ts,feature_json,surface_vector,materialized_at)
        VALUES (%s,%s,%s::jsonb,%s::jsonb,%s)
        ON CONFLICT (ticker,ts) DO UPDATE SET feature_json=EXCLUDED.feature_json,
        surface_vector=EXCLUDED.surface_vector, materialized_at=EXCLUDED.materialized_at
        """,
        (ticker, ts, json.dumps(feature_json), json.dumps(surface_vector), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def upsert_daily_insight(conn, *, ticker, market_date, kind, payload):
    import json
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO daily_insights (ticker,market_date,kind,payload_json,created_at,updated_at)
        VALUES (%s,%s,%s,%s::jsonb,%s,%s)
        ON CONFLICT (ticker,market_date,kind) DO UPDATE SET payload_json=EXCLUDED.payload_json, updated_at=EXCLUDED.updated_at
        """,
        (ticker, market_date, kind, json.dumps(payload), now, now),
    )
    conn.commit()


def fetch_daily_insights(conn, ticker, *, market_date=None, limit=10):
    with conn.cursor(row_factory=dict_row) as cur:
        if market_date:
            cur.execute("SELECT * FROM daily_insights WHERE ticker=%s AND market_date=%s ORDER BY kind", (ticker, market_date))
        else:
            cur.execute(
                "SELECT * FROM daily_insights WHERE ticker=%s ORDER BY market_date DESC,kind LIMIT %s", (ticker, limit)
            )
        return list(cur.fetchall())


def ensure_extensions(conn):
    from pathlib import Path

    path = Path(__file__).resolve().parent / "schema_extensions.sql"
    if not path.exists():
        return
    for stmt in path.read_text().split(";"):
        stmt = stmt.strip()
        if stmt and not stmt.startswith("--"):
            try:
                conn.execute(stmt)
            except psycopg.Error:
                pass
    conn.commit()
