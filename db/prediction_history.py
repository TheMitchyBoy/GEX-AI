"""Fetch recent resolved LLM predictions with outcomes."""

from __future__ import annotations

import json
from typing import Any

import psycopg
from psycopg.rows import dict_row


def fetch_recent_resolved_outcomes(
    conn: psycopg.Connection,
    ticker: str,
    *,
    limit: int = 10,
    source: str | None = None,
) -> list[dict[str, Any]]:
    clauses, params = ["ticker=%s", "resolved_at IS NOT NULL", "outcome_json IS NOT NULL"], [ticker]
    if source:
        clauses.append("source=%s")
        params.append(source)
    params.append(limit)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT snapshot_ts, payload_json, outcome_json, resolved_at
            FROM llm_predictions
            WHERE {' AND '.join(clauses)}
            ORDER BY resolved_at DESC LIMIT %s
            """,
            params,
        )
        rows = list(cur.fetchall())
    results = []
    for row in rows:
        try:
            payload = row["payload_json"] if isinstance(row["payload_json"], dict) else json.loads(row["payload_json"])
            outcome = row["outcome_json"] if isinstance(row["outcome_json"], dict) else json.loads(row["outcome_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        results.append(
            {
                "snapshot_ts": row["snapshot_ts"],
                "resolved_at": row["resolved_at"],
                "predicted_delta_gex_bn": payload.get("predicted_delta_gex_bn"),
                "predicted_regime": payload.get("predicted_regime"),
                "confidence": payload.get("confidence"),
                "actual_delta_gex_bn": outcome.get("delta_gex_bn"),
                "sign_hit": outcome.get("sign_hit"),
                "regime_hit": outcome.get("regime_hit"),
                "bias_hit": outcome.get("bias_hit"),
                "spot_move_pct": outcome.get("spot_move_pct"),
            }
        )
    return results
