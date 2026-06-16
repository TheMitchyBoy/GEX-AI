"""Load snapshots from Postgres into prediction-ready history dicts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config
from db.connection import get_connection
from db.features import (
    apply_summary_fields,
    enrich_snapshot_metrics,
    expiration_series_from_json,
    strike_series_from_strikes_df,
    term_structure_breakdown,
)
from db.json_fallbacks import resolve_strike_series
from db.queries import fetch_snapshot_strikes_bulk, fetch_snapshots


def snapshot_to_history_dict(
    row: dict[str, Any],
    strikes_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    summary = row.get("summary_json") or {}
    if isinstance(summary, str):
        import json

        summary = json.loads(summary)

    strike, cumulative = resolve_strike_series(strikes_df, row)

    exp = expiration_series_from_json(row.get("expiration_json"))
    market_date = row.get("market_date")
    snap_date = pd.Timestamp(market_date) if market_date else None
    term = term_structure_breakdown(exp, snapshot_date=snap_date)

    metrics: dict[str, Any] = {
        "ticker": row["ticker"],
        "ts": row["ts"],
        "market_date": market_date,
        "spot": row.get("spot"),
        "total_gex": row.get("total_gex"),
        "regime": row.get("regime") or summary.get("net_gamma_regime"),
        "strike": strike,
        "cumulative": cumulative,
        "summary": summary,
        "surface_json": row.get("surface_json"),
        "greek_exposure_json": row.get("greek_exposure_json"),
        "expiration_json": row.get("expiration_json"),
        **term,
    }
    apply_summary_fields(metrics, summary)
    return metrics


def load_snapshot_history(
    ticker: str | None = None,
    *,
    lookback_days: int | None = None,
    include_strikes: bool = True,
) -> list[dict[str, Any]]:
    ticker = ticker or config.DEFAULT_TICKER
    lookback_days = lookback_days if lookback_days is not None else config.LOOKBACK_DAYS

    with get_connection() as conn:
        rows = fetch_snapshots(conn, ticker, lookback_days=lookback_days)
        strikes_by_ts: dict[str, pd.DataFrame] = {}
        if include_strikes and rows:
            strikes_by_ts = fetch_snapshot_strikes_bulk(conn, ticker, [r["ts"] for r in rows])

        history = [snapshot_to_history_dict(row, strikes_by_ts.get(row["ts"])) for row in rows]
    return history


def history_to_dataframe(history: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for h in history:
        enriched = enrich_snapshot_metrics(h.copy())
        records.append(
            {
                "ts": enriched["ts"],
                "market_date": enriched.get("market_date"),
                "spot": enriched["spot"],
                "total_gex": enriched["total_gex"],
                "regime": enriched.get("regime"),
                "gamma_flip": enriched.get("gamma_flip"),
                "call_wall": enriched.get("call_wall"),
                "put_wall": enriched.get("put_wall"),
                "flip_distance_pct": enriched.get("flip_distance_pct"),
                "near_term_ratio": enriched.get("near_term_ratio"),
                "flow_event_count": enriched.get("flow_event_count"),
                "event_risk_score": enriched.get("event_risk_score"),
            }
        )
    return pd.DataFrame(records)


def materialize_features_for_history(history: list[dict[str, Any]]) -> int:
    """Write precomputed features to snapshot_features table."""
    from db.queries import upsert_snapshot_features

    count = 0
    ticker = history[0].get("ticker", config.DEFAULT_TICKER) if history else config.DEFAULT_TICKER
    with get_connection() as conn:
        for h in history:
            enriched = enrich_snapshot_metrics(h.copy())
            sv = enriched.get("surface_vector", np.zeros(config.SURFACE_BINS))
            feature_json = {
                "total_gex": enriched["total_gex"],
                "gamma_flip": enriched.get("gamma_flip"),
                "call_wall": enriched.get("call_wall"),
                "put_wall": enriched.get("put_wall"),
                "near_term_ratio": enriched.get("near_term_ratio"),
                "flip_distance_pct": enriched.get("flip_distance_pct"),
            }
            upsert_snapshot_features(
                conn,
                ticker=ticker,
                ts=enriched["ts"],
                feature_json=feature_json,
                surface_vector=sv.tolist() if hasattr(sv, "tolist") else list(sv),
            )
            count += 1
    return count
