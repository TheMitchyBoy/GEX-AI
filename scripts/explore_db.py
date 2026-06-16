#!/usr/bin/env python3
"""Explore live database schema and row counts."""

from __future__ import annotations

import json
import sys

import config
from db.connection import get_connection, require_database_url
from db.queries import fetch_latest_snapshot, get_row_counts, get_latest_ts


def main() -> int:
    try:
        require_database_url()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    ticker = config.DEFAULT_TICKER
    with get_connection() as conn:
        counts = get_row_counts(conn)
        latest_ts = get_latest_ts(conn, ticker)
        latest = fetch_latest_snapshot(conn, ticker)

    print("=== GEX Database Explorer ===")
    print(f"Ticker: {ticker}")
    print(f"Lookback days: {config.LOOKBACK_DAYS}")
    print("\nRow counts:")
    for table, count in counts.items():
        print(f"  {table}: {count}")

    print(f"\nLatest ts: {latest_ts}")
    if latest:
        summary = latest.get("summary_json") or {}
        print(f"Spot: {latest.get('spot')}")
        print(f"Total GEX: {latest.get('total_gex')} Bn$/1%")
        print(f"Regime: {latest.get('regime')}")
        if summary:
            print(f"Gamma flip: {summary.get('gamma_flip')}")
            print(f"Flow events: {summary.get('flow_event_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
