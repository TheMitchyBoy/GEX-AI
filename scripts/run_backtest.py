#!/usr/bin/env python3
"""Run walk-forward backtest and print metrics."""

from __future__ import annotations

import argparse
import json
import sys

import config
from db.connection import require_database_url
from db.loader import load_snapshot_history
from models.backtest import run_backtest


def main() -> int:
    parser = argparse.ArgumentParser(description="GEX forecast backtest")
    parser.add_argument("--ticker", default=config.DEFAULT_TICKER)
    parser.add_argument("--lookback-days", type=int, default=30)
    args = parser.parse_args()

    try:
        require_database_url()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    history = load_snapshot_history(args.ticker, lookback_days=args.lookback_days)
    report = run_backtest(history, lookback_days=args.lookback_days)
    print(json.dumps(report.to_dict(), indent=2))
    if report.n_forecasts == 0:
        print("No forecasts produced — need more snapshot history.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
