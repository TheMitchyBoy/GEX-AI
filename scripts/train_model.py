#!/usr/bin/env python3
"""Train gradient boosting overlay model for a ticker."""

from __future__ import annotations

import argparse
import json
import sys

import config
from db.connection import require_database_url
from db.loader import load_snapshot_history
from models.gboost import train_gboost


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=config.DEFAULT_TICKER)
    parser.add_argument("--lookback-days", type=int, default=config.LOOKBACK_DAYS)
    args = parser.parse_args()
    try:
        require_database_url()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    history = load_snapshot_history(args.ticker, lookback_days=args.lookback_days)
    result = train_gboost(history, args.ticker)
    if not result:
        print("Training failed — insufficient data", file=sys.stderr)
        return 2
    print(json.dumps({"n_train": result["n_train"], "cv_mae": result["cv_mae"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
