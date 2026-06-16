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
from models.online_learn import warm_start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=config.DEFAULT_TICKER)
    parser.add_argument("--lookback-days", type=int, default=config.LOOKBACK_DAYS)
    parser.add_argument("--online", action="store_true", help="Bootstrap River online learner")
    parser.add_argument("--gboost", action="store_true", help="Train gradient boosting overlay")
    args = parser.parse_args()
    if not args.online and not args.gboost:
        args.gboost = True
    try:
        require_database_url()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    history = load_snapshot_history(args.ticker, lookback_days=args.lookback_days)
    out: dict[str, object] = {}
    if args.gboost:
        result = train_gboost(history, args.ticker)
        if not result:
            print("GBoost training failed — insufficient data", file=sys.stderr)
            return 2
        out["gboost"] = {"n_train": result["n_train"], "cv_mae": result["cv_mae"]}
    if args.online:
        result = warm_start(history, args.ticker)
        if not result.get("ok"):
            print("Online bootstrap failed — insufficient data", file=sys.stderr)
            return 2
        out["online"] = result
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
