#!/usr/bin/env python3
"""Backfill 90d option quotes from GEX snapshots + UW historical intraday."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from services.option_backfill import backfill_option_quotes


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill option quotes from GEX + UW")
    parser.add_argument("ticker", nargs="?", default=config.DEFAULT_TICKER)
    parser.add_argument("--lookback-days", type=int, default=config.OPTION_BACKFILL_LOOKBACK_DAYS)
    parser.add_argument("--step", type=int, default=config.OPTION_BACKFILL_STEP, help="Use every Nth snapshot")
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    result = backfill_option_quotes(
        args.ticker.upper(),
        lookback_days=args.lookback_days,
        step=args.step,
        train=not args.no_train,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
