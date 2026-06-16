#!/usr/bin/env python3
"""Evaluate GEX agent intelligence / grounding."""

from __future__ import annotations

import argparse
import json
import sys

import config
from db.connection import require_database_url
from db.loader import load_snapshot_history
from models.llm_eval import evaluate_agent_grounding


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=config.DEFAULT_TICKER)
    parser.add_argument("--lookback-days", type=int, default=30)
    args = parser.parse_args()
    try:
        require_database_url()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    history = load_snapshot_history(args.ticker, lookback_days=args.lookback_days)
    report = evaluate_agent_grounding(history, lookback_days=args.lookback_days)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
