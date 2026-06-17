#!/usr/bin/env python3
"""Create option_quotes / option_price_predictions tables in Postgres."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow: python3 scripts/ensure_option_schema.py from /app
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from db.connection import get_connection
from db.option_queries import _option_tables_exist, ensure_option_schema


def main() -> int:
    try:
        with get_connection() as conn:
            ensure_option_schema(conn)
            ok = _option_tables_exist(conn)
        if ok:
            print("OK: option_quotes table exists")
            return 0
        print("FAIL: option_quotes table still missing", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
