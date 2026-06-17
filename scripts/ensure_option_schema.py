#!/usr/bin/env python3
"""Create option_quotes / option_price_predictions tables in Postgres."""

from __future__ import annotations

import sys

from db.connection import get_connection
from db.option_queries import ensure_option_schema, _option_tables_exist


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
