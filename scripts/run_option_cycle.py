#!/usr/bin/env python3
"""Run option ingest → learn → predict cycle (for Railway shell without curl)."""

from __future__ import annotations

import json
import os
import sys

import httpx


def main() -> int:
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "SPX").upper()
    port = os.environ.get("PORT", "8000")
    base = f"http://127.0.0.1:{port}"
    r = httpx.post(f"{base}/options/cycle/{ticker}", timeout=120)
    print("Status:", r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)
    return 0 if r.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
