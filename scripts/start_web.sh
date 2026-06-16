#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8000}"
PYTHON="$(command -v python3 || command -v python)"
exec "$PYTHON" -m uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
