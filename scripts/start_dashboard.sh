#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8501}"
PYTHON="$(command -v python3 || command -v python)"
exec "$PYTHON" -m streamlit run dashboard/app.py \
  --server.port="$PORT" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
