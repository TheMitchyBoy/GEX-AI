"""Application configuration from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL", "")
DEFAULT_TICKER = os.environ.get("DEFAULT_TICKER", "SPX")
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "90"))
FORECAST_POLL_SEC = int(os.environ.get("FORECAST_POLL_SEC", "60"))
PROCESSOR_HEALTH_URL = os.environ.get("PROCESSOR_HEALTH_URL", "")
WRITE_PREDICTIONS = os.environ.get("WRITE_PREDICTIONS", "0") in ("1", "true", "True")
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
PREDICTION_SOURCE = os.environ.get("PREDICTION_SOURCE", "gex-ai-dashboard")

MIN_KNN_SNAPSHOTS = 4
RECENCY_DECAY = 0.92
SURFACE_BINS = 32
INTERVAL_Z = 1.0
