"""Application configuration from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL", "")
DEFAULT_TICKER = os.environ.get("DEFAULT_TICKER", "SPX")
SUPPORTED_TICKERS = [t.strip().upper() for t in os.environ.get("SUPPORTED_TICKERS", "SPX,SPY,NDX").split(",") if t.strip()]
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "90"))
FORECAST_POLL_SEC = int(os.environ.get("FORECAST_POLL_SEC", "60"))
PROCESSOR_HEALTH_URL = os.environ.get("PROCESSOR_HEALTH_URL", "")
WRITE_PREDICTIONS = os.environ.get("WRITE_PREDICTIONS", "0") in ("1", "true", "True")
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
PREDICTION_SOURCE = os.environ.get("PREDICTION_SOURCE", "gex-ai-dashboard")
LLM_PREDICTION_SOURCE = os.environ.get("LLM_PREDICTION_SOURCE", "gex-ai-llm")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", os.environ.get("GEX_AGENT_MODEL", "gpt-4o-mini"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", os.environ.get("GEX_AI_MAX_TOKENS", "1200")))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", os.environ.get("GEX_AI_TEMPERATURE", "0.35")))
LLM_CACHE_ENABLED = os.environ.get("LLM_CACHE_ENABLED", "1") in ("1", "true", "True")

API_KEY = os.environ.get("API_KEY", "")
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "120"))
CACHE_MAX_AGE_SEC = int(os.environ.get("CACHE_MAX_AGE_SEC", "30"))

ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "")
ALERT_REGIME_FLIP_THRESHOLD = float(os.environ.get("ALERT_REGIME_FLIP_THRESHOLD", "0.4"))
ALERT_FLIP_DISTANCE_PCT = float(os.environ.get("ALERT_FLIP_DISTANCE_PCT", "0.003"))
ALERT_DELTA_GEX_THRESHOLD = float(os.environ.get("ALERT_DELTA_GEX_THRESHOLD", "0.05"))

USE_LISTEN_NOTIFY = os.environ.get("USE_LISTEN_NOTIFY", "0") in ("1", "true", "True")
PG_NOTIFY_CHANNEL = os.environ.get("PG_NOTIFY_CHANNEL", "gex_snapshot_insert")
RUN_LLM_ON_POLL = os.environ.get("RUN_LLM_ON_POLL", "1") in ("1", "true", "True")
MATERIALIZE_FEATURES = os.environ.get("MATERIALIZE_FEATURES", "0") in ("1", "true", "True")

GBOOST_BLEND_WEIGHT = float(os.environ.get("GBOOST_BLEND_WEIGHT", "0.3"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(Path(__file__).resolve().parent / "artifacts" / "models")))
MULTI_HORIZONS = tuple(int(x) for x in os.environ.get("MULTI_HORIZONS", "1,3,6").split(",") if x.strip())

MIN_KNN_SNAPSHOTS = 4
RECENCY_DECAY = 0.92
SURFACE_BINS = 32
INTERVAL_Z = 1.0
