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
LLM_MODEL = os.environ.get("LLM_MODEL", os.environ.get("GEX_AGENT_MODEL", "gpt-4o"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", os.environ.get("GEX_AI_MAX_TOKENS", "2000")))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", os.environ.get("GEX_AI_TEMPERATURE", "0.25")))
LLM_CACHE_ENABLED = os.environ.get("LLM_CACHE_ENABLED", "1") in ("1", "true", "True")
LLM_TWO_PASS = os.environ.get("LLM_TWO_PASS", "1") in ("1", "true", "True")
LLM_USE_TOOLS = os.environ.get("LLM_USE_TOOLS", "1") in ("1", "true", "True")
LLM_RICH_CONTEXT = os.environ.get("LLM_RICH_CONTEXT", "1") in ("1", "true", "True")
LLM_MAX_TOOL_ROUNDS = int(os.environ.get("LLM_MAX_TOOL_ROUNDS", "2"))
LLM_AGENT_FAST = os.environ.get("LLM_AGENT_FAST", "1") in ("1", "true", "True")
LLM_AGENT_TIMEOUT_SEC = int(os.environ.get("LLM_AGENT_TIMEOUT_SEC", "120"))
LLM_MODEL_FAST = os.environ.get("LLM_MODEL_FAST", "gpt-4o-mini")
LLM_AGENT_TEMPERATURE = float(os.environ.get("LLM_AGENT_TEMPERATURE", os.environ.get("LLM_TEMPERATURE", "0.45")))
LLM_STRUCTURED_OUTPUT = os.environ.get("LLM_STRUCTURED_OUTPUT", "0") in ("1", "true", "True")
LLM_CONVERSATIONAL = os.environ.get("LLM_CONVERSATIONAL", "1") in ("1", "true", "True")
LLM_CONTEXT_COMPRESS = os.environ.get("LLM_CONTEXT_COMPRESS", "1") in ("1", "true", "True")
ENSEMBLE_ENABLED = os.environ.get("ENSEMBLE_ENABLED", "1") in ("1", "true", "True")
ENSEMBLE_WEIGHT_KNN = float(os.environ.get("ENSEMBLE_WEIGHT_KNN", "0.5"))
ENSEMBLE_WEIGHT_GBOOST = float(os.environ.get("ENSEMBLE_WEIGHT_GBOOST", "0.25"))
ENSEMBLE_WEIGHT_ONLINE = float(os.environ.get("ENSEMBLE_WEIGHT_ONLINE", "0.25"))
AGENT_MEMORY_ENABLED = os.environ.get("AGENT_MEMORY_ENABLED", "1") in ("1", "true", "True")
AUTO_TRAIN_GBOOST = os.environ.get("AUTO_TRAIN_GBOOST", "1") in ("1", "true", "True")

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
ONLINE_LEARNING_ENABLED = os.environ.get("ONLINE_LEARNING_ENABLED", "1") in ("1", "true", "True")
ONLINE_BLEND_WEIGHT = float(os.environ.get("ONLINE_BLEND_WEIGHT", "0.15"))
ONLINE_AUTO_BOOTSTRAP = os.environ.get("ONLINE_AUTO_BOOTSTRAP", "1") in ("1", "true", "True")
ONLINE_MIN_UPDATES = int(os.environ.get("ONLINE_MIN_UPDATES", "20"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(Path(__file__).resolve().parent / "artifacts" / "models")))
MULTI_HORIZONS = tuple(int(x) for x in os.environ.get("MULTI_HORIZONS", "1,3,6").split(",") if x.strip())

MIN_KNN_SNAPSHOTS = 4
RECENCY_DECAY = 0.92
SURFACE_BINS = 32
INTERVAL_Z = 1.0
