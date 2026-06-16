# GEX Prediction & Analytics Dashboard

Standalone read-only analytics and prediction app for gamma exposure (GEX) data written by the [TheMitchyBoy/GEX](https://github.com/TheMitchyBoy/GEX) processor into Railway PostgreSQL.

## Features

- **Postgres consumer** — loads `snapshots` and `snapshot_strikes` (no CSV exports, no UW API key)
- **Feature pipeline** — scalar regime features + 32-bin ATM surface vectors from strike profiles
- **Weighted KNN baseline** — z-scored features, cosine surface similarity, exponential recency decay
- **Forecasts** — next-snapshot ΔGEX, total GEX, regime, gamma flip, spot bias, confidence interval
- **Similar setups** — nearest historical analogs
- **Walk-forward backtest** — MAE on ΔGEX, regime accuracy, spot bias hit rate, interval coverage
- **Streamlit dashboard** — live regime, forecast card, intraday charts, strike heatmap
- **FastAPI** — `/forecast`, `/history`, `/similar`, `/backtest`, `/strikes`, `/llm/*`
- **Forecast poller** — optional job to detect new `ts` and write to `llm_predictions`

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Railway DATABASE_URL
```

### Explore database

```bash
python scripts/explore_db.py
```

### Run API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | App + DB row counts |
| GET | `/forecast/{ticker}` | Next-snapshot forecast |
| GET | `/history/{ticker}` | Snapshot timeline |
| GET | `/similar/{ticker}` | Historical analogs |
| GET | `/strikes/{ticker}?ts=` | Strike profile |
| GET | `/backtest/{ticker}` | Walk-forward metrics |
| GET | `/llm/status` | LLM configuration status |
| GET | `/llm/forecast/{ticker}` | LLM-enhanced next-snapshot forecast |
| POST | `/llm/forecast/{ticker}` | LLM forecast with optional extra instructions |
| GET | `/llm/predictions/{ticker}` | Logged predictions from `llm_predictions` |

### LLM forecast

Requires `OPENAI_API_KEY`. Builds a context bundle from Postgres (current snapshot, strikes, term structure, intraday timeline, KNN forecast, similar setups) and returns structured JSON:

```bash
curl http://localhost:8000/llm/forecast/SPX

curl -X POST http://localhost:8000/llm/forecast/SPX \
  -H 'Content-Type: application/json' \
  -d '{"extra_instructions": "Focus on 0DTE pin risk near spot."}'
```

Without an API key, `/llm/forecast` falls back to the KNN baseline with rule-based narrative.

Set `WRITE_PREDICTIONS=1` or pass `?persist=true` to log forecasts into `llm_predictions`.

### Run dashboard

```bash
streamlit run dashboard/app.py
```

### Forecast poller

```bash
# Set WRITE_PREDICTIONS=1 to insert into llm_predictions
python jobs/forecast_poll.py
```

### Backtest CLI

```bash
python scripts/run_backtest.py --ticker SPX --lookback-days 30
```

### Tests (synthetic data, no DB required)

```bash
pytest -q
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | Railway PostgreSQL connection string |
| `DEFAULT_TICKER` | `SPX` | Primary ticker |
| `LOOKBACK_DAYS` | `90` | Training window for KNN |
| `FORECAST_POLL_SEC` | `60` | Dashboard/poller refresh interval |
| `PROCESSOR_HEALTH_URL` | — | Optional upstream `GET /health/live` |
| `WRITE_PREDICTIONS` | `0` | Insert forecasts into `llm_predictions` |
| `OPENAI_API_KEY` | — | OpenAI API key for LLM forecasts |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `LLM_MAX_TOKENS` | `1200` | Max tokens per LLM response |
| `LLM_TEMPERATURE` | `0.35` | LLM sampling temperature |
| `LLM_PREDICTION_SOURCE` | `gex-ai-llm` | `source` column when logging LLM forecasts |

## Database schema

Canonical tables (from upstream processor):

### `snapshots`

Primary key: `(ticker, ts)` where `ts` is `YYYY-MM-DD_HHMMSS` UTC.

| Column | Use |
|--------|-----|
| `spot`, `total_gex`, `regime` | Headline metrics |
| `summary_json` | Flip, flow, greeks, calendar flags |
| `expiration_json` | Term structure |
| `surface_json` | Optional surface rows |
| `greek_exposure_json` | UW greek exposure |

### `snapshot_strikes`

Per-strike `gex_bn_per_pct` and `cumulative_gex_bn_per_pct` for heatmaps and surface vectors.

### Example queries

```sql
-- Latest snapshot
SELECT ticker, ts, spot, total_gex, regime, summary_json
FROM snapshots WHERE ticker = 'SPX' ORDER BY ts DESC LIMIT 1;

-- Intraday timeline
SELECT ts, spot, total_gex, regime
FROM snapshots
WHERE ticker = 'SPX' AND market_date = CURRENT_DATE::text
ORDER BY ts;

-- Strike profile
SELECT strike, gex_bn_per_pct, cumulative_gex_bn_per_pct
FROM snapshot_strikes
WHERE ticker = 'SPX' AND ts = '2026-06-15_225554'
ORDER BY strike;
```

See [docs/DASHBOARD_SCHEMA.md](https://github.com/TheMitchyBoy/GEX/blob/main/docs/DASHBOARD_SCHEMA.md) in the upstream repo.

## Prediction model

Baseline **weighted KNN** (aligned with upstream `gex_core/predict.py`):

1. Enrich each snapshot with walls, flip, term ratios, surface vector
2. Build training pairs `(snapshot_t → snapshot_{t+1})`
3. Z-score regime features; blend L2 distance with strike-surface cosine distance (35%)
4. Weight neighbors by inverse distance × recency decay (`0.92` per step)
5. Output weighted mean targets + empirical prediction interval

Requires **≥ 4 snapshots**; returns `null` / HTTP 422 when insufficient.

### Forecast outputs

| Field | Description |
|-------|-------------|
| `predicted_delta_gex` | Change in net GEX (Bn$/1%) |
| `predicted_total_gex` | Next total GEX |
| `predicted_regime` | LONG vs SHORT gamma |
| `predicted_flip` | Gamma flip strike |
| `spot_bias` | `up` / `down` / `neutral` toward magnets/flip |
| `confidence` | 0–1 calibrated score |
| `prediction_interval` | Low/high band on ΔGEX |

Optional write-back:

```sql
INSERT INTO llm_predictions (ticker, source, snapshot_ts, market_date, created_at, payload_json)
VALUES ('SPX', 'gex-ai-dashboard', '2026-06-15_225554', '2026-06-15', NOW()::text, '{"predicted_delta_gex_bn": -0.01, "confidence": 0.72}');
```

## Project layout

```
db/           # Postgres queries, feature engineering, loader
models/       # KNN predict, walk-forward backtest
api/          # FastAPI service
dashboard/    # Streamlit UI
jobs/         # Snapshot poller
scripts/      # explore_db, run_backtest
tests/        # Unit tests with synthetic snapshots
```

## Processor health

Upstream processor exposes:

```
GET https://<processor-service>/health/live
→ {"mode":"processor","status":"ok","latest_ts":"..."}
```

Set `PROCESSOR_HEALTH_URL` to surface this in `/health`.

## Units

All GEX values are in **billions of dollars per 1% spot move** (`Bn$/1%`).

## License

MIT
