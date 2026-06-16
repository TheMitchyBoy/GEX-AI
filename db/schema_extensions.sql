-- Optional extensions for GEX-AI dashboard (idempotent)
-- Run: psql $DATABASE_URL -f db/schema_extensions.sql

CREATE TABLE IF NOT EXISTS snapshot_features (
    ticker TEXT NOT NULL,
    ts TEXT NOT NULL,
    feature_json JSONB NOT NULL,
    surface_vector JSONB,
    materialized_at TEXT NOT NULL,
    PRIMARY KEY (ticker, ts)
);

CREATE INDEX IF NOT EXISTS idx_snapshot_features_ticker_ts
    ON snapshot_features (ticker, ts DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_predictions_dedupe
    ON llm_predictions (ticker, snapshot_ts, source)
    WHERE snapshot_ts IS NOT NULL;

CREATE TABLE IF NOT EXISTS daily_insights (
    ticker TEXT NOT NULL,
    market_date TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (ticker, market_date, kind)
);

CREATE INDEX IF NOT EXISTS idx_daily_insights_ticker_date
    ON daily_insights (ticker, market_date DESC);

CREATE TABLE IF NOT EXISTS agent_feedback (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    session_id TEXT NOT NULL,
    rating INTEGER NOT NULL,
    message TEXT,
    reply_preview TEXT,
    snapshot_ts TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_feedback_ticker
    ON agent_feedback (ticker, created_at DESC);

-- Optional: NOTIFY on new snapshots (processor can also emit manually)
CREATE OR REPLACE FUNCTION notify_gex_snapshot_insert() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('gex_snapshot_insert', NEW.ticker || '|' || NEW.ts);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_gex_snapshot_notify ON snapshots;
CREATE TRIGGER trg_gex_snapshot_notify
    AFTER INSERT ON snapshots
    FOR EACH ROW
    EXECUTE FUNCTION notify_gex_snapshot_insert();
