-- Option tables for UW ingest + price learning (idempotent)
CREATE TABLE IF NOT EXISTS option_quotes (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    snapshot_ts TEXT,
    quote_ts TEXT NOT NULL,
    slot TEXT NOT NULL,
    uw_ticker TEXT NOT NULL,
    option_symbol TEXT NOT NULL,
    option_type TEXT NOT NULL,
    expiry TEXT NOT NULL,
    strike DOUBLE PRECISION,
    spot DOUBLE PRECISION,
    mid_price DOUBLE PRECISION,
    last_price DOUBLE PRECISION,
    nbbo_bid DOUBLE PRECISION,
    nbbo_ask DOUBLE PRECISION,
    implied_volatility DOUBLE PRECISION,
    volume BIGINT,
    open_interest BIGINT,
    moneyness DOUBLE PRECISION,
    dte INTEGER,
    gex_at_strike DOUBLE PRECISION,
    total_gex DOUBLE PRECISION,
    gamma_flip DOUBLE PRECISION,
    flow_features JSONB,
    source TEXT NOT NULL DEFAULT 'unusual_whales',
    UNIQUE (ticker, quote_ts, slot)
);

CREATE INDEX IF NOT EXISTS idx_option_quotes_ticker_ts
    ON option_quotes (ticker, quote_ts DESC);

CREATE INDEX IF NOT EXISTS idx_option_quotes_ticker_slot
    ON option_quotes (ticker, slot, quote_ts DESC);

CREATE TABLE IF NOT EXISTS option_price_predictions (
    ticker TEXT NOT NULL,
    snapshot_ts TEXT NOT NULL,
    slot TEXT NOT NULL,
    option_symbol TEXT,
    predicted_delta_mid DOUBLE PRECISION,
    predicted_pct_change DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    model TEXT,
    features_json JSONB,
    created_at TEXT NOT NULL,
    PRIMARY KEY (ticker, snapshot_ts, slot)
);
