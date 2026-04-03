-- 004_multi_timeframe_tables.sql: Add 15m, 1H, and daily OHLCV tables

CREATE TABLE IF NOT EXISTS ohlcv_15m (
    timestamp TIMESTAMPTZ NOT NULL,
    instrument TEXT NOT NULL DEFAULT 'GC',
    open NUMERIC(10,2) NOT NULL,
    high NUMERIC(10,2) NOT NULL,
    low NUMERIC(10,2) NOT NULL,
    close NUMERIC(10,2) NOT NULL,
    volume BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (timestamp, instrument)
);
SELECT create_hypertable('ohlcv_15m', 'timestamp', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS ohlcv_1h (
    timestamp TIMESTAMPTZ NOT NULL,
    instrument TEXT NOT NULL DEFAULT 'GC',
    open NUMERIC(10,2) NOT NULL,
    high NUMERIC(10,2) NOT NULL,
    low NUMERIC(10,2) NOT NULL,
    close NUMERIC(10,2) NOT NULL,
    volume BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (timestamp, instrument)
);
SELECT create_hypertable('ohlcv_1h', 'timestamp', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS ohlcv_daily (
    timestamp TIMESTAMPTZ NOT NULL,
    instrument TEXT NOT NULL DEFAULT 'GC',
    open NUMERIC(10,2) NOT NULL,
    high NUMERIC(10,2) NOT NULL,
    low NUMERIC(10,2) NOT NULL,
    close NUMERIC(10,2) NOT NULL,
    volume BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (timestamp, instrument)
);
SELECT create_hypertable('ohlcv_daily', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_ohlcv_15m_instrument ON ohlcv_15m (instrument, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_1h_instrument ON ohlcv_1h (instrument, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_daily_instrument ON ohlcv_daily (instrument, timestamp DESC);
