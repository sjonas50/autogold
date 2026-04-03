-- 001_initial.sql: Create all tables for the gold trading system
-- TimescaleDB hypertables + pgvector indexes
-- Note: Hypertables require the time column in the PRIMARY KEY.

-- Ensure extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================================
-- trade_journal: completed trades with full context
-- ============================================================
CREATE TABLE IF NOT EXISTS trade_journal (
    id              UUID NOT NULL DEFAULT gen_random_uuid(),
    strategy_id     TEXT NOT NULL,
    instrument      TEXT NOT NULL,
    direction       TEXT NOT NULL,
    contracts       INTEGER NOT NULL,
    entry_price     NUMERIC(10,2) NOT NULL,
    exit_price      NUMERIC(10,2),
    entry_time      TIMESTAMPTZ NOT NULL,
    closed_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pnl_usd         NUMERIC(10,2),
    r_multiple      NUMERIC(6,3),
    max_adverse_exc NUMERIC(10,2),
    regime_at_entry TEXT,
    sentiment_score NUMERIC(4,3),
    macro_bias      TEXT,
    session         TEXT,
    atr_at_entry    NUMERIC(10,4),
    notes           TEXT,
    PRIMARY KEY (id, closed_at)
);
SELECT create_hypertable('trade_journal', 'closed_at', if_not_exists => TRUE);

-- ============================================================
-- decision_log: all agent decisions with reasoning
-- ============================================================
CREATE TABLE IF NOT EXISTS decision_log (
    id              UUID NOT NULL DEFAULT gen_random_uuid(),
    agent_name      TEXT NOT NULL,
    decision_type   TEXT NOT NULL,
    inputs_summary  JSONB NOT NULL DEFAULT '{}',
    decision        TEXT NOT NULL,
    reasoning       TEXT,
    confidence      NUMERIC(3,2),
    outcome_tag     TEXT DEFAULT 'pending',
    related_trade   UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
);
SELECT create_hypertable('decision_log', 'created_at', if_not_exists => TRUE);

-- ============================================================
-- lessons: extracted learnings with pgvector embeddings
-- (NOT a hypertable — queried by vector similarity, not time)
-- ============================================================
CREATE TABLE IF NOT EXISTS lessons (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content         TEXT NOT NULL,
    embedding       VECTOR(1536),
    regime_tags     TEXT[],
    strategy_class  TEXT,
    macro_context   TEXT,
    confidence      NUMERIC(3,2),
    source_trades   UUID[],
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS lessons_embedding_idx ON lessons
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================
-- regime_state: market regime classifications
-- ============================================================
CREATE TABLE IF NOT EXISTS regime_state (
    id              UUID NOT NULL DEFAULT gen_random_uuid(),
    regime          TEXT NOT NULL,
    hmm_state       INTEGER,
    hmm_confidence  NUMERIC(5,4),
    atr_14          NUMERIC(10,4),
    adx_14          NUMERIC(6,3),
    timeframe       TEXT NOT NULL DEFAULT '5m',
    dxy_change_1d   NUMERIC(6,4),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
);
SELECT create_hypertable('regime_state', 'created_at', if_not_exists => TRUE);

-- ============================================================
-- sentiment_scores: gold news sentiment
-- ============================================================
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id              UUID NOT NULL DEFAULT gen_random_uuid(),
    headline        TEXT NOT NULL,
    source          TEXT,
    url             TEXT,
    published_at    TIMESTAMPTZ NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sentiment       NUMERIC(4,3) NOT NULL,
    gold_relevance  NUMERIC(3,2),
    catalyst_tags   TEXT[],
    raw_response    JSONB,
    PRIMARY KEY (id, published_at)
);
SELECT create_hypertable('sentiment_scores', 'published_at', if_not_exists => TRUE);

-- ============================================================
-- macro_data: FRED economic data
-- ============================================================
CREATE TABLE IF NOT EXISTS macro_data (
    id              UUID NOT NULL DEFAULT gen_random_uuid(),
    observation_date DATE NOT NULL,
    dxy             NUMERIC(8,4),
    real_yield_10y  NUMERIC(6,4),
    cpi_yoy         NUMERIC(6,4),
    breakeven_10y   NUMERIC(6,4),
    oil_wti         NUMERIC(8,2),
    gold_fix_pm     NUMERIC(8,2),
    macro_regime    TEXT,
    reasoning       TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, observation_date)
);
SELECT create_hypertable('macro_data', 'observation_date', if_not_exists => TRUE);

-- ============================================================
-- strategies: strategy definitions and backtest metrics
-- (NOT a hypertable — keyed by strategy ID)
-- ============================================================
CREATE TABLE IF NOT EXISTS strategies (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    version             INTEGER NOT NULL DEFAULT 1,
    pine_script         TEXT NOT NULL,
    instrument          TEXT NOT NULL DEFAULT 'GC',
    timeframe           TEXT NOT NULL DEFAULT '5m',
    strategy_class      TEXT,
    vbt_sharpe          NUMERIC(6,3),
    vbt_win_rate        NUMERIC(5,4),
    vbt_expectancy      NUMERIC(8,2),
    vbt_max_drawdown    NUMERIC(5,4),
    mc_sharpe_p5        NUMERIC(6,3),
    mc_sharpe_p50       NUMERIC(6,3),
    status              TEXT NOT NULL DEFAULT 'pending_deployment',
    is_active           BOOLEAN NOT NULL DEFAULT false,
    cio_recommendation  TEXT,
    tradingview_alert_id TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deployed_at         TIMESTAMPTZ,
    retired_at          TIMESTAMPTZ
);

-- ============================================================
-- paper_trades: live and closed paper positions
-- (NOT a hypertable — needs fast lookup by status)
-- ============================================================
CREATE TABLE IF NOT EXISTS paper_trades (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id     TEXT NOT NULL,
    instrument      TEXT NOT NULL,
    direction       TEXT NOT NULL,
    contracts       INTEGER NOT NULL,
    entry_price     NUMERIC(10,2) NOT NULL,
    exit_price      NUMERIC(10,2),
    entry_time      TIMESTAMPTZ NOT NULL,
    exit_time       TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'open',
    stop_loss_price NUMERIC(10,2),
    take_profit_price NUMERIC(10,2),
    pnl_usd         NUMERIC(10,2),
    r_multiple      NUMERIC(6,3),
    regime_at_entry TEXT,
    sentiment_at_entry NUMERIC(4,3),
    macro_at_entry  TEXT,
    idempotency_key TEXT UNIQUE NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- pinescript_corpus: RAG corpus for Pine Script v6
-- (NOT a hypertable — queried by vector similarity)
-- ============================================================
CREATE TABLE IF NOT EXISTS pinescript_corpus (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_file     TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    embedding       VECTOR(1536),
    token_count     INTEGER,
    chunk_type      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_file, chunk_index)
);
CREATE INDEX IF NOT EXISTS pinescript_corpus_embedding_idx ON pinescript_corpus
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================
-- ohlcv_5m: 5-minute gold futures price data
-- ============================================================
CREATE TABLE IF NOT EXISTS ohlcv_5m (
    timestamp       TIMESTAMPTZ NOT NULL,
    instrument      TEXT NOT NULL DEFAULT 'GC',
    open            NUMERIC(10,2) NOT NULL,
    high            NUMERIC(10,2) NOT NULL,
    low             NUMERIC(10,2) NOT NULL,
    close           NUMERIC(10,2) NOT NULL,
    volume          BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (timestamp, instrument)
);
SELECT create_hypertable('ohlcv_5m', 'timestamp', if_not_exists => TRUE);

-- ============================================================
-- Indexes for common query patterns
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_trade_journal_strategy ON trade_journal (strategy_id, closed_at DESC);
CREATE INDEX IF NOT EXISTS idx_decision_log_agent ON decision_log (agent_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades (status) WHERE status = 'open';
CREATE INDEX IF NOT EXISTS idx_paper_trades_strategy ON paper_trades (strategy_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_scores_ingested ON sentiment_scores (ingested_at DESC);
CREATE INDEX IF NOT EXISTS idx_regime_state_latest ON regime_state (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_macro_data_latest ON macro_data (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_strategies_active ON strategies (is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_ohlcv_5m_instrument ON ohlcv_5m (instrument, timestamp DESC);
