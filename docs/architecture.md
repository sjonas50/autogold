# Architecture: Autonomous Gold Futures Trading System

**Date:** 2026-04-03 | **Scope:** Backtest + paper trading only. No live broker execution.

---

## System Overview

A Paperclip-orchestrated company of 7 AI agents (all Claude Sonnet 4.6) that research gold market conditions, develop and validate Pine Script trading strategies via vectorbt, and execute paper trades on GC/MGC futures through a TradingView webhook pipeline. Strategy development is fully automated; Pine Script deployment to TradingView requires a human-in-the-loop step because TradingView has no programmatic deployment API. All agent state persists in TimescaleDB + pgvector, eliminating dependency on in-memory session context.

```
                         ┌─────────────────────────────────────────────────────┐
                         │                  PAPERCLIP SERVER                   │
                         │              (Node.js 20 + React UI)                │
                         │                                                     │
                         │  ┌──────────┐   ┌──────────────────────────────┐   │
                         │  │   CIO    │   │       Paperclip              │   │
                         │  │claude_   │◄──│       PostgreSQL             │   │
                         │  │local     │   │     (Paperclip internal)     │   │
                         │  │7200s     │   └──────────────────────────────┘   │
                         │  └────┬─────┘                                       │
                         │       │ delegates / reads reports                   │
                         │  ┌────▼──────────────────────────────────────────┐ │
                         │  │              AGENT LAYER                      │ │
                         │  │                                               │ │
                         │  │  ┌─────────────┐   ┌─────────────────────┐  │ │
                         │  │  │   Macro     │   │    Technical        │  │ │
                         │  │  │  Analyst    │   │    Analyst          │  │ │
                         │  │  │ process     │   │  claude_local       │  │ │
                         │  │  │  14400s     │   │    3600s            │  │ │
                         │  │  └──────┬──────┘   └──────┬──────────────┘  │ │
                         │  │         │                  │                 │ │
                         │  │  ┌──────▼──────┐   ┌──────▼──────────────┐  │ │
                         │  │  │   Regime    │   │   Quant Researcher  │  │ │
                         │  │  │  Analyst    │   │      process        │  │ │
                         │  │  │  process    │   │       28800s        │  │ │
                         │  │  │   1800s     │   └──────┬──────────────┘  │ │
                         │  │  └──────┬──────┘          │                 │ │
                         │  │         │         ┌────────▼──────────────┐  │ │
                         │  │  ┌──────▼──────┐  │  Sentiment Analyst   │  │ │
                         │  │  │    Risk     │  │      process         │  │ │
                         │  │  │   Manager   │  │       1800s          │  │ │
                         │  │  │  process    │  └──────────────────────┘  │ │
                         │  │  │    900s     │                             │ │
                         │  │  └─────────────┘                             │ │
                         │  └───────────────────────────────────────────────┘ │
                         └────────────────────────┬────────────────────────────┘
                                                  │ reads/writes
                         ┌────────────────────────▼────────────────────────────┐
                         │              TimescaleDB + pgvector                  │
                         │              (Docker: port 5433)                     │
                         │                                                      │
                         │  trade_journal │ decision_log │ lessons (pgvector)  │
                         │  regime_state  │ sentiment_scores │ macro_data      │
                         │  strategies   │ paper_trades │ pinescript_corpus    │
                         └────────────────────────────────────────────────────┘

External Data Feeds:                    Paper Trading Pipeline:
┌────────────────────┐                  ┌───────────────────────────────┐
│  FRED API (fedfred) │──► Macro Agent  │  TradingView Premium          │
│  Polygon.io         │──► Sentiment    │  (GC/MGC 5m live chart)       │
│  FirstRate Data     │──► Quant (hist) │  Pine Script strategy alert   │
│  Twelve Data        │──► Quant (val)  └──────────────┬────────────────┘
└────────────────────┘                                 │ POST (webhook)
                                                       ▼
                                        ┌──────────────────────────────┐
                                        │  FastAPI Webhook Receiver    │
                                        │  (src/gold_trading/webhook/) │
                                        │  - validates shared secret   │
                                        │  - idempotency check         │
                                        │  - risk rule enforcement     │
                                        │  - paper trade simulation    │
                                        └──────────────────────────────┘

Human-in-the-Loop Gate:
  Quant Researcher generates Pine Script
       ↓
  vectorbt backtest + Monte Carlo passes
       ↓
  [HUMAN: copy Pine Script → paste into TradingView IDE → set webhook alert]
       ↓
  TradingView fires strategy alerts → webhook receiver
```

---

## Components

### CIO (Chief Investment Officer)
- **Purpose:** Synthesizes all agent reports, activates/deactivates strategies, extracts lessons from trade outcomes, maintains overall market positioning stance
- **Technology:** Paperclip `claude_local` adapter, Claude Sonnet 4.6, heartbeat 7200s (2hr)
- **Inputs:** Decision log entries from all agents, recent trade journal rows, lessons store similarity query results, regime state, sentiment scores, macro data
- **Outputs:** Strategy activation flags written to `strategies` table, lessons written to `lessons` table, CIO decisions written to `decision_log`
- **Key Decisions:** `claude_local` chosen over `process` because CIO needs multi-turn reasoning across its own prior synthesis. Session context persists between heartbeats, enabling coherent review of trend patterns without re-reading the full DB each time.

### Macro Analyst
- **Purpose:** Fetches FRED economic series and classifies the macro regime for gold as bullish, neutral, or bearish
- **Technology:** Paperclip `process` adapter, Python 3.11, `fedfred` async client, heartbeat 14400s (4hr)
- **Script:** `scripts/macro_analyst.py`
- **Inputs:** FRED series — DXY (USD index), DFII10 (10yr real yield), CPIAUCSL (CPI), DCOILWTICO (oil), T10YIE (breakeven inflation), GOLDAMGBD228NLBM (gold fix)
- **Outputs:** Writes `macro_data` row with regime classification + raw series values + reasoning summary
- **Key Decisions:** FRED data lags 1-2 days; used for daily macro regime classification only, not intraday signals. `fedfred` provides async access and handles pagination. Process adapter loses session context — all state persists to DB.

### Technical Analyst
- **Purpose:** Analyzes multi-timeframe charts for GC/MGC: identifies support/resistance levels, session high/low ranges, trend structure, and key price zones
- **Technology:** Paperclip `claude_local` adapter, Claude Sonnet 4.6, heartbeat 3600s (1hr)
- **Inputs:** OHLCV data from TimescaleDB for 1m, 5m, 15m, 1H timeframes; prior S/R levels from decision log
- **Outputs:** Structured technical analysis written to `decision_log` — S/R levels, trend bias, key zones, session ranges
- **Key Decisions:** `claude_local` chosen because multi-timeframe chart reasoning benefits from persistent context — the agent can compare current structure to what it saw one hour ago without reconstructing full context from DB. No external charting API needed; agent reasons from raw OHLCV data.

### Quant Researcher
- **Purpose:** Generates Pine Script v6 strategies using RAG over the pinescriptv6 corpus, runs vectorbt backtests, validates with Monte Carlo simulation, and packages passing strategies for human deployment
- **Technology:** Paperclip `process` adapter, Python 3.11, `vectorbt`, `numpy`, `pgvector` (similarity search over pine script corpus), `anthropic` SDK, heartbeat 28800s (8hr)
- **Script:** `scripts/quant_researcher.py`
- **Inputs:** Current regime state, macro bias, technical S/R levels, lessons store (what has worked/failed in similar conditions), Pine Script v6 corpus (RAG via pgvector)
- **Outputs:** Passing strategies written to `strategies` table with Pine Script code, vectorbt backtest metrics (Sharpe, win rate, max drawdown, expectancy), Monte Carlo confidence intervals; failing strategies logged to `decision_log`
- **Key Decisions:** RAG uses hybrid BM25 + pgvector cosine similarity over the `codenamedevan/pinescriptv6` corpus (chunked, embedded with `text-embedding-3-small`). Fitness metric is Sharpe ratio + win rate, not max drawdown — vectorbt and TradingView diverge on drawdown calculation due to intrabar vs bar-boundary equity. `process_orders_on_close=true` in Pine Script and `price="close"` in vectorbt align execution models.

### Sentiment Analyst
- **Purpose:** Ingests gold-related financial news from Polygon.io, scores each headline/article for gold sentiment using Claude Sonnet 4.6, and maintains a rolling sentiment signal
- **Technology:** Paperclip `process` adapter, Python 3.11, `polygon-api-client`, `anthropic` SDK, heartbeat 1800s (30min)
- **Script:** `scripts/sentiment_analyst.py`
- **Inputs:** Polygon.io news feed filtered for gold-relevant tickers (GC, MGC, GLD, XAUUSD) and keywords (gold, FOMC, CPI, real yields, safe haven, USD)
- **Outputs:** Writes `sentiment_scores` rows: headline, source, timestamp, sentiment score (-1.0 to +1.0), gold_relevance score, catalyst tags (FOMC/CPI/geopolitical/USD/risk-off/risk-on)
- **Key Decisions:** Claude Sonnet 4.6 over FinBERT because it handles gold-specific nuance (safe-haven dynamics, USD inverse correlation, real yield sensitivity) without fine-tuning. Cost is approximately $5-8/month at 72K headlines/month, cheaper than self-hosted FinBERT on GPU.

### Risk Manager
- **Purpose:** Enforces all risk rules deterministically — position sizing, drawdown monitoring, strategy kill-switch. This is the safety layer; it must never depend on LLM judgment for hard risk decisions.
- **Technology:** Paperclip `process` adapter, Python 3.11, deterministic arithmetic only, heartbeat 900s (15min)
- **Script:** `scripts/risk_manager.py`
- **Inputs:** `paper_trades` table (open positions, P&L), `trade_journal` (historical performance), `strategies` table (active strategies)
- **Outputs:** Updates `paper_trades` with stop-loss triggers; writes kill-switch flags to `strategies.is_active`; logs decisions to `decision_log`; logs completed trade outcomes to `trade_journal`
- **Key Decisions:** Deterministic Python — no LLM calls. Risk rules: max 1 position at a time, max 0.5% account risk per trade, max 2% drawdown before full halt. Position sizing uses ATR-based stop distance: `contracts = floor((account * 0.005) / (atr_distance * contract_multiplier))`. GC multiplier = $100/oz; MGC multiplier = $10/oz.

### Regime Analyst
- **Purpose:** Classifies the current market regime (trending/ranging/volatile/quiet) using ATR, ADX, and a Hidden Markov Model trained on historical regime transitions
- **Technology:** Paperclip `process` adapter, Python 3.11, `hmmlearn` (GaussianHMM), `pandas-ta`, heartbeat 1800s (30min)
- **Script:** `scripts/regime_analyst.py`
- **Inputs:** OHLCV data (5m and 1H candles) from TimescaleDB; DXY and TIPS data from `macro_data` table as auxiliary HMM features
- **Outputs:** Writes `regime_state` row: regime label, ATR value, ADX value, HMM state probability, confidence score, timestamp
- **Key Decisions:** GaussianHMM with 4 hidden states (trending-up, trending-down, ranging, high-volatility) trained offline on 15 years of gold data. Online inference at each heartbeat. Features: ATR(14), ADX(14), log-return volatility, DXY daily change. HMM chosen over threshold-only rules because it captures non-linear state transitions and persistence probability.

### FastAPI Webhook Receiver
- **Purpose:** Receives TradingView strategy alert webhooks, validates payloads, enforces risk rules before simulating paper trades, and provides a REST API for reading trade state
- **Technology:** Python 3.11, `fastapi`, `uvicorn`, `asyncpg`, deployed as a long-running Docker service
- **Module:** `src/gold_trading/webhook/`
- **Inputs:** HTTP POST from TradingView with JSON payload containing order action, contracts, price, timestamp, strategy ID, and shared secret
- **Outputs:** Writes `paper_trades` rows; returns 200 OK (accepted) or 422 (rejected with reason); emits structured log via `loguru`
- **Key Decisions:** TradingView paper trading simulator does not fire webhooks — strategy alerts on a live chart do. Shared secret in payload for auth (no built-in TradingView webhook auth). Idempotency key on `(strategy_id, bar_time, action)` to handle TradingView's occasional duplicate delivery. All fills simulated at close price with 0.02% slippage.

---

## Data Flow

### Loop A: Strategy Development (runs every 8 hours, Quant Researcher heartbeat)

```
1. Quant Researcher wakes → queries lessons store:
   "What has worked in the current regime + macro context?"
   SELECT lesson, metadata FROM lessons
   ORDER BY embedding <=> $context_embedding LIMIT 5

2. Quant Researcher queries current state:
   - regime_state (latest row)
   - macro_data (latest row)
   - sentiment_scores (rolling 4hr average)
   - strategies (active strategies + performance)

3. Quant Researcher RAG lookup:
   - Builds query from desired strategy characteristics
   - Hybrid BM25 + vector search over pinescript_corpus
   - Retrieves top-10 relevant Pine Script v6 code chunks

4. Quant Researcher calls Claude Sonnet 4.6:
   - Context: regime, macro, sentiment, lessons, retrieved Pine Script chunks
   - Task: generate a new Pine Script v6 strategy or mutate an existing one
   - Output: structured Pine Script code block + parameter rationale

5. vectorbt backtest:
   - Translate Pine Script logic to vectorbt signal arrays
   - Run on 2 years of 5m GC data from TimescaleDB
   - Calculate Sharpe ratio, win rate, expectancy, max drawdown
   - Run 1000-iteration Monte Carlo (block bootstrap, 6-month blocks)

6. Gate check:
   - Sharpe >= 1.0
   - Win rate >= 45%
   - Monte Carlo 5th percentile Sharpe >= 0.5
   - Max drawdown < 15% (vectorbt will underestimate; TradingView will be higher)
   - If pass: write to strategies table with status='pending_deployment'
   - If fail: log to decision_log, adjust and retry up to 3 times

7. [HUMAN GATE] CIO notifies via Paperclip task:
   "Strategy {id} passed backtest. Deploy to TradingView GC 5m chart.
    Pine Script: [code attached]. Set webhook to https://[host]/webhook/signal"
```

### Loop B: Paper Trading Execution (event-driven, real-time)

```
1. Human deploys Pine Script to TradingView on GC or MGC 5m live chart
   → Creates strategy alert with webhook URL + shared secret in payload

2. TradingView fires HTTP POST to FastAPI webhook receiver:
   {
     "secret": "...",
     "strategy_id": "gs_v3_momentum",
     "action": "buy",
     "contracts": 1,
     "price": "{{close}}",
     "bar_time": "{{time}}",
     "instrument": "GC1!",
     "atr": "{{strategy.atr}}"
   }

3. Webhook receiver validates:
   a. Secret matches WEBHOOK_SECRET env var
   b. Idempotency: (strategy_id, bar_time, action) not already in paper_trades
   c. Risk rules (via direct DB query, not Risk Manager agent):
      - No existing open position (max 1 at a time)
      - strategy.is_active = true in strategies table
      - Current drawdown below 2% threshold

4. If all checks pass:
   - Simulate fill at close price + 0.02% slippage
   - Write to paper_trades: entry_price, contracts, instrument,
     strategy_id, regime_at_entry, sentiment_at_entry, macro_at_entry

5. Risk Manager wakes every 15 min:
   - Reads all open paper_trades
   - Checks account drawdown: (peak_equity - current_equity) / peak_equity
   - If drawdown > 2%: set strategies.is_active = false for all active strategies
   - Tags each open trade with current regime + sentiment snapshot
   - Logs action to decision_log

6. Trade closes (TradingView fires exit signal):
   - Webhook receiver processes exit
   - Updates paper_trades: exit_price, exit_time, pnl, r_multiple
   - Writes complete trade record to trade_journal with full context
```

### Loop C: Learning / Feedback (runs every 2 hours, CIO heartbeat)

```
1. CIO wakes → queries trade_journal for trades closed in past 2 hours:
   SELECT * FROM trade_journal
   WHERE closed_at > NOW() - INTERVAL '2 hours'
   ORDER BY closed_at DESC

2. CIO queries decision_log for agent predictions at time of each trade:
   - What regime did Regime Analyst predict?
   - What sentiment score did Sentiment Analyst produce?
   - What technical structure did Technical Analyst describe?
   - Did predictions match outcome?

3. CIO calls Claude Sonnet 4.6 for lesson extraction:
   - Input: trade details, agent predictions, actual outcome, lessons already stored
   - Task: extract 1-3 lessons as structured text blobs
   - Output: lesson text + metadata (regime tags, strategy class, conditions)

4. Lessons embedded and stored:
   embedding = text-embedding-3-small(lesson_text)
   INSERT INTO lessons (content, embedding, metadata, created_at, confidence)

5. CIO evaluates strategy performance patterns:
   - If 3+ consecutive losses on a strategy in the same regime: flag for deactivation
   - Writes strategy recommendation to decision_log
   - Sets strategies.cio_recommendation = 'deactivate' if threshold breached

6. All agents query lessons at their own heartbeat start:
   SELECT content, metadata FROM lessons
   ORDER BY embedding <=> $current_context_embedding LIMIT 5
   → Top-5 lessons injected into agent prompt as memory context
```

---

## Database Schema

All tables live in TimescaleDB (PostgreSQL 16 + TimescaleDB extension + pgvector extension).

### `trade_journal`
Hypertable partitioned by `closed_at`.
```sql
CREATE TABLE trade_journal (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id     TEXT NOT NULL,
    instrument      TEXT NOT NULL,          -- 'GC' or 'MGC'
    direction       TEXT NOT NULL,          -- 'long' or 'short'
    contracts       INTEGER NOT NULL,
    entry_price     NUMERIC(10,2) NOT NULL,
    exit_price      NUMERIC(10,2),
    entry_time      TIMESTAMPTZ NOT NULL,
    closed_at       TIMESTAMPTZ,
    pnl_usd         NUMERIC(10,2),
    r_multiple      NUMERIC(6,3),           -- profit / initial risk
    max_adverse_exc NUMERIC(10,2),          -- MAE in USD
    regime_at_entry TEXT,                   -- trending_up/trending_down/ranging/volatile
    sentiment_score NUMERIC(4,3),           -- -1.0 to +1.0
    macro_bias      TEXT,                   -- bullish/neutral/bearish
    session         TEXT,                   -- london/new_york/asia/overnight
    atr_at_entry    NUMERIC(10,4),
    notes           TEXT
);
SELECT create_hypertable('trade_journal', 'closed_at');
```

### `decision_log`
Hypertable partitioned by `created_at`.
```sql
CREATE TABLE decision_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name      TEXT NOT NULL,          -- 'cio', 'macro_analyst', etc.
    decision_type   TEXT NOT NULL,          -- 'regime_classification', 'strategy_activation', etc.
    inputs_summary  JSONB NOT NULL,         -- key inputs considered
    decision        TEXT NOT NULL,          -- what was decided
    reasoning       TEXT,                   -- agent's reasoning
    confidence      NUMERIC(3,2),           -- 0.0 to 1.0
    outcome_tag     TEXT,                   -- 'correct'/'incorrect'/'pending' — tagged post-hoc
    related_trade   UUID REFERENCES trade_journal(id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
SELECT create_hypertable('decision_log', 'created_at');
```

### `lessons`
pgvector table for similarity search.
```sql
CREATE TABLE lessons (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content         TEXT NOT NULL,          -- natural language lesson
    embedding       VECTOR(1536),           -- text-embedding-3-small output
    regime_tags     TEXT[],                 -- ['trending_up', 'high_atr']
    strategy_class  TEXT,                   -- 'breakout', 'mean_reversion', 'momentum'
    macro_context   TEXT,                   -- 'bullish'/'neutral'/'bearish'
    confidence      NUMERIC(3,2),
    source_trades   UUID[],                 -- trade_journal IDs that generated this lesson
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX lessons_embedding_idx ON lessons
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

### `regime_state`
Hypertable partitioned by `created_at`.
```sql
CREATE TABLE regime_state (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    regime          TEXT NOT NULL,          -- 'trending_up'/'trending_down'/'ranging'/'volatile'
    hmm_state       INTEGER,                -- raw HMM state index (0-3)
    hmm_confidence  NUMERIC(5,4),           -- probability of current state
    atr_14          NUMERIC(10,4),
    adx_14          NUMERIC(6,3),
    timeframe       TEXT NOT NULL DEFAULT '5m',
    dxy_change_1d   NUMERIC(6,4),           -- DXY 1-day change (from macro_data)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
SELECT create_hypertable('regime_state', 'created_at');
```

### `sentiment_scores`
Hypertable partitioned by `published_at`.
```sql
CREATE TABLE sentiment_scores (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    headline        TEXT NOT NULL,
    source          TEXT,
    url             TEXT,
    published_at    TIMESTAMPTZ NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sentiment       NUMERIC(4,3) NOT NULL,  -- -1.0 to +1.0
    gold_relevance  NUMERIC(3,2),           -- 0.0 to 1.0
    catalyst_tags   TEXT[],                 -- ['fomc', 'cpi', 'usd', 'geopolitical', 'risk_off']
    raw_response    JSONB                   -- full Claude response for audit
);
SELECT create_hypertable('sentiment_scores', 'published_at');
```

### `macro_data`
Hypertable partitioned by `observation_date`.
```sql
CREATE TABLE macro_data (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    observation_date DATE NOT NULL,
    dxy             NUMERIC(8,4),           -- US Dollar Index
    real_yield_10y  NUMERIC(6,4),           -- DFII10 (FRED)
    cpi_yoy         NUMERIC(6,4),           -- CPIAUCSL YoY %
    breakeven_10y   NUMERIC(6,4),           -- T10YIE
    oil_wti         NUMERIC(8,2),           -- DCOILWTICO
    gold_fix_pm     NUMERIC(8,2),           -- GOLDAMGBD228NLBM
    macro_regime    TEXT,                   -- 'bullish'/'neutral'/'bearish' (for gold)
    reasoning       TEXT,                   -- Claude Sonnet 4.6 classification rationale
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
SELECT create_hypertable('macro_data', 'observation_date');
```

### `strategies`
Standard table (not hypertable — strategies are not time-series).
```sql
CREATE TABLE strategies (
    id                  TEXT PRIMARY KEY,   -- human-readable e.g. 'gs_v3_momentum_atr'
    name                TEXT NOT NULL,
    version             INTEGER NOT NULL DEFAULT 1,
    pine_script         TEXT NOT NULL,      -- full Pine Script v6 code
    instrument          TEXT NOT NULL,      -- 'GC' or 'MGC'
    timeframe           TEXT NOT NULL DEFAULT '5m',
    vbt_sharpe          NUMERIC(6,3),       -- vectorbt backtest Sharpe
    vbt_win_rate        NUMERIC(5,4),       -- vectorbt win rate
    vbt_expectancy      NUMERIC(8,2),       -- USD per trade expectancy
    vbt_max_drawdown    NUMERIC(5,4),
    mc_sharpe_p5        NUMERIC(6,3),       -- Monte Carlo 5th percentile Sharpe
    mc_sharpe_p50       NUMERIC(6,3),       -- Monte Carlo median Sharpe
    status              TEXT NOT NULL DEFAULT 'pending_deployment',
                                            -- pending_deployment/active/paused/retired
    is_active           BOOLEAN NOT NULL DEFAULT false,
    cio_recommendation  TEXT,               -- 'activate'/'deactivate'/'hold'
    tradingview_alert_id TEXT,              -- TV alert ID once deployed
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deployed_at         TIMESTAMPTZ,
    retired_at          TIMESTAMPTZ
);
```

### `paper_trades`
Live and closed paper trades.
```sql
CREATE TABLE paper_trades (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id     TEXT REFERENCES strategies(id),
    instrument      TEXT NOT NULL,
    direction       TEXT NOT NULL,
    contracts       INTEGER NOT NULL,
    entry_price     NUMERIC(10,2) NOT NULL,
    exit_price      NUMERIC(10,2),
    entry_time      TIMESTAMPTZ NOT NULL,
    exit_time       TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'open',  -- 'open'/'closed'/'stopped'
    stop_loss_price NUMERIC(10,2),
    take_profit_price NUMERIC(10,2),
    pnl_usd         NUMERIC(10,2),
    r_multiple      NUMERIC(6,3),
    idempotency_key TEXT UNIQUE NOT NULL,   -- hash of (strategy_id, bar_time, action)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### `pinescript_corpus`
RAG corpus for Pine Script v6 knowledge retrieval.
```sql
CREATE TABLE pinescript_corpus (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_file     TEXT NOT NULL,          -- relative path in codenamedevan/pinescriptv6
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    embedding       VECTOR(1536),
    token_count     INTEGER,
    chunk_type      TEXT,                   -- 'concept'/'reference'/'example'
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_file, chunk_index)
);
CREATE INDEX pinescript_corpus_embedding_idx ON pinescript_corpus
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

---

## External Dependencies

| Dependency | Auth Method | Used By | Failure Mode |
|---|---|---|---|
| Anthropic API (`claude-sonnet-4-6`) | `ANTHROPIC_API_KEY` header | CIO, Technical Analyst, Sentiment Analyst, Quant Researcher | Agents halt heartbeat; log error to decision_log; retry on next heartbeat |
| Polygon.io | `POLYGON_API_KEY` query param | Sentiment Analyst | Fall back to RSS feeds (Reuters, Bloomberg); log gap in sentiment_scores |
| FRED API | `FRED_API_KEY` query param | Macro Analyst | Use previous macro_data row; log stale data warning; FRED rarely goes down |
| Anthropic Embeddings API (`text-embedding-3-small`) | `ANTHROPIC_API_KEY` (same key) | Quant Researcher (corpus ingestion), CIO (lesson embedding) | Retry with exponential backoff; block lesson write if unavailable |
| TimescaleDB | `DATABASE_URL` connection string | All agents | Hard failure — all agents depend on DB; Docker health check + restart policy |
| TradingView Premium | Human-managed credentials | Human operator only | N/A — human-in-the-loop step |
| FirstRate Data (historical 5m GC data) | One-time download, stored locally | Quant Researcher | Pre-loaded into TimescaleDB at setup; no runtime dependency |
| Paperclip server | Internal (localhost:3000) | All agents via adapter | All agents halt; Paperclip Docker restart policy handles recovery |

**TradingView webhook shared secret:** Not a standard API key — it is a plain string embedded in the Pine Script alert payload and validated by the FastAPI receiver. Stored as `WEBHOOK_SECRET` env var.

---

## Environment Variables

```bash
# Anthropic
ANTHROPIC_API_KEY=sk-ant-...          # All LLM calls + embeddings

# Database
DATABASE_URL=postgresql://gold:gold@localhost:5433/gold_trading
PAPERCLIP_DB_URL=postgresql://paperclip:paperclip@localhost:5432/paperclip

# External APIs
POLYGON_API_KEY=...                   # Sentiment Analyst news feed
FRED_API_KEY=...                      # Macro Analyst FRED series

# TradingView
WEBHOOK_SECRET=...                    # Shared secret validated in every TV alert payload

# Trading Parameters
ACCOUNT_SIZE_USD=50000                # Simulated paper trading account size
MAX_RISK_PER_TRADE=0.005             # 0.5% of account per trade
MAX_DRAWDOWN_LIMIT=0.02              # 2% drawdown triggers full halt
MAX_POSITIONS=1                       # Max concurrent open positions
GC_CONTRACT_MULTIPLIER=100           # $100/oz for full GC
MGC_CONTRACT_MULTIPLIER=10           # $10/oz for micro MGC

# FastAPI Webhook Receiver
WEBHOOK_HOST=0.0.0.0
WEBHOOK_PORT=8080
LOG_LEVEL=INFO

# Paperclip
PAPERCLIP_URL=http://localhost:3000
PAPERCLIP_API_KEY=...                 # If Paperclip auth is enabled

# Embeddings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
```

---

## Directory Structure

```
tradingview/
├── src/
│   └── gold_trading/
│       ├── __init__.py
│       ├── main.py                   # FastAPI app factory
│       ├── webhook/
│       │   ├── __init__.py
│       │   ├── receiver.py           # FastAPI router: POST /webhook/signal
│       │   ├── validator.py          # Secret check, idempotency, risk rules
│       │   └── simulator.py         # Paper trade fill simulation
│       ├── db/
│       │   ├── __init__.py
│       │   ├── client.py             # asyncpg pool factory
│       │   ├── migrations/           # SQL migration files (numbered)
│       │   └── queries/
│       │       ├── trades.py         # trade_journal + paper_trades queries
│       │       ├── decisions.py      # decision_log queries
│       │       ├── lessons.py        # lessons similarity search queries
│       │       ├── regime.py         # regime_state queries
│       │       ├── sentiment.py      # sentiment_scores queries
│       │       ├── macro.py          # macro_data queries
│       │       └── strategies.py     # strategies CRUD
│       ├── risk/
│       │   ├── __init__.py
│       │   ├── calculator.py         # Position sizing (ATR-based), drawdown math
│       │   └── rules.py              # Rule enforcement (max drawdown, max positions)
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── client.py             # Anthropic embeddings API wrapper
│       │   └── corpus.py            # Pine Script corpus ingestion + chunking
│       └── models/
│           ├── __init__.py
│           ├── trade.py              # Pydantic models: Trade, PaperTrade, TradeOutcome
│           ├── signal.py             # Pydantic models: WebhookPayload, SignalValidation
│           ├── regime.py             # Pydantic models: RegimeState, RegimeClassification
│           ├── sentiment.py          # Pydantic models: SentimentScore, NewsItem
│           ├── macro.py              # Pydantic models: MacroData, MacroRegime
│           ├── strategy.py           # Pydantic models: Strategy, BacktestResult, MonteCarloResult
│           └── lesson.py             # Pydantic models: Lesson, LessonQuery
├── scripts/                          # Paperclip process adapter agent scripts
│   ├── macro_analyst.py              # FRED data fetch + macro regime classification
│   ├── quant_researcher.py           # Pine Script RAG + vectorbt backtest + Monte Carlo
│   ├── sentiment_analyst.py          # Polygon.io news + Claude Sonnet 4.6 scoring
│   ├── risk_manager.py               # Drawdown check + kill-switch + trade tagging
│   └── regime_analyst.py             # ATR/ADX/HMM regime classification
├── pine/
│   ├── README.md                     # Deployment instructions for human operator
│   └── generated/                    # Pine Script files output by Quant Researcher
│       └── *.pine                    # Named by strategy ID
├── .paperclip/
│   ├── company.json                  # Paperclip company config
│   └── skills/
│       ├── gold-trading-context.md   # Shared skill: instruments, risk rules, DB schema
│       ├── pine-script-v6.md         # Shared skill: Pine Script v6 RAG instructions
│       └── memory-protocol.md        # Shared skill: how agents query/write lessons
├── tests/
│   ├── conftest.py                   # pytest fixtures: DB connection, mock data
│   ├── test_webhook_receiver.py      # FastAPI webhook endpoint tests
│   ├── test_validator.py             # Signal validation + risk rule tests
│   ├── test_risk_calculator.py       # Position sizing math tests
│   ├── test_regime_analyst.py        # HMM regime classification tests
│   ├── test_sentiment_analyst.py     # Sentiment scoring tests (mocked Claude)
│   ├── test_macro_analyst.py         # FRED data processing tests (mocked fedfred)
│   └── test_lessons_store.py         # pgvector similarity search tests
├── docs/
│   ├── research.md                   # Research findings (source of truth)
│   ├── architecture.md               # This file
│   └── build-plan.md                 # Phased implementation plan
├── docker-compose.yml                # TimescaleDB + pgvector service
├── Dockerfile                        # FastAPI webhook receiver image
├── pyproject.toml                    # Dependencies, ruff config, pytest config
├── .env.example                      # All env vars with descriptions (never commit .env)
├── CLAUDE.md                         # Project-specific instructions for Claude Code
└── README.md                         # Setup instructions for human operator
```

---

## Scaling Considerations

**What breaks first:** The Sentiment Analyst processes up to 72K headlines/month via Claude Sonnet 4.6 API calls. At 30-minute heartbeats this is approximately 2,400 API calls/day. If gold becomes a high-news event (geopolitical crisis, FOMC surprise), Polygon.io can spike to thousands of articles per hour and the sentiment pipeline will queue behind API rate limits.

**Mitigation:** Batch headlines into groups of 10-20 per API call using Claude's context window. Pre-filter by gold relevance score before scoring (use keyword filter first, then LLM). This reduces API calls by approximately 60-70%.

**Second constraint:** The vectorbt backtest in the Quant Researcher processes 2 years of 5m bars (~105,000 bars) per strategy. At 8-hour heartbeats this is low-frequency enough that vectorbt's in-memory performance (sub-second for simple signal arrays) is not a bottleneck. Monte Carlo with 1,000 iterations adds approximately 2-5 seconds. No scaling concern at current volume.

**Third constraint:** pgvector HNSW index performance degrades gracefully up to approximately 1M vectors. The Pine Script corpus has approximately 5,000 vectors; the lessons store will accumulate perhaps 10,000 lessons per year. Both are well within HNSW's sweet spot. No action needed until year 3+.

**TimescaleDB chunk management:** trade_journal and decision_log hypertables default to 7-day chunks. At estimated volume (<100 trades/day), annual storage is under 1GB including all tables. No compression policy needed for 12 months; configure `add_compression_policy('trade_journal', INTERVAL '30 days')` at launch as a precaution.

**Paperclip heartbeat deadlock:** Paperclip has a known bug (#2516) where `maxConcurrentRuns=1` agents can deadlock if a previous run is still executing when the next heartbeat fires. All process adapter scripts must complete within their heartbeat interval. Risk Manager (900s interval) must complete in under 5 minutes — designed to be so since it runs only DB queries and arithmetic. If a script overruns, add `timeoutSec` in the agent runtimeConfig to force-terminate the run.
