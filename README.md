# AutoGoldFutures

Autonomous gold futures (GC/MGC) trading system powered by 12 AI agents orchestrated through [Paperclip](https://github.com/paperclipai/paperclip). Agents research market conditions across multiple timeframes, develop and validate Pine Script strategies, and execute paper trades on TradingView.

**Status:** Backtest + paper trading. No live broker execution.

## How It Works

```
  Risk Manager (Independent — can veto CIO)
  │ 15-min drawdown checks, kill-switch authority
  │
  CIO (claude_local, 30min)
  ├── Technical Analyst — Strategy Team Lead (claude_local, 1hr)
  │   ├── Quant Researcher ──── Pine Script generation + backtest (30min)
  │   ├── Walk-Forward Validator ── OOS overfitting detection (30min)
  │   └── Strategy Monitor ──── Live performance tracking (30min)
  │
  ├── Regime Analyst — Intelligence Lead (process, 30min)
  │   ├── Macro Analyst ──── FRED economic data (1hr)
  │   └── Sentiment Analyst ── Gold news scoring (30min)
  │
  └── Data Pipeline — Operations Lead (process, 1hr)
      ├── Economic Calendar ── FOMC/CPI/NFP alerts (1hr)
      └── Skill Optimizer ──── Meta-learning, agent improvement (4hr)

         ┌──────────────────────────────┐
         │  TimescaleDB + pgvector      │
         │  135K bars across 4 TFs      │
         │  (5m, 15m, 1H, Daily)        │
         │  Trade journal, lessons,     │
         │  regime, sentiment, etc      │
         └──────────────┬───────────────┘
                        │
  TradingView (GC/MGC 5m chart)
    │ strategy alert fires webhook
    ▼
  FastAPI Webhook Receiver
    → validates secret + idempotency
    → enforces risk rules
    → simulates paper trade fill
    → logs to trade journal
```

## Quick Start

### Prerequisites

- Python 3.12+
- Docker (for TimescaleDB)
- Node.js 20+ (for Paperclip)
- [uv](https://github.com/astral-sh/uv) package manager

### 1. Clone and install

```bash
git clone https://github.com/sjonas50/autogold.git
cd autogold
uv sync
```

### 2. Start infrastructure

```bash
# TimescaleDB + pgvector
docker compose up -d

# Run database migrations
uv run python -m gold_trading.db.client --migrate
```

### 3. Configure environment

```bash
cp .env.example .env
# Required:
#   ANTHROPIC_API_KEY   — Claude Sonnet 4.6 for agent reasoning
#   FRED_API_KEY        — Federal Reserve economic data (free)
#   OPENAI_API_KEY      — Embeddings (text-embedding-3-small)
#   WEBHOOK_SECRET      — Shared secret for TradingView alerts
#
# Optional (more data = better backtests):
#   TWELVEDATA_API_KEY  — 1+ year of 5m gold data (free at twelvedata.com)
#   POLYGON_API_KEY     — Real-time gold news ($29/mo)
#   OANDA_API_KEY       — 6+ months XAUUSD data (free practice account)
```

### 4. Install and configure Paperclip

Follow the [Paperclip installation guide](https://docs.paperclip.ing). The AutoGoldFutures company with 12 agents is configured via the API. Verify at `http://localhost:3100`.

### 5. Ingest data

```bash
# Pull 1 year of multi-timeframe data (5m, 15m, 1H, daily) from Twelve Data
uv run python scripts/ingest_ohlcv.py --source twelvedata --days 365 --multi-tf

# Or pull free 60-day data from yfinance (no API key needed)
uv run python scripts/ingest_ohlcv.py --source yfinance

# Train HMM regime model on historical data
uv run python scripts/train_hmm.py

# Ingest Pine Script v6 RAG corpus
uv run python -m gold_trading.embeddings.corpus
```

### 6. Start services

```bash
# Webhook receiver
uv run uvicorn gold_trading.main:app --host 0.0.0.0 --port 8080

# Heartbeat scheduler (triggers process agents on their intervals)
nohup bash scripts/heartbeat_cron.sh &

# Monitor at http://localhost:3100
```

## Org Structure

The system uses a tiered hierarchy — CIO delegates to 3 team leads, not 11 individual agents:

| Agent | Role | Reports To | Heartbeat |
|-------|------|-----------|-----------|
| **Risk Manager** | Independent Safety Officer | Board (no one) | 15 min |
| **CIO** | Chief Investment Officer | Board | 30 min |
| **Technical Analyst** | Strategy Team Lead | CIO | 1 hr |
| ↳ Quant Researcher | Strategy Development | Technical Analyst | 30 min |
| ↳ Walk-Forward Validator | OOS Validation | Technical Analyst | 30 min |
| ↳ Strategy Monitor | Performance Tracking | Technical Analyst | 30 min |
| **Regime Analyst** | Intelligence Lead | CIO | 30 min |
| ↳ Macro Analyst | FRED Economic Data | Regime Analyst | 1 hr |
| ↳ Sentiment Analyst | Gold News Scoring | Regime Analyst | 30 min |
| **Data Pipeline** | Operations Lead | CIO | 1 hr |
| ↳ Economic Calendar | Event Monitoring | Data Pipeline | 1 hr |
| ↳ Skill Optimizer | Meta-Learning | Data Pipeline | 4 hr |

**Key design decisions:**
- Risk Manager is independent — can kill all strategies regardless of CIO's opinion
- CIO has 3 direct reports (was 11) — manageable context window
- Team leads delegate tasks via Paperclip issues to their reports
- Process agents use a shared `paperclip.py` helper for task routing

## Instruments

| Symbol | Contract | Multiplier | Tick Value |
|--------|----------|-----------|-----------|
| **GC** | 100 troy oz | $100/point | $10/tick |
| **MGC** | 10 troy oz | $10/point | $1/tick |

Execution timeframe: **5-minute candles**. Analysis across 5m, 15m, 1H, Daily.

## Multi-Timeframe Data

| Timeframe | Bars | Coverage | Use |
|-----------|------|----------|-----|
| **5m** | 90,000 | 1 year | Execution, ATR, VWAP |
| **15m** | 30,000 | 1 year | Session ranges, patterns |
| **1H** | 10,000 | 1.5 years | Trend structure, major S/R |
| **Daily** | 5,000 | 19 years | Long-term context |

The Regime Analyst classifies market conditions using all 4 timeframes with a trained HMM model (4 states: ranging 39%, trending_up 39%, trending_down 10%, volatile 12%).

## Risk Rules

Hard-coded and enforced by the independent Risk Manager (deterministic Python, no LLM):

| Rule | Limit | Effect |
|------|-------|--------|
| Max risk per trade | 0.5% of account | Position sizing capped |
| Max drawdown | 2% | **All strategies halted** |
| Max positions | 1 | No new entries |
| Daily loss limit | 1% | No more trades that session |
| Max trade duration | 120 minutes | Flagged for auto-exit |

## Strategy Development Pipeline

```
CIO identifies regime + market conditions
  → Delegates to Technical Analyst (Strategy Team Lead)
    → Quant Researcher:
        1. Checks for manager directives (Paperclip tasks)
        2. Queries lessons store for past learnings
        3. Finds best retired strategy to mutate (or generates new)
        4. RAG search over Pine Script v6 corpus (517 chunks)
        5. Claude Sonnet 4.6 generates Pine Script
        6. vectorbt backtest on 90K bars of 5m GC data
        7. Monte Carlo simulation (1000 iterations, block bootstrap)
        8. Backtest gate: Sharpe ≥ 1.5, WR ≥ 50%, DD < 5%, PF ≥ 1.3
        9. Monte Carlo gate: p5 Sharpe ≥ 0.5, p95 DD < 5%, ruin < 5%
       10. BOTH gates must pass for pending_deployment
    → Walk-Forward Validator:
        11. 3-window train/test split (70/30)
        12. Rejects if OOS Sharpe degrades >50%
    → Strategy Monitor:
        13. Tracks live performance vs backtest
        14. Alerts on win rate or expectancy degradation
  → CIO reviews and activates/deactivates
```

## Agent Memory System

Three-layer learning system in PostgreSQL + pgvector:

| Layer | Purpose | Writers | Readers |
|-------|---------|---------|---------|
| **Trade Journal** | Every trade with full context | Webhook receiver | All agents |
| **Decision Log** | Agent decisions with reasoning | All agents | Team leads + CIO |
| **Lessons Store** | Extracted learnings (pgvector) | CIO | All agents |

The CIO reviews outcomes, extracts lessons, and stores them as vector embeddings. All agents query the lessons store at heartbeat start to inform their decisions. The Skill Optimizer analyzes agent accuracy and proposes improvements.

## Self-Improvement

The system improves itself through multiple feedback loops:

1. **Strategy Mutation** — Quant Researcher mutates the best retired strategy's parameters instead of starting from scratch
2. **Walk-Forward Validation** — rejects overfit strategies before deployment
3. **CIO Active Delegation** — creates targeted Paperclip tasks based on failure patterns
4. **Skill Optimizer** — analyzes decision accuracy, proposes agent skill updates
5. **Multi-TF Confidence** — regime confidence boosted when 5m/15m/1H/Daily align, reduced on conflict
6. **Economic Calendar** — pre-event strategy deactivation prevents news-driven losses

## Development

```bash
# Run tests (130 tests, all against real TimescaleDB)
uv run python -m pytest tests/ -v

# Lint
ruff check src/ tests/ scripts/
ruff format src/ tests/ scripts/

# Run a single agent manually
uv run python scripts/risk_manager.py
uv run python scripts/regime_analyst.py

# Trigger agent via Paperclip API
curl -X POST http://localhost:3100/api/agents/{agent_id}/heartbeat/invoke

# Ingest more data
uv run python scripts/ingest_ohlcv.py --source twelvedata --days 365 --multi-tf

# Retrain HMM after ingesting new data
uv run python scripts/train_hmm.py

# Monitor heartbeat scheduler
tail -f logs/heartbeat_cron.log
```

## Project Structure

```
src/gold_trading/
├── main.py                  # FastAPI app with webhook router
├── paperclip.py             # Shared Paperclip API helpers (task routing)
├── webhook/                 # Signal receiver, validator, paper trade simulator
├── risk/                    # Position sizing, drawdown, risk rules
├── backtest/                # vectorbt engine, Monte Carlo simulation
├── embeddings/              # OpenAI embeddings client, Pine Script corpus
├── db/                      # asyncpg pool, migrations, query modules
│   └── queries/ohlcv.py     # Multi-timeframe OHLCV data access
└── models/                  # Pydantic schemas for all data types

scripts/
├── macro_analyst.py         # FRED data → macro regime
├── sentiment_analyst.py     # RSS/Polygon.io → Claude scoring
├── regime_analyst.py        # HMM + multi-TF → regime state (Intelligence Lead)
├── risk_manager.py          # Drawdown enforcement — independent, no LLM
├── quant_researcher.py      # RAG + mutation + backtest + Monte Carlo
├── walk_forward_validator.py # OOS validation, overfitting detection
├── strategy_monitor.py      # Live performance vs backtest expectations
├── data_pipeline.py         # OHLCV refresh, data quality (Ops Lead)
├── economic_calendar.py     # FOMC/CPI/NFP monitoring
├── skill_optimizer.py       # Meta-learning, agent improvement proposals
├── train_hmm.py             # HMM regime model training
├── ingest_ohlcv.py          # Multi-source, multi-TF data ingestion
├── heartbeat_cron.sh        # External scheduler for process agents
└── run_*.sh                 # Shell wrappers for Paperclip

pine/generated/              # Output Pine Script strategies
tests/                       # 130 tests against real TimescaleDB
docs/
├── research.md              # Technology evaluation + decisions
├── architecture.md          # System design + database schema
└── build-plan.md            # Phased implementation plan
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Orchestration | [Paperclip](https://github.com/paperclipai/paperclip) | Agent management, heartbeats, task delegation, governance |
| LLM | Claude Sonnet 4.6 | Agent reasoning, sentiment scoring, Pine Script generation |
| Database | TimescaleDB + pgvector | Multi-TF time-series, vector similarity search |
| Backtesting | vectorbt | Strategy optimization, signal generation |
| Monte Carlo | numpy (block bootstrap) | Strategy validation, ruin probability |
| Regime Detection | hmmlearn (GaussianHMM) | 4-state market regime classification |
| News | RSS feeds + Polygon.io | Gold-related financial news (RSS is free) |
| Macro Data | FRED API (via fedfred) | Economic indicators (DXY, real yields, CPI) |
| Historical Data | Twelve Data + yfinance | Multi-TF OHLCV (free) |
| Embeddings | text-embedding-3-small | Pine Script RAG, lessons store similarity search |
| Webhook | FastAPI + uvicorn | TradingView alert receiver |
| Charting | TradingView Premium | Pine Script strategies, paper trading |

## Estimated Monthly Cost

| Item | Cost |
|------|------|
| Claude Sonnet 4.6 (12 agents) | ~$150-200 |
| TradingView Premium | $42-60 |
| Polygon.io Starter (optional) | $0-29 |
| VPS hosting (4 vCPU / 8GB) | $24-48 |
| OpenAI embeddings | ~$1-5 |
| FRED API + Twelve Data + yfinance | Free |
| **Total** | **~$215-340/mo** |

## License

Private repository.
