# AutoGoldFutures

Autonomous gold futures (GC/MGC) trading system powered by 7 AI agents orchestrated through [Paperclip](https://github.com/paperclipai/paperclip). Agents research market conditions, develop Pine Script strategies, validate via Monte Carlo simulation, and execute paper trades on TradingView.

**Status:** Backtest + paper trading. No live broker execution.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    PAPERCLIP SERVER                          │
│                                                             │
│  CIO ──────────── Synthesizes all inputs, activates         │
│  │                 strategies every 2 hours                  │
│  ├── Technical Analyst ── Multi-TF chart analysis (1hr)     │
│  ├── Macro Analyst ────── FRED economic data (4hr)          │
│  ├── Sentiment Analyst ── Gold news scoring (30min)         │
│  ├── Regime Analyst ───── ATR/ADX/HMM classifier (30min)   │
│  ├── Risk Manager ─────── Drawdown enforcement (15min)      │
│  └── Quant Researcher ─── Pine Script generation (8hr)      │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │ reads/writes
         ┌─────────────▼─────────────┐
         │  TimescaleDB + pgvector   │
         │  (trade journal, lessons, │
         │   regime, sentiment, etc) │
         └───────────────────────────┘

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

# Run database migrations (creates all tables)
uv run python -m gold_trading.db.client --migrate
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys:
#   ANTHROPIC_API_KEY  — Claude Sonnet 4.6 for agent reasoning
#   POLYGON_API_KEY    — Real-time gold news
#   FRED_API_KEY       — Federal Reserve economic data
#   OPENAI_API_KEY     — Embeddings (text-embedding-3-small)
#   WEBHOOK_SECRET     — Shared secret for TradingView alerts
```

### 4. Install and configure Paperclip

Follow the [Paperclip installation guide](https://docs.paperclip.ing). The AutoGoldFutures company with 7 agents should already be configured if you ran the setup scripts. Verify at `http://localhost:3100`.

### 5. Ingest data

```bash
# Historical 5m GC OHLCV data (required for Regime Analyst + Quant Researcher)
# Purchase from FirstRate Data (~$30) or pull from your broker, then load into ohlcv_5m table

# Pine Script v6 RAG corpus (required for Quant Researcher)
uv run python -m gold_trading.embeddings.corpus
```

### 6. Start services

```bash
# Webhook receiver (receives TradingView strategy alerts)
uv run uvicorn gold_trading.main:app --host 0.0.0.0 --port 8080

# Enable agent heartbeats in the Paperclip UI at http://localhost:3100
# Recommended order: Risk Manager → Regime Analyst → Sentiment → Macro → Quant → CIO
```

## Instruments

| Symbol | Contract | Multiplier | Tick Value | Use Case |
|--------|----------|-----------|-----------|----------|
| **GC** | 100 troy oz | $100/point | $10/tick | Full-size futures |
| **MGC** | 10 troy oz | $10/point | $1/tick | Micro futures (preferred for smaller accounts) |

Execution timeframe: **5-minute candles**. Analysis across 1m, 5m, 15m, 1H.

## Risk Rules

These are hard-coded and enforced by the Risk Manager (deterministic Python, no LLM):

| Rule | Limit | Effect |
|------|-------|--------|
| Max risk per trade | 0.5% of account | Position sizing capped |
| Max drawdown | 2% | **All strategies halted** |
| Max positions | 1 | No new entries until current closes |
| Daily loss limit | 1% | No more trades that session |
| Max trade duration | 120 minutes | Flagged for auto-exit |

## Agent Roles

| Agent | Type | Heartbeat | What It Does |
|-------|------|-----------|-------------|
| **CIO** | claude_local | 2 hours | Reads all reports, activates/deactivates strategies, extracts lessons |
| **Technical Analyst** | claude_local | 1 hour | Multi-timeframe S/R levels, VWAP, trend structure |
| **Macro Analyst** | process (Python) | 4 hours | FRED data (DXY, real yields, CPI), macro regime classification |
| **Sentiment Analyst** | process (Python) | 30 min | Polygon.io news, Claude Sonnet 4.6 sentiment scoring |
| **Regime Analyst** | process (Python) | 30 min | ATR/ADX/HMM market regime (trending/ranging/volatile) |
| **Risk Manager** | process (Python) | 15 min | Drawdown tracking, kill-switch, position sizing |
| **Quant Researcher** | process (Python) | 8 hours | Pine Script generation (RAG), vectorbt backtesting, Monte Carlo |

## Strategy Development Pipeline

```
Quant Researcher:
  1. Query lessons store for past learnings
  2. RAG search over Pine Script v6 corpus
  3. Claude Sonnet 4.6 generates Pine Script strategy
  4. vectorbt backtest on 5m GC historical data
  5. Monte Carlo simulation (1000 iterations, block bootstrap)
  6. Gate check: Sharpe ≥ 1.0, Win Rate ≥ 45%, MC 5th %ile Sharpe ≥ 0.5
  7. If PASS → writes Pine Script to pine/generated/
  8. [HUMAN] copies Pine Script to TradingView, sets webhook alert
```

## Agent Memory System

Agents share a three-layer learning system in PostgreSQL:

| Layer | Purpose | Writers | Readers |
|-------|---------|---------|---------|
| **Trade Journal** | Every trade with full context | Webhook receiver | All agents |
| **Decision Log** | Agent decisions with reasoning | All agents | CIO |
| **Lessons Store** | Extracted learnings (pgvector) | CIO | All agents |

The CIO reviews trade outcomes every 2 hours, extracts lessons from unexpected results, and stores them as vector embeddings. All agents query the lessons store at the start of each heartbeat to inform their decisions.

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
```

## Project Structure

```
src/gold_trading/
├── main.py                  # FastAPI app with webhook router
├── webhook/                 # Signal receiver, validator, paper trade simulator
├── risk/                    # Position sizing, drawdown, risk rules
├── backtest/                # vectorbt engine, Monte Carlo simulation
├── embeddings/              # OpenAI embeddings client, Pine Script corpus
├── db/                      # asyncpg pool, migrations, query modules
└── models/                  # Pydantic schemas for all data types

scripts/                     # Paperclip process adapter agent scripts
├── macro_analyst.py         # FRED data → macro regime
├── sentiment_analyst.py     # Polygon.io → Claude scoring
├── regime_analyst.py        # ATR/ADX/HMM → regime state
├── risk_manager.py          # Drawdown enforcement (no LLM)
├── quant_researcher.py      # RAG + backtest + Monte Carlo
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
| Orchestration | [Paperclip](https://github.com/paperclipai/paperclip) | Agent management, heartbeats, governance |
| LLM | Claude Sonnet 4.6 | Agent reasoning, sentiment scoring, Pine Script generation |
| Database | TimescaleDB + pgvector | Time-series data, vector similarity search |
| Backtesting | vectorbt | Strategy optimization, signal generation |
| Monte Carlo | numpy (block bootstrap) | Strategy validation, ruin probability |
| News | Polygon.io | Real-time gold-related financial news |
| Macro Data | FRED API (via fedfred) | Economic indicators (DXY, real yields, CPI) |
| Embeddings | text-embedding-3-small | Pine Script RAG, lessons store similarity search |
| Webhook | FastAPI + uvicorn | TradingView alert receiver |
| Charting | TradingView Premium | Pine Script strategies, paper trading |

## Estimated Monthly Cost

| Item | Cost |
|------|------|
| Claude Sonnet 4.6 (7 agents) | ~$120-150 |
| TradingView Premium | $42-60 |
| Polygon.io Starter | $29 |
| VPS hosting (4 vCPU / 8GB) | $24-48 |
| OpenAI embeddings | ~$1-5 |
| FRED API | Free |
| **Total** | **~$215-290/mo** |

## License

Private repository.
