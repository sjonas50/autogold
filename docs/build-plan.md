# Build Plan: Autonomous Gold Futures Trading System

**Date:** 2026-04-03 | **Phases:** 6 | **Total Tasks:** 28

Prerequisites: `docs/research.md` and `docs/architecture.md` completed.

---

## Phase 0: Scaffold + Paperclip (M)

Project structure, dependencies, Docker, linting, **and Paperclip installation with company shell**.

| # | Task | Files | Complexity |
|---|---|---|---|
| 0.1 | Create `pyproject.toml` with all deps (fastapi, uvicorn, asyncpg, anthropic, vectorbt, hmmlearn, fedfred, polygon-api-client, quantstats, pandas-ta, loguru, pydantic, httpx, numpy, sentence-transformers) | `pyproject.toml` | S |
| 0.2 | Create `docker-compose.yml` — TimescaleDB + pgvector on port 5433 | `docker-compose.yml` | S |
| 0.3 | Create project directory structure per architecture.md | `src/gold_trading/**/__init__.py` | S |
| 0.4 | Create `.env.example` with all env vars from architecture | `.env.example` | S |
| 0.5 | Create `Dockerfile` for FastAPI webhook receiver | `Dockerfile` | S |
| 0.6 | Install Paperclip (`npx paperclipai onboard --yes`), verify server starts on port 3100 | N/A (system install) | S |
| 0.7 | Create Paperclip company shell — company name, mission, org chart with 7 agent slots (CIO as CEO, 6 reports). Configure adapter types and heartbeat intervals per agent. No skills/prompts yet — just the structure. | `.paperclip/company.json` | M |
| 0.8 | Create a "hello world" test agent using the `process` adapter — a minimal Python script that reads `PAPERCLIP_AGENT_ID` and `PAPERCLIP_API_URL` env vars, calls the Paperclip API to fetch its identity, and logs success. Verify one heartbeat cycle completes. | `scripts/test_agent.py` | S |

**Test Gate:**
```bash
cd /Users/sjonas/tradingview && \
  uv sync && \
  ruff check src/ tests/ scripts/ && \
  docker compose up -d && \
  sleep 3 && \
  docker compose exec timescaledb pg_isready -U gold && \
  curl -s http://localhost:3100/api/health | grep -q ok && \
  echo "Paperclip + TimescaleDB running"
```

---

## Phase 1: Database + Pydantic Models (M)

Schemas, migrations, async DB client, all Pydantic models.

| # | Task | Files | Complexity |
|---|---|---|---|
| 1.1 | Create asyncpg connection pool factory | `src/gold_trading/db/client.py` | S |
| 1.2 | Create SQL migrations — all 9 tables from architecture (trade_journal, decision_log, lessons, regime_state, sentiment_scores, macro_data, strategies, paper_trades, pinescript_corpus) + TimescaleDB hypertables + pgvector indexes | `src/gold_trading/db/migrations/001_initial.sql` | M |
| 1.3 | Create all Pydantic models — Trade, PaperTrade, Signal, RegimeState, SentimentScore, MacroData, Strategy, BacktestResult, MonteCarloResult, Lesson | `src/gold_trading/models/*.py` | M |
| 1.4 | Create query modules — typed async functions for each table (CRUD + specialized queries like lessons similarity search) | `src/gold_trading/db/queries/*.py` | M |
| 1.5 | Write DB tests — migration applies cleanly, CRUD works, pgvector similarity search returns correct ordering | `tests/conftest.py`, `tests/test_db.py` | M |

**Test Gate:**
```bash
cd /Users/sjonas/tradingview && \
  docker compose up -d && \
  uv run python -m gold_trading.db.client --migrate && \
  uv run pytest tests/test_db.py -v
```

---

## Phase 2: Risk Engine + Webhook Receiver (M)

FastAPI service, signal validation, paper trade simulation, risk rules.
**Paperclip wiring:** None yet — webhook receiver runs independently. Risk Manager agent script wired in Phase 5.

| # | Task | Files | Complexity |
|---|---|---|---|
| 2.1 | Create risk calculator — ATR-based position sizing for GC ($100/oz) and MGC ($10/oz), drawdown math, daily loss tracking | `src/gold_trading/risk/calculator.py` | M |
| 2.2 | Create risk rules enforcement — max 1 position, 0.5% per trade, 2% drawdown halt, strategy kill-switch | `src/gold_trading/risk/rules.py` | S |
| 2.3 | Create webhook receiver — FastAPI POST `/webhook/signal`, shared secret validation, idempotency check, structured logging via loguru | `src/gold_trading/webhook/receiver.py`, `src/gold_trading/main.py` | M |
| 2.4 | Create signal validator — orchestrates secret check → idempotency → risk rules → accept/reject | `src/gold_trading/webhook/validator.py` | S |
| 2.5 | Create paper trade simulator — fill simulation at close + 0.02% slippage, open/close position tracking, P&L calculation, trade journal write on close | `src/gold_trading/webhook/simulator.py` | M |
| 2.6 | Write tests — position sizing math (parametrized for GC/MGC), drawdown calculation, webhook endpoint (accept/reject scenarios), idempotency | `tests/test_risk_calculator.py`, `tests/test_validator.py`, `tests/test_webhook_receiver.py` | M |

**Test Gate:**
```bash
cd /Users/sjonas/tradingview && \
  uv run pytest tests/test_risk_calculator.py tests/test_validator.py tests/test_webhook_receiver.py -v
```

---

## Phase 3: Data Pipelines + Sentiment (M)

FRED macro data, Polygon.io news, Claude Sonnet 4.6 sentiment scoring, regime classification.
**Paperclip wiring:** After each script passes tests, wire it into the corresponding Paperclip agent slot and verify one heartbeat cycle completes via `POST /api/agents/:id/heartbeat/invoke`.

| # | Task | Files | Complexity |
|---|---|---|---|
| 3.1 | Create macro analyst script — fetches FRED series (DXY, DFII10, T10YIE, CPI, gold fix), classifies macro regime via Claude Sonnet 4.6, writes to `macro_data` | `scripts/macro_analyst.py` | M |
| 3.2 | Create sentiment analyst script — polls Polygon.io news, filters for gold relevance (keywords + ticker), batches 10-20 headlines per Claude Sonnet 4.6 call, writes scored results to `sentiment_scores` | `scripts/sentiment_analyst.py` | M |
| 3.3 | Create regime analyst script — loads 5m OHLCV from DB, calculates ATR(14)/ADX(14) via pandas-ta, runs GaussianHMM inference (4 states), writes to `regime_state` | `scripts/regime_analyst.py` | L |
| 3.4 | Create embeddings client — wrapper for text-embedding-3-small API, handles batching and retry | `src/gold_trading/embeddings/client.py` | S |
| 3.5 | Write tests — macro data processing (mocked FRED), sentiment scoring (mocked Claude), regime classification (known ATR/ADX inputs → expected regime) | `tests/test_macro_analyst.py`, `tests/test_sentiment_analyst.py`, `tests/test_regime_analyst.py` | M |

**Test Gate:**
```bash
cd /Users/sjonas/tradingview && \
  uv run pytest tests/test_macro_analyst.py tests/test_sentiment_analyst.py tests/test_regime_analyst.py -v
```

---

## Phase 4: Backtesting Engine + Pine Script RAG (L)

vectorbt strategy backtesting, Monte Carlo, Pine Script corpus ingestion, Quant Researcher agent.
**Paperclip wiring:** Wire Quant Researcher script into its Paperclip agent slot and verify one heartbeat cycle completes.

| # | Task | Files | Complexity |
|---|---|---|---|
| 4.1 | Create Pine Script corpus ingestion — clone codenamedevan/pinescriptv6, code-aware chunking (never split fenced code blocks), embed chunks via text-embedding-3-small, load into `pinescript_corpus` table | `src/gold_trading/embeddings/corpus.py` | M |
| 4.2 | Create vectorbt backtesting module — translate strategy parameters into vectorbt signal arrays on 5m GC data, run backtest with `price="close"` and 0.02% slippage, calculate Sharpe/win rate/expectancy/max DD | `src/gold_trading/backtest/engine.py` | L |
| 4.3 | Create Monte Carlo simulation — block bootstrap (6-month blocks) on trade returns, 1000 iterations, output 5th/50th/95th percentile Sharpe and max drawdown | `src/gold_trading/backtest/montecarlo.py` | M |
| 4.4 | Create quant researcher script — queries lessons + regime + RAG corpus, calls Claude Sonnet 4.6 to generate Pine Script, runs vectorbt + Monte Carlo gate, writes passing strategies to `strategies` table | `scripts/quant_researcher.py` | L |
| 4.5 | Write tests — corpus chunking (code blocks intact), backtest on known signal array (expected Sharpe), Monte Carlo distribution shape, lessons store similarity search | `tests/test_corpus.py`, `tests/test_backtest.py`, `tests/test_lessons_store.py` | M |

**Test Gate:**
```bash
cd /Users/sjonas/tradingview && \
  uv run pytest tests/test_corpus.py tests/test_backtest.py tests/test_lessons_store.py -v
```

---

## Phase 5: Skills, Prompts + Integration (L)

Shared Paperclip skills, CIO/Technical Analyst prompts, risk manager agent, end-to-end test. Paperclip company already running from Phase 0; agents wired incrementally in Phases 2-4.

| # | Task | Files | Complexity |
|---|---|---|---|
| 5.1 | Create shared Paperclip skills — gold-trading-context (instruments, risk rules, DB schema), pine-script-v6 (RAG instructions), memory-protocol (how to query/write lessons) | `.paperclip/skills/*.md` | M |
| 5.2 | Create risk manager agent script — deterministic Python only (no LLM), checks drawdown every heartbeat, kills strategies if limits breached, tags open trades with regime/sentiment. Wire into Paperclip Risk Manager agent slot. | `scripts/risk_manager.py` | M |
| 5.3 | Create CIO and Technical Analyst HEARTBEAT.md prompts — system prompts for claude_local agents with decision frameworks, memory query instructions, strategy activation logic | `.paperclip/agents/cio/HEARTBEAT.md`, `.paperclip/agents/technical_analyst/HEARTBEAT.md` | M |
| 5.4 | Enable all agent heartbeats in Paperclip — verify each agent completes at least one heartbeat cycle successfully | N/A (Paperclip UI) | S |
| 5.5 | End-to-end integration test — all services up, run migration, ingest sample data, trigger each agent via Paperclip `POST /api/agents/:id/heartbeat/invoke`, verify DB state after each run, send test webhook, verify paper trade created | `tests/test_integration.py` | L |

**Test Gate:**
```bash
cd /Users/sjonas/tradingview && \
  docker compose up -d && \
  uv run python -m gold_trading.db.client --migrate && \
  uv run pytest tests/test_integration.py -v && \
  echo "All phases complete. System operational."
```

---

## Phase Summary

| Phase | Name | Tasks | Gate |
|---|---|---|---|
| 0 | Scaffold + Paperclip | 8 | `ruff check` + Docker up + Paperclip health check |
| 1 | Database + Models | 5 | Migrations apply + DB tests pass |
| 2 | Risk Engine + Webhook | 6 | Webhook + risk tests pass |
| 3 | Data Pipelines + Sentiment | 5 | Agent script tests pass + Paperclip heartbeat verified |
| 4 | Backtesting + RAG | 5 | Backtest + corpus tests pass + Paperclip heartbeat verified |
| 5 | Skills, Prompts + Integration | 5 | End-to-end integration test passes |
| **Total** | | **34** | |

---

## Post-Build Checklist

After all phases pass (Paperclip already running from Phase 0):

1. [ ] Purchase historical 5m GC data from FirstRate Data, ingest into TimescaleDB
2. [ ] Ingest Pine Script v6 corpus: `uv run python -m gold_trading.embeddings.corpus`
3. [ ] Set up TradingView Premium account with GC/MGC 5m charts
4. [ ] Enable all heartbeats in Paperclip UI — monitor dashboard for first 48 hours
5. [ ] Once Quant Researcher produces a passing strategy: deploy Pine Script to TradingView manually
6. [ ] Configure TradingView strategy alert → webhook URL → begin paper trading
