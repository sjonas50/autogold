# Gold Futures Trading System

Autonomous gold futures (GC/MGC) trading system orchestrated by Paperclip with 7 AI agents (all Claude Sonnet 4.6). Backtest + paper trade scope only.

## Quick Reference

```bash
# Install deps
uv sync

# Start database
docker compose up -d

# Run migrations
uv run python -m gold_trading.db.client --migrate

# Run tests
uv run pytest -v

# Lint
ruff check src/ tests/ scripts/
ruff format src/ tests/ scripts/

# Start webhook receiver
uv run uvicorn gold_trading.main:app --host 0.0.0.0 --port 8080

# Run individual agent scripts (normally triggered by Paperclip)
uv run python scripts/macro_analyst.py
uv run python scripts/sentiment_analyst.py
uv run python scripts/regime_analyst.py
uv run python scripts/risk_manager.py
uv run python scripts/quant_researcher.py

# Ingest Pine Script RAG corpus
uv run python -m gold_trading.embeddings.corpus
```

## Architecture

- **Orchestration:** Paperclip (Node.js) manages 7 agents via heartbeat scheduling
- **LLM:** Claude Sonnet 4.6 (`claude-sonnet-4-6`) for all agents. No Haiku.
- **Database:** TimescaleDB (PostgreSQL 16) + pgvector on Docker port 5433
- **Webhook:** FastAPI receiver for TradingView strategy alerts
- **Backtesting:** vectorbt (Python) for optimization loop. TradingView = visual validation only.
- **Instruments:** GC (gold futures, $100/oz) and MGC (micro gold, $10/oz). NOT XAUUSD spot.
- **Timeframe:** 5-minute execution. Analysis across 1m, 5m, 15m, 1H.

## Agent Adapters

| Agent | Adapter | Script |
|---|---|---|
| CIO | `claude_local` | N/A (Paperclip-managed) |
| Macro Analyst | `process` | `scripts/macro_analyst.py` |
| Technical Analyst | `claude_local` | N/A (Paperclip-managed) |
| Quant Researcher | `process` | `scripts/quant_researcher.py` |
| Sentiment Analyst | `process` | `scripts/sentiment_analyst.py` |
| Risk Manager | `process` | `scripts/risk_manager.py` |
| Regime Analyst | `process` | `scripts/regime_analyst.py` |

## Risk Rules (hard-coded, never override)

- Max risk per trade: **0.5%** of account
- Max drawdown before full halt: **2%**
- Max concurrent positions: **1**
- Max trade duration: **120 minutes** (auto-exit)
- Daily loss limit: **1%** — no more trades that session
- Risk Manager is **deterministic Python only** — no LLM calls in the risk path

## Key Constraints

- **TradingView has no API for deploying Pine Script** — human must paste scripts into TradingView IDE
- **TradingView backtest results cannot be exported** — use vectorbt for automated optimization
- **Process adapter agents lose session context** — all state persists to TimescaleDB
- **TradingView paper trading does NOT fire webhooks** — use strategy alerts on live chart instead
- **vectorbt and TradingView diverge on max drawdown** — use Sharpe/win rate as fitness metric

## Database

9 tables in TimescaleDB. See `docs/architecture.md` for full schema.

Key tables: `trade_journal`, `decision_log`, `lessons` (pgvector), `regime_state`, `sentiment_scores`, `macro_data`, `strategies`, `paper_trades`, `pinescript_corpus`

## Vectorbt ↔ TradingView Alignment

- Pine Script: `process_orders_on_close=true`
- vectorbt: `price="close"`
- Both: 0.02% slippage, matching commission values
- Never use `calc_on_every_tick` in Pine Script
- Drop 22:00-23:00 UTC maintenance bars from vectorbt data

## Project Structure

```
src/gold_trading/       # FastAPI webhook + shared libraries
scripts/                # Paperclip process adapter agent scripts
pine/generated/         # Output Pine Script strategies
.paperclip/             # Paperclip company config + skills
tests/                  # pytest tests
docs/                   # research.md, architecture.md, build-plan.md
```

## Stack Pitfalls

- `fedfred` for FRED API (not `fredapi` — it's unmaintained)
- Polygon.io for news (NOT NewsAPI.org — free tier has 24h delay)
- `text-embedding-3-small` for embeddings (1536 dimensions)
- pgvector HNSW index: use `vector_cosine_ops` for similarity search
- Paperclip heartbeat deadlock bug #2516: set `timeoutSec` in agent runtimeConfig
- Paperclip imported companies have heartbeats disabled by default — re-enable manually
