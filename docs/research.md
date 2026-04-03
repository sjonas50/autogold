# Research: Autonomous Gold Trading Firm via Paperclip

**Date:** 2026-04-03 | **Scope:** Backtest + paper trade only. No live broker execution. 5m execution timeframe.

## Executive Summary

Paperclip is a viable orchestration platform for this system — it provides org-chart governance, heartbeat scheduling, budget controls, and a monitoring dashboard out of the box. Its Process adapter runs Python scripts natively, and its HTTP adapter can call FastAPI services. The critical architectural constraint is that **TradingView has no programmatic API for deploying Pine Script** — a human must paste scripts into the IDE. Design around this: agents generate and optimize Pine Script code, but deployment is a human-in-the-loop step. For the backtest/paper-trade scope, this is acceptable. The second major finding is that the original plan's "Paperclip writes Pine Script → TradingView backtests → webhook fires" loop has a gap: **TradingView backtest results cannot be exported programmatically** — they must be scraped or manually copied. Consider running parallel backtests in Python (vectorbt) for the automated optimization loop, with TradingView as the final visual validation step.

## Problem Statement

Build a Paperclip-orchestrated "company" of 7 AI agents that research gold market conditions, develop Pine Script strategies, validate them via Monte Carlo simulation, and deploy them on TradingView for paper trading. Primary execution timeframe: 5-minute. Max drawdown: 2%. Live news analysis is a priority.

---

## Technology Evaluation

### Agent Orchestration: Paperclip

| Attribute | Detail |
|---|---|
| Runtime | Node.js 20+ server, React UI, embedded PostgreSQL |
| License | MIT, self-hosted, no account required |
| Stars | ~44K (launched March 4, 2026 — 1 month old) |
| Python support | Process adapter (`python3 /path/to/agent.py`) or HTTP adapter (POST to FastAPI) |
| Heartbeat | 9-step protocol: identity → work discovery → checkout → execute → status update → delegate |
| Budget controls | Per-agent monthly caps, 80% warning, 100% hard stop |
| Session persistence | Claude Local adapter preserves context across heartbeats; Process adapter does NOT |
| Skills | Markdown-based SKILL.md files with YAML frontmatter — instructions, not executable code |

**How our 7 agents map to Paperclip:**

| Agent | Adapter | Why |
|---|---|---|
| CIO | `claude_local` | Needs reasoning + session persistence for synthesis |
| Macro Analyst | `process` | Python script calling FRED API + analysis |
| Technical Analyst | `claude_local` | Multi-timeframe chart reasoning, session context |
| Quant Researcher | `process` | Python: Pine Script generation (RAG), vectorbt backtesting, Monte Carlo |
| Sentiment Analyst | `process` | Python: news ingestion + FinBERT inference |
| Risk Manager | `process` | Python: deterministic drawdown tracking, position sizing |
| Regime Analyst | `process` | Python: ATR/ADX/HMM regime classification |

**Key gap:** Process adapter agents lose context between heartbeats. Mitigate by persisting agent state to the database — each Python agent reads/writes its own state file or DB row.

**Estimated LLM cost:** ~$120-150/mo for 7 agents at planned heartbeat frequencies (Claude Sonnet 4.6 for all agents).

### Backtesting & Validation

| Library | Best For | Notes |
|---|---|---|
| **vectorbt PRO** | Primary backtesting + Monte Carlo | 1M simulations in 20s, walk-forward CV, block bootstrap. Paid license. |
| **vectorbt** (open) | Budget alternative | Fewer features, no walk-forward built-in |
| **quantstats** | Performance reporting | Sharpe, drawdown, tearsheets. Use alongside vectorbt |
| **hmmlearn** | Regime detection | Gaussian HMM, unsupervised regime discovery |

**Recommendation:** Use vectorbt (open-source initially, upgrade to PRO if needed) for the automated backtest loop. TradingView remains the visual validation + paper trading platform. Pine Script strategies are the deployment target, but Python backtesting runs the optimization loop.

### Pine Script Generation

**FaustoS88/Pydantic-AI-Pinescript-Expert** (GitHub) demonstrates the right pattern: RAG over Pine Script v6 documentation corpus with code-aware chunking. LLMs hallucinate Pine Script v5 syntax under v6 at a high rate without retrieval grounding.

**Approach:** Build a RAG skill for the Quant Researcher agent using the `codenamedevan/pinescriptv6` documentation corpus. Generate → validate syntax → backtest in vectorbt → if passing, present to human for TradingView deployment.

### News & Sentiment Pipeline (Priority)

| Component | Choice | Price | Rationale |
|---|---|---|---|
| Financial news | **Polygon.io Starter** | $29/mo | Real-time, financial-focused, clean Python client |
| Backup/supplement | **NewsAPI.ai** (Event Registry) | Free 2K searches/mo | Entity tagging, near real-time |
| Sentiment model | **Claude Sonnet 4.6** API | ~$5/mo | Handles gold-specific nuance natively, no fine-tuning needed |
| Economic calendar | **FRED API** via `fedfred` | $0 | 120 req/min, async, gold-relevant series |
| Breaking events | RSS feeds + Polygon.io | Included | FOMC, CPI, NFP, geopolitical |

**Critical note:** NewsAPI.org free tier has 24-hour delay — useless for live news. Use Polygon.io.

**Sentiment approach:** Claude Sonnet 4.6 handles gold-specific nuance (safe-haven dynamics, USD inverse correlation, real yield sensitivity) natively without fine-tuning. FinBERT-FOMC available as a low-cost fallback if API costs need to be reduced later.

### Database

**QuestDB** (6-13x faster than TimescaleDB for time-series queries) or **TimescaleDB** (safer bet, PostgreSQL extension). Both work. At our volume (~100 trades/day), either is fine. TimescaleDB has the advantage of being a PostgreSQL extension — Paperclip already uses PostgreSQL.

**Recommendation:** TimescaleDB on Docker (`timescale/timescaledb:latest-pg16`). Storage: <1GB/year at our volume.

### TradingView

- **Plan required:** Premium ($42-60/mo) for 800 alerts + unlimited duration
- **Pine Script v6 limits:** 100K token compiled script, 20K historical bars (Premium), 500 max drawing objects
- **No API for script deployment** — human-in-the-loop required
- **No programmatic backtest export** — scrape or copy manually
- **Webhook rate:** ~1/sec sustained before throttling
- **Webhook auth:** No built-in auth — validate a shared secret in every payload

---

## Reference Implementations

| Project | Relevance | Key Takeaway |
|---|---|---|
| [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) | HIGH | 7-agent trading firm with bull/bear debate protocol. Adopt org structure. |
| [FaustoS88/Pydantic-AI-Pinescript-Expert](https://github.com/FaustoS88/Pydantic-AI-Pinescript-Expert) | HIGH | RAG over Pine Script v6 docs. Adopt for Quant Researcher agent. |
| [Sakeeb91/market-regime-detection](https://github.com/Sakeeb91/market-regime-detection) | HIGH | HMM regime classifier with hmmlearn. Use DXY + TIPS + ATR/ADX as inputs for gold. |
| [harmtemolder/tradingview-to-oanda](https://github.com/harmtemolder/tradingview-to-oanda) | MEDIUM | OANDA webhook pattern (useful when live execution is added later). |
| [je-suis-tm/quant-trading](https://github.com/je-suis-tm/quant-trading) | HIGH | Monte Carlo via return resampling. Use vectorbt block bootstrap instead for production. |
| [paperclipai/paperclip](https://github.com/paperclipai/paperclip) | CORE | Our orchestration platform. "Crypto Trading Desk" template (12 agents) coming to Cliphub. |

---

## Known Pitfalls & Risks

| Risk | Severity | Mitigation |
|---|---|---|
| TradingView has no deployment API | HIGH | Human-in-the-loop for Pine Script deployment. Automate everything up to that step. |
| TradingView backtest results not exportable | HIGH | Run primary optimization loop in vectorbt. TradingView = final visual validation. |
| Paperclip is 1 month old | MEDIUM | Keep risk-critical logic (drawdown checks, position sizing) in deterministic Python, not LLM agents. |
| Process adapter loses session context | MEDIUM | Persist agent state to TimescaleDB between heartbeats. |
| No gold-specific sentiment model | LOW | Claude Sonnet 4.6 handles gold nuance natively. Fine-tune FinBERT-FOMC later if cost reduction needed. |
| TV paper trading doesn't fire webhooks | LOW | Use strategy alerts on live chart → own paper trading layer. Already planned. |
| vectorbt/TV backtest divergence | MEDIUM | Use Sharpe/win rate as fitness metric (not drawdown). Validate rank order on top-3 strategies. |
| FRED data is 1-2 day lagged | LOW | Use for daily macro regime only, not intraday signals. |
| TradingView alerts silently expire | LOW | Monitor alert count on a schedule. |

---

## Recommended Stack

| Layer | Choice | Cost/mo |
|---|---|---|
| Orchestration | **Paperclip** (self-hosted) | $0 + LLM costs |
| LLM for all agents | **Claude Sonnet 4.6** (`claude-sonnet-4-6`) | ~$120-150/mo |
| Backtesting engine | **vectorbt** (open-source) | $0 |
| Monte Carlo | vectorbt + numpy | $0 |
| Regime detection | **hmmlearn** + ATR/ADX | $0 |
| Pine Script generation | RAG over v6 docs (Pydantic AI pattern) | $0 |
| Sentiment | **Claude Sonnet 4.6** API (fallback: fine-tuned FinBERT-FOMC) | included above |
| News feed | **Polygon.io Starter** | $29 |
| Macro data | **FRED** via `fedfred` | $0 |
| Database | **TimescaleDB** (Docker) | $0 |
| Vector search | **pgvector** extension on TimescaleDB | $0 |
| Embeddings | `text-embedding-3-small` or local `sentence-transformers` | ~$1-5 |
| Agent memory | Trade Journal + Decision Log + Lessons Store (PostgreSQL) | $0 |
| Charting/paper trade | **TradingView Premium** | $42-60 |
| Hosting | VPS 4 vCPU / 8GB (NYC) | $24-48 |
| **Total** | | **~$215-290/mo** |

---

## Agent Memory & Learning System

The system needs a closed-loop memory so agents improve over time — not just log trades, but extract lessons and adjust behavior based on accumulated experience.

### Architecture: Three Memory Layers (all PostgreSQL/TimescaleDB + pgvector)

| Layer | Schema Purpose | Writers | Readers |
|---|---|---|---|
| **Trade Journal** | Every trade with full context: regime state, sentiment score, macro bias, strategy ID, entry/exit prices, R-multiple outcome, slippage, time-of-day, session | Risk Manager, Quant Researcher | All agents — primary feedback signal |
| **Decision Log** | Every agent decision with structured reasoning: what was decided, inputs considered, confidence level, outcome (correct/incorrect tagged post-hoc) | All agents (each logs own decisions) | CIO (reviews team performance), individual agents (self-review) |
| **Lessons Store** (pgvector) | Extracted learnings as vector embeddings with structured metadata: condition tags, strategy class, regime, date range, confidence score | CIO (extracts lessons during review heartbeat) | All agents query before decisions via similarity search |

### Feedback Loop

```
Trade completes
  → Outcome + full context logged to Trade Journal
  → Risk Manager tags win/loss, R-multiple, regime at time of trade
  → CIO review heartbeat (every 2 hours):
      - Queries recent trades + decision log
      - Compares outcome to agent predictions
      - If unexpected outcome → extracts lesson → embeds → stores in Lessons Store
      - If pattern emerges (e.g., 3+ losses in same regime/strategy combo) → flags for strategy deactivation
  → All agents query Lessons Store at heartbeat start:
      - "Given current regime + sentiment + macro, what have we learned?"
      - Top-K similar lessons injected into agent prompt context
      - Agent adjusts behavior accordingly
```

### Why pgvector (not a separate vector DB)

- Already running PostgreSQL via TimescaleDB — pgvector is a single `CREATE EXTENSION`
- No extra infrastructure, no extra backup strategy, no extra failure mode
- Supports HNSW indexes for fast approximate nearest neighbor at our scale (<100K lessons)
- Agents query with: `SELECT lesson, metadata FROM lessons ORDER BY embedding <=> $current_context LIMIT 5`
- JOIN with structured data: "find lessons where regime was trending AND strategy was breakout AND outcome was loss"

### What Gets Embedded

Each lesson is a structured text blob + metadata, embedded via a small model (e.g., `text-embedding-3-small` or local `sentence-transformers`):

- **Condition snapshot:** regime state, sentiment score, macro bias, time-of-day, day-of-week, recent ATR
- **Decision made:** which strategy activated, position size, entry/exit logic
- **Outcome:** P&L, R-multiple, max adverse excursion
- **Extracted lesson:** natural language summary (generated by CIO agent)
- **Tags:** strategy class, regime type, confidence level, date

### Answers Open Question #4

This memory system also solves the Process adapter session persistence problem — agents don't need in-memory session context when they can query their own Decision Log and the shared Lessons Store at each heartbeat. State lives in the database, not the adapter.

---

## Resolved Questions

### Q1: Paperclip Heartbeat Interval Config — RESOLVED

Per-agent intervals are fully supported via `runtimeConfig.schedule.intervalSec`:

```json
{
  "runtimeConfig": {
    "schedule": {
      "enabled": true,
      "intervalSec": 900,
      "wakeOnAssignment": true
    },
    "timeoutSec": 300
  }
}
```

- Risk Manager at 15 min = `intervalSec: 900` — works today
- CIO at 2 hours = `intervalSec: 7200` — works today
- Minimum interval: 30 seconds. Set to 0 to disable timer (event-only).
- **No cron syntax yet.** PR [#1172](https://github.com/paperclipai/paperclip/issues/219) ("Routines") is open but not merged. For wall-clock determinism (e.g., "run at market open 9:30 ET"), use external cron calling `POST /api/agents/:agentId/heartbeat/invoke`.
- **Gotchas:** Skipped heartbeats (paused/budget-blocked) produce no receipt ([#1120](https://github.com/paperclipai/paperclip/issues/1120)). Deadlock possible on `maxConcurrentRuns=1` agents ([#2516](https://github.com/paperclipai/paperclip/issues/2516)). Imported companies have heartbeats disabled by default.

### Q2: Pine Script v6 RAG Corpus — RESOLVED

**Use [`codenamedevan/pinescriptv6`](https://github.com/codenamedevan/pinescriptv6)** — the de facto standard.

- 169 stars, 98 forks, 79 commits. Last updated January 24, 2026.
- Structured for LLM/RAG: 4 folders (`concepts/`, `reference/`, `visuals/`, `writing_scripts/`) + `LLM_MANIFEST.md` index.
- Corpus size: ~4,910 chunks after processing, ~2.5–5M tokens total.
- Reference RAG implementation: [FaustoS88/Pydantic-AI-Pinescript-Expert](https://github.com/FaustoS88/Pydantic-AI-Pinescript-Expert) — pgvector, hybrid BM25+vector search, code-aware chunking. Scored 0.919 context relevance, 0.833 faithfulness (RAGAS benchmark, 40 questions).
- Supplement with a fresh crawl via [FaustoS88/PinescriptV6-docs-crawler](https://github.com/FaustoS88/PinescriptV6-docs-crawler) before launch to catch recent v6 additions.

### Q3: vectorbt vs TradingView Backtest Fidelity — RESOLVED

**They will not match exactly.** Close enough for rank-ordering strategies (valid for optimization), not for precise P&L figures.

**Alignment protocol (minimizes divergence):**

| Setting | Pine Script | vectorbt |
|---|---|---|
| Execution timing | `process_orders_on_close=true` | `price="close"` |
| Commission | Fixed $ or % in strategy settings | Same value in `fees` param |
| Slippage | 0.02% | 0.02% |
| Session gap | Time filter blocking 22:00-23:00 UTC | Drop maintenance bars from data |
| `calc_on_every_tick` | **Never enable** (breaks historical consistency) | N/A |
| Bar Magnifier | Off during prototyping | N/A |

**Known structural divergences:**
- **Max drawdown:** TradingView uses intrabar lows; vectorbt uses bar-boundary equity. Expect 2–8% relative difference. **Use Sharpe ratio or win rate as fitness metric, not drawdown.**
- **Fee rounding:** ~0.1% absolute difference (floating-point vs TV's rounding).
- **Trade count and win rate:** Match exactly with correct execution model alignment.

**5-minute XAUUSD data sources:**

| Source | Coverage | Cost | Recommended Use |
|---|---|---|---|
| **FirstRate Data** | 15 years | ~$30/dataset | Optimization corpus |
| **OANDA v20 API** | ~6 months rolling | Free (account holders) | Live-adjacent validation |
| **Twelve Data** | 1 year (free) | Free–$50/mo | Supplement |

**Validation step:** After optimizing in vectorbt, run top-3 strategies in TradingView strategy tester on most recent 3 months. If rank order is preserved, the loop is valid.

### Q4: Agent State Persistence — RESOLVED

Agents persist state via Trade Journal + Decision Log + Lessons Store in PostgreSQL. See Agent Memory & Learning System section above.

### Q5: Gold-Specific Sentiment Model — RESOLVED

**No gold-specific BERT model exists.** Recommendation: **Use Claude Sonnet 4.6 for all sentiment analysis.**

**Cost comparison at 50 headlines / 30 min (~72K headlines/month):**

| Approach | Monthly Cost | Accuracy on Gold Text | Setup Effort |
|---|---|---|---|
| **Claude Sonnet 4.6 API** | **~$5-8/mo** | Highest (handles gold-specific nuance natively) | Prompt engineering only |
| Self-hosted FinBERT (GPU) | ~$380/mo (always-on) or ~$5/mo (on-demand batch) | Lower on gold text without fine-tuning | Model deployment |
| Fine-tuned FinBERT-FOMC | ~$5/mo (on-demand) + one-time $1 training | Better on FOMC text, unknown on general gold | 1-2 weeks fine-tuning work |

**If you do fine-tune later:**
- Base model: [`Incredible88/FinBERT-FOMC`](https://github.com/Incredible88/FinBERT-FOMC) — fine-tuned on FOMC minutes, +17.4% accuracy on complex sentences vs base FinBERT.
- Dataset: [Kaggle gold commodity sentiment](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-in-commodity-market-gold) (~10K labeled examples, 2000-2021, CC BY-NC-ND 4.0).
- Augment with 500-1,000 Claude-labeled gold headlines (validated approach per TinyFinBERT research).
- Training: 3 epochs, lr 2e-5, batch 16, <2 hours on a T4.

### Q6: TradingView Paper Trading Webhooks — RESOLVED

**TradingView's built-in "Paper Trading" simulator does NOT fire webhooks.**

**The correct approach for our paper trading phase:**
1. Apply Pine Script strategy to a **live** XAUUSD chart
2. Create a **strategy alert** with your webhook URL
3. TradingView fires HTTP POST on every order execution in real-time
4. Your webhook receiver logs the signal and simulates the trade (your own paper trading layer)

**Key details:**
- Webhook requires Essential/Pro plan minimum ($14.95+/mo). Premium already in our stack.
- Strategy alerts run server-side — browser doesn't need to be open.
- Use `process_orders_on_close=true` to prevent repainting mid-bar signals.
- Implement idempotent handling — TradingView may occasionally duplicate or miss alerts.
- Webhook payload is user-defined JSON with Pine Script placeholders: `{{strategy.order.action}}`, `{{strategy.order.contracts}}`, `{{close}}`, `{{time}}`, etc.

## Remaining Open Questions

None — all research questions resolved. Ready for architecture planning.

---

## Sources

- [Paperclip GitHub](https://github.com/paperclipai/paperclip) | [Docs](https://docs.paperclip.ing)
- [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) | [arXiv](https://arxiv.org/abs/2412.20138)
- [Pydantic-AI-Pinescript-Expert](https://github.com/FaustoS88/Pydantic-AI-Pinescript-Expert)
- [market-regime-detection](https://github.com/Sakeeb91/market-regime-detection)
- [TradingView Pricing](https://www.tradingview.com/pricing/) | [Pine Script Limits](https://www.tradingview.com/pine-script-docs/writing/limitations/)
- [OANDA v20 API](https://developer.oanda.com/rest-live-v20/introduction/)
- [ib_async](https://github.com/ib-api-reloaded/ib_async)
- [FRED API](https://fred.stlouisfed.org/docs/api/fred/) | [fedfred](https://nikhilxsunder.github.io/fedfred/)
- [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- [Polygon.io](https://polygon.io)
- [vectorbt](https://github.com/polakowo/vectorbt)
- [quantstats](https://github.com/ranaroussi/quantstats)
- [hmmlearn](https://github.com/hmmlearn/hmmlearn)
- [Walk-Forward Validation arXiv:2512.12924](https://arxiv.org/html/2512.12924v1)
- [Paperclip Heartbeat Protocol](https://docs.paperclip.ing/guides/agent-developer/heartbeat-protocol)
- [Paperclip RFC: Agent Routines #219](https://github.com/paperclipai/paperclip/issues/219)
- [codenamedevan/pinescriptv6](https://github.com/codenamedevan/pinescriptv6)
- [PinescriptV6-docs-crawler](https://github.com/FaustoS88/PinescriptV6-docs-crawler)
- [FinBERT-FOMC](https://github.com/Incredible88/FinBERT-FOMC) | [ACM Paper](https://dl.acm.org/doi/fullHtml/10.1145/3604237.3626843)
- [Kaggle Gold Commodity Sentiment Dataset](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-in-commodity-market-gold)
- [vectorbt GitHub Issue #771 (TV fidelity)](https://github.com/polakowo/vectorbt/issues/771)
- [FirstRate Data (historical 5m XAUUSD)](https://firstratedata.com/)
- [TradingView Strategy Alerts](https://www.tradingview.com/support/solutions/43000481368-strategy-alerts/)
- [TradingView Webhook Config](https://www.tradingview.com/support/solutions/43000529348-how-to-configure-webhook-alerts/)
- [pgvector](https://github.com/pgvector/pgvector)
