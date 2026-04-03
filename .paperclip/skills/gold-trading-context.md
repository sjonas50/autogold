---
name: gold-trading-context
description: Core context for the AutoGoldFutures trading system. Invoke when you need to understand instruments, risk rules, database schema, or agent roles.
---

# AutoGoldFutures — Trading Context

## Instruments

| Instrument | Contract Size | Multiplier | Tick Size | Tick Value |
|---|---|---|---|---|
| **GC** | 100 troy oz | $100/point | $0.10 | $10 |
| **MGC** | 10 troy oz | $10/point | $0.10 | $1 |

Primary execution timeframe: **5-minute candles**.
Analysis timeframes: 1m, 5m, 15m, 1H.

## Trading Sessions (ET)

| Session | Time | Character |
|---|---|---|
| Asia/Sydney | 6pm-3am | Low volume, range-bound |
| London open | 3am-8:30am | First momentum burst |
| NY open | 8:30am-9:30am | News reactions, second burst |
| London/NY overlap | 8:30am-11:30am | Highest volume, strongest trends |
| NY afternoon | 11:30am-1:30pm | Mean-reversion, chop |
| COMEX close | 1:30pm | Volume spike |
| Maintenance | 5pm-6pm ET (22:00-23:00 UTC) | **Market closed** |

## Risk Rules (absolute, never override)

- Max risk per trade: **0.5%** of account
- Max drawdown before full halt: **2%**
- Max concurrent positions: **1**
- Daily loss limit: **1%** — no more trades that session
- Max trade duration: **120 minutes** (auto-exit)
- No trading during first 60 seconds of a news event
- No trading when regime = "volatile" or "choppy"
- Risk Manager kill-switch overrides CIO

## Position Sizing Formula

```
contracts = floor(account_equity * 0.005 / (stop_distance * contract_multiplier))
```

Where:
- `stop_distance` = ATR-based distance in price points
- GC: `contract_multiplier` = $100/point
- MGC: `contract_multiplier` = $10/point

## Strategy Classes

| Class | Best Regime | Duration | Entry Logic |
|---|---|---|---|
| Session Open Breakout | trending | 15-90 min | Break above prior 30-60 min range |
| VWAP/Key Level Reversion | ranging | 5-45 min | Extended move from VWAP, reversal pattern |
| News Momentum | event-driven | 1-30 min | Post-NFP/CPI/FOMC, 60s delay, volume confirm |
| Liquidity Sweep Reversal | overlap | 10-60 min | Sweep of session high/low, then reverse |

## Validation Gates (all four must pass)

1. **Backtest**: Sharpe >= 1.0, win rate >= 45%, trades >= 50, max DD < 15%
2. **Walk-forward**: Profitable out-of-sample across 3+ windows
3. **Monte Carlo**: 5th percentile Sharpe >= 0.5, 95th percentile DD < 15%
4. **Regime filter**: Positive expectancy in target regime, flat in others

## Agent Org Chart

```
CIO (claude_local, 2hr)
├── Technical Analyst (claude_local, 1hr)
├── Macro Analyst (process, 4hr)
├── Sentiment Analyst (process, 30min)
├── Regime Analyst (process, 30min)
├── Risk Manager (process, 15min)
└── Quant Researcher (process, 8hr)
```

## Database (TimescaleDB + pgvector)

| Table | Purpose |
|---|---|
| `trade_journal` | Completed trades with full context |
| `decision_log` | All agent decisions with reasoning |
| `lessons` | Extracted learnings (pgvector similarity search) |
| `regime_state` | Current market regime classification |
| `sentiment_scores` | Gold news sentiment scores |
| `macro_data` | FRED economic indicators |
| `strategies` | Strategy definitions + backtest metrics |
| `paper_trades` | Open and closed paper positions |
| `ohlcv_5m` | 5-minute gold futures price data |
| `pinescript_corpus` | Pine Script v6 RAG embeddings |

Connection: `$DATABASE_URL` (postgresql://gold:gold@localhost:5433/gold_trading)
