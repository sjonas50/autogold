---
name: pine-script-v6
description: Pine Script v6 coding conventions and RAG-assisted strategy development. Invoke when writing, reviewing, or debugging Pine Script v6 for TradingView.
---

# Pine Script v6 — Strategy Development Guide

## Mandatory Strategy Settings

Every strategy MUST include these settings:

```pinescript
//@version=6
strategy("Strategy Name",
    overlay=true,
    process_orders_on_close=true,  // REQUIRED for vectorbt alignment
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    commission_type=strategy.commission.cash_per_contract,
    commission_value=2.05,         // IBKR GC commission
    slippage=3)                    // ~$0.30 = 3 ticks on GC
```

**Never use `calc_on_every_tick=true`** — it breaks historical backtest consistency.
**Never use Bar Magnifier** during prototyping — it changes fill logic.

## Required Components in Every Strategy

### 1. Time Filter
```pinescript
// Block trading during maintenance (22:00-23:00 UTC)
is_maintenance = (hour(time, "UTC") == 22)
// Block 60 seconds before/after major news (manual input)
news_blackout = input.bool(false, "News Blackout Active")
can_trade = not is_maintenance and not news_blackout
```

### 2. ATR-Based Stop
```pinescript
atr_val = ta.atr(14)
stop_distance = atr_val * input.float(1.5, "ATR Stop Multiple")
```

### 3. Max Duration Auto-Exit
```pinescript
var int entry_bar = 0
if strategy.position_size != 0 and entry_bar == 0
    entry_bar := bar_index
if strategy.position_size != 0 and (bar_index - entry_bar) >= 24  // 24 bars * 5min = 120 min
    strategy.close_all("Max Duration")
    entry_bar := 0
if strategy.position_size == 0
    entry_bar := 0
```

### 4. Webhook Alert Messages
```pinescript
if entry_condition and can_trade
    strategy.entry("Long", strategy.long)
    alert('{"secret":"' + input.string("", "Secret") + '","strategy_id":"gs_v1_breakout","action":"buy","contracts":1,"price":' + str.tostring(close) + ',"bar_time":"' + str.tostring(time) + '","instrument":"GC"}', alert.freq_once_per_bar_close)

if exit_condition
    strategy.close("Long")
    alert('{"secret":"' + input.string("", "Secret") + '","strategy_id":"gs_v1_breakout","action":"close_long","contracts":1,"price":' + str.tostring(close) + ',"bar_time":"' + str.tostring(time) + '","instrument":"GC"}', alert.freq_once_per_bar_close)
```

## RAG Corpus

The Pine Script v6 documentation is embedded in the `pinescript_corpus` table (pgvector).

To search:
```sql
SELECT content, source_file, chunk_type
FROM pinescript_corpus
ORDER BY embedding <=> $query_embedding::vector
LIMIT 10;
```

Chunk types: `concept` (explanations), `reference` (API docs), `example` (code samples).

## Pine Script v6 Key Differences from v5

- `strategy()` requires explicit parameter names
- `input.*` functions (input.int, input.float, input.string, input.bool)
- `ta.*` namespace for all technical indicators
- `str.*` for string operations
- `array.*` and `map.*` for collections
- Plot functions require `title` parameter

## Alignment with vectorbt

For backtest results to be comparable between Pine Script and vectorbt:
- Pine: `process_orders_on_close=true` ↔ vectorbt: `price="close"`
- Match commission and slippage exactly
- Never use `calc_on_every_tick`
- Drop 22:00-23:00 UTC bars from vectorbt data
- Use Sharpe ratio and win rate as fitness metrics (NOT max drawdown)
