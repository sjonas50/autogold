---
name: memory-protocol
description: How agents query and write to the shared learning system (lessons store + decision log). Invoke when you need to read past lessons, write new lessons, or log decisions.
---

# Memory Protocol — Shared Learning System

All agents share a three-layer memory system in TimescaleDB + pgvector.

## Layer 1: Trade Journal (read-only for most agents)

Completed trades with full context. Written by the webhook receiver when trades close.

```sql
SELECT strategy_id, direction, pnl_usd, r_multiple,
       regime_at_entry, sentiment_score, macro_bias, session
FROM trade_journal
WHERE closed_at > NOW() - INTERVAL '24 hours'
ORDER BY closed_at DESC;
```

## Layer 2: Decision Log (all agents write, CIO reviews)

Every agent logs decisions with structured reasoning.

**Writing a decision:**
```sql
INSERT INTO decision_log
    (agent_name, decision_type, inputs_summary, decision, reasoning, confidence)
VALUES
    ('your_agent_name', 'classification_type',
     '{"key": "value"}'::jsonb,
     'your decision text',
     'why you decided this, citing specific data points',
     0.75);
```

**Reading decisions:**
```sql
SELECT agent_name, decision, reasoning, confidence, outcome_tag
FROM decision_log
WHERE agent_name = 'regime_analyst'
  AND created_at > NOW() - INTERVAL '4 hours'
ORDER BY created_at DESC;
```

## Layer 3: Lessons Store (pgvector similarity search)

Extracted learnings from past trade outcomes. Written by the CIO during review heartbeats.

**Querying lessons (at the start of every heartbeat):**

1. Build a text description of current conditions:
   `"regime: trending_up, macro: bullish, sentiment: 0.4, session: london_ny_overlap"`

2. Embed the text via text-embedding-3-small (1536 dimensions)

3. Search:
```sql
SELECT content, regime_tags, strategy_class, macro_context, confidence
FROM lessons
ORDER BY embedding <=> $context_embedding::vector
LIMIT 5;
```

**Writing a lesson (CIO only):**

1. Extract the lesson as structured text:
   `"Session open breakouts fail during NFP week pre-positioning. In 3 of 4 cases, the breakout reversed within 15 minutes of NFP release."`

2. Embed the text

3. Insert:
```sql
INSERT INTO lessons
    (content, embedding, regime_tags, strategy_class, macro_context, confidence, source_trades)
VALUES
    ($lesson_text, $embedding::vector,
     ARRAY['trending_up', 'pre_news'],
     'breakout', 'neutral', 0.8,
     ARRAY['trade_uuid_1', 'trade_uuid_2']::uuid[]);
```

## Every Heartbeat Must:

1. **Read**: Query top-5 similar lessons at the start
2. **Decide**: Make your classification/analysis/decision
3. **Log**: Write a decision_log entry with inputs, decision, reasoning, and confidence
4. **Learn**: (CIO only) Extract lessons from unexpected trade outcomes

## Confidence Scoring

- 0.9+: Very high confidence, multiple confirming signals
- 0.7-0.9: High confidence, primary signals aligned
- 0.5-0.7: Moderate, some conflicting signals
- 0.3-0.5: Low confidence, transition zone or mixed data
- <0.3: Very low, should probably not act
