"""Quant Researcher agent — generates Pine Script strategies via RAG, backtests with vectorbt.

Paperclip process adapter script. Runs on heartbeat (every 8 hours).
Queries lessons + regime + RAG corpus, calls Claude Sonnet 4.6 to generate Pine Script,
runs vectorbt backtest + Monte Carlo gate, writes passing strategies to strategies table.

Run manually: uv run python scripts/quant_researcher.py
"""

import asyncio
import json
import os

import anthropic
import asyncpg
import pandas as pd
from loguru import logger

from gold_trading.backtest.engine import (
    StrategySignals,
    generate_breakout_signals,
    generate_mean_reversion_signals,
    run_backtest,
)
from gold_trading.backtest.montecarlo import run_monte_carlo
from gold_trading.db.client import get_database_url
from gold_trading.db.queries.decisions import insert_decision
from gold_trading.db.queries.lessons import search_similar_lessons
from gold_trading.db.queries.macro import get_latest_macro
from gold_trading.db.queries.regime import get_latest_regime
from gold_trading.db.queries.strategies import get_strategies_by_status, upsert_strategy
from gold_trading.embeddings.client import embed_text
from gold_trading.embeddings.corpus import search_corpus
from gold_trading.models.lesson import DecisionLogEntry
from gold_trading.models.strategy import Strategy

ANTHROPIC_MODEL = "claude-sonnet-4-6"


def _extract_trade_pnls(
    ohlcv: pd.DataFrame,
    signals: "StrategySignals",
    instrument: str = "GC",
) -> list[float]:
    """Extract actual per-trade P&Ls from a vectorbt backtest.

    Returns list of P&L values in USD, preserving autocorrelation and
    tail distribution for realistic Monte Carlo simulation.
    """
    import vectorbt as vbt

    from gold_trading.backtest.engine import COMMISSION_PER_CONTRACT, SLIPPAGE_PCT

    close = ohlcv["close"].astype(float)
    multiplier = 100.0 if instrument == "GC" else 10.0

    pf = vbt.Portfolio.from_signals(
        close,
        entries=signals.entries,
        exits=signals.exits,
        size=1.0,
        size_type="amount",
        init_cash=50_000.0,
        fees=COMMISSION_PER_CONTRACT / (close.mean() * multiplier),
        slippage=SLIPPAGE_PCT,
        freq="5min",
    )

    trades = pf.trades.records_readable
    if trades is None or len(trades) == 0:
        return []

    return [float(pnl) for pnl in trades["PnL"]]


async def load_ohlcv_data(conn: asyncpg.Connection, bars: int = 25000) -> pd.DataFrame | None:
    """Load 5m OHLCV data for backtesting."""
    rows = await conn.fetch(
        """
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv_5m
        WHERE instrument = 'GC'
        ORDER BY timestamp DESC
        LIMIT $1
        """,
        bars,
    )
    if not rows:
        return None

    df = pd.DataFrame([dict(r) for r in rows])
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(int)
    return df


async def gather_context(conn: asyncpg.Connection) -> dict:
    """Gather current market context for strategy generation."""
    regime = await get_latest_regime(conn)
    macro = await get_latest_macro(conn)

    # Get relevant lessons
    context_text = f"regime: {regime.regime if regime else 'unknown'}, macro: {macro.macro_regime if macro else 'unknown'}"
    try:
        context_embedding = await embed_text(context_text)
        lessons = await search_similar_lessons(conn, context_embedding, limit=5)
    except Exception as e:
        logger.warning(f"Could not query lessons: {e}")
        lessons = []

    # Get ALL strategies (including retired) to learn from failures
    existing = await get_strategies_by_status(conn, ["active", "pending_deployment", "paused"])
    all_strategies = await conn.fetch(
        """
        SELECT id, strategy_class, vbt_sharpe, vbt_win_rate, vbt_max_drawdown, status
        FROM strategies ORDER BY created_at DESC LIMIT 10
        """
    )
    past_results = [
        f"{r['id']}: {r['strategy_class']}, Sharpe={r['vbt_sharpe']}, WR={r['vbt_win_rate']}, status={r['status']}"
        for r in all_strategies
    ]

    return {
        "regime": regime.regime if regime else "unknown",
        "regime_confidence": float(regime.hmm_confidence)
        if regime and regime.hmm_confidence
        else 0.5,
        "atr_14": float(regime.atr_14) if regime and regime.atr_14 else None,
        "adx_14": float(regime.adx_14) if regime and regime.adx_14 else None,
        "macro_regime": macro.macro_regime if macro else "unknown",
        "lessons": [lesson.content for lesson in lessons],
        "existing_strategies": [s.id for s in existing],
        "past_results": past_results,
    }


async def generate_pine_script(context: dict, corpus_chunks: list[dict]) -> dict | None:
    """Use Claude Sonnet 4.6 + RAG to generate a Pine Script strategy.

    Returns dict with 'pine_script', 'name', 'id', 'strategy_class', 'description'.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Build RAG context from corpus chunks
    rag_context = "\n\n---\n\n".join(
        f"[{c['source_file']} ({c['chunk_type']})]:\n{c['content']}" for c in corpus_chunks[:8]
    )

    # Build lessons context
    lessons_text = "\n".join(f"- {item}" for item in context.get("lessons", []))

    # Determine which strategy class to focus on
    regime = context.get("regime", "unknown")
    if regime in ("trending_up", "trending_down"):
        strategy_focus = "session open breakout"
    elif regime == "ranging":
        strategy_focus = "VWAP/key level mean reversion"
    else:
        strategy_focus = "conservative mean reversion with tight stops"

    prompt = f"""You are a quantitative Pine Script v6 developer for gold futures (GC/MGC).

## Current Market Context
- Regime: {context.get("regime")} (confidence: {context.get("regime_confidence", 0.5):.0%})
- ATR(14): {context.get("atr_14", "N/A")}
- ADX(14): {context.get("adx_14", "N/A")}
- Macro regime: {context.get("macro_regime", "unknown")}
- Strategy focus: {strategy_focus}

## Lessons from Past Trades
{lessons_text or "No lessons yet."}

## Existing Strategies (avoid duplicating)
{", ".join(context.get("existing_strategies", [])) or "None"}

## Past Strategy Results (learn from failures — try a DIFFERENT approach)
{chr(10).join(context.get("past_results", [])) or "No past strategies yet."}

## Pine Script v6 Reference
{rag_context}

## Requirements
Write a Pine Script v6 strategy for GC gold futures on the 5-minute timeframe that:
1. Uses `process_orders_on_close=true` in the strategy() declaration
2. Calculates position size based on ATR for stop distance
3. Has a max trade duration of 120 minutes (auto-exit)
4. Includes clear entry and exit conditions
5. Adds alert conditions for TradingView webhooks
6. Is optimized for the current regime ({regime})

## Output Format
Respond with EXACTLY this structure — metadata as JSON, then Pine Script in a separate fenced block:

METADATA:
```json
{{"id": "gs_v1_breakout", "name": "Strategy Name", "strategy_class": "breakout", "description": "2-3 sentences."}}
```

PINESCRIPT:
```pinescript
// Full Pine Script v6 code here
```"""

    response = await client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Parse metadata JSON block
    metadata = None
    pine_script = None

    # Extract JSON block
    import re

    json_match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    if json_match:
        try:
            metadata = json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            logger.error(f"Failed to parse metadata JSON: {json_match.group(1)[:200]}")

    # Extract Pine Script block
    pine_match = re.search(r"```pinescript\s*\n(.*?)\n```", text, re.DOTALL)
    if not pine_match:
        # Fallback: try any fenced block that looks like Pine Script
        pine_match = re.search(r"```(?:pine)?\s*\n(.*?//@version.*?)\n```", text, re.DOTALL)
    if pine_match:
        pine_script = pine_match.group(1).strip()

    if not metadata or not pine_script:
        # Last resort: try parsing entire response as JSON (in case Claude ignored the format)
        try:
            parsed = json.loads(text)
            return parsed
        except json.JSONDecodeError:
            pass
        logger.error(
            f"Failed to parse generation response. "
            f"Has metadata: {metadata is not None}, has pine: {pine_script is not None}. "
            f"Response start: {text[:200]}"
        )
        return None

    metadata["pine_script"] = pine_script
    return metadata


async def main() -> None:
    """Main heartbeat execution."""
    logger.info("Quant Researcher heartbeat starting")

    conn = await asyncpg.connect(get_database_url())
    try:
        # Load OHLCV data for backtesting
        ohlcv = await load_ohlcv_data(conn)
        if ohlcv is None or len(ohlcv) < 500:
            logger.warning(
                f"Insufficient OHLCV data ({len(ohlcv) if ohlcv is not None else 0} bars). "
                "Need at least 500 bars for meaningful backtesting. "
                "Ingest historical data first."
            )
            # Still log a decision
            await insert_decision(
                conn,
                DecisionLogEntry(
                    agent_name="quant_researcher",
                    decision_type="strategy_development",
                    inputs_summary={"bars_available": len(ohlcv) if ohlcv is not None else 0},
                    decision="Skipped — insufficient data",
                    reasoning="Need at least 500 bars of 5m GC data for backtesting.",
                    confidence=0.0,
                ),
            )
            return

        # Gather context
        context = await gather_context(conn)
        logger.info(f"Context: regime={context['regime']}, macro={context['macro_regime']}")

        # Search Pine Script corpus for relevant code
        try:
            regime = context.get("regime", "breakout")
            query_text = f"Pine Script strategy for gold {regime} {context.get('macro_regime', '')}"
            query_embedding = await embed_text(query_text)
            corpus_chunks = await search_corpus(query_embedding, limit=8)
        except Exception as e:
            logger.warning(f"Corpus search failed: {e}. Proceeding without RAG context.")
            corpus_chunks = []

        # Generate Pine Script
        generated = await generate_pine_script(context, corpus_chunks)
        if not generated:
            logger.error("Pine Script generation failed")
            return

        # Generate unique strategy ID (append attempt number)
        base_id = generated.get("id", f"gs_auto_{context['regime']}")
        existing_ids = context.get("existing_strategies", [])
        attempt = sum(1 for eid in existing_ids if eid.startswith(base_id.rsplit("_a", 1)[0])) + 1
        strategy_id = f"{base_id}_a{attempt}" if attempt > 1 else base_id
        pine_script = generated.get("pine_script", "")
        strategy_class = generated.get("strategy_class", "breakout")

        logger.info(f"Generated strategy: {strategy_id} ({strategy_class}), attempt #{attempt}")

        # Vary signal parameters based on attempt number to explore the parameter space
        # Each attempt uses different lookback/threshold combos
        import random

        rng = random.Random(attempt + hash(strategy_id))

        if strategy_class in ("breakout", "momentum"):
            lookback = rng.choice([6, 9, 12, 18, 24])
            atr_mult = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])
            signals = generate_breakout_signals(ohlcv, lookback=lookback, atr_multiplier=atr_mult)
            logger.info(f"Breakout params: lookback={lookback}, atr_mult={atr_mult}")
        else:
            vwap_period = rng.choice([24, 36, 48, 72, 96])
            entry_thresh = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])
            signals = generate_mean_reversion_signals(
                ohlcv, vwap_period=vwap_period, entry_threshold=entry_thresh
            )
            logger.info(f"MeanRev params: vwap_period={vwap_period}, entry_thresh={entry_thresh}")

        bt_result = run_backtest(ohlcv, signals, instrument="GC")
        bt_result.strategy_id = strategy_id

        # Run Monte Carlo if backtest has enough trades
        mc_result = None
        if bt_result.total_trades >= 20:
            # Extract actual trade P&Ls from the vectorbt backtest
            trade_pnls = _extract_trade_pnls(ohlcv, signals, instrument="GC")

            if len(trade_pnls) >= 10:
                mc_result = run_monte_carlo(
                    trade_pnls,
                    strategy_id=strategy_id,
                    n_iterations=1000,
                )

        # Save strategy
        strategy = Strategy(
            id=strategy_id,
            name=generated.get("name", strategy_id),
            pine_script=pine_script,
            instrument="GC",
            strategy_class=strategy_class,
            vbt_sharpe=bt_result.sharpe_ratio,
            vbt_win_rate=bt_result.win_rate,
            vbt_expectancy=bt_result.expectancy_usd,
            vbt_max_drawdown=bt_result.max_drawdown,
            mc_sharpe_p5=mc_result.sharpe_p5 if mc_result else None,
            mc_sharpe_p50=mc_result.sharpe_p50 if mc_result else None,
            status="pending_deployment" if bt_result.passed_gate else "retired",
        )
        await upsert_strategy(conn, strategy)

        # Log decision
        await insert_decision(
            conn,
            DecisionLogEntry(
                agent_name="quant_researcher",
                decision_type="strategy_development",
                inputs_summary={
                    "regime": context["regime"],
                    "macro": context["macro_regime"],
                    "strategy_class": strategy_class,
                    "corpus_chunks_used": len(corpus_chunks),
                    "lessons_used": len(context.get("lessons", [])),
                },
                decision=f"{'PASS' if bt_result.passed_gate else 'FAIL'}: {strategy_id}",
                reasoning=(
                    f"Sharpe={bt_result.sharpe_ratio:.2f}, WR={bt_result.win_rate:.1%}, "
                    f"DD={bt_result.max_drawdown:.1%}, Trades={bt_result.total_trades}. "
                    f"{'Monte Carlo: Sharpe_p5=' + str(mc_result.sharpe_p5) if mc_result else 'No MC (insufficient trades)'}."
                ),
                confidence=0.7 if bt_result.passed_gate else 0.3,
            ),
        )

        # Write Pine Script to file for human deployment
        if bt_result.passed_gate and pine_script:
            pine_dir = os.path.join(os.path.dirname(__file__), "..", "pine", "generated")
            os.makedirs(pine_dir, exist_ok=True)
            pine_path = os.path.join(pine_dir, f"{strategy_id}.pine")
            with open(pine_path, "w") as f:
                f.write(pine_script)
            logger.info(f"Pine Script written to {pine_path}")

        logger.info(
            f"Quant Researcher heartbeat complete. "
            f"Strategy {strategy_id}: {'PASS' if bt_result.passed_gate else 'FAIL'}"
        )

    finally:
        await conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
