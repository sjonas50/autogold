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
import httpx
import pandas as pd
from loguru import logger

from gold_trading.backtest.engine import (
    StrategySignals,
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

## CIO Directive (if any — follow this guidance)
{context.get("cio_directive") or "No specific directive. Use your best judgment based on current regime and past results."}

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


async def check_manager_tasks(conn: asyncpg.Connection) -> str | None:
    """Check for tasks from Strategy Team Lead (Technical Analyst) or CIO.

    Returns the task description if found, else None.
    """
    from gold_trading.paperclip import get_my_tasks

    tasks = await get_my_tasks("quant_researcher")
    if tasks:
        task = tasks[0]
        logger.info(f"Found task from manager: {task.get('title', '')}")
        return task.get("description") or task.get("title", "")
    return None


async def _legacy_check_cio_tasks(conn: asyncpg.Connection) -> str | None:
    """Legacy: direct CIO task check (kept as fallback)."""
    quant_id = "8b85ccad-060e-4dd8-b007-4eced8098223"
    paperclip_url = os.environ.get("PAPERCLIP_URL", "http://localhost:3100")
    company_id = os.environ.get("PAPERCLIP_COMPANY_ID", "ea9cb59f-a7c4-4c1b-856f-173ea1d5bddf")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{paperclip_url}/api/companies/{company_id}/issues",
                params={
                    "assigneeAgentId": quant_id,
                    "status": "todo",
                },
            )
            if resp.status_code == 200:
                issues = resp.json()
                if issues:
                    # Take the highest priority task
                    task = issues[0]
                    logger.info(f"Found CIO task: {task.get('title', '')}")
                    return task.get("description") or task.get("title", "")
    except Exception as e:
        logger.debug(f"Could not check Paperclip tasks: {e}")

    return None


async def find_best_strategy_to_mutate(conn: asyncpg.Connection) -> dict | None:
    """Find the best-performing retired strategy to use as a mutation base.

    Returns dict with strategy details and params, or None if no candidates.
    """
    row = await conn.fetchrow(
        """
        SELECT id, strategy_class, vbt_sharpe, vbt_win_rate, backtest_params
        FROM strategies
        WHERE status = 'retired'
          AND vbt_sharpe IS NOT NULL
          AND backtest_params IS NOT NULL
        ORDER BY vbt_sharpe DESC
        LIMIT 1
        """
    )
    if row and row["backtest_params"]:
        import json as _json

        return {
            "id": row["id"],
            "strategy_class": row["strategy_class"],
            "sharpe": float(row["vbt_sharpe"]),
            "win_rate": float(row["vbt_win_rate"]) if row["vbt_win_rate"] else None,
            "params": _json.loads(row["backtest_params"]),
        }
    return None


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

        # Check for tasks from Strategy Team Lead (Technical Analyst) or CIO
        cio_directive = await check_manager_tasks(conn)
        if cio_directive:
            logger.info(f"Manager directive: {cio_directive}")

        # Gather context
        context = await gather_context(conn)
        context["cio_directive"] = cio_directive
        logger.info(f"Context: regime={context['regime']}, macro={context['macro_regime']}")

        # === TOURNAMENT-BASED STRATEGY DEVELOPMENT ===
        from gold_trading.backtest.dynamic_strategy import (
            STRATEGY_CODE_PROMPT,
            execute_generated_strategy,
        )
        from gold_trading.backtest.tournament import (
            format_leaderboard_for_prompt,
            get_leaderboard,
            should_explore,
        )

        # Get leaderboard and decide: explore (new) or exploit (refine)?
        leaderboard = await get_leaderboard(conn, limit=10)
        explore = should_explore(leaderboard)

        # Get Strategy Analyst guidance
        analyst_guidance = ""
        analyst_row = await conn.fetchrow(
            "SELECT decision, reasoning FROM decision_log "
            "WHERE agent_name = 'strategy_analyst' ORDER BY created_at DESC LIMIT 1"
        )
        if analyst_row:
            analyst_guidance = f"{analyst_row['decision']}\n{analyst_row['reasoning']}"

        # Strategy ID
        existing_count = await conn.fetchrow("SELECT COUNT(*) as cnt FROM strategies")
        strategy_num = (existing_count["cnt"] or 0) + 1

        context_str = (
            f"Regime: {context['regime']} (confidence: {context.get('regime_confidence', 0.5):.0%})\n"
            f"ADX: {context.get('adx_14', 'N/A')} | ATR: {context.get('atr_14', 'N/A')}\n"
            f"Macro: {context['macro_regime']}\n"
            f"CIO directive: {cio_directive or 'None'}"
        )

        leaderboard_str = format_leaderboard_for_prompt(leaderboard)

        if explore or not leaderboard:
            # === EXPLORE: generate a completely new strategy ===
            strategy_id = f"gs_v{strategy_num}_new"
            logger.info(f"MODE: EXPLORE — generating new strategy {strategy_id}")

            task_str = (
                f"Generate a NEW trading strategy for GC gold futures on 5-minute bars.\n"
                f"Market: {context['regime']}, ADX={context.get('adx_14', '?')}, macro={context['macro_regime']}.\n"
                f"Match the strategy to current conditions. Be creative — try different indicators.\n"
                f"Do NOT copy any existing strategy. Try a completely different approach."
            )
        else:
            # === EXPLOIT: refine the best existing strategy ===
            best = leaderboard[0]
            strategy_id = f"gs_v{strategy_num}_ref"
            logger.info(
                f"MODE: EXPLOIT — refining {best['id']} (fitness={best['fitness']:.3f}, "
                f"Sharpe={best['sharpe']:.2f}, WR={best['win_rate']:.0%})"
            )

            task_str = (
                f"REFINE this existing strategy that has fitness={best['fitness']:.3f}:\n"
                f"- Sharpe: {best['sharpe']:.2f} (target: >= 1.5)\n"
                f"- Win Rate: {best['win_rate']:.0%} (target: >= 50%)\n"
                f"- Profit Factor: {best.get('profit_factor', 0):.2f} (target: >= 1.3)\n"
                f"- Max Drawdown: {best.get('max_drawdown', 0):.2%} (target: < 5%)\n"
                f"- Trades: {best.get('trades', 0)}\n\n"
                f"Here is the current code:\n```python\n{best.get('code', '# No code')}\n```\n\n"
                f"Make SPECIFIC improvements:\n"
                f"- If win rate is low: tighten entry conditions, add confirmation filters\n"
                f"- If Sharpe is negative: adjust stop/target ratio, add trend filter\n"
                f"- If drawdown is high: reduce position frequency, add regime filter\n"
                f"- If profit factor < 1: improve R:R ratio, widen targets or tighten stops\n\n"
                f"Keep what works. Change what doesn't. Small improvements compound."
            )

        prompt = STRATEGY_CODE_PROMPT.format(
            context=context_str,
            guidance=analyst_guidance or "No specific guidance.",
            past_results=leaderboard_str,
            task=task_str,
        )

        # Call Claude
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.AsyncAnthropic(api_key=api_key)

        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )

        strategy_code = response.content[0].text.strip()
        if strategy_code.startswith("```"):
            lines = strategy_code.split("\n")
            strategy_code = "\n".join(line for line in lines if not line.strip().startswith("```"))

        mode = "EXPLORE" if explore or not leaderboard else "EXPLOIT"
        logger.info(f"[{mode}] Claude generated strategy code ({len(strategy_code)} chars)")

        # Execute the generated strategy against historical data
        signals = execute_generated_strategy(strategy_code, ohlcv)

        if signals is None:
            logger.error("Generated strategy failed to execute — retrying next heartbeat")
            await insert_decision(
                conn,
                DecisionLogEntry(
                    agent_name="quant_researcher",
                    decision_type="strategy_development",
                    inputs_summary={"regime": context["regime"], "code_length": len(strategy_code)},
                    decision=f"FAIL: {strategy_id} — execution error",
                    reasoning="Claude-generated strategy code failed to execute. Will retry.",
                    confidence=0.1,
                ),
            )
            return

        strategy_class = signals.direction + "_dynamic"
        logger.info(
            f"Strategy {strategy_id}: {signals.entries.sum()} entries, direction={signals.direction}"
        )

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

        # Calculate fitness for tournament ranking
        from gold_trading.backtest.tournament import calculate_fitness

        fitness = calculate_fitness(
            bt_result.sharpe_ratio,
            bt_result.win_rate,
            bt_result.profit_factor,
            bt_result.max_drawdown,
            bt_result.total_trades,
        )
        logger.info(f"[{mode}] Fitness: {fitness:+.3f}")

        # Save strategy with full metrics
        bt_params = {
            "type": "dynamic",
            "mode": mode.lower(),
            "direction": signals.direction,
            "code_length": len(strategy_code),
            "fitness": fitness,
            "refined_from": leaderboard[0]["id"] if mode == "EXPLOIT" and leaderboard else None,
        }
        strategy = Strategy(
            id=strategy_id,
            name=f"{mode} {context['regime']} #{strategy_num}",
            pine_script=strategy_code,  # Store the Python code (Pine Script generated later if it passes)
            instrument="GC",
            strategy_class=strategy_class,
            vbt_sharpe=bt_result.sharpe_ratio,
            vbt_win_rate=bt_result.win_rate,
            vbt_expectancy=bt_result.expectancy_usd,
            vbt_max_drawdown=bt_result.max_drawdown,
            vbt_total_trades=bt_result.total_trades,
            vbt_profit_factor=bt_result.profit_factor,
            vbt_avg_duration_min=bt_result.avg_trade_duration_minutes,
            backtest_params=bt_params,
            mc_sharpe_p5=mc_result.sharpe_p5 if mc_result else None,
            mc_sharpe_p50=mc_result.sharpe_p50 if mc_result else None,
            status="pending_deployment"
            if bt_result.passed_gate and (mc_result is None or mc_result.passed_gate)
            else "retired",
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
                    "code_length": len(strategy_code),
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

        # Save strategy code to file for review
        if bt_result.passed_gate:
            code_dir = os.path.join(os.path.dirname(__file__), "..", "pine", "generated")
            os.makedirs(code_dir, exist_ok=True)
            code_path = os.path.join(code_dir, f"{strategy_id}.py")
            with open(code_path, "w") as f:
                f.write(f"# Strategy: {strategy_id}\n")
                f.write(f"# Sharpe: {bt_result.sharpe_ratio}, WR: {bt_result.win_rate}\n")
                f.write(f"# Regime: {context['regime']}, Macro: {context['macro_regime']}\n\n")
                f.write(strategy_code)
            logger.info(f"Strategy code saved to {code_path}")

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
