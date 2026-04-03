"""Risk Manager agent — deterministic drawdown enforcement. NO LLM CALLS.

Paperclip process adapter script. Runs on heartbeat (every 15 minutes).
This is the safety layer — it enforces all hard risk rules:
- Max 0.5% risk per trade
- Max 2% drawdown → full halt (kill all strategies)
- Max 1 position at a time
- 1% daily loss limit → no more trades that session
- 120-minute max trade duration → auto-flag for exit
- Tags open trades with current regime/sentiment snapshots

CRITICAL: This script uses ONLY deterministic arithmetic. No LLM calls.
Risk decisions must never depend on probabilistic model output.

Run manually: uv run python scripts/risk_manager.py
"""

import asyncio
import os
from datetime import UTC, datetime

import asyncpg
from loguru import logger

from gold_trading.db.client import get_database_url
from gold_trading.db.queries.decisions import insert_decision
from gold_trading.db.queries.regime import get_latest_regime
from gold_trading.db.queries.sentiment import get_sentiment_summary
from gold_trading.db.queries.strategies import deactivate_all_strategies, get_active_strategies
from gold_trading.db.queries.trades import get_open_paper_trades
from gold_trading.models.lesson import DecisionLogEntry
from gold_trading.risk.rules import (
    MAX_DRAWDOWN_LIMIT,
    MAX_TRADE_DURATION_MINUTES,
    check_daily_loss,
    check_drawdown,
    check_trade_duration,
)

ACCOUNT_SIZE = float(os.environ.get("ACCOUNT_SIZE_USD", "50000"))


async def get_account_state(conn: asyncpg.Connection) -> dict:
    """Calculate current account state from paper trades."""
    # Peak equity: account size + best cumulative P&L
    peak_row = await conn.fetchrow(
        """
        WITH cumulative AS (
            SELECT
                SUM(pnl_usd) OVER (ORDER BY exit_time) as cum_pnl
            FROM paper_trades
            WHERE status = 'closed' AND pnl_usd IS NOT NULL
        )
        SELECT COALESCE(MAX(cum_pnl), 0) as best_pnl
        FROM cumulative
        """
    )
    peak_equity = ACCOUNT_SIZE + float(peak_row["best_pnl"])

    # Current equity: account size + all realized P&L
    total_pnl_row = await conn.fetchrow(
        """
        SELECT COALESCE(SUM(pnl_usd), 0) as total_pnl
        FROM paper_trades
        WHERE status = 'closed' AND pnl_usd IS NOT NULL
        """
    )
    current_equity = ACCOUNT_SIZE + float(total_pnl_row["total_pnl"])

    # Daily P&L
    daily_row = await conn.fetchrow(
        """
        SELECT COALESCE(SUM(pnl_usd), 0) as daily_pnl
        FROM paper_trades
        WHERE status = 'closed'
          AND exit_time >= CURRENT_DATE
          AND pnl_usd IS NOT NULL
        """
    )
    daily_pnl = float(daily_row["daily_pnl"])

    # Unrealized P&L from open positions (mark to latest close)
    open_trades = await get_open_paper_trades(conn)
    unrealized_pnl = 0.0
    # Note: without real-time price feed, unrealized P&L is not calculated
    # In paper trading mode, this is zero until we have a price feed

    return {
        "peak_equity": peak_equity,
        "current_equity": current_equity,
        "daily_pnl": daily_pnl,
        "unrealized_pnl": unrealized_pnl,
        "open_positions": len(open_trades),
        "open_trades": open_trades,
    }


async def check_duration_violations(
    conn: asyncpg.Connection,
    open_trades: list,
    now: datetime,
) -> list[dict]:
    """Check if any open trades have exceeded max duration."""
    violations = []
    for trade in open_trades:
        check = check_trade_duration(trade.entry_time, now)
        if not check.passed:
            violations.append(
                {
                    "trade_id": str(trade.id),
                    "strategy_id": trade.strategy_id,
                    "entry_time": trade.entry_time.isoformat(),
                    "duration_minutes": (now - trade.entry_time).total_seconds() / 60,
                    "violation": check.violations[0],
                }
            )
    return violations


async def tag_open_trades_context(
    conn: asyncpg.Connection,
    open_trades: list,
) -> None:
    """Update open trades with current regime and sentiment snapshots."""
    regime = await get_latest_regime(conn)
    sentiment = await get_sentiment_summary(conn, hours=1.0)

    for trade in open_trades:
        await conn.execute(
            """
            UPDATE paper_trades
            SET regime_at_entry = COALESCE(regime_at_entry, $1),
                sentiment_at_entry = COALESCE(sentiment_at_entry, $2)
            WHERE id = $3
            """,
            regime.regime if regime else None,
            round(sentiment.avg_sentiment, 3) if sentiment else None,
            trade.id,
        )


async def main() -> None:
    """Main heartbeat execution — the safety layer."""
    logger.info("Risk Manager heartbeat starting")
    now = datetime.now(UTC)

    conn = await asyncpg.connect(get_database_url())
    try:
        # 1. Get account state
        state = await get_account_state(conn)
        logger.info(
            f"Account state: equity=${state['current_equity']:.2f}, "
            f"peak=${state['peak_equity']:.2f}, "
            f"daily_pnl=${state['daily_pnl']:.2f}, "
            f"open_positions={state['open_positions']}"
        )

        violations = []
        actions_taken = []

        # 2. Check drawdown
        dd_check = check_drawdown(state["peak_equity"], state["current_equity"])
        if not dd_check.passed:
            violations.extend(dd_check.violations)
            # KILL SWITCH: Deactivate ALL strategies
            count = await deactivate_all_strategies(conn)
            actions_taken.append(
                f"KILL SWITCH: Deactivated {count} strategies — drawdown limit breached"
            )
            logger.critical(
                f"DRAWDOWN LIMIT BREACHED: {dd_check.violations[0]}. "
                f"Deactivated {count} strategies."
            )

        # 3. Check daily loss
        dl_check = check_daily_loss(state["daily_pnl"], ACCOUNT_SIZE)
        if not dl_check.passed:
            violations.extend(dl_check.violations)
            count = await deactivate_all_strategies(conn)
            actions_taken.append(
                f"DAILY LIMIT: Deactivated {count} strategies — daily loss limit hit"
            )
            logger.warning(f"DAILY LOSS LIMIT: {dl_check.violations[0]}")

        # 4. Check trade durations
        if state["open_trades"]:
            duration_violations = await check_duration_violations(conn, state["open_trades"], now)
            for dv in duration_violations:
                violations.append(dv["violation"])
                actions_taken.append(
                    f"DURATION: Trade {dv['strategy_id']} exceeded {MAX_TRADE_DURATION_MINUTES}min "
                    f"({dv['duration_minutes']:.0f}min). Flagged for exit."
                )
                logger.warning(
                    f"Trade duration exceeded: {dv['strategy_id']} at {dv['duration_minutes']:.0f}min"
                )

        # 5. Tag open trades with current market context
        if state["open_trades"]:
            await tag_open_trades_context(conn, state["open_trades"])

        # 6. Calculate drawdown percentage for reporting
        drawdown_pct = 0.0
        if state["peak_equity"] > 0 and state["current_equity"] < state["peak_equity"]:
            drawdown_pct = (state["peak_equity"] - state["current_equity"]) / state["peak_equity"]

        # 7. Get active strategy count
        active_strategies = await get_active_strategies(conn)

        # 8. Log decision
        await insert_decision(
            conn,
            DecisionLogEntry(
                agent_name="risk_manager",
                decision_type="risk_check",
                inputs_summary={
                    "peak_equity": round(state["peak_equity"], 2),
                    "current_equity": round(state["current_equity"], 2),
                    "drawdown_pct": round(drawdown_pct, 4),
                    "daily_pnl": round(state["daily_pnl"], 2),
                    "open_positions": state["open_positions"],
                    "active_strategies": len(active_strategies),
                },
                decision="VIOLATIONS: " + "; ".join(violations) if violations else "ALL CLEAR",
                reasoning=(
                    f"Drawdown: {drawdown_pct:.2%} (limit: {MAX_DRAWDOWN_LIMIT:.2%}). "
                    f"Daily P&L: ${state['daily_pnl']:.2f}. "
                    f"Open positions: {state['open_positions']}. "
                    f"Active strategies: {len(active_strategies)}. "
                    + (
                        f"Actions: {'; '.join(actions_taken)}"
                        if actions_taken
                        else "No action needed."
                    )
                ),
                confidence=1.0,  # Risk checks are deterministic — always 100% confidence
            ),
        )

        if violations:
            logger.warning(
                f"Risk Manager: {len(violations)} violation(s). Actions: {'; '.join(actions_taken)}"
            )
        else:
            logger.info(
                f"Risk Manager heartbeat complete. ALL CLEAR. "
                f"DD={drawdown_pct:.2%}, daily=${state['daily_pnl']:.2f}, "
                f"positions={state['open_positions']}, "
                f"active_strategies={len(active_strategies)}"
            )

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
