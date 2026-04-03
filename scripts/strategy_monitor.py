"""Strategy Monitor agent — tracks live performance vs backtest expectations.

Paperclip process adapter script. Runs on heartbeat (every 30 minutes).
Compares paper trade results to backtest metrics for active strategies.
Creates Paperclip alerts when performance degrades below thresholds.

Run manually: uv run python scripts/strategy_monitor.py
"""

import asyncio
import os

import asyncpg
import httpx
from loguru import logger

from gold_trading.db.client import get_database_url
from gold_trading.db.queries.decisions import insert_decision
from gold_trading.db.queries.strategies import get_active_strategies, get_strategies_by_status
from gold_trading.models.lesson import DecisionLogEntry

PAPERCLIP_URL = os.environ.get("PAPERCLIP_URL", "http://localhost:3100")
COMPANY_ID = os.environ.get("PAPERCLIP_COMPANY_ID", "3422f81a-8ca2-4ce1-aae5-5cf8ce34fa0e")
CIO_AGENT_ID = "37bbe408-e573-4598-a374-cc369bad0258"

# Degradation thresholds
MIN_TRADES_FOR_EVALUATION = 10
WIN_RATE_DEGRADATION_THRESHOLD = 0.15  # Alert if WR drops >15% vs backtest
SHARPE_DEGRADATION_THRESHOLD = 0.50  # Alert if Sharpe drops >50%


async def get_strategy_live_performance(conn: asyncpg.Connection, strategy_id: str) -> dict | None:
    """Calculate live paper trading performance for a strategy."""
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) as total_trades,
            COUNT(*) FILTER (WHERE pnl_usd > 0) as wins,
            COALESCE(SUM(pnl_usd), 0) as total_pnl,
            COALESCE(AVG(pnl_usd), 0) as avg_pnl,
            COALESCE(AVG(r_multiple), 0) as avg_r
        FROM trade_journal
        WHERE strategy_id = $1
        """,
        strategy_id,
    )
    if not row or row["total_trades"] == 0:
        return None

    total = row["total_trades"]
    wins = row["wins"]
    return {
        "total_trades": total,
        "win_rate": wins / total if total > 0 else 0,
        "total_pnl": float(row["total_pnl"]),
        "avg_pnl": float(row["avg_pnl"]),
        "avg_r": float(row["avg_r"]),
    }


async def create_degradation_alert(
    strategy_id: str, metric: str, backtest_val: float, live_val: float
) -> None:
    """Alert CIO about strategy performance degradation."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{PAPERCLIP_URL}/api/companies/{COMPANY_ID}/issues",
                json={
                    "title": f"[DEGRADATION] {strategy_id} — {metric} declining",
                    "description": (
                        f"Strategy **{strategy_id}** is underperforming vs backtest:\n\n"
                        f"- **{metric}** backtest: {backtest_val:.3f}\n"
                        f"- **{metric}** live: {live_val:.3f}\n"
                        f"- Degradation: {abs(live_val - backtest_val) / max(abs(backtest_val), 0.001):.0%}\n\n"
                        f"**Recommendation:** Consider deactivating this strategy or "
                        f"requesting the Quant Researcher to re-optimize."
                    ),
                    "assigneeAgentId": CIO_AGENT_ID,
                    "priority": "high",
                },
            )
            logger.info(f"Created degradation alert for {strategy_id}: {metric}")
    except Exception as e:
        logger.warning(f"Could not create Paperclip task: {e}")


async def main() -> None:
    """Main heartbeat execution."""
    logger.info("Strategy Monitor heartbeat starting")

    conn = await asyncpg.connect(get_database_url())
    try:
        # Check all active + pending strategies
        strategies = await get_active_strategies(conn)
        strategies.extend(await get_strategies_by_status(conn, ["pending_deployment"]))

        if not strategies:
            logger.info("No active or pending strategies to monitor")
            await insert_decision(
                conn,
                DecisionLogEntry(
                    agent_name="strategy_monitor",
                    decision_type="performance_check",
                    inputs_summary={"strategies_checked": 0},
                    decision="No strategies to monitor",
                    confidence=1.0,
                ),
            )
            return

        alerts = []
        for strategy in strategies:
            live = await get_strategy_live_performance(conn, strategy.id)

            if not live or live["total_trades"] < MIN_TRADES_FOR_EVALUATION:
                logger.debug(
                    f"{strategy.id}: {live['total_trades'] if live else 0} trades "
                    f"(need {MIN_TRADES_FOR_EVALUATION} for evaluation)"
                )
                continue

            # Compare win rate
            bt_wr = strategy.vbt_win_rate or 0
            live_wr = live["win_rate"]
            if bt_wr > 0 and (bt_wr - live_wr) > WIN_RATE_DEGRADATION_THRESHOLD:
                alerts.append(f"{strategy.id}: WR dropped {bt_wr:.1%} → {live_wr:.1%}")
                await create_degradation_alert(strategy.id, "win_rate", bt_wr, live_wr)

            # Compare via expected profitability
            if live["avg_pnl"] < 0 and (strategy.vbt_expectancy or 0) > 0:
                alerts.append(
                    f"{strategy.id}: Expected +${strategy.vbt_expectancy:.0f}/trade, "
                    f"actual ${live['avg_pnl']:.0f}/trade"
                )
                await create_degradation_alert(
                    strategy.id,
                    "expectancy",
                    strategy.vbt_expectancy or 0,
                    live["avg_pnl"],
                )

            logger.info(
                f"{strategy.id}: {live['total_trades']} trades, "
                f"WR={live['win_rate']:.1%} (bt: {bt_wr:.1%}), "
                f"PnL=${live['total_pnl']:.0f}"
            )

        await insert_decision(
            conn,
            DecisionLogEntry(
                agent_name="strategy_monitor",
                decision_type="performance_check",
                inputs_summary={
                    "strategies_checked": len(strategies),
                    "alerts": len(alerts),
                },
                decision=f"{len(alerts)} alerts" if alerts else "All strategies within tolerance",
                reasoning="; ".join(alerts) if alerts else "No degradation detected.",
                confidence=1.0,
            ),
        )

        logger.info(f"Strategy Monitor complete. {len(strategies)} checked, {len(alerts)} alerts.")

    finally:
        await conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
