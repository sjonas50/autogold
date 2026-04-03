"""Database queries for strategies table."""

import json as _json

import asyncpg

from gold_trading.models.strategy import Strategy


def _parse_strategy_row(row: asyncpg.Record) -> Strategy:
    """Parse a strategy row, handling JSONB backtest_params."""
    data = dict(row)
    if isinstance(data.get("backtest_params"), str):
        data["backtest_params"] = _json.loads(data["backtest_params"])
    return Strategy(**data)


async def upsert_strategy(conn: asyncpg.Connection, strategy: Strategy) -> str:
    """Insert or update a strategy."""
    import json

    await conn.execute(
        """
        INSERT INTO strategies
            (id, name, version, pine_script, instrument, timeframe, strategy_class,
             vbt_sharpe, vbt_win_rate, vbt_expectancy, vbt_max_drawdown,
             vbt_total_trades, vbt_profit_factor, vbt_avg_duration_min, backtest_params,
             mc_sharpe_p5, mc_sharpe_p50, status, is_active, cio_recommendation)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20)
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            version = EXCLUDED.version,
            pine_script = EXCLUDED.pine_script,
            vbt_sharpe = EXCLUDED.vbt_sharpe,
            vbt_win_rate = EXCLUDED.vbt_win_rate,
            vbt_expectancy = EXCLUDED.vbt_expectancy,
            vbt_max_drawdown = EXCLUDED.vbt_max_drawdown,
            vbt_total_trades = EXCLUDED.vbt_total_trades,
            vbt_profit_factor = EXCLUDED.vbt_profit_factor,
            vbt_avg_duration_min = EXCLUDED.vbt_avg_duration_min,
            backtest_params = EXCLUDED.backtest_params,
            mc_sharpe_p5 = EXCLUDED.mc_sharpe_p5,
            mc_sharpe_p50 = EXCLUDED.mc_sharpe_p50,
            status = EXCLUDED.status,
            is_active = EXCLUDED.is_active,
            cio_recommendation = EXCLUDED.cio_recommendation
        """,
        strategy.id,
        strategy.name,
        strategy.version,
        strategy.pine_script,
        strategy.instrument,
        strategy.timeframe,
        strategy.strategy_class,
        strategy.vbt_sharpe,
        strategy.vbt_win_rate,
        strategy.vbt_expectancy,
        strategy.vbt_max_drawdown,
        strategy.vbt_total_trades,
        strategy.vbt_profit_factor,
        strategy.vbt_avg_duration_min,
        json.dumps(strategy.backtest_params) if strategy.backtest_params else None,
        strategy.mc_sharpe_p5,
        strategy.mc_sharpe_p50,
        strategy.status,
        strategy.is_active,
        strategy.cio_recommendation,
    )
    return strategy.id


async def get_strategy(conn: asyncpg.Connection, strategy_id: str) -> Strategy | None:
    """Get a strategy by ID."""
    row = await conn.fetchrow("SELECT * FROM strategies WHERE id = $1", strategy_id)
    if row is None:
        return None
    return _parse_strategy_row(row)


async def get_active_strategies(conn: asyncpg.Connection) -> list[Strategy]:
    """Get all active strategies."""
    rows = await conn.fetch(
        "SELECT * FROM strategies WHERE is_active = true ORDER BY created_at DESC"
    )
    return [_parse_strategy_row(r) for r in rows]


async def get_strategies_by_status(conn: asyncpg.Connection, statuses: list[str]) -> list[Strategy]:
    """Get strategies matching any of the given statuses."""
    rows = await conn.fetch(
        "SELECT * FROM strategies WHERE status = ANY($1) ORDER BY created_at DESC",
        statuses,
    )
    return [_parse_strategy_row(r) for r in rows]


async def set_strategy_active(conn: asyncpg.Connection, strategy_id: str, active: bool) -> None:
    """Activate or deactivate a strategy."""
    await conn.execute(
        """
        UPDATE strategies SET is_active = $1,
            deployed_at = CASE WHEN $1 = true AND deployed_at IS NULL THEN NOW() ELSE deployed_at END
        WHERE id = $2
        """,
        active,
        strategy_id,
    )


async def set_cio_recommendation(
    conn: asyncpg.Connection, strategy_id: str, recommendation: str
) -> None:
    """Set the CIO's recommendation for a strategy."""
    await conn.execute(
        "UPDATE strategies SET cio_recommendation = $1 WHERE id = $2",
        recommendation,
        strategy_id,
    )


async def deactivate_all_strategies(conn: asyncpg.Connection) -> int:
    """Emergency: deactivate all strategies. Returns count affected."""
    result = await conn.execute("UPDATE strategies SET is_active = false WHERE is_active = true")
    return int(result.split()[-1])


async def is_strategy_active(conn: asyncpg.Connection, strategy_id: str) -> bool:
    """Check if a specific strategy is currently active."""
    row = await conn.fetchrow("SELECT is_active FROM strategies WHERE id = $1", strategy_id)
    if row is None:
        return False
    return row["is_active"]
