"""Database queries for macro_data table."""

from uuid import UUID

import asyncpg

from gold_trading.models.macro import MacroData


async def insert_macro_data(conn: asyncpg.Connection, data: MacroData) -> UUID:
    """Insert a macro data snapshot."""
    row = await conn.fetchrow(
        """
        INSERT INTO macro_data
            (observation_date, dxy, real_yield_10y, cpi_yoy,
             breakeven_10y, oil_wti, gold_fix_pm, macro_regime, reasoning)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
        RETURNING id
        """,
        data.observation_date,
        data.dxy,
        data.real_yield_10y,
        data.cpi_yoy,
        data.breakeven_10y,
        data.oil_wti,
        data.gold_fix_pm,
        data.macro_regime,
        data.reasoning,
    )
    return row["id"]


async def get_latest_macro(conn: asyncpg.Connection) -> MacroData | None:
    """Get the most recent macro data snapshot."""
    row = await conn.fetchrow("SELECT * FROM macro_data ORDER BY created_at DESC LIMIT 1")
    if row is None:
        return None
    return MacroData(**dict(row))


async def get_macro_history(
    conn: asyncpg.Connection, days: int = 30, limit: int = 30
) -> list[MacroData]:
    """Get macro data history."""
    rows = await conn.fetch(
        """
        SELECT * FROM macro_data
        WHERE observation_date > CURRENT_DATE - $1
        ORDER BY observation_date DESC LIMIT $2
        """,
        days,
        limit,
    )
    return [MacroData(**dict(r)) for r in rows]
