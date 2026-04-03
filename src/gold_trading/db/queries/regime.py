"""Database queries for regime_state table."""

from uuid import UUID

import asyncpg

from gold_trading.models.regime import RegimeState


async def insert_regime_state(conn: asyncpg.Connection, state: RegimeState) -> UUID:
    """Insert a new regime classification."""
    row = await conn.fetchrow(
        """
        INSERT INTO regime_state
            (regime, hmm_state, hmm_confidence, atr_14, adx_14,
             timeframe, dxy_change_1d)
        VALUES ($1,$2,$3,$4,$5,$6,$7)
        RETURNING id
        """,
        state.regime,
        state.hmm_state,
        state.hmm_confidence,
        state.atr_14,
        state.adx_14,
        state.timeframe,
        state.dxy_change_1d,
    )
    return row["id"]


async def get_latest_regime(conn: asyncpg.Connection) -> RegimeState | None:
    """Get the most recent regime classification."""
    row = await conn.fetchrow("SELECT * FROM regime_state ORDER BY created_at DESC LIMIT 1")
    if row is None:
        return None
    return RegimeState(**dict(row))


async def get_regime_history(
    conn: asyncpg.Connection, hours: int = 24, limit: int = 48
) -> list[RegimeState]:
    """Get regime history for the last N hours."""
    rows = await conn.fetch(
        """
        SELECT * FROM regime_state
        WHERE created_at > NOW() - make_interval(hours => $1)
        ORDER BY created_at DESC LIMIT $2
        """,
        hours,
        limit,
    )
    return [RegimeState(**dict(r)) for r in rows]
