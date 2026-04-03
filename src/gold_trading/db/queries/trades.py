"""Database queries for trade_journal and paper_trades tables."""

from datetime import datetime
from uuid import UUID

import asyncpg

from gold_trading.models.trade import PaperTrade, TradeJournalEntry


async def insert_trade_journal(conn: asyncpg.Connection, trade: TradeJournalEntry) -> UUID:
    """Insert a completed trade into trade_journal."""
    row = await conn.fetchrow(
        """
        INSERT INTO trade_journal
            (strategy_id, instrument, direction, contracts, entry_price, exit_price,
             entry_time, closed_at, pnl_usd, r_multiple, max_adverse_exc,
             regime_at_entry, sentiment_score, macro_bias, session, atr_at_entry, notes)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)
        RETURNING id
        """,
        trade.strategy_id,
        trade.instrument,
        trade.direction,
        trade.contracts,
        trade.entry_price,
        trade.exit_price,
        trade.entry_time,
        trade.closed_at or datetime.now(),
        trade.pnl_usd,
        trade.r_multiple,
        trade.max_adverse_exc,
        trade.regime_at_entry,
        trade.sentiment_score,
        trade.macro_bias,
        trade.session,
        trade.atr_at_entry,
        trade.notes,
    )
    return row["id"]


async def get_recent_trades(
    conn: asyncpg.Connection, hours: int = 24, limit: int = 50
) -> list[TradeJournalEntry]:
    """Get trades closed in the last N hours."""
    rows = await conn.fetch(
        """
        SELECT * FROM trade_journal
        WHERE closed_at > NOW() - make_interval(hours => $1)
        ORDER BY closed_at DESC
        LIMIT $2
        """,
        hours,
        limit,
    )
    return [TradeJournalEntry(**dict(r)) for r in rows]


async def insert_paper_trade(conn: asyncpg.Connection, trade: PaperTrade) -> UUID:
    """Insert a new paper trade."""
    row = await conn.fetchrow(
        """
        INSERT INTO paper_trades
            (strategy_id, instrument, direction, contracts, entry_price,
             entry_time, status, stop_loss_price, take_profit_price,
             regime_at_entry, sentiment_at_entry, macro_at_entry, idempotency_key)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
        RETURNING id
        """,
        trade.strategy_id,
        trade.instrument,
        trade.direction,
        trade.contracts,
        trade.entry_price,
        trade.entry_time,
        trade.status,
        trade.stop_loss_price,
        trade.take_profit_price,
        trade.regime_at_entry,
        trade.sentiment_at_entry,
        trade.macro_at_entry,
        trade.idempotency_key,
    )
    return row["id"]


async def get_open_paper_trades(conn: asyncpg.Connection) -> list[PaperTrade]:
    """Get all currently open paper trades."""
    rows = await conn.fetch(
        "SELECT * FROM paper_trades WHERE status = 'open' ORDER BY entry_time DESC"
    )
    return [PaperTrade(**dict(r)) for r in rows]


async def close_paper_trade(
    conn: asyncpg.Connection,
    trade_id: UUID,
    exit_price: float,
    exit_time: datetime,
    pnl_usd: float,
    r_multiple: float | None = None,
    status: str = "closed",
) -> None:
    """Close a paper trade with exit details."""
    await conn.execute(
        """
        UPDATE paper_trades
        SET exit_price = $1, exit_time = $2, pnl_usd = $3,
            r_multiple = $4, status = $5
        WHERE id = $6
        """,
        exit_price,
        exit_time,
        pnl_usd,
        r_multiple,
        status,
        trade_id,
    )


async def check_idempotency(conn: asyncpg.Connection, key: str) -> bool:
    """Check if an idempotency key already exists. Returns True if duplicate."""
    row = await conn.fetchrow("SELECT 1 FROM paper_trades WHERE idempotency_key = $1", key)
    return row is not None


async def get_open_position_count(conn: asyncpg.Connection) -> int:
    """Get count of currently open positions."""
    row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM paper_trades WHERE status = 'open'")
    return row["cnt"]
