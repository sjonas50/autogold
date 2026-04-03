"""Paper trade fill simulation — simulates fills at close + slippage."""

from datetime import UTC, datetime

import asyncpg
from loguru import logger

from gold_trading.db.queries.macro import get_latest_macro
from gold_trading.db.queries.regime import get_latest_regime
from gold_trading.db.queries.sentiment import get_sentiment_summary
from gold_trading.db.queries.trades import (
    close_paper_trade,
    get_open_paper_trades,
    insert_paper_trade,
    insert_trade_journal,
)
from gold_trading.models.signal import WebhookPayload
from gold_trading.models.trade import PaperTrade, TradeJournalEntry
from gold_trading.risk.calculator import apply_slippage, calculate_pnl, calculate_r_multiple


async def simulate_entry(
    conn: asyncpg.Connection,
    payload: WebhookPayload,
    idempotency_key: str,
) -> PaperTrade:
    """Simulate a trade entry — apply slippage, capture market context, write to DB."""
    direction = "long" if payload.action == "buy" else "short"
    fill_price = apply_slippage(payload.price, direction)

    # Capture current market context
    regime = await get_latest_regime(conn)
    sentiment = await get_sentiment_summary(conn, hours=4.0)
    macro = await get_latest_macro(conn)

    trade = PaperTrade(
        strategy_id=payload.strategy_id,
        instrument=payload.instrument,
        direction=direction,
        contracts=payload.contracts,
        entry_price=round(fill_price, 2),
        entry_time=datetime.now(UTC),
        stop_loss_price=payload.stop_loss,
        take_profit_price=payload.take_profit,
        regime_at_entry=regime.regime if regime else None,
        sentiment_at_entry=round(sentiment.avg_sentiment, 3) if sentiment else None,
        macro_at_entry=macro.macro_regime if macro else None,
        idempotency_key=idempotency_key,
    )

    trade_id = await insert_paper_trade(conn, trade)
    trade.id = trade_id

    logger.info(
        f"Paper trade opened: {direction} {payload.contracts}x {payload.instrument} "
        f"@ {fill_price:.2f} (strategy={payload.strategy_id})"
    )
    return trade


async def simulate_exit(
    conn: asyncpg.Connection,
    payload: WebhookPayload,
    idempotency_key: str,
) -> TradeJournalEntry | None:
    """Simulate a trade exit — find the matching open trade, close it, write to journal."""
    open_trades = await get_open_paper_trades(conn)

    # Find the open trade for this strategy
    matching = [t for t in open_trades if t.strategy_id == payload.strategy_id]
    if not matching:
        logger.warning(
            f"Exit signal for {payload.strategy_id} but no open trade found. "
            "Signal may be stale or duplicate."
        )
        return None

    trade = matching[0]

    # Apply slippage in the exit direction (opposite of entry)
    exit_direction = "short" if trade.direction == "long" else "long"
    exit_price = apply_slippage(payload.price, exit_direction)
    exit_time = datetime.now(UTC)

    pnl = calculate_pnl(
        entry_price=float(trade.entry_price),
        exit_price=exit_price,
        contracts=trade.contracts,
        direction=trade.direction,
        instrument=trade.instrument,
    )

    r_mult = None
    if trade.stop_loss_price:
        r_mult = calculate_r_multiple(
            entry_price=float(trade.entry_price),
            exit_price=exit_price,
            stop_price=float(trade.stop_loss_price),
            direction=trade.direction,
        )

    # Close the paper trade
    await close_paper_trade(
        conn,
        trade_id=trade.id,
        exit_price=round(exit_price, 2),
        exit_time=exit_time,
        pnl_usd=round(pnl, 2),
        r_multiple=round(r_mult, 3) if r_mult is not None else None,
    )

    # Write to trade journal with full context
    journal_entry = TradeJournalEntry(
        strategy_id=trade.strategy_id,
        instrument=trade.instrument,
        direction=trade.direction,
        contracts=trade.contracts,
        entry_price=float(trade.entry_price),
        exit_price=round(exit_price, 2),
        entry_time=trade.entry_time,
        closed_at=exit_time,
        pnl_usd=round(pnl, 2),
        r_multiple=round(r_mult, 3) if r_mult is not None else None,
        regime_at_entry=trade.regime_at_entry,
        sentiment_score=float(trade.sentiment_at_entry) if trade.sentiment_at_entry else None,
        macro_bias=trade.macro_at_entry,
    )
    await insert_trade_journal(conn, journal_entry)

    r_str = f" | R: {r_mult:.2f}R" if r_mult else ""
    logger.info(
        f"Paper trade closed: {trade.direction} {trade.contracts}x {trade.instrument} "
        f"@ {exit_price:.2f} | PnL: ${pnl:.2f}{r_str}"
    )
    return journal_entry
