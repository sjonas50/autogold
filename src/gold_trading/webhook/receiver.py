"""FastAPI webhook receiver — POST /webhook/signal endpoint."""

import os

from fastapi import APIRouter

from gold_trading.db.client import get_pool
from gold_trading.models.signal import SignalResponse, WebhookPayload
from gold_trading.webhook.simulator import simulate_entry, simulate_exit
from gold_trading.webhook.validator import validate_signal

router = APIRouter(prefix="/webhook", tags=["webhook"])

ACCOUNT_SIZE = float(os.environ.get("ACCOUNT_SIZE_USD", "50000"))


@router.post("/signal", response_model=SignalResponse)
async def receive_signal(payload: WebhookPayload) -> SignalResponse:
    """Receive a TradingView strategy alert webhook.

    Flow:
    1. Validate secret, idempotency, risk rules
    2. If entry signal (buy/sell): simulate fill, open paper trade
    3. If exit signal (close_long/close_short): close matching trade, write to journal
    """
    pool = await get_pool()

    async with pool.acquire() as conn:
        # Calculate current equity state for risk checks
        # In a full system, this would track peak equity from a separate table
        # For now, use account size as peak (conservative)
        peak_equity = ACCOUNT_SIZE
        daily_pnl_row = await conn.fetchrow(
            """
            SELECT COALESCE(SUM(pnl_usd), 0) as daily_pnl
            FROM paper_trades
            WHERE status = 'closed'
              AND exit_time > CURRENT_DATE
            """
        )
        daily_pnl = float(daily_pnl_row["daily_pnl"])
        current_equity = ACCOUNT_SIZE + daily_pnl

        # Validate
        validation = await validate_signal(
            conn=conn,
            payload=payload,
            peak_equity=peak_equity,
            current_equity=current_equity,
            daily_pnl=daily_pnl,
        )

        if not validation.accepted:
            return SignalResponse(
                status="rejected",
                rejection_reason=validation.rejection_reason,
            )

        # Process the signal
        if payload.action in ("buy", "sell"):
            trade = await simulate_entry(conn, payload, validation.idempotency_key)
            return SignalResponse(
                status="accepted",
                signal_id=str(trade.id),
            )

        elif payload.action in ("close_long", "close_short"):
            journal_entry = await simulate_exit(conn, payload, validation.idempotency_key)
            return SignalResponse(
                status="accepted",
                signal_id=str(journal_entry.id) if journal_entry else None,
            )

        return SignalResponse(
            status="rejected", rejection_reason=f"Unknown action: {payload.action}"
        )
