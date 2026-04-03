"""Signal validation — orchestrates secret check, idempotency, and risk rules."""

import hashlib
import os

import asyncpg
from loguru import logger

from gold_trading.db.queries.strategies import is_strategy_active
from gold_trading.db.queries.trades import check_idempotency, get_open_position_count
from gold_trading.models.signal import SignalValidation, WebhookPayload
from gold_trading.risk.rules import (
    check_daily_loss,
    check_drawdown,
    check_max_positions,
    check_strategy_active,
)

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
ACCOUNT_SIZE = float(os.environ.get("ACCOUNT_SIZE_USD", "50000"))


def make_idempotency_key(payload: WebhookPayload) -> str:
    """Generate idempotency key from strategy_id + bar_time + action."""
    raw = f"{payload.strategy_id}|{payload.bar_time.isoformat()}|{payload.action}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def validate_secret(payload: WebhookPayload) -> bool:
    """Check that the webhook secret matches."""
    expected = os.environ.get("WEBHOOK_SECRET", "")
    if not expected:
        logger.warning("WEBHOOK_SECRET not set — accepting all signals (dev mode)")
        return True
    return payload.secret == expected


async def validate_signal(
    conn: asyncpg.Connection,
    payload: WebhookPayload,
    peak_equity: float,
    current_equity: float,
    daily_pnl: float,
) -> SignalValidation:
    """Run the full validation pipeline on an incoming signal.

    Checks in order:
    1. Secret validation
    2. Idempotency (duplicate detection)
    3. Strategy is active
    4. Risk rules (position count, drawdown, daily loss)
    """
    idem_key = make_idempotency_key(payload)

    # 1. Secret check
    if not validate_secret(payload):
        logger.warning(f"Invalid webhook secret for signal {payload.strategy_id}")
        return SignalValidation(
            accepted=False,
            rejection_reason="Invalid webhook secret",
            idempotency_key=idem_key,
        )

    # 2. Idempotency
    is_duplicate = await check_idempotency(conn, idem_key)
    if is_duplicate:
        logger.info(f"Duplicate signal rejected: {idem_key}")
        return SignalValidation(
            accepted=False,
            rejection_reason="Duplicate signal (idempotency key already exists)",
            idempotency_key=idem_key,
        )

    # For close signals, skip strategy active and risk checks
    if payload.action in ("close_long", "close_short"):
        return SignalValidation(
            accepted=True,
            idempotency_key=idem_key,
        )

    # 3. Strategy active check
    strategy_active = await is_strategy_active(conn, payload.strategy_id)
    strat_check = check_strategy_active(strategy_active)
    if not strat_check.passed:
        logger.info(f"Signal rejected — strategy not active: {payload.strategy_id}")
        return SignalValidation(
            accepted=False,
            rejection_reason=strat_check.violations[0],
            idempotency_key=idem_key,
            strategy_active=False,
        )

    # 4. Risk checks
    open_count = await get_open_position_count(conn)

    pos_check = check_max_positions(open_count)
    if not pos_check.passed:
        logger.info(f"Signal rejected — {pos_check.violations[0]}")
        return SignalValidation(
            accepted=False,
            rejection_reason=pos_check.violations[0],
            idempotency_key=idem_key,
            position_check_passed=False,
        )

    dd_check = check_drawdown(peak_equity, current_equity)
    if not dd_check.passed:
        logger.warning(f"Signal rejected — {dd_check.violations[0]}")
        return SignalValidation(
            accepted=False,
            rejection_reason=dd_check.violations[0],
            idempotency_key=idem_key,
            drawdown_check_passed=False,
        )

    dl_check = check_daily_loss(daily_pnl, ACCOUNT_SIZE)
    if not dl_check.passed:
        logger.warning(f"Signal rejected — {dl_check.violations[0]}")
        return SignalValidation(
            accepted=False,
            rejection_reason=dl_check.violations[0],
            idempotency_key=idem_key,
            risk_check_passed=False,
        )

    logger.info(f"Signal accepted: {payload.strategy_id} {payload.action} {payload.contracts}x {payload.instrument}")
    return SignalValidation(
        accepted=True,
        idempotency_key=idem_key,
    )
