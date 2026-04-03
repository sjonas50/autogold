"""Tests for signal validation — secret check, idempotency, risk rules.

All tests run against real TimescaleDB.
"""

import os
from datetime import UTC, datetime

from gold_trading.db.queries.strategies import upsert_strategy
from gold_trading.db.queries.trades import insert_paper_trade
from gold_trading.models.signal import WebhookPayload
from gold_trading.models.strategy import Strategy
from gold_trading.models.trade import PaperTrade
from gold_trading.webhook.validator import make_idempotency_key, validate_signal


def _make_payload(**overrides) -> WebhookPayload:
    defaults = {
        "secret": os.environ.get("WEBHOOK_SECRET", ""),
        "strategy_id": "gs_v1_breakout",
        "action": "buy",
        "contracts": 1,
        "price": 3125.00,
        "bar_time": datetime(2026, 4, 3, 15, 0, 0),
        "instrument": "GC",
    }
    defaults.update(overrides)
    return WebhookPayload(**defaults)


async def _setup_active_strategy(conn, strategy_id: str = "gs_v1_breakout") -> None:
    """Create an active strategy in the DB."""
    await upsert_strategy(
        conn,
        Strategy(
            id=strategy_id,
            name="Test Strategy",
            pine_script="// test",
            is_active=True,
            status="active",
        ),
    )


class TestIdempotencyKey:
    def test_deterministic(self):
        """Same inputs produce same key."""
        p1 = _make_payload()
        p2 = _make_payload()
        assert make_idempotency_key(p1) == make_idempotency_key(p2)

    def test_different_action(self):
        """Different actions produce different keys."""
        p1 = _make_payload(action="buy")
        p2 = _make_payload(action="sell")
        assert make_idempotency_key(p1) != make_idempotency_key(p2)

    def test_different_time(self):
        """Different bar times produce different keys."""
        p1 = _make_payload(bar_time=datetime(2026, 4, 3, 15, 0, 0))
        p2 = _make_payload(bar_time=datetime(2026, 4, 3, 15, 5, 0))
        assert make_idempotency_key(p1) != make_idempotency_key(p2)


class TestValidateSignal:
    async def test_accept_valid_signal(self, conn):
        """Valid signal with active strategy is accepted."""
        await _setup_active_strategy(conn)
        payload = _make_payload()

        result = await validate_signal(
            conn=conn,
            payload=payload,
            peak_equity=50_000,
            current_equity=50_000,
            daily_pnl=0,
        )
        assert result.accepted is True

    async def test_reject_bad_secret(self, conn):
        """Signal with wrong secret is rejected."""
        os.environ["WEBHOOK_SECRET"] = "correct_secret"
        try:
            payload = _make_payload(secret="wrong_secret")
            result = await validate_signal(conn, payload, 50_000, 50_000, 0)
            assert result.accepted is False
            assert "secret" in result.rejection_reason.lower()
        finally:
            os.environ.pop("WEBHOOK_SECRET", None)

    async def test_reject_duplicate(self, conn):
        """Duplicate signal (same idempotency key) is rejected."""
        await _setup_active_strategy(conn)
        payload = _make_payload()
        idem_key = make_idempotency_key(payload)

        # Insert a trade with this idempotency key
        await insert_paper_trade(
            conn,
            PaperTrade(
                strategy_id="gs_v1_breakout",
                instrument="GC",
                direction="long",
                contracts=1,
                entry_price=3125.00,
                entry_time=datetime.now(UTC),
                idempotency_key=idem_key,
            ),
        )

        result = await validate_signal(conn, payload, 50_000, 50_000, 0)
        assert result.accepted is False
        assert "duplicate" in result.rejection_reason.lower()

    async def test_reject_inactive_strategy(self, conn):
        """Signal for inactive strategy is rejected."""
        await upsert_strategy(
            conn,
            Strategy(
                id="gs_v1_breakout",
                name="Test",
                pine_script="// test",
                is_active=False,
            ),
        )
        payload = _make_payload()
        result = await validate_signal(conn, payload, 50_000, 50_000, 0)
        assert result.accepted is False
        assert result.strategy_active is False

    async def test_reject_max_positions(self, conn):
        """Signal rejected when max positions reached."""
        await _setup_active_strategy(conn)

        # Open an existing position
        await insert_paper_trade(
            conn,
            PaperTrade(
                strategy_id="gs_v1_breakout",
                instrument="GC",
                direction="long",
                contracts=1,
                entry_price=3120.00,
                entry_time=datetime.now(UTC),
                idempotency_key="existing_position",
            ),
        )

        payload = _make_payload(bar_time=datetime(2026, 4, 3, 16, 0, 0))
        result = await validate_signal(conn, payload, 50_000, 50_000, 0)
        assert result.accepted is False
        assert result.position_check_passed is False

    async def test_reject_drawdown_breached(self, conn):
        """Signal rejected when 2% drawdown breached."""
        await _setup_active_strategy(conn)
        payload = _make_payload()

        result = await validate_signal(
            conn=conn,
            payload=payload,
            peak_equity=50_000,
            current_equity=48_900,  # 2.2% drawdown
            daily_pnl=-1_100,
        )
        assert result.accepted is False
        assert result.drawdown_check_passed is False

    async def test_accept_close_signal_regardless(self, conn):
        """Close signals skip strategy active and risk checks."""
        payload = _make_payload(action="close_long")
        result = await validate_signal(conn, payload, 50_000, 48_000, -2_000)
        assert result.accepted is True
