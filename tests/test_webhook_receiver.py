"""Tests for the FastAPI webhook endpoint — integration tests against real DB."""

import pytest
from httpx import ASGITransport, AsyncClient

# We need to patch get_pool to use our test connection
# For endpoint tests, we use httpx AsyncClient with the real app


@pytest.fixture
def signal_payload():
    """Valid signal payload."""
    return {
        "secret": "",
        "strategy_id": "gs_test_webhook",
        "action": "buy",
        "contracts": 1,
        "price": 3125.00,
        "bar_time": "2026-04-03T15:00:00",
        "instrument": "GC",
    }


class TestWebhookEndpoint:
    async def test_health_endpoint(self):
        """Health check returns ok."""
        from gold_trading.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    async def test_invalid_payload_returns_422(self):
        """Missing required fields return 422."""
        from gold_trading.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/webhook/signal", json={"secret": "test"})
            assert resp.status_code == 422

    async def test_valid_payload_structure(self):
        """Valid payload shape is accepted by Pydantic (even if risk rejects it)."""
        from gold_trading.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/signal",
                json={
                    "secret": "",
                    "strategy_id": "gs_nonexistent",
                    "action": "buy",
                    "contracts": 1,
                    "price": 3125.00,
                    "bar_time": "2026-04-03T15:00:00",
                    "instrument": "GC",
                },
            )
            # Should get 200 (with rejected status) not 422
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "rejected"
