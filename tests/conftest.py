"""Shared pytest fixtures for gold trading tests — all against real TimescaleDB."""

from datetime import UTC, datetime

import asyncpg
import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def db_pool():
    """Create a connection pool to the real test database."""
    pool = await asyncpg.create_pool(
        "postgresql://gold:gold@localhost:5433/gold_trading",
        min_size=1,
        max_size=5,
    )
    yield pool
    await pool.close()


@pytest_asyncio.fixture
async def conn(db_pool):
    """Get a connection with automatic transaction rollback for test isolation."""
    async with db_pool.acquire() as connection:
        tx = connection.transaction()
        await tx.start()
        yield connection
        await tx.rollback()


@pytest.fixture
def webhook_secret() -> str:
    return "test_secret_123"


@pytest.fixture
def account_size() -> float:
    return 50_000.0


@pytest.fixture
def now() -> datetime:
    return datetime.now(UTC)
