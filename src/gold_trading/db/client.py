"""Async PostgreSQL connection pool factory for TimescaleDB."""

import asyncio
import os
import sys
from pathlib import Path

import asyncpg
from loguru import logger

_pool: asyncpg.Pool | None = None


def get_database_url() -> str:
    """Get database URL from environment."""
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://gold:gold@localhost:5433/gold_trading",
    )


async def get_pool() -> asyncpg.Pool:
    """Get or create the connection pool."""
    global _pool
    if _pool is None or getattr(_pool, "_closed", True):
        _pool = await asyncpg.create_pool(
            get_database_url(),
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        logger.info("Database connection pool created")
    return _pool


async def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool is not None and not getattr(_pool, "_closed", True):
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed")


async def run_migrations() -> None:
    """Apply all SQL migration files in order."""
    migrations_dir = Path(__file__).parent / "migrations"
    pool = await get_pool()

    async with pool.acquire() as conn:
        # Create migrations tracking table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                filename TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        # Get already-applied migrations
        applied = {row["filename"] for row in await conn.fetch("SELECT filename FROM _migrations")}

        # Find and sort migration files
        migration_files = sorted(migrations_dir.glob("*.sql"))

        for migration_file in migration_files:
            if migration_file.name in applied:
                logger.debug(f"Skipping already-applied migration: {migration_file.name}")
                continue

            logger.info(f"Applying migration: {migration_file.name}")
            sql = migration_file.read_text()

            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO _migrations (filename) VALUES ($1)",
                    migration_file.name,
                )

            logger.info(f"Migration applied: {migration_file.name}")


async def main() -> None:
    """CLI entrypoint for running migrations."""
    if "--migrate" in sys.argv:
        await run_migrations()
        logger.info("All migrations applied successfully")
        await close_pool()
    else:
        print("Usage: python -m gold_trading.db.client --migrate")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
