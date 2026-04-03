"""FastAPI application factory for the gold trading webhook receiver."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from gold_trading.db.client import close_pool, get_pool, run_migrations
from gold_trading.webhook.receiver import router as webhook_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage DB pool lifecycle."""
    await get_pool()
    await run_migrations()
    yield
    await close_pool()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Gold Trading Webhook Receiver",
        description="Receives TradingView strategy alerts and manages paper trades",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(webhook_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
