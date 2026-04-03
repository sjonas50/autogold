"""FastAPI application factory for the gold trading webhook receiver."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Gold Trading Webhook Receiver",
        description="Receives TradingView strategy alerts and manages paper trades",
        version="0.1.0",
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
