"""Pydantic models for webhook signals and validation."""

from datetime import datetime

from pydantic import BaseModel, Field


class WebhookPayload(BaseModel):
    """Incoming TradingView strategy alert webhook payload."""

    secret: str
    strategy_id: str
    action: str = Field(pattern=r"^(buy|sell|close_long|close_short)$")
    contracts: int = Field(ge=1)
    price: float = Field(gt=0)
    bar_time: datetime
    instrument: str = Field(default="GC", pattern=r"^(GC|MGC)$")
    stop_loss: float | None = None
    take_profit: float | None = None
    atr: float | None = None


class SignalValidation(BaseModel):
    """Result of validating an incoming signal."""

    accepted: bool
    rejection_reason: str | None = None
    idempotency_key: str
    risk_check_passed: bool = True
    position_check_passed: bool = True
    strategy_active: bool = True
    drawdown_check_passed: bool = True


class SignalResponse(BaseModel):
    """HTTP response for webhook endpoint."""

    status: str  # 'accepted' or 'rejected'
    signal_id: str | None = None
    rejection_reason: str | None = None
