"""Pydantic models for trades and paper trades."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class TradeJournalEntry(BaseModel):
    """A completed trade with full context."""

    id: UUID | None = None
    strategy_id: str
    instrument: str = Field(pattern=r"^(GC|MGC)$")
    direction: str = Field(pattern=r"^(long|short)$")
    contracts: int = Field(ge=1)
    entry_price: float = Field(gt=0)
    exit_price: float | None = None
    entry_time: datetime
    closed_at: datetime | None = None
    pnl_usd: float | None = None
    r_multiple: float | None = None
    max_adverse_exc: float | None = None
    regime_at_entry: str | None = None
    sentiment_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    macro_bias: str | None = None
    session: str | None = None
    atr_at_entry: float | None = None
    notes: str | None = None


class PaperTrade(BaseModel):
    """A live or closed paper trade position."""

    id: UUID | None = None
    strategy_id: str
    instrument: str = Field(pattern=r"^(GC|MGC)$")
    direction: str = Field(pattern=r"^(long|short)$")
    contracts: int = Field(ge=1)
    entry_price: float = Field(gt=0)
    exit_price: float | None = None
    entry_time: datetime
    exit_time: datetime | None = None
    status: str = Field(default="open", pattern=r"^(open|closed|stopped)$")
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    pnl_usd: float | None = None
    r_multiple: float | None = None
    regime_at_entry: str | None = None
    sentiment_at_entry: float | None = None
    macro_at_entry: str | None = None
    idempotency_key: str
    created_at: datetime | None = None


class TradeOutcome(BaseModel):
    """Summary of a trade result for CIO review."""

    trade_id: UUID
    strategy_id: str
    instrument: str
    direction: str
    pnl_usd: float
    r_multiple: float
    regime_at_entry: str | None
    sentiment_score: float | None
    session: str | None
    duration_minutes: float | None = None
