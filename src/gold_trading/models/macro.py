"""Pydantic models for macroeconomic data."""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field


class MacroData(BaseModel):
    """A snapshot of gold-relevant macroeconomic indicators."""

    id: UUID | None = None
    observation_date: date
    dxy: float | None = None
    real_yield_10y: float | None = None
    cpi_yoy: float | None = None
    breakeven_10y: float | None = None
    oil_wti: float | None = None
    gold_fix_pm: float | None = None
    macro_regime: str | None = Field(default=None, pattern=r"^(bullish|neutral|bearish)$")
    reasoning: str | None = None
    created_at: datetime | None = None


class MacroRegime(BaseModel):
    """Macro regime classification output."""

    regime: str = Field(pattern=r"^(bullish|neutral|bearish)$")
    confidence: float = Field(ge=0.0, le=1.0)
    key_drivers: list[str] = Field(default_factory=list)
    reasoning: str
