"""Pydantic models for market regime classification."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class RegimeState(BaseModel):
    """A point-in-time market regime classification."""

    id: UUID | None = None
    regime: str = Field(pattern=r"^(trending_up|trending_down|ranging|volatile)$")
    hmm_state: int | None = Field(default=None, ge=0, le=3)
    hmm_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    atr_14: float | None = Field(default=None, gt=0)
    adx_14: float | None = Field(default=None, ge=0)
    timeframe: str = "5m"
    dxy_change_1d: float | None = None
    created_at: datetime | None = None


class RegimeClassification(BaseModel):
    """Output of the regime classifier for a given moment."""

    regime: str
    hmm_state: int
    hmm_confidence: float
    atr_14: float
    adx_14: float
    features: dict[str, float] = Field(default_factory=dict)
