"""Pydantic models for news sentiment scoring."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    """A raw news item before sentiment scoring."""

    headline: str
    source: str | None = None
    url: str | None = None
    published_at: datetime
    tickers: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class SentimentScore(BaseModel):
    """A scored news headline with gold sentiment."""

    id: UUID | None = None
    headline: str
    source: str | None = None
    url: str | None = None
    published_at: datetime
    ingested_at: datetime | None = None
    sentiment: float = Field(ge=-1.0, le=1.0)
    gold_relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    catalyst_tags: list[str] = Field(default_factory=list)
    raw_response: dict | None = None


class SentimentSummary(BaseModel):
    """Rolling sentiment summary for a time window."""

    avg_sentiment: float
    headline_count: int
    active_catalysts: list[str] = Field(default_factory=list)
    window_hours: float = 4.0
