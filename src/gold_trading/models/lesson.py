"""Pydantic models for the agent learning/memory system."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class Lesson(BaseModel):
    """An extracted learning stored in the lessons table with embedding."""

    id: UUID | None = None
    content: str
    embedding: list[float] | None = None
    regime_tags: list[str] = Field(default_factory=list)
    strategy_class: str | None = None
    macro_context: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source_trades: list[UUID] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class LessonQuery(BaseModel):
    """Query parameters for similarity search over lessons."""

    embedding: list[float] = Field(min_length=1536, max_length=1536)
    limit: int = Field(default=5, ge=1, le=20)
    regime_filter: str | None = None
    strategy_class_filter: str | None = None


class DecisionLogEntry(BaseModel):
    """An agent decision recorded in the decision log."""

    id: UUID | None = None
    agent_name: str
    decision_type: str
    inputs_summary: dict = Field(default_factory=dict)
    decision: str
    reasoning: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    outcome_tag: str = Field(default="pending", pattern=r"^(pending|correct|incorrect)$")
    related_trade: UUID | None = None
    created_at: datetime | None = None
