"""Database queries for the lessons store (pgvector similarity search)."""

from uuid import UUID

import asyncpg

from gold_trading.models.lesson import Lesson


async def insert_lesson(conn: asyncpg.Connection, lesson: Lesson) -> UUID:
    """Insert a new lesson with embedding."""
    embedding_str = None
    if lesson.embedding:
        embedding_str = "[" + ",".join(str(x) for x in lesson.embedding) + "]"

    row = await conn.fetchrow(
        """
        INSERT INTO lessons
            (content, embedding, regime_tags, strategy_class,
             macro_context, confidence, source_trades)
        VALUES ($1, $2::vector, $3, $4, $5, $6, $7)
        RETURNING id
        """,
        lesson.content,
        embedding_str,
        lesson.regime_tags,
        lesson.strategy_class,
        lesson.macro_context,
        lesson.confidence,
        lesson.source_trades,
    )
    return row["id"]


async def search_similar_lessons(
    conn: asyncpg.Connection,
    embedding: list[float],
    limit: int = 5,
    regime_filter: str | None = None,
    strategy_class_filter: str | None = None,
) -> list[Lesson]:
    """Search for similar lessons using pgvector cosine distance."""
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    conditions = []
    params: list = [embedding_str, limit]

    if regime_filter:
        params.append(regime_filter)
        conditions.append("$3 = ANY(regime_tags)")

    if strategy_class_filter:
        idx = len(params) + 1
        params.append(strategy_class_filter)
        conditions.append(f"${idx} = strategy_class")

    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    rows = await conn.fetch(
        f"""
        SELECT id, content, regime_tags, strategy_class, macro_context,
               confidence, source_trades, created_at, updated_at
        FROM lessons
        {where}
        ORDER BY embedding <=> $1::vector
        LIMIT $2
        """,
        *params,
    )
    return [Lesson(**dict(r)) for r in rows]


async def get_lesson_count(conn: asyncpg.Connection) -> int:
    """Get total lesson count."""
    row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM lessons")
    return row["cnt"]
