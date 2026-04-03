"""Database queries for sentiment_scores table."""

from uuid import UUID

import asyncpg

from gold_trading.models.sentiment import SentimentScore, SentimentSummary


async def insert_sentiment_score(conn: asyncpg.Connection, score: SentimentScore) -> UUID:
    """Insert a scored news headline."""
    import json

    row = await conn.fetchrow(
        """
        INSERT INTO sentiment_scores
            (headline, source, url, published_at, sentiment,
             gold_relevance, catalyst_tags, raw_response)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
        RETURNING id
        """,
        score.headline,
        score.source,
        score.url,
        score.published_at,
        score.sentiment,
        score.gold_relevance,
        score.catalyst_tags,
        json.dumps(score.raw_response) if score.raw_response else None,
    )
    return row["id"]


async def insert_sentiment_scores_batch(
    conn: asyncpg.Connection, scores: list[SentimentScore]
) -> int:
    """Insert multiple sentiment scores. Returns count inserted."""
    import json

    records = [
        (
            s.headline,
            s.source,
            s.url,
            s.published_at,
            s.sentiment,
            s.gold_relevance,
            s.catalyst_tags,
            json.dumps(s.raw_response) if s.raw_response else None,
        )
        for s in scores
    ]
    await conn.executemany(
        """
        INSERT INTO sentiment_scores
            (headline, source, url, published_at, sentiment,
             gold_relevance, catalyst_tags, raw_response)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
        """,
        records,
    )
    return len(records)


async def get_sentiment_summary(conn: asyncpg.Connection, hours: float = 4.0) -> SentimentSummary:
    """Get rolling sentiment summary for the last N hours."""
    row = await conn.fetchrow(
        """
        SELECT
            COALESCE(AVG(sentiment), 0) as avg_sentiment,
            COUNT(*) as headline_count
        FROM sentiment_scores
        WHERE ingested_at > NOW() - make_interval(hours => $1)
        """,
        hours,
    )
    # Get unique catalyst tags
    tag_rows = await conn.fetch(
        """
        SELECT DISTINCT unnest(catalyst_tags) as tag
        FROM sentiment_scores
        WHERE ingested_at > NOW() - make_interval(hours => $1)
        """,
        hours,
    )
    return SentimentSummary(
        avg_sentiment=float(row["avg_sentiment"]),
        headline_count=row["headline_count"],
        active_catalysts=[r["tag"] for r in tag_rows],
        window_hours=hours,
    )


async def get_recent_scores(conn: asyncpg.Connection, limit: int = 20) -> list[SentimentScore]:
    """Get the most recent sentiment scores."""
    rows = await conn.fetch(
        "SELECT * FROM sentiment_scores ORDER BY ingested_at DESC LIMIT $1",
        limit,
    )
    return [
        SentimentScore(
            **{
                **dict(r),
                "raw_response": (
                    __import__("json").loads(r["raw_response"]) if r["raw_response"] else None
                ),
            }
        )
        for r in rows
    ]
