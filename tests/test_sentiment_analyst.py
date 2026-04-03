"""Tests for sentiment analyst news processing and scoring logic."""

import os
from datetime import UTC, datetime, timedelta

import pytest

from gold_trading.db.queries.sentiment import (
    get_sentiment_summary,
    insert_sentiment_scores_batch,
)
from gold_trading.models.sentiment import SentimentScore


class TestSentimentModel:
    def test_valid_score(self):
        s = SentimentScore(
            headline="Gold rallies on dovish Fed",
            published_at=datetime.now(UTC),
            sentiment=0.7,
            gold_relevance=0.9,
            catalyst_tags=["fomc"],
        )
        assert s.sentiment == 0.7
        assert s.catalyst_tags == ["fomc"]

    def test_sentiment_bounds(self):
        """Sentiment must be between -1 and 1."""
        with pytest.raises(ValueError):
            SentimentScore(
                headline="Test",
                published_at=datetime.now(UTC),
                sentiment=1.5,
            )

    def test_gold_relevance_bounds(self):
        """Gold relevance must be between 0 and 1."""
        with pytest.raises(ValueError):
            SentimentScore(
                headline="Test",
                published_at=datetime.now(UTC),
                sentiment=0.5,
                gold_relevance=1.5,
            )


class TestGoldKeywordFiltering:
    def test_keyword_matching(self):
        """Verify gold keyword filtering logic from sentiment_analyst.py."""
        from scripts.sentiment_analyst import GOLD_KEYWORDS

        # These should match
        assert any(kw.lower() in "gold prices surge on fomc" for kw in GOLD_KEYWORDS)
        assert any(kw.lower() in "federal reserve holds rates" for kw in GOLD_KEYWORDS)
        assert any(kw.lower() in "cpi data comes in hot" for kw in GOLD_KEYWORDS)
        assert any(kw.lower() in "safe haven demand rises" for kw in GOLD_KEYWORDS)

        # These should not match
        assert not any(kw.lower() in "apple stock rises 5%" for kw in GOLD_KEYWORDS)
        assert not any(kw.lower() in "bitcoin hits new high" for kw in GOLD_KEYWORDS)


class TestSentimentDBIntegration:
    async def test_batch_insert_and_summary(self, conn):
        """Insert a batch and verify summary calculation."""
        now = datetime.now(UTC)
        scores = [
            SentimentScore(
                headline=f"Headline {i}",
                source="Reuters",
                published_at=now - timedelta(minutes=i * 10),
                sentiment=0.3 + (i * 0.1),  # 0.3, 0.4, 0.5, 0.6, 0.7
                gold_relevance=0.8,
                catalyst_tags=["fomc"] if i % 2 == 0 else ["usd"],
            )
            for i in range(5)
        ]
        count = await insert_sentiment_scores_batch(conn, scores)
        assert count == 5

        summary = await get_sentiment_summary(conn, hours=4.0)
        assert summary.headline_count >= 5
        # Avg includes our 5 inserts plus any from live agent runs
        assert -1.0 <= summary.avg_sentiment <= 1.0
        assert "fomc" in summary.active_catalysts
        assert "usd" in summary.active_catalysts

    async def test_empty_summary(self, conn):
        """Summary with no data returns zero sentiment."""
        summary = await get_sentiment_summary(conn, hours=0.001)  # Tiny window
        assert summary.headline_count == 0
        assert summary.avg_sentiment == 0.0


class TestPolygonIntegration:
    @pytest.mark.skipif(
        not os.environ.get("POLYGON_API_KEY"),
        reason="POLYGON_API_KEY not set — skipping live API test",
    )
    async def test_fetch_real_polygon_news(self):
        """Test fetching real Polygon.io news (requires POLYGON_API_KEY)."""
        from scripts.sentiment_analyst import fetch_polygon_news

        api_key = os.environ["POLYGON_API_KEY"]
        articles = await fetch_polygon_news(api_key, limit=10, lookback_minutes=1440)
        # We should get at least some financial news in the last 24 hours
        assert isinstance(articles, list)
        # Don't assert count since gold-relevant articles may vary
