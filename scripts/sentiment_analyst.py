"""Sentiment Analyst agent — ingests Polygon.io gold news and scores with Claude Sonnet 4.6.

Paperclip process adapter script. Runs on heartbeat (every 30 minutes).
Fetches gold-related news from Polygon.io, batches headlines for sentiment scoring
via Claude Sonnet 4.6, writes results to sentiment_scores and decision_log.

Run manually: uv run python scripts/sentiment_analyst.py
"""

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta

import anthropic
import asyncpg
import httpx
from loguru import logger

from gold_trading.db.client import get_database_url
from gold_trading.db.queries.decisions import insert_decision
from gold_trading.db.queries.sentiment import (
    get_sentiment_summary,
    insert_sentiment_scores_batch,
)
from gold_trading.models.lesson import DecisionLogEntry
from gold_trading.models.sentiment import SentimentScore

ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Gold-related tickers and keywords for Polygon.io news filtering
GOLD_TICKERS = ["GC", "MGC", "GLD", "IAU", "XAUUSD"]
GOLD_KEYWORDS = [
    "gold",
    "FOMC",
    "federal reserve",
    "interest rate",
    "CPI",
    "inflation",
    "NFP",
    "nonfarm",
    "safe haven",
    "real yield",
    "treasury",
    "USD",
    "dollar",
    "central bank",
    "geopolitical",
    "bullion",
    "precious metal",
]


async def fetch_polygon_news(
    api_key: str,
    limit: int = 50,
    lookback_minutes: int = 60,
) -> list[dict]:
    """Fetch recent gold-related news from Polygon.io."""
    since = datetime.now(UTC) - timedelta(minutes=lookback_minutes)
    published_after = since.strftime("%Y-%m-%dT%H:%M:%SZ")

    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "apiKey": api_key,
        "limit": limit,
        "published_utc.gte": published_after,
        "sort": "published_utc",
        "order": "desc",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
        if resp.status_code != 200:
            logger.error(f"Polygon.io API error: {resp.status_code} {resp.text}")
            return []

        data = resp.json()
        articles = data.get("results", [])

    # Filter for gold relevance
    gold_articles = []
    for article in articles:
        title = (article.get("title") or "").lower()
        description = (article.get("description") or "").lower()
        tickers = [t.get("ticker", "") for t in article.get("tickers", [])]

        # Check ticker match
        ticker_match = any(t in GOLD_TICKERS for t in tickers)

        # Check keyword match
        combined_text = f"{title} {description}"
        keyword_match = any(kw.lower() in combined_text for kw in GOLD_KEYWORDS)

        if ticker_match or keyword_match:
            gold_articles.append(article)

    logger.info(f"Polygon.io: {len(articles)} total articles, {len(gold_articles)} gold-relevant")
    return gold_articles


async def score_headlines_batch(
    headlines: list[dict],
) -> list[dict]:
    """Score a batch of headlines for gold sentiment using Claude Sonnet 4.6.

    Batches up to 20 headlines per API call to minimize costs.

    Args:
        headlines: List of dicts with 'title', 'source', 'url', 'published_utc'.

    Returns:
        List of dicts with added 'sentiment', 'gold_relevance', 'catalyst_tags'.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    if not headlines:
        return []

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Format headlines for the prompt
    headline_list = "\n".join(
        f"{i + 1}. [{h.get('source', {}).get('name', 'Unknown')}] {h.get('title', '')}"
        for i, h in enumerate(headlines)
    )

    prompt = f"""You are a gold market sentiment analyst. Score each headline for its impact on gold prices.

Headlines:
{headline_list}

For each headline, provide:
1. sentiment: float from -1.0 (very bearish for gold) to +1.0 (very bullish for gold). 0.0 = neutral.
2. gold_relevance: float from 0.0 (not about gold) to 1.0 (directly about gold prices/demand).
3. catalyst_tags: list of applicable tags from: ["fomc", "cpi", "nfp", "geopolitical", "usd", "risk_off", "risk_on", "central_bank", "inflation", "yields", "technical"]

Key sentiment drivers for gold:
- Fed dovish / rate cuts → bullish (+)
- Fed hawkish / rate hikes → bearish (-)
- High CPI / rising inflation → bullish (+)
- Strong USD → bearish (-)
- Geopolitical tension → bullish (+)
- Risk-off sentiment → bullish (+)
- Falling real yields → bullish (+)
- Central bank gold buying → bullish (+)

Respond with a JSON array, one object per headline, in order:
[{{"sentiment": 0.5, "gold_relevance": 0.8, "catalyst_tags": ["fomc"]}}, ...]"""

    response = await client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Parse JSON from response
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        scores = json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse Claude response: {text[:200]}")
        return []

    # Merge scores back into headlines
    results = []
    for i, headline in enumerate(headlines):
        if i < len(scores):
            score = scores[i]
            headline["sentiment"] = max(-1.0, min(1.0, float(score.get("sentiment", 0))))
            headline["gold_relevance"] = max(0.0, min(1.0, float(score.get("gold_relevance", 0.5))))
            headline["catalyst_tags"] = score.get("catalyst_tags", [])
            results.append(headline)

    return results


async def main() -> None:
    """Main heartbeat execution."""
    logger.info("Sentiment Analyst heartbeat starting")

    polygon_key = os.environ.get("POLYGON_API_KEY")
    if not polygon_key:
        logger.error("POLYGON_API_KEY not set — cannot fetch news")
        return

    # Fetch news
    articles = await fetch_polygon_news(polygon_key, limit=50, lookback_minutes=60)

    if not articles:
        logger.info("No gold-relevant news in the last 60 minutes")
        return

    # Score in batches of 20
    scored = []
    batch_size = 20
    for i in range(0, len(articles), batch_size):
        batch = articles[i : i + batch_size]
        batch_scores = await score_headlines_batch(batch)
        scored.extend(batch_scores)

    logger.info(f"Scored {len(scored)} headlines")

    # Convert to SentimentScore models
    sentiment_scores = []
    for article in scored:
        published_str = article.get("published_utc", "")
        try:
            published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            published = datetime.now(UTC)

        sentiment_scores.append(
            SentimentScore(
                headline=article.get("title", "")[:500],
                source=article.get("publisher", {}).get("name")
                if isinstance(article.get("publisher"), dict)
                else str(article.get("publisher", "")),
                url=article.get("article_url"),
                published_at=published,
                sentiment=article.get("sentiment", 0.0),
                gold_relevance=article.get("gold_relevance", 0.5),
                catalyst_tags=article.get("catalyst_tags", []),
                raw_response={
                    "sentiment": article.get("sentiment"),
                    "gold_relevance": article.get("gold_relevance"),
                    "catalyst_tags": article.get("catalyst_tags"),
                },
            )
        )

    # Write to DB
    conn = await asyncpg.connect(get_database_url())
    try:
        if sentiment_scores:
            count = await insert_sentiment_scores_batch(conn, sentiment_scores)
            logger.info(f"Wrote {count} sentiment scores to DB")

        # Get rolling summary for decision log
        summary = await get_sentiment_summary(conn, hours=4.0)

        # Write decision log
        await insert_decision(
            conn,
            DecisionLogEntry(
                agent_name="sentiment_analyst",
                decision_type="sentiment_update",
                inputs_summary={
                    "articles_fetched": len(articles),
                    "articles_scored": len(scored),
                    "avg_sentiment": round(summary.avg_sentiment, 3),
                    "headline_count_4h": summary.headline_count,
                    "active_catalysts": summary.active_catalysts,
                },
                decision=f"avg_sentiment={summary.avg_sentiment:.3f} over {summary.headline_count} headlines",
                reasoning=f"Scored {len(scored)} new headlines. Rolling 4h average: {summary.avg_sentiment:.3f}. Active catalysts: {', '.join(summary.active_catalysts) or 'none'}.",
                confidence=0.7,
            ),
        )

        logger.info(
            f"Sentiment Analyst heartbeat complete. "
            f"4h avg: {summary.avg_sentiment:.3f}, catalysts: {summary.active_catalysts}"
        )

    finally:
        await conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
