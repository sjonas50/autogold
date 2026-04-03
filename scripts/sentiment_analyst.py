"""Sentiment Analyst agent — ingests gold news and scores with Claude Sonnet 4.6.

Paperclip process adapter script. Runs on heartbeat (every 30 minutes).

News sources (in priority order):
1. Polygon.io — if POLYGON_API_KEY is set (best quality, real-time)
2. RSS feeds — free, no API key: Google News, Reuters, MarketWatch, CNBC
   Always available as fallback or primary source.

Batches headlines for sentiment scoring via Claude Sonnet 4.6,
writes results to sentiment_scores and decision_log.

Run manually: uv run python scripts/sentiment_analyst.py
"""

import asyncio
import json
import os
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

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

# Free RSS feeds for gold/financial news
RSS_FEEDS = [
    # Google News — gold-specific search
    "https://news.google.com/rss/search?q=gold+futures+OR+gold+price+OR+XAUUSD+OR+FOMC+gold&hl=en-US&gl=US&ceid=US:en",
    # Reuters commodities
    "https://www.reutersagency.com/feed/?best-topics=commodities&post_type=best",
    # MarketWatch gold/metals
    "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    # CNBC economy
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    # Investing.com commodities
    "https://www.investing.com/rss/news_285.rss",
]


async def fetch_rss_news(lookback_minutes: int = 120) -> list[dict]:
    """Fetch gold-related news from multiple RSS feeds. Free, no API key."""
    since = datetime.now(UTC) - timedelta(minutes=lookback_minutes)
    all_articles = []

    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        for feed_url in RSS_FEEDS:
            try:
                resp = await client.get(
                    feed_url,
                    headers={"User-Agent": "AutoGoldFutures/1.0 (RSS Reader)"},
                )
                if resp.status_code != 200:
                    logger.debug(f"RSS feed returned {resp.status_code}: {feed_url[:60]}")
                    continue

                root = ET.fromstring(resp.text)

                # Handle both RSS 2.0 (<channel><item>) and Atom (<entry>)
                items = root.findall(".//item") or root.findall(
                    ".//{http://www.w3.org/2005/Atom}entry"
                )

                for item in items:
                    title = (
                        item.findtext("title")
                        or item.findtext("{http://www.w3.org/2005/Atom}title")
                        or ""
                    )
                    link = item.findtext("link") or (
                        item.find("{http://www.w3.org/2005/Atom}link") or {}
                    ).get("href", "")
                    pub_date_str = (
                        item.findtext("pubDate")
                        or item.findtext("{http://www.w3.org/2005/Atom}published")
                        or item.findtext("{http://www.w3.org/2005/Atom}updated")
                    )
                    source_name = _extract_source_name(feed_url)

                    # Parse publication date
                    pub_date = None
                    if pub_date_str:
                        try:
                            pub_date = parsedate_to_datetime(pub_date_str)
                            if pub_date.tzinfo is None:
                                pub_date = pub_date.replace(tzinfo=UTC)
                        except Exception:
                            try:
                                pub_date = datetime.fromisoformat(
                                    pub_date_str.replace("Z", "+00:00")
                                )
                            except Exception:
                                pub_date = None

                    # Skip old articles (ensure both are tz-aware for comparison)
                    if pub_date:
                        if pub_date.tzinfo is None:
                            pub_date = pub_date.replace(tzinfo=UTC)
                        if pub_date < since:
                            continue

                    if not pub_date:
                        pub_date = datetime.now(UTC)

                    all_articles.append(
                        {
                            "title": title.strip(),
                            "source": source_name,
                            "url": link.strip() if isinstance(link, str) else "",
                            "published_utc": pub_date.isoformat(),
                        }
                    )

            except Exception as e:
                logger.debug(f"RSS feed error ({feed_url[:50]}): {e}")
                continue

    # Filter for gold relevance
    gold_articles = []
    for article in all_articles:
        title_lower = article["title"].lower()
        if any(kw.lower() in title_lower for kw in GOLD_KEYWORDS):
            gold_articles.append(article)

    # Deduplicate by title similarity (exact match)
    seen_titles = set()
    unique_articles = []
    for article in gold_articles:
        title_key = article["title"].lower().strip()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_articles.append(article)

    logger.info(
        f"RSS feeds: {len(all_articles)} total, {len(gold_articles)} gold-relevant, "
        f"{len(unique_articles)} unique"
    )
    return unique_articles


def _extract_source_name(feed_url: str) -> str:
    """Extract a human-readable source name from a feed URL."""
    if "google.com" in feed_url:
        return "Google News"
    if "reuters" in feed_url:
        return "Reuters"
    if "marketwatch" in feed_url:
        return "MarketWatch"
    if "cnbc" in feed_url:
        return "CNBC"
    if "investing.com" in feed_url:
        return "Investing.com"
    return "RSS"


async def fetch_polygon_news(
    api_key: str,
    limit: int = 50,
    lookback_minutes: int = 60,
) -> list[dict]:
    """Fetch recent gold-related news from Polygon.io (requires API key)."""
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

    gold_tickers = {"GC", "MGC", "GLD", "IAU", "XAUUSD"}
    gold_articles = []
    for article in articles:
        title = (article.get("title") or "").lower()
        description = (article.get("description") or "").lower()
        tickers = [t.get("ticker", "") for t in article.get("tickers", [])]

        ticker_match = any(t in gold_tickers for t in tickers)
        combined_text = f"{title} {description}"
        keyword_match = any(kw.lower() in combined_text for kw in GOLD_KEYWORDS)

        if ticker_match or keyword_match:
            gold_articles.append(
                {
                    "title": article.get("title", ""),
                    "source": article.get("publisher", {}).get("name", "Polygon")
                    if isinstance(article.get("publisher"), dict)
                    else "Polygon",
                    "url": article.get("article_url", ""),
                    "published_utc": article.get("published_utc", ""),
                }
            )

    logger.info(f"Polygon.io: {len(articles)} total, {len(gold_articles)} gold-relevant")
    return gold_articles


async def score_headlines_batch(headlines: list[dict]) -> list[dict]:
    """Score a batch of headlines for gold sentiment using Claude Sonnet 4.6."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    if not headlines:
        return []

    client = anthropic.AsyncAnthropic(api_key=api_key)

    headline_list = "\n".join(
        f"{i + 1}. [{h.get('source', 'Unknown')}] {h.get('title', '')}"
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

    # Fetch news from available sources
    articles = []

    # Try Polygon.io first (best quality)
    polygon_key = os.environ.get("POLYGON_API_KEY")
    if polygon_key:
        polygon_articles = await fetch_polygon_news(polygon_key, limit=50, lookback_minutes=60)
        articles.extend(polygon_articles)

    # Always fetch RSS (free, supplements Polygon or serves as primary)
    rss_articles = await fetch_rss_news(lookback_minutes=120)
    articles.extend(rss_articles)

    # Deduplicate across sources
    seen = set()
    unique = []
    for a in articles:
        key = a["title"].lower().strip()
        if key not in seen and key:
            seen.add(key)
            unique.append(a)
    articles = unique

    if not articles:
        logger.info("No gold-relevant news found")
        # Still log a decision
        conn = await asyncpg.connect(get_database_url())
        try:
            await insert_decision(
                conn,
                DecisionLogEntry(
                    agent_name="sentiment_analyst",
                    decision_type="sentiment_update",
                    inputs_summary={"articles_fetched": 0, "source": "rss+polygon"},
                    decision="No gold-relevant news found",
                    reasoning="No headlines matched gold keywords in RSS feeds or Polygon.io.",
                    confidence=0.5,
                ),
            )
        finally:
            await conn.close()
        return

    logger.info(f"Total unique articles to score: {len(articles)}")

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
            pub = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pub = datetime.now(UTC)

        sentiment_scores.append(
            SentimentScore(
                headline=article.get("title", "")[:500],
                source=article.get("source", "RSS"),
                url=article.get("url"),
                published_at=pub,
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

        summary = await get_sentiment_summary(conn, hours=4.0)

        source_str = "polygon+rss" if polygon_key else "rss"
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
                    "source": source_str,
                },
                decision=f"avg_sentiment={summary.avg_sentiment:.3f} over {summary.headline_count} headlines",
                reasoning=(
                    f"Scored {len(scored)} new headlines from {source_str}. "
                    f"Rolling 4h average: {summary.avg_sentiment:.3f}. "
                    f"Active catalysts: {', '.join(summary.active_catalysts) or 'none'}."
                ),
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
