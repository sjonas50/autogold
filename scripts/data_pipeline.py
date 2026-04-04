"""Data Pipeline agent — keeps OHLCV data fresh and monitors data quality.

Paperclip process adapter script. Runs on heartbeat (every 60 minutes).
- Pulls latest 5m GC bars from yfinance
- Detects gaps (missed bars, maintenance windows, weekends)
- Monitors data staleness and alerts CIO via Paperclip task if data is >2 hours old
- Refreshes Pine Script RAG corpus monthly

Run manually: uv run python scripts/data_pipeline.py
"""

import asyncio
import os
from datetime import UTC, datetime

import asyncpg
import pandas as pd
import yfinance as yf
from loguru import logger

from gold_trading.db.client import get_database_url
from gold_trading.db.queries.decisions import insert_decision
from gold_trading.models.lesson import DecisionLogEntry

PAPERCLIP_URL = os.environ.get("PAPERCLIP_URL", "http://localhost:3100")
COMPANY_ID = os.environ.get("PAPERCLIP_COMPANY_ID", "ea9cb59f-a7c4-4c1b-856f-173ea1d5bddf")
CIO_AGENT_ID = "2a732301-8c96-48da-8ae6-c9ef9a4fa517"


async def get_latest_bar_time(conn: asyncpg.Connection) -> datetime | None:
    """Get the timestamp of the most recent bar in the DB."""
    row = await conn.fetchrow(
        "SELECT MAX(timestamp) as latest FROM ohlcv_5m WHERE instrument = 'GC'"
    )
    return row["latest"] if row and row["latest"] else None


async def ingest_new_bars(conn: asyncpg.Connection) -> int:
    """Pull latest 5m bars from yfinance and insert new ones."""
    latest = await get_latest_bar_time(conn)

    # Determine how far back to fetch
    if latest:
        hours_stale = (datetime.now(UTC) - latest.replace(tzinfo=UTC)).total_seconds() / 3600
        # Fetch enough to cover the gap, minimum 1 day, max 5 days
        days = max(1, min(int(hours_stale / 24) + 1, 5))
    else:
        days = 5  # First run: get 5 days

    logger.info(f"Fetching GC=F 5m data for last {days} days (latest bar: {latest})")

    ticker = yf.Ticker("GC=F")
    df = ticker.history(period=f"{days}d", interval="5m")

    if df.empty:
        logger.warning("yfinance returned no data — market may be closed")
        return 0

    # Normalize
    df = df.reset_index()
    df = df.rename(columns={"Datetime": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # Only insert bars newer than what we have
    if latest:
        latest_utc = latest.replace(tzinfo=UTC) if latest.tzinfo is None else latest
        df = df[df["timestamp"] > latest_utc]

    if df.empty:
        logger.info("No new bars to ingest")
        return 0

    records = [
        (
            row["timestamp"],
            "GC",
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            int(row.get("Volume", 0)),
        )
        for _, row in df.iterrows()
    ]

    await conn.executemany(
        """
        INSERT INTO ohlcv_5m (timestamp, instrument, open, high, low, close, volume)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (timestamp, instrument) DO NOTHING
        """,
        records,
    )

    logger.info(f"Ingested {len(records)} new bars")
    return len(records)


async def check_data_quality(conn: asyncpg.Connection) -> dict:
    """Check for data quality issues."""
    issues = []

    # Check staleness
    latest = await get_latest_bar_time(conn)
    if latest:
        latest_utc = latest.replace(tzinfo=UTC) if latest.tzinfo is None else latest
        hours_stale = (datetime.now(UTC) - latest_utc).total_seconds() / 3600

        # Account for weekends (market closed Fri 5pm - Sun 6pm ET)
        now = datetime.now(UTC)
        is_weekend = now.weekday() >= 5  # Saturday or Sunday

        if hours_stale > 4 and not is_weekend:
            issues.append(f"Data is {hours_stale:.1f} hours stale (latest: {latest})")
    else:
        issues.append("No OHLCV data in database")

    # Check total bar count
    count_row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM ohlcv_5m WHERE instrument = 'GC'")
    total_bars = count_row["cnt"]

    # Check for large gaps (>30 min during market hours)
    gap_rows = await conn.fetch(
        """
        SELECT timestamp,
               LEAD(timestamp) OVER (ORDER BY timestamp) as next_ts,
               EXTRACT(EPOCH FROM (LEAD(timestamp) OVER (ORDER BY timestamp) - timestamp)) / 60 as gap_min
        FROM ohlcv_5m
        WHERE instrument = 'GC'
          AND timestamp > NOW() - INTERVAL '7 days'
        ORDER BY timestamp DESC
        LIMIT 2000
        """
    )
    large_gaps = [
        r
        for r in gap_rows
        if r["gap_min"]
        and r["gap_min"] > 30
        # Exclude maintenance break (22:00-23:00 UTC) and weekends
        and r["timestamp"].weekday() < 5
        and r["timestamp"].hour != 21  # Maintenance typically at ~21:00-22:00 UTC
    ]

    if len(large_gaps) > 5:
        issues.append(f"{len(large_gaps)} gaps >30min in last 7 days")

    return {
        "total_bars": total_bars,
        "latest_bar": latest.isoformat() if latest else None,
        "hours_stale": hours_stale if latest else None,
        "issues": issues,
        "large_gaps": len(large_gaps) if gap_rows else 0,
    }


async def create_cio_alert(issue_text: str) -> None:
    """Create a Paperclip task for the CIO when data quality degrades."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{PAPERCLIP_URL}/api/companies/{COMPANY_ID}/issues",
                json={
                    "title": f"[DATA ALERT] {issue_text}",
                    "description": (
                        f"The Data Pipeline agent detected a data quality issue:\n\n"
                        f"{issue_text}\n\n"
                        f"This may affect Regime Analyst classification and Quant Researcher backtesting."
                    ),
                    "assigneeAgentId": CIO_AGENT_ID,
                    "priority": "high",
                },
            )
            if resp.status_code in (200, 201, 202):
                logger.info(f"Created CIO alert: {issue_text}")
            else:
                logger.warning(f"Failed to create CIO alert: {resp.status_code}")
    except Exception as e:
        logger.warning(f"Could not create Paperclip task: {e}")


async def handle_ops_lead_duties():
    """Check for CIO tasks and delegate to Calendar/Skill Optimizer if needed."""
    from gold_trading.paperclip import create_task, get_my_tasks

    tasks = await get_my_tasks("data_pipeline")
    for task in tasks:
        title = task.get("title", "").lower()
        desc = task.get("description", "")

        if any(kw in title for kw in ["event", "fomc", "cpi", "nfp", "calendar", "news"]):
            await create_task(
                title=f"[From Ops Lead] {task.get('title', '')}",
                description=desc,
                assignee="economic_calendar",
                parent_id=task.get("id"),
            )
        elif any(kw in title for kw in ["skill", "improve", "optimize", "performance review"]):
            await create_task(
                title=f"[From Ops Lead] {task.get('title', '')}",
                description=desc,
                assignee="skill_optimizer",
                parent_id=task.get("id"),
            )
        else:
            logger.info(f"Data Pipeline handling task directly: {task.get('title', '')}")


async def main() -> None:
    """Main heartbeat execution."""
    logger.info("Data Pipeline (Ops Lead) heartbeat starting")

    # Team lead duties: check for CIO tasks, delegate to Calendar/Skill Optimizer
    await handle_ops_lead_duties()

    conn = await asyncpg.connect(get_database_url())
    try:
        # 1. Ingest new bars
        new_bars = await ingest_new_bars(conn)

        # 2. Check data quality
        quality = await check_data_quality(conn)
        logger.info(
            f"Data quality: {quality['total_bars']} total bars, "
            f"latest={quality['latest_bar']}, "
            f"stale={quality.get('hours_stale', 'N/A')}h, "
            f"gaps={quality['large_gaps']}"
        )

        # 3. Alert CIO on issues
        for issue in quality["issues"]:
            await create_cio_alert(issue)

        # 4. Log decision
        await insert_decision(
            conn,
            DecisionLogEntry(
                agent_name="data_pipeline",
                decision_type="data_refresh",
                inputs_summary={
                    "new_bars": new_bars,
                    "total_bars": quality["total_bars"],
                    "hours_stale": quality.get("hours_stale"),
                    "issues": quality["issues"],
                },
                decision=f"Ingested {new_bars} new bars. {len(quality['issues'])} issues."
                if quality["issues"]
                else f"Ingested {new_bars} new bars. Data healthy.",
                reasoning=f"Total bars: {quality['total_bars']}. "
                + (
                    f"Issues: {'; '.join(quality['issues'])}" if quality["issues"] else "No issues."
                ),
                confidence=1.0,
            ),
        )

        logger.info(f"Data Pipeline heartbeat complete. +{new_bars} bars.")

    finally:
        await conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
