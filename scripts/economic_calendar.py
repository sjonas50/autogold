"""Economic Calendar agent — monitors upcoming events and creates pre-event alerts.

Paperclip process adapter script. Runs on heartbeat (every 60 minutes).
Checks for upcoming high-impact economic events (FOMC, CPI, NFP, etc.)
and creates Paperclip tasks for the CIO to deactivate strategies before events.

Uses the FRED release calendar and a hardcoded list of recurring events.

Run manually: uv run python scripts/economic_calendar.py
"""

import asyncio
import os
from datetime import UTC, datetime, timedelta

import asyncpg
import httpx
from loguru import logger

from gold_trading.db.client import get_database_url
from gold_trading.db.queries.decisions import insert_decision
from gold_trading.models.lesson import DecisionLogEntry

PAPERCLIP_URL = os.environ.get("PAPERCLIP_URL", "http://localhost:3100")
COMPANY_ID = os.environ.get("PAPERCLIP_COMPANY_ID", "3422f81a-8ca2-4ce1-aae5-5cf8ce34fa0e")
CIO_AGENT_ID = "37bbe408-e573-4598-a374-cc369bad0258"

# High-impact events that move gold
HIGH_IMPACT_EVENTS = {
    "FOMC": "Federal Reserve interest rate decision and statement",
    "CPI": "Consumer Price Index — inflation data",
    "NFP": "Nonfarm Payrolls — employment data",
    "PPI": "Producer Price Index",
    "PCE": "Personal Consumption Expenditures — Fed's preferred inflation gauge",
    "GDP": "Gross Domestic Product",
    "RETAIL": "Retail Sales",
    "ISM": "ISM Manufacturing/Services PMI",
}


async def fetch_upcoming_fred_releases() -> list[dict]:
    """Fetch upcoming FRED data releases (next 7 days)."""
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.warning("FRED_API_KEY not set — using hardcoded calendar only")
        return []

    now = datetime.now(UTC)
    end = now + timedelta(days=7)

    url = "https://api.stlouisfed.org/fred/releases/dates"
    params = {
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": now.strftime("%Y-%m-%d"),
        "realtime_end": end.strftime("%Y-%m-%d"),
        "include_release_dates_with_no_data": "true",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
        if resp.status_code != 200:
            logger.warning(f"FRED releases API error: {resp.status_code}")
            return []

        data = resp.json()
        releases = data.get("release_dates", [])

    # Filter for gold-relevant releases
    gold_relevant = []
    for release in releases:
        name = release.get("release_name", "").upper()
        for event_key, description in HIGH_IMPACT_EVENTS.items():
            if event_key in name or any(word in name for word in description.upper().split()[:2]):
                gold_relevant.append(
                    {
                        "event": event_key,
                        "name": release.get("release_name", ""),
                        "date": release.get("date", ""),
                        "description": description,
                    }
                )
                break

    return gold_relevant


async def check_imminent_events(upcoming: list[dict]) -> list[dict]:
    """Find events happening within the next 2 hours."""
    now = datetime.now(UTC)
    imminent = []

    for event in upcoming:
        event_date = event.get("date", "")
        try:
            # FRED dates are date-only; economic releases are typically at 8:30 AM ET
            event_dt = datetime.strptime(event_date, "%Y-%m-%d").replace(
                hour=12,
                minute=30,
                tzinfo=UTC,  # 8:30 AM ET = 12:30 UTC
            )
            hours_until = (event_dt - now).total_seconds() / 3600

            if 0 < hours_until <= 2:
                event["hours_until"] = round(hours_until, 1)
                imminent.append(event)
        except (ValueError, TypeError):
            continue

    return imminent


async def create_cio_event_alert(event: dict) -> None:
    """Create a Paperclip task for CIO to deactivate strategies before an event."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{PAPERCLIP_URL}/api/companies/{COMPANY_ID}/issues",
                json={
                    "title": f"[EVENT] {event['event']} in {event.get('hours_until', '?')}h — deactivate all strategies",
                    "description": (
                        f"**{event['name']}** is scheduled within the next {event.get('hours_until', '?')} hours.\n\n"
                        f"Description: {event['description']}\n"
                        f"Date: {event['date']}\n\n"
                        f"**Action required:** Deactivate all active strategies at least 60 minutes before the release. "
                        f"Gold can move $20+ in minutes during {event['event']}.\n\n"
                        f"Re-enable strategies 30 minutes after the release if regime + sentiment are favorable."
                    ),
                    "assigneeAgentId": CIO_AGENT_ID,
                    "priority": "urgent",
                },
            )
            if resp.status_code in (200, 201, 202):
                logger.info(
                    f"Created CIO event alert: {event['event']} in {event.get('hours_until')}h"
                )
            else:
                logger.warning(f"Failed to create event alert: {resp.status_code}")
    except Exception as e:
        logger.warning(f"Could not create Paperclip task: {e}")


async def main() -> None:
    """Main heartbeat execution."""
    logger.info("Economic Calendar heartbeat starting")

    # Fetch upcoming releases
    upcoming = await fetch_upcoming_fred_releases()
    logger.info(f"Found {len(upcoming)} gold-relevant releases in next 7 days")

    # Check for imminent events
    imminent = await check_imminent_events(upcoming)

    if imminent:
        for event in imminent:
            logger.warning(
                f"IMMINENT EVENT: {event['event']} ({event['name']}) in {event.get('hours_until')}h"
            )
            await create_cio_event_alert(event)

    # Log to DB
    conn = await asyncpg.connect(get_database_url())
    try:
        await insert_decision(
            conn,
            DecisionLogEntry(
                agent_name="economic_calendar",
                decision_type="calendar_check",
                inputs_summary={
                    "upcoming_events": len(upcoming),
                    "imminent_events": len(imminent),
                    "events": [{"event": e["event"], "date": e["date"]} for e in upcoming[:5]],
                },
                decision=f"{len(imminent)} imminent events"
                if imminent
                else f"No imminent events. {len(upcoming)} upcoming in 7 days.",
                reasoning="; ".join(
                    f"{e['event']} in {e.get('hours_until', '?')}h" for e in imminent
                )
                or "All clear — no high-impact events within 2 hours.",
                confidence=1.0,
            ),
        )

        logger.info(
            f"Economic Calendar heartbeat complete. "
            f"{len(upcoming)} upcoming, {len(imminent)} imminent."
        )
    finally:
        await conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
