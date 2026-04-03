"""Macro Analyst agent — fetches FRED economic data and classifies gold macro regime.

Paperclip process adapter script. Runs on heartbeat (every 4 hours).
Reads FRED series: DXY proxy, real yields, CPI, breakeven inflation, oil, gold fix.
Classifies macro regime as bullish/neutral/bearish for gold via Claude Sonnet 4.6.
Writes results to macro_data and decision_log tables.

Run manually: uv run python scripts/macro_analyst.py
"""

import asyncio
import json
import os
from datetime import date

import anthropic
import asyncpg
from loguru import logger

from gold_trading.db.client import get_database_url
from gold_trading.db.queries.decisions import insert_decision
from gold_trading.db.queries.macro import get_latest_macro, insert_macro_data
from gold_trading.models.lesson import DecisionLogEntry
from gold_trading.models.macro import MacroData

# FRED series IDs relevant to gold
FRED_SERIES = {
    "DTWEXBGS": "dxy",  # Broad USD Index (DXY proxy)
    "DFII10": "real_yield_10y",  # 10-Year TIPS Real Yield
    "CPIAUCSL": "cpi_yoy",  # CPI (we'll compute YoY)
    "T10YIE": "breakeven_10y",  # 10-Year Breakeven Inflation
    "DCOILWTICO": "oil_wti",  # WTI Crude Oil
    "GOLDAMGBD228NLBM": "gold_fix_pm",  # London Gold PM Fix
}

ANTHROPIC_MODEL = "claude-sonnet-4-6"


async def fetch_fred_series(series_id: str, api_key: str) -> float | None:
    """Fetch the most recent observation for a FRED series."""
    import httpx

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 5,
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
        if resp.status_code != 200:
            logger.warning(f"FRED API error for {series_id}: {resp.status_code}")
            return None

        data = resp.json()
        observations = data.get("observations", [])

        for obs in observations:
            if obs["value"] != ".":
                return float(obs["value"])

    return None


async def fetch_all_fred_data(api_key: str) -> dict[str, float | None]:
    """Fetch all FRED series concurrently."""
    import asyncio

    async def _fetch(field_name: str, series_id: str) -> tuple[str, float | None]:
        value = await fetch_fred_series(series_id, api_key)
        return field_name, value

    tasks = [_fetch(field_name, series_id) for series_id, field_name in FRED_SERIES.items()]
    results_list = await asyncio.gather(*tasks)
    return dict(results_list)


async def classify_macro_regime(
    data: dict[str, float | None],
    previous_regime: str | None,
) -> tuple[str, str]:
    """Use Claude Sonnet 4.6 to classify gold macro regime.

    Returns:
        Tuple of (regime, reasoning) where regime is bullish/neutral/bearish.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    prompt = f"""You are a gold macro analyst. Based on the following economic data, classify the current macro regime for gold as one of: bullish, neutral, or bearish.

Current data:
- USD Index (DXY proxy): {data.get("dxy", "N/A")}
- 10-Year Real Yield (TIPS): {data.get("real_yield_10y", "N/A")}%
- CPI YoY: {data.get("cpi_yoy", "N/A")}%
- 10-Year Breakeven Inflation: {data.get("breakeven_10y", "N/A")}%
- WTI Oil: ${data.get("oil_wti", "N/A")}
- London Gold PM Fix: ${data.get("gold_fix_pm", "N/A")}

Previous regime: {previous_regime or "None (first classification)"}

Key relationships for gold:
- Gold is inversely correlated with real yields (falling real yields = bullish gold)
- Gold is inversely correlated with USD strength (weak USD = bullish gold)
- Rising inflation expectations (breakeven) support gold as an inflation hedge
- Gold acts as a safe haven during geopolitical/economic uncertainty

Respond in this exact JSON format:
{{"regime": "bullish|neutral|bearish", "reasoning": "2-3 sentence explanation citing specific data points"}}"""

    response = await client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Parse JSON from response
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    parsed = json.loads(text)
    regime = parsed["regime"]
    reasoning = parsed["reasoning"]

    if regime not in ("bullish", "neutral", "bearish"):
        logger.warning(f"Unexpected regime: {regime}, defaulting to neutral")
        regime = "neutral"

    return regime, reasoning


async def main() -> None:
    """Main heartbeat execution."""
    logger.info("Macro Analyst heartbeat starting")

    fred_api_key = os.environ.get("FRED_API_KEY")
    if not fred_api_key:
        logger.error("FRED_API_KEY not set — cannot fetch economic data")
        return

    # Fetch FRED data
    fred_data = await fetch_all_fred_data(fred_api_key)
    logger.info(
        f"FRED data fetched: {json.dumps({k: v for k, v in fred_data.items() if v is not None})}"
    )

    # Get previous regime for context
    conn = await asyncpg.connect(get_database_url())
    try:
        previous = await get_latest_macro(conn)
        previous_regime = previous.macro_regime if previous else None

        # Classify regime
        regime, reasoning = await classify_macro_regime(fred_data, previous_regime)
        logger.info(f"Macro regime: {regime} — {reasoning}")

        # Write to macro_data
        macro = MacroData(
            observation_date=date.today(),
            dxy=fred_data.get("dxy"),
            real_yield_10y=fred_data.get("real_yield_10y"),
            cpi_yoy=fred_data.get("cpi_yoy"),
            breakeven_10y=fred_data.get("breakeven_10y"),
            oil_wti=fred_data.get("oil_wti"),
            gold_fix_pm=fred_data.get("gold_fix_pm"),
            macro_regime=regime,
            reasoning=reasoning,
        )
        await insert_macro_data(conn, macro)

        # Write to decision_log
        await insert_decision(
            conn,
            DecisionLogEntry(
                agent_name="macro_analyst",
                decision_type="macro_classification",
                inputs_summary={k: v for k, v in fred_data.items() if v is not None},
                decision=regime,
                reasoning=reasoning,
                confidence=0.7,  # FRED data is reliable but lagged 1-2 days
            ),
        )

        logger.info("Macro Analyst heartbeat complete")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
