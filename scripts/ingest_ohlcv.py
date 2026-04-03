"""Ingest historical 5-minute GC (gold futures) OHLCV data into TimescaleDB.

Sources (in priority order):
1. yfinance — free, ~60 days of 5m data for GC=F (continuous gold futures)
2. Polygon.io — if POLYGON_API_KEY is set, pulls more history

Usage:
    # Pull free data via yfinance (~60 days of 5m bars)
    uv run python scripts/ingest_ohlcv.py

    # Pull from Polygon.io (requires API key, more history)
    uv run python scripts/ingest_ohlcv.py --source polygon --days 365

    # Load from a local CSV file (e.g., FirstRate Data purchase)
    uv run python scripts/ingest_ohlcv.py --source csv --file data/gc_5m.csv
"""

import argparse
import asyncio
import os
from datetime import UTC, datetime, timedelta

import asyncpg
import pandas as pd
from loguru import logger

from gold_trading.db.client import get_database_url


async def ingest_dataframe(df: pd.DataFrame, instrument: str = "GC") -> int:
    """Load a DataFrame of OHLCV data into the ohlcv_5m table.

    Expected columns: timestamp (or index), open, high, low, close, volume.
    Returns number of rows inserted.
    """
    conn = await asyncpg.connect(get_database_url())
    try:
        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]

        if "timestamp" not in df.columns and df.index.name == "Datetime":
            df = df.reset_index()
            df = df.rename(columns={"Datetime": "timestamp"})
        elif "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "timestamp"})

        # Ensure timestamp is timezone-aware
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Drop rows with NaN prices
        df = df.dropna(subset=["open", "high", "low", "close"])

        # Remove duplicates
        df = df.drop_duplicates(subset=["timestamp"])

        logger.info(f"Preparing to ingest {len(df)} bars for {instrument}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Batch insert with ON CONFLICT to handle re-runs
        records = [
            (
                row["timestamp"],
                instrument,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row.get("volume", 0)),
            )
            for _, row in df.iterrows()
        ]

        await conn.executemany(
            """
            INSERT INTO ohlcv_5m (timestamp, instrument, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (timestamp, instrument) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
            """,
            records,
        )

        # Verify
        count = await conn.fetchrow(
            "SELECT COUNT(*) as cnt FROM ohlcv_5m WHERE instrument = $1", instrument
        )
        logger.info(f"Total bars in ohlcv_5m for {instrument}: {count['cnt']}")
        return len(records)

    finally:
        await conn.close()


def fetch_yfinance(days: int = 60) -> pd.DataFrame:
    """Fetch 5-minute GC=F data from Yahoo Finance.

    Free, no API key needed. Limited to ~60 days of intraday data.
    """
    import yfinance as yf

    ticker = yf.Ticker("GC=F")
    logger.info(f"Fetching GC=F 5m data from yfinance (last {days} days)...")

    # yfinance max intraday period is 60 days for 5m
    df = ticker.history(period=f"{min(days, 60)}d", interval="5m")

    if df.empty:
        raise RuntimeError("yfinance returned no data for GC=F. Market may be closed.")

    logger.info(f"yfinance returned {len(df)} bars")
    return df


async def fetch_polygon(days: int = 365) -> pd.DataFrame:
    """Fetch 5-minute GC data from Polygon.io.

    Requires POLYGON_API_KEY. Supports longer history than yfinance.
    """
    import httpx

    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set")

    end = datetime.now(UTC)
    start = end - timedelta(days=days)

    all_bars = []
    current_start = start

    async with httpx.AsyncClient(timeout=30) as client:
        while current_start < end:
            chunk_end = min(current_start + timedelta(days=30), end)

            url = (
                f"https://api.polygon.io/v2/aggs/ticker/C:XAUUSD/range/5/minute/"
                f"{current_start.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}"
            )
            params = {"apiKey": api_key, "limit": 50000, "sort": "asc"}

            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                logger.warning(f"Polygon.io API error: {resp.status_code} {resp.text[:200]}")
                break

            data = resp.json()
            results = data.get("results", [])
            if not results:
                logger.info(f"No data for {current_start.date()} to {chunk_end.date()}")
                current_start = chunk_end
                continue

            for bar in results:
                all_bars.append(
                    {
                        "timestamp": pd.Timestamp(bar["t"], unit="ms", tz="UTC"),
                        "open": bar["o"],
                        "high": bar["h"],
                        "low": bar["l"],
                        "close": bar["c"],
                        "volume": bar.get("v", 0),
                    }
                )

            logger.info(f"Fetched {len(results)} bars for {current_start.date()} to {chunk_end.date()}")
            current_start = chunk_end

    if not all_bars:
        raise RuntimeError("Polygon.io returned no data")

    df = pd.DataFrame(all_bars)
    logger.info(f"Polygon.io total: {len(df)} bars")
    return df


def load_csv(filepath: str) -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Supports common formats from FirstRate Data, Twelve Data, etc.
    Expected columns: timestamp/date/datetime, open, high, low, close, volume.
    """
    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip() for c in df.columns]

    # Try to find the timestamp column
    for col in ["timestamp", "datetime", "date", "time"]:
        if col in df.columns:
            df = df.rename(columns={col: "timestamp"})
            break

    if "timestamp" not in df.columns:
        raise ValueError(f"Could not find timestamp column in CSV. Columns: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    logger.info(f"Loaded {len(df)} bars from {filepath}")
    return df


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest GC OHLCV data into TimescaleDB")
    parser.add_argument(
        "--source",
        choices=["yfinance", "polygon", "csv"],
        default="yfinance",
        help="Data source (default: yfinance)",
    )
    parser.add_argument("--days", type=int, default=60, help="Days of history to fetch")
    parser.add_argument("--file", type=str, help="CSV file path (for --source csv)")
    parser.add_argument("--instrument", type=str, default="GC", help="Instrument symbol")
    args = parser.parse_args()

    if args.source == "yfinance":
        df = fetch_yfinance(args.days)
    elif args.source == "polygon":
        df = await fetch_polygon(args.days)
    elif args.source == "csv":
        if not args.file:
            parser.error("--file is required when --source csv")
        df = load_csv(args.file)
    else:
        parser.error(f"Unknown source: {args.source}")

    count = await ingest_dataframe(df, instrument=args.instrument)
    logger.info(f"Ingestion complete: {count} bars loaded for {args.instrument}")


if __name__ == "__main__":
    asyncio.run(main())
