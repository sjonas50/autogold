"""Database queries for multi-timeframe OHLCV data."""

import asyncpg
import pandas as pd

TIMEFRAME_TABLES = {
    "5m": "ohlcv_5m",
    "15m": "ohlcv_15m",
    "1h": "ohlcv_1h",
    "1d": "ohlcv_daily",
    "daily": "ohlcv_daily",
}


async def get_ohlcv(
    conn: asyncpg.Connection,
    timeframe: str = "5m",
    instrument: str = "GC",
    bars: int = 200,
) -> pd.DataFrame | None:
    """Load OHLCV data for any timeframe.

    Args:
        conn: Database connection.
        timeframe: One of '5m', '15m', '1h', '1d'/'daily'.
        instrument: 'GC' or 'MGC'.
        bars: Number of most recent bars to load.

    Returns:
        DataFrame with timestamp, open, high, low, close, volume columns.
        Sorted oldest → newest. None if no data.
    """
    table = TIMEFRAME_TABLES.get(timeframe)
    if not table:
        raise ValueError(f"Unknown timeframe: {timeframe}. Valid: {list(TIMEFRAME_TABLES.keys())}")

    rows = await conn.fetch(
        f"""
        SELECT timestamp, open, high, low, close, volume
        FROM {table}
        WHERE instrument = $1
        ORDER BY timestamp DESC
        LIMIT $2
        """,
        instrument,
        bars,
    )

    if not rows:
        return None

    df = pd.DataFrame([dict(r) for r in rows])
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(int)
    return df


async def get_multi_timeframe(
    conn: asyncpg.Connection,
    instrument: str = "GC",
    bars_5m: int = 200,
    bars_15m: int = 100,
    bars_1h: int = 100,
    bars_daily: int = 50,
) -> dict[str, pd.DataFrame | None]:
    """Load OHLCV data across all timeframes.

    Returns dict mapping timeframe → DataFrame (or None if no data).
    """
    return {
        "5m": await get_ohlcv(conn, "5m", instrument, bars_5m),
        "15m": await get_ohlcv(conn, "15m", instrument, bars_15m),
        "1h": await get_ohlcv(conn, "1h", instrument, bars_1h),
        "daily": await get_ohlcv(conn, "daily", instrument, bars_daily),
    }


async def get_bar_counts(conn: asyncpg.Connection, instrument: str = "GC") -> dict[str, int]:
    """Get total bar count for each timeframe."""
    counts = {}
    for tf, table in TIMEFRAME_TABLES.items():
        if tf == "1d":
            continue  # Skip alias
        row = await conn.fetchrow(
            f"SELECT COUNT(*) as cnt FROM {table} WHERE instrument = $1",
            instrument,
        )
        counts[tf] = row["cnt"]
    return counts
