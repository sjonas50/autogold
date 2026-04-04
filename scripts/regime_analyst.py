"""Regime Analyst agent — classifies gold market regime using ATR, ADX, and HMM.

Paperclip process adapter script. Runs on heartbeat (every 30 minutes).
Loads 5m OHLCV data from TimescaleDB, calculates ATR(14) and ADX(14),
runs GaussianHMM inference, writes regime classification to regime_state and decision_log.

Regime states: trending_up, trending_down, ranging, volatile

Run manually: uv run python scripts/regime_analyst.py
"""

import asyncio
import hashlib
import pickle
from pathlib import Path

import asyncpg
import numpy as np
import pandas as pd
from loguru import logger

from gold_trading.db.client import get_database_url
from gold_trading.db.queries.decisions import insert_decision
from gold_trading.db.queries.macro import get_latest_macro
from gold_trading.db.queries.regime import get_latest_regime, insert_regime_state
from gold_trading.models.lesson import DecisionLogEntry
from gold_trading.models.regime import RegimeState

# HMM model path (trained offline, loaded at runtime)
HMM_MODEL_PATH = Path(__file__).parent.parent / "data" / "hmm_regime_model.pkl"

# Regime labels mapped from HMM states
# These get calibrated during initial model training
REGIME_LABELS = {
    0: "trending_up",
    1: "trending_down",
    2: "ranging",
    3: "volatile",
}

# Fallback thresholds when HMM model is not yet trained
ADX_TRENDING_THRESHOLD = 25.0
ATR_VOLATILE_THRESHOLD_MULTIPLIER = 2.0  # >2x 50-bar ATR average = volatile


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)

    tr = pd.concat(
        [high - low, (high - close).abs(), (low - close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.rolling(window=period).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Plus/Minus Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index
    )

    # True Range
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)

    # Smoothed
    atr_smooth = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_smooth)

    # DX and ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()

    return adx


def classify_regime_thresholds(
    atr_14: float,
    adx_14: float,
    atr_50_avg: float,
    return_5bar: float,
) -> tuple[str, float]:
    """Classify regime using simple thresholds (fallback when HMM not available).

    Returns:
        Tuple of (regime_label, confidence).
    """
    # Volatile: ATR is 2x+ the 50-bar average
    if atr_50_avg > 0 and atr_14 > atr_50_avg * ATR_VOLATILE_THRESHOLD_MULTIPLIER:
        return "volatile", 0.8

    # Trending: ADX > 25
    if adx_14 > ADX_TRENDING_THRESHOLD:
        if return_5bar > 0:
            return "trending_up", min(0.5 + adx_14 / 100, 0.9)
        else:
            return "trending_down", min(0.5 + adx_14 / 100, 0.9)

    # Ranging: ADX < 20 and not volatile
    if adx_14 < 20:
        return "ranging", min(0.5 + (20 - adx_14) / 40, 0.85)

    # Transition zone: ADX between 20-25
    if return_5bar > 0:
        return "trending_up", 0.45
    else:
        return "ranging", 0.45


def classify_regime_hmm(
    features: np.ndarray,
    model_path: Path,
) -> tuple[int, float]:
    """Classify regime using pre-trained GaussianHMM.

    Args:
        features: 2D array of shape (n_samples, n_features).
            Features: [log_return, atr_14, adx_14, dxy_change]
        model_path: Path to pickled HMM model.

    Returns:
        Tuple of (hmm_state_index, confidence).
    """
    # Validate model file hash before deserializing
    model_hash_path = model_path.with_suffix(".sha256")
    if model_hash_path.exists():
        expected_hash = model_hash_path.read_text().strip()
        actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"HMM model file hash mismatch. Expected {expected_hash}, got {actual_hash}. "
                "File may be corrupted or tampered with."
            )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Normalize features using the model's stored params (from training)
    norm_means = getattr(model, "_norm_means", None)
    norm_stds = getattr(model, "_norm_stds", None)
    if norm_means is not None and norm_stds is not None:
        features = (features - norm_means) / norm_stds

    # Predict most likely state for the latest observation
    state_probs = model.predict_proba(features)
    latest_probs = state_probs[-1]
    state = int(np.argmax(latest_probs))
    confidence = float(latest_probs[state])

    return state, confidence


async def handle_team_lead_duties(conn):
    """Check for CIO tasks and delegate to Macro/Sentiment if needed."""
    from gold_trading.paperclip import create_task, get_my_tasks

    tasks = await get_my_tasks("regime_analyst")
    for task in tasks:
        title = task.get("title", "").lower()
        desc = task.get("description", "")

        if any(kw in title for kw in ["macro", "fred", "yield", "dxy", "cpi"]):
            await create_task(
                title=f"[From Intelligence Lead] {task.get('title', '')}",
                description=desc,
                assignee="macro_analyst",
                parent_id=task.get("id"),
            )
        elif any(kw in title for kw in ["sentiment", "news", "headline", "fomc"]):
            await create_task(
                title=f"[From Intelligence Lead] {task.get('title', '')}",
                description=desc,
                assignee="sentiment_analyst",
                parent_id=task.get("id"),
            )
        else:
            logger.info(f"Regime Analyst handling task directly: {task.get('title', '')}")


async def main() -> None:
    """Main heartbeat execution."""
    logger.info("Regime Analyst (Intelligence Lead) heartbeat starting")

    conn = await asyncpg.connect(get_database_url())
    try:
        # Team lead duties: check for CIO tasks, delegate to Macro/Sentiment
        await handle_team_lead_duties(conn)

        # Load multi-timeframe OHLCV data
        from gold_trading.db.queries.ohlcv import get_multi_timeframe

        mtf = await get_multi_timeframe(conn, bars_5m=200, bars_15m=200, bars_1h=200, bars_daily=50)
        df = mtf.get("5m")

        if df is None or len(df) < 30:
            logger.warning(
                f"Insufficient OHLCV data ({len(df) if df is not None else 0} bars). "
                "Need at least 30 bars for regime classification. "
                "Ingest historical data first."
            )
            return

        # Calculate indicators
        df["atr_14"] = calculate_atr(df, period=14)
        df["adx_14"] = calculate_adx(df, period=14)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        latest = df.iloc[-1]
        atr_14 = float(latest["atr_14"]) if pd.notna(latest["atr_14"]) else 0.0
        adx_14 = float(latest["adx_14"]) if pd.notna(latest["adx_14"]) else 0.0

        # 50-bar ATR average for volatility context
        atr_50_avg = float(df["atr_14"].tail(50).mean()) if len(df) >= 50 else atr_14

        # 5-bar return for direction
        return_5bar = float(df["close"].iloc[-1] - df["close"].iloc[-6]) if len(df) >= 6 else 0.0

        # Get DXY change from macro data
        macro = await get_latest_macro(conn)
        dxy_change = None
        if macro and macro.dxy:
            dxy_change = 0.0  # Would need previous day's DXY to compute change

        # Try HMM first, fall back to threshold-based
        hmm_state = None
        hmm_confidence = None

        if HMM_MODEL_PATH.exists():
            try:
                # Must match training features: log_return, atr_14, adx_14, volatility_20
                df["volatility_20"] = df["log_return"].rolling(20).std()
                feature_df = df[["log_return", "atr_14", "adx_14", "volatility_20"]].dropna()
                features = feature_df.values
                if len(features) >= 10:
                    hmm_state, hmm_confidence = classify_regime_hmm(features, HMM_MODEL_PATH)
                    # Use model's own label mapping if available
                    with open(HMM_MODEL_PATH, "rb") as _f:
                        _model = pickle.load(_f)
                    model_labels = getattr(_model, "_state_labels", REGIME_LABELS)
                    regime = model_labels.get(hmm_state, REGIME_LABELS.get(hmm_state, "ranging"))
                    confidence = hmm_confidence
                    logger.info(f"HMM regime: {regime} (state={hmm_state}, conf={confidence:.3f})")
                else:
                    raise ValueError("Insufficient features for HMM")
            except Exception as e:
                logger.warning(f"HMM classification failed: {e}. Using threshold fallback.")
                regime, confidence = classify_regime_thresholds(
                    atr_14, adx_14, atr_50_avg, return_5bar
                )
        else:
            logger.info("HMM model not found — using threshold-based classification")
            regime, confidence = classify_regime_thresholds(atr_14, adx_14, atr_50_avg, return_5bar)

        # Write to regime_state
        # Multi-timeframe confirmation BEFORE writing to DB
        htf_context = {}
        df_15m = mtf.get("15m")
        df_1h = mtf.get("1h")
        df_daily = mtf.get("daily")

        if df_15m is not None and len(df_15m) >= 20:
            df_15m["atr_14"] = calculate_atr(df_15m, period=14)
            df_15m["adx_14"] = calculate_adx(df_15m, period=14)
            m15_adx = float(df_15m["adx_14"].iloc[-1]) if pd.notna(df_15m["adx_14"].iloc[-1]) else 0
            m15_return = (
                float(df_15m["close"].iloc[-1] - df_15m["close"].iloc[-6])
                if len(df_15m) >= 6
                else 0
            )
            m15_trend = "up" if m15_return > 0 else "down" if m15_return < 0 else "flat"
            htf_context["15m"] = {
                "adx": round(m15_adx, 1),
                "trend": m15_trend,
                "return": round(m15_return, 2),
            }
            logger.info(
                f"15m context: ADX={m15_adx:.1f}, trend={m15_trend}, return={m15_return:.2f}"
            )

        if df_1h is not None and len(df_1h) >= 20:
            df_1h["atr_14"] = calculate_atr(df_1h, period=14)
            df_1h["adx_14"] = calculate_adx(df_1h, period=14)
            h1_adx = float(df_1h["adx_14"].iloc[-1]) if pd.notna(df_1h["adx_14"].iloc[-1]) else 0
            h1_return = (
                float(df_1h["close"].iloc[-1] - df_1h["close"].iloc[-6]) if len(df_1h) >= 6 else 0
            )
            h1_trend = "up" if h1_return > 0 else "down" if h1_return < 0 else "flat"
            htf_context["1h"] = {
                "adx": round(h1_adx, 1),
                "trend": h1_trend,
                "return": round(h1_return, 2),
            }
            logger.info(f"1H context: ADX={h1_adx:.1f}, trend={h1_trend}, return={h1_return:.2f}")

        if df_daily is not None and len(df_daily) >= 10:
            daily_return_5d = (
                float(df_daily["close"].iloc[-1] - df_daily["close"].iloc[-6])
                if len(df_daily) >= 6
                else 0
            )
            daily_return_20d = (
                float(df_daily["close"].iloc[-1] - df_daily["close"].iloc[-21])
                if len(df_daily) >= 21
                else 0
            )
            htf_context["daily"] = {
                "return_5d": round(daily_return_5d, 2),
                "return_20d": round(daily_return_20d, 2),
                "trend": "up" if daily_return_20d > 0 else "down",
            }
            logger.info(
                f"Daily context: 5d return=${daily_return_5d:.2f}, 20d return=${daily_return_20d:.2f}"
            )

        # Adjust confidence based on multi-TF alignment
        agreement_count = 0
        total_htf = 0
        for tf_key in ["15m", "1h"]:
            if htf_context.get(tf_key):
                total_htf += 1
                tf_trend = htf_context[tf_key]["trend"]
                if regime in ("trending_up", "trending_down"):
                    expected = "up" if regime == "trending_up" else "down"
                    if tf_trend == expected:
                        agreement_count += 1
        if htf_context.get("daily"):
            total_htf += 1
            if regime in ("trending_up", "trending_down") and htf_context["daily"]["trend"] == (
                "up" if regime == "trending_up" else "down"
            ):
                agreement_count += 1
        if total_htf > 0:
            alignment_ratio = agreement_count / total_htf
            if alignment_ratio >= 0.67:
                confidence = min(confidence * 1.2, 0.99)
                logger.info(
                    f"Multi-TF alignment: {agreement_count}/{total_htf} agree — boosting confidence"
                )
            elif alignment_ratio == 0:
                confidence *= 0.6
                logger.warning(f"Multi-TF conflict: 0/{total_htf} agree — reducing confidence")

        state = RegimeState(
            regime=regime,
            hmm_state=hmm_state,
            hmm_confidence=hmm_confidence or confidence,
            atr_14=round(atr_14, 4),
            adx_14=round(adx_14, 3),
            dxy_change_1d=dxy_change,
        )
        await insert_regime_state(conn, state)

        # Write to decision_log
        await insert_decision(
            conn,
            DecisionLogEntry(
                agent_name="regime_analyst",
                decision_type="regime_classification",
                inputs_summary={
                    "atr_14": round(atr_14, 4),
                    "adx_14": round(adx_14, 3),
                    "atr_50_avg": round(atr_50_avg, 4),
                    "return_5bar": round(return_5bar, 4),
                    "bars_loaded": len(df),
                    "method": "hmm" if hmm_state is not None else "threshold",
                    "htf_context": htf_context,
                },
                decision=regime,
                reasoning=(
                    f"5m: ATR={atr_14:.4f}, ADX={adx_14:.1f}, return={return_5bar:.2f}. "
                    f"{'HMM state ' + str(hmm_state) if hmm_state is not None else 'Threshold'}. "
                    f"15m: {htf_context.get('15m', 'N/A')}. "
                    f"1H: {htf_context.get('1h', 'N/A')}. "
                    f"Daily: {htf_context.get('daily', 'N/A')}. "
                    f"MTF alignment: {agreement_count}/{total_htf}. "
                    f"Regime: {regime}, confidence {confidence:.2f}."
                ),
                confidence=round(confidence, 2),
            ),
        )

        # Check if regime changed
        previous = await get_latest_regime(conn)
        if previous and previous.regime != regime:
            logger.warning(
                f"Regime changed: {previous.regime} → {regime} (ATR={atr_14:.4f}, ADX={adx_14:.1f})"
            )

        logger.info(
            f"Regime Analyst heartbeat complete. "
            f"Regime={regime}, ATR={atr_14:.4f}, ADX={adx_14:.1f}, Confidence={confidence:.2f}"
        )

    finally:
        await conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
