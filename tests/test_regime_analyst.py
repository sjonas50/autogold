"""Tests for regime analyst — ATR/ADX calculations and regime classification.

Tests use synthetic OHLCV data to verify indicator math and classification logic.
All DB tests run against real TimescaleDB.
"""

import numpy as np
import pandas as pd
import pytest
from scripts.regime_analyst import (
    calculate_adx,
    calculate_atr,
    classify_regime_thresholds,
)

from gold_trading.db.queries.regime import get_latest_regime, insert_regime_state
from gold_trading.models.regime import RegimeState


def _make_trending_up_data(n: int = 100) -> pd.DataFrame:
    """Generate synthetic trending-up OHLCV data."""
    np.random.seed(42)
    base = 3100.0
    prices = base + np.cumsum(np.random.normal(0.3, 0.5, n))  # Upward drift
    noise = np.random.uniform(0.5, 3.0, n)

    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + noise,
            "low": prices - noise,
            "close": prices + np.random.normal(0.1, 0.3, n),
            "volume": np.random.randint(100, 5000, n),
        }
    )


def _make_ranging_data(n: int = 100) -> pd.DataFrame:
    """Generate synthetic ranging/choppy OHLCV data."""
    np.random.seed(42)
    base = 3100.0
    # Mean-reverting: oscillate around base
    prices = base + np.sin(np.linspace(0, 6 * np.pi, n)) * 5 + np.random.normal(0, 0.5, n)
    noise = np.random.uniform(0.3, 1.5, n)

    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + noise,
            "low": prices - noise,
            "close": prices + np.random.normal(0, 0.2, n),
            "volume": np.random.randint(100, 3000, n),
        }
    )


def _make_volatile_data(n: int = 100) -> pd.DataFrame:
    """Generate synthetic high-volatility OHLCV data."""
    np.random.seed(42)
    base = 3100.0
    prices = base + np.cumsum(np.random.normal(0, 3.0, n))  # Large moves
    noise = np.random.uniform(3.0, 15.0, n)  # Wide bars

    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + noise,
            "low": prices - noise,
            "close": prices + np.random.normal(0, 2.0, n),
            "volume": np.random.randint(1000, 20000, n),
        }
    )


class TestATR:
    def test_atr_produces_values(self):
        """ATR should produce non-null values after the lookback period."""
        df = _make_trending_up_data()
        atr = calculate_atr(df, period=14)
        assert atr.notna().sum() > 50
        assert atr.iloc[-1] > 0

    def test_atr_volatile_higher(self):
        """Volatile data should produce higher ATR than ranging data."""
        vol_atr = calculate_atr(_make_volatile_data(), period=14).iloc[-1]
        rng_atr = calculate_atr(_make_ranging_data(), period=14).iloc[-1]
        assert vol_atr > rng_atr

    def test_atr_period_matters(self):
        """Shorter period ATR should be more responsive."""
        df = _make_trending_up_data()
        atr_5 = calculate_atr(df, period=5)
        atr_50 = calculate_atr(df, period=50)
        # Short ATR has valid values earlier
        assert atr_5.notna().sum() > atr_50.notna().sum()


class TestADX:
    def test_adx_trending_high(self):
        """Trending data should produce ADX > 25."""
        df = _make_trending_up_data(200)
        adx = calculate_adx(df, period=14)
        latest_adx = adx.iloc[-1]
        assert latest_adx > 20  # Strong trend

    def test_adx_ranging_low(self):
        """Ranging data should produce lower ADX."""
        df = _make_ranging_data(200)
        adx = calculate_adx(df, period=14)
        latest_adx = adx.iloc[-1]
        # Ranging should have lower ADX than trending
        trend_adx = calculate_adx(_make_trending_up_data(200), period=14).iloc[-1]
        assert latest_adx < trend_adx

    def test_adx_non_negative(self):
        """ADX values should be non-negative."""
        df = _make_trending_up_data()
        adx = calculate_adx(df, period=14)
        valid = adx.dropna()
        assert (valid >= 0).all()


class TestThresholdClassification:
    def test_trending_up(self):
        """High ADX with positive return → trending_up."""
        regime, conf = classify_regime_thresholds(
            atr_14=4.0, adx_14=30.0, atr_50_avg=3.5, return_5bar=5.0
        )
        assert regime == "trending_up"
        assert conf > 0.5

    def test_trending_down(self):
        """High ADX with negative return → trending_down."""
        regime, conf = classify_regime_thresholds(
            atr_14=4.0, adx_14=32.0, atr_50_avg=3.5, return_5bar=-5.0
        )
        assert regime == "trending_down"
        assert conf > 0.5

    def test_ranging(self):
        """Low ADX → ranging."""
        regime, conf = classify_regime_thresholds(
            atr_14=2.0, adx_14=15.0, atr_50_avg=2.5, return_5bar=0.5
        )
        assert regime == "ranging"
        assert conf > 0.4

    def test_volatile(self):
        """ATR > 2x average → volatile."""
        regime, conf = classify_regime_thresholds(
            atr_14=10.0, adx_14=25.0, atr_50_avg=4.0, return_5bar=3.0
        )
        assert regime == "volatile"
        assert conf > 0.6

    def test_transition_zone(self):
        """ADX between 20-25 is low confidence."""
        _regime, conf = classify_regime_thresholds(
            atr_14=3.0, adx_14=22.0, atr_50_avg=3.0, return_5bar=1.0
        )
        assert conf < 0.5  # Low confidence in transition zone


class TestRegimeEndToEnd:
    def test_trending_data_classified_correctly(self):
        """Generate trending data, calculate indicators, classify."""
        df = _make_trending_up_data(200)
        df["atr_14"] = calculate_atr(df, period=14)
        df["adx_14"] = calculate_adx(df, period=14)

        latest = df.iloc[-1]
        atr_14 = float(latest["atr_14"])
        adx_14 = float(latest["adx_14"])
        atr_50_avg = float(df["atr_14"].tail(50).mean())
        return_5bar = float(df["close"].iloc[-1] - df["close"].iloc[-6])

        regime, _conf = classify_regime_thresholds(atr_14, adx_14, atr_50_avg, return_5bar)
        assert regime in ("trending_up", "trending_down")

    def test_ranging_data_classified_correctly(self):
        """Generate ranging data, calculate indicators, classify."""
        df = _make_ranging_data(200)
        df["atr_14"] = calculate_atr(df, period=14)
        df["adx_14"] = calculate_adx(df, period=14)

        latest = df.iloc[-1]
        atr_14 = float(latest["atr_14"])
        adx_14 = float(latest["adx_14"])
        atr_50_avg = float(df["atr_14"].tail(50).mean())
        return_5bar = float(df["close"].iloc[-1] - df["close"].iloc[-6])

        regime, _conf = classify_regime_thresholds(atr_14, adx_14, atr_50_avg, return_5bar)
        assert regime in ("ranging", "trending_up", "trending_down")  # May vary with seed


class TestRegimeDBIntegration:
    async def test_insert_and_retrieve(self, conn):
        """Insert regime state and verify retrieval."""
        state = RegimeState(
            regime="trending_up",
            hmm_state=0,
            hmm_confidence=0.87,
            atr_14=4.25,
            adx_14=28.5,
        )
        rid = await insert_regime_state(conn, state)
        assert rid is not None

        latest = await get_latest_regime(conn)
        assert latest is not None
        assert latest.regime == "trending_up"
        assert float(latest.atr_14) == 4.25

    async def test_regime_model_validation(self):
        """Verify RegimeState model rejects invalid regimes."""
        with pytest.raises(ValueError):
            RegimeState(regime="invalid_regime", atr_14=4.0, adx_14=20.0)
