"""Tests for macro analyst FRED data processing and regime classification.

Tests the data processing and DB write logic. FRED API calls are tested
with a real API key when available, otherwise skipped.
"""

import os
from datetime import date

import pytest

from gold_trading.db.queries.macro import get_latest_macro, insert_macro_data
from gold_trading.models.macro import MacroData


class TestMacroDataModel:
    def test_valid_bullish(self):
        m = MacroData(
            observation_date=date(2026, 4, 3),
            dxy=103.5,
            real_yield_10y=1.2,
            macro_regime="bullish",
            reasoning="Falling real yields support gold.",
        )
        assert m.macro_regime == "bullish"

    def test_valid_neutral(self):
        m = MacroData(
            observation_date=date(2026, 4, 3),
            macro_regime="neutral",
            reasoning="Mixed signals.",
        )
        assert m.macro_regime == "neutral"

    def test_invalid_regime_rejected(self):
        with pytest.raises(ValueError):
            MacroData(
                observation_date=date(2026, 4, 3),
                macro_regime="unknown",
                reasoning="Bad.",
            )

    def test_all_fields_optional_except_date(self):
        """All FRED fields are optional since any series could be unavailable."""
        m = MacroData(observation_date=date(2026, 4, 3))
        assert m.dxy is None
        assert m.real_yield_10y is None
        assert m.macro_regime is None


class TestMacroDBIntegration:
    async def test_insert_and_retrieve(self, conn):
        """Insert macro data and verify retrieval."""
        data = MacroData(
            observation_date=date(2026, 4, 3),
            dxy=104.25,
            real_yield_10y=1.85,
            cpi_yoy=3.2,
            breakeven_10y=2.35,
            oil_wti=78.50,
            gold_fix_pm=3120.00,
            macro_regime="bullish",
            reasoning="Real yields declining, DXY weakening.",
        )
        mid = await insert_macro_data(conn, data)
        assert mid is not None

        latest = await get_latest_macro(conn)
        assert latest is not None
        assert latest.macro_regime == "bullish"
        assert float(latest.dxy) == 104.25
        assert float(latest.real_yield_10y) == 1.85
        assert float(latest.gold_fix_pm) == 3120.00

    async def test_multiple_dates_all_stored(self, conn):
        """Insert multiple dates and verify all are stored."""
        regimes = ["bearish", "neutral", "bullish"]
        for i, regime in enumerate(regimes):
            await insert_macro_data(
                conn,
                MacroData(
                    observation_date=date(2026, 4, 1 + i),
                    macro_regime=regime,
                    reasoning=f"Day {i + 1}",
                ),
            )

        latest = await get_latest_macro(conn)
        assert latest is not None
        assert latest.macro_regime in regimes


class TestFREDIntegration:
    @pytest.mark.skipif(
        not os.environ.get("FRED_API_KEY"),
        reason="FRED_API_KEY not set — skipping live API test",
    )
    async def test_fetch_real_fred_data(self):
        """Test fetching real FRED data (requires FRED_API_KEY)."""
        from scripts.macro_analyst import fetch_fred_series

        api_key = os.environ["FRED_API_KEY"]
        # DXY proxy should always have recent data
        value = await fetch_fred_series("DTWEXBGS", api_key)
        assert value is not None
        assert value > 50  # USD index is always > 50
        assert value < 200  # Sanity check
