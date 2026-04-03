"""Tests for position sizing, P&L, drawdown, and slippage calculations."""

import pytest

from gold_trading.risk.calculator import (
    apply_slippage,
    calculate_drawdown,
    calculate_pnl,
    calculate_position_size,
    calculate_r_multiple,
)


class TestPositionSizing:
    def test_gc_basic(self):
        """$50k account, 0.5% risk, 5-point stop on GC."""
        result = calculate_position_size(
            account_equity=50_000,
            risk_per_trade=0.005,
            entry_price=3125.00,
            stop_price=3120.00,
            instrument="GC",
        )
        # Max risk = $250, risk per contract = 5 * $100 = $500
        # contracts = floor(250 / 500) = 0
        assert result.contracts == 0
        assert result.stop_distance_points == 5.0
        assert result.multiplier == 100.0

    def test_gc_wide_stop(self):
        """$100k account, 0.5% risk, 2-point stop on GC."""
        result = calculate_position_size(
            account_equity=100_000,
            risk_per_trade=0.005,
            entry_price=3125.00,
            stop_price=3123.00,
            instrument="GC",
        )
        # Max risk = $500, risk per contract = 2 * $100 = $200
        # contracts = floor(500 / 200) = 2
        assert result.contracts == 2
        assert result.risk_usd == 400.0  # 2 * $200

    def test_mgc_basic(self):
        """$50k account, 0.5% risk, 5-point stop on MGC."""
        result = calculate_position_size(
            account_equity=50_000,
            risk_per_trade=0.005,
            entry_price=3125.00,
            stop_price=3120.00,
            instrument="MGC",
        )
        # Max risk = $250, risk per contract = 5 * $10 = $50
        # contracts = floor(250 / 50) = 5
        assert result.contracts == 5
        assert result.risk_usd == 250.0

    def test_short_position(self):
        """Short position — stop is above entry."""
        result = calculate_position_size(
            account_equity=50_000,
            risk_per_trade=0.005,
            entry_price=3125.00,
            stop_price=3130.00,
            instrument="MGC",
        )
        assert result.contracts == 5
        assert result.stop_distance_points == 5.0

    def test_zero_account_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calculate_position_size(0, 0.005, 3125, 3120, "GC")

    def test_equal_entry_stop_raises(self):
        with pytest.raises(ValueError, match="equal"):
            calculate_position_size(50_000, 0.005, 3125, 3125, "GC")

    def test_unknown_instrument_raises(self):
        with pytest.raises(ValueError, match="Unknown instrument"):
            calculate_position_size(50_000, 0.005, 3125, 3120, "ES")


class TestPnL:
    def test_long_profit(self):
        """Long 1 GC: buy 3120, sell 3130 = $1000 profit."""
        pnl = calculate_pnl(3120, 3130, 1, "long", "GC")
        assert pnl == 1000.0

    def test_long_loss(self):
        """Long 1 GC: buy 3120, sell 3115 = -$500 loss."""
        pnl = calculate_pnl(3120, 3115, 1, "long", "GC")
        assert pnl == -500.0

    def test_short_profit(self):
        """Short 2 MGC: sell 3130, cover 3120 = $200 profit."""
        pnl = calculate_pnl(3130, 3120, 2, "short", "MGC")
        assert pnl == 200.0

    def test_short_loss(self):
        """Short 1 GC: sell 3120, cover 3125 = -$500 loss."""
        pnl = calculate_pnl(3120, 3125, 1, "short", "GC")
        assert pnl == -500.0

    def test_multiple_contracts(self):
        """3 MGC long: 3120 → 3125 = $150."""
        pnl = calculate_pnl(3120, 3125, 3, "long", "MGC")
        assert pnl == 150.0


class TestRMultiple:
    def test_2r_winner(self):
        """Long, 5pt stop, 10pt win = 2R."""
        r = calculate_r_multiple(3120, 3130, 3115, "long")
        assert r == 2.0

    def test_1r_loser(self):
        """Long, 5pt stop, hit stop = -1R."""
        r = calculate_r_multiple(3120, 3115, 3115, "long")
        assert r == -1.0

    def test_short_winner(self):
        """Short, 3pt stop, 6pt win = 2R."""
        r = calculate_r_multiple(3130, 3124, 3133, "short")
        assert r == 2.0


class TestDrawdown:
    def test_no_drawdown(self):
        assert calculate_drawdown(50_000, 50_000) == 0.0
        assert calculate_drawdown(50_000, 52_000) == 0.0

    def test_two_percent(self):
        dd = calculate_drawdown(50_000, 49_000)
        assert abs(dd - 0.02) < 0.0001

    def test_zero_peak(self):
        assert calculate_drawdown(0, 1000) == 0.0


class TestSlippage:
    def test_long_entry_slips_up(self):
        """Buying gets a worse (higher) price."""
        price = apply_slippage(3125.00, "long", 0.0002)
        assert price > 3125.00
        assert abs(price - 3125.625) < 0.01

    def test_short_entry_slips_down(self):
        """Selling gets a worse (lower) price."""
        price = apply_slippage(3125.00, "short", 0.0002)
        assert price < 3125.00
