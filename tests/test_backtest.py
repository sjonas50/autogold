"""Tests for vectorbt backtesting engine and Monte Carlo simulation.

Uses synthetic gold OHLCV data to test the backtest pipeline and
Monte Carlo statistics.
"""

import numpy as np
import pandas as pd

from gold_trading.backtest.engine import (
    StrategySignals,
    generate_breakout_signals,
    generate_mean_reversion_signals,
    run_backtest,
)
from gold_trading.backtest.montecarlo import (
    block_bootstrap_trades,
    compute_equity_curves,
    compute_max_drawdowns,
    compute_sharpe_ratios,
    run_monte_carlo,
)


def _make_gold_ohlcv(n: int = 2000, trend: float = 0.05) -> pd.DataFrame:
    """Generate synthetic 5m gold OHLCV data.

    Args:
        n: Number of bars.
        trend: Drift per bar (positive = uptrend).
    """
    np.random.seed(42)
    base = 3100.0
    returns = np.random.normal(trend, 1.5, n)
    close = base + np.cumsum(returns)
    noise = np.random.uniform(0.5, 3.0, n)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="5min"),
            "open": close - np.random.uniform(-0.5, 0.5, n),
            "high": close + noise,
            "low": close - noise,
            "close": close,
            "volume": np.random.randint(500, 5000, n),
        }
    )


# ============================================================
# Backtest Engine
# ============================================================


class TestBacktestEngine:
    def test_backtest_runs_without_error(self):
        """Basic smoke test — backtest completes."""
        ohlcv = _make_gold_ohlcv(2000, trend=0.05)
        signals = generate_breakout_signals(ohlcv)
        result = run_backtest(ohlcv, signals, instrument="GC")

        assert result.total_trades >= 0
        assert -5.0 <= result.sharpe_ratio <= 10.0
        assert 0.0 <= result.win_rate <= 1.0
        assert 0.0 <= result.max_drawdown <= 1.0

    def test_backtest_mean_reversion(self):
        """Mean reversion strategy runs on ranging data."""
        ohlcv = _make_gold_ohlcv(2000, trend=0.0)  # No trend = ranging
        signals = generate_mean_reversion_signals(ohlcv)
        result = run_backtest(ohlcv, signals, instrument="GC")

        assert result.total_trades >= 0
        assert result.sharpe_ratio is not None

    def test_backtest_no_signals(self):
        """Backtest with no entry signals produces 0 trades."""
        ohlcv = _make_gold_ohlcv(500)
        signals = StrategySignals(
            entries=pd.Series(False, index=ohlcv.index),
            exits=pd.Series(False, index=ohlcv.index),
        )
        result = run_backtest(ohlcv, signals)
        assert result.total_trades == 0

    def test_backtest_result_fields(self):
        """All BacktestResult fields are populated."""
        ohlcv = _make_gold_ohlcv(2000, trend=0.1)
        signals = generate_breakout_signals(ohlcv)
        result = run_backtest(ohlcv, signals)

        assert result.strategy_id is not None
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.win_rate, float)
        assert isinstance(result.expectancy_usd, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.total_trades, int)
        assert isinstance(result.passed_gate, bool)

    def test_gate_check_logic(self):
        """Verify the gate check criteria."""
        ohlcv = _make_gold_ohlcv(5000, trend=0.1)  # Strong uptrend
        signals = generate_breakout_signals(ohlcv, lookback=6)
        result = run_backtest(ohlcv, signals)

        # With strong trend and enough data, we should get trades
        if result.total_trades >= 50:
            # TradingView gate: Sharpe >= 1.5, WR >= 50%, DD < 5%, PF >= 1.3
            pf = result.profit_factor if result.profit_factor else 0.0
            expected_pass = (
                result.sharpe_ratio >= 1.5
                and result.win_rate >= 0.50
                and result.max_drawdown < 0.05
                and pf >= 1.3
            )
            assert result.passed_gate == expected_pass


class TestSignalGeneration:
    def test_breakout_signals_shape(self):
        """Breakout signals have same length as input data."""
        ohlcv = _make_gold_ohlcv(500)
        signals = generate_breakout_signals(ohlcv)
        assert len(signals.entries) == len(ohlcv)
        assert len(signals.exits) == len(ohlcv)
        assert signals.direction == "long"

    def test_mean_reversion_signals_shape(self):
        """Mean reversion signals have same length as input data."""
        ohlcv = _make_gold_ohlcv(500)
        signals = generate_mean_reversion_signals(ohlcv)
        assert len(signals.entries) == len(ohlcv)
        assert len(signals.exits) == len(ohlcv)

    def test_breakout_produces_entries(self):
        """Breakout on trending data should produce some entries."""
        ohlcv = _make_gold_ohlcv(2000, trend=0.2)  # Strong uptrend
        signals = generate_breakout_signals(ohlcv)
        assert signals.entries.sum() > 0

    def test_mean_reversion_produces_entries(self):
        """Mean reversion on any data should produce some entries."""
        ohlcv = _make_gold_ohlcv(2000, trend=0.0)
        signals = generate_mean_reversion_signals(ohlcv, entry_threshold=1.0)
        assert signals.entries.sum() > 0


# ============================================================
# Monte Carlo
# ============================================================


class TestBlockBootstrap:
    def test_output_shape(self):
        """Bootstrap output has correct shape."""
        pnls = np.random.normal(50, 100, 100)
        result = block_bootstrap_trades(pnls, n_iterations=500)
        assert result.shape == (500, 100)

    def test_values_from_original(self):
        """Bootstrapped values come from the original distribution."""
        pnls = np.array([100.0, -50.0, 200.0, -30.0, 150.0])
        result = block_bootstrap_trades(pnls, n_iterations=10, block_size=2)
        # All values should be from the original set
        original_set = set(pnls)
        for row in result:
            for val in row:
                assert val in original_set

    def test_block_size_auto(self):
        """Auto block size = sqrt(n_trades)."""
        pnls = np.random.normal(0, 100, 100)
        # Should not raise with default block_size
        result = block_bootstrap_trades(pnls, n_iterations=10)
        assert result.shape[0] == 10


class TestEquityCurves:
    def test_initial_capital(self):
        """First value of each equity curve is initial capital."""
        sims = np.array([[100, -50, 200], [50, 50, -100]])
        equity = compute_equity_curves(sims, initial_capital=50000)
        assert equity[0, 0] == 50000
        assert equity[1, 0] == 50000

    def test_cumulative_pnl(self):
        """Equity curve follows cumulative P&L."""
        sims = np.array([[100, 200, -50]])
        equity = compute_equity_curves(sims, initial_capital=10000)
        np.testing.assert_array_equal(equity[0], [10000, 10100, 10300, 10250])


class TestMaxDrawdown:
    def test_no_drawdown(self):
        """Strictly increasing equity has zero drawdown."""
        equity = np.array([[1000, 1100, 1200, 1300]])
        dd = compute_max_drawdowns(equity)
        assert dd[0] == 0.0

    def test_known_drawdown(self):
        """Known drawdown is calculated correctly."""
        # Peak at 1200, trough at 1000 → 200/1200 = 16.67%
        equity = np.array([[1000, 1200, 1000, 1100]])
        dd = compute_max_drawdowns(equity)
        assert abs(dd[0] - 200 / 1200) < 0.001


class TestSharpeRatio:
    def test_positive_sharpe(self):
        """Consistently positive P&L gives positive Sharpe."""
        sims = np.array([[100, 150, 120, 130, 140]] * 10)
        sharpes = compute_sharpe_ratios(sims)
        assert all(s > 0 for s in sharpes)

    def test_negative_sharpe(self):
        """Consistently negative P&L gives negative Sharpe."""
        sims = np.array([[-100, -150, -120, -130, -140]] * 10)
        sharpes = compute_sharpe_ratios(sims)
        assert all(s < 0 for s in sharpes)


class TestRunMonteCarlo:
    def test_full_pipeline(self):
        """Full Monte Carlo runs and produces valid result."""
        np.random.seed(42)
        pnls = list(np.random.normal(50, 100, 200))
        result = run_monte_carlo(pnls, strategy_id="test_mc", n_iterations=500)

        assert result.strategy_id == "test_mc"
        assert result.iterations == 500
        assert result.sharpe_p5 <= result.sharpe_p50 <= result.sharpe_p95
        assert result.max_drawdown_p5 <= result.max_drawdown_p50 <= result.max_drawdown_p95
        assert 0.0 <= result.ruin_probability <= 1.0

    def test_profitable_strategy(self):
        """Highly profitable strategy should have good Monte Carlo stats."""
        pnls = list(np.random.normal(200, 50, 300))  # Very profitable
        result = run_monte_carlo(pnls, strategy_id="profitable", n_iterations=500)

        assert result.sharpe_p50 > 0
        assert result.ruin_probability < 0.5

    def test_losing_strategy(self):
        """Losing strategy should have poor Monte Carlo stats."""
        pnls = list(np.random.normal(-100, 50, 200))  # Losing
        result = run_monte_carlo(pnls, strategy_id="loser", n_iterations=500)

        assert result.sharpe_p50 < 0

    def test_small_sample(self):
        """Works with small samples (with warning)."""
        pnls = [100.0, -50.0, 200.0, -30.0, 150.0]
        result = run_monte_carlo(pnls, strategy_id="small", n_iterations=100)
        assert result.iterations == 100

    def test_gate_check(self):
        """Verify gate passes for good strategy."""
        pnls = list(np.random.normal(150, 60, 500))
        result = run_monte_carlo(pnls, strategy_id="gate_test", n_iterations=500)

        # Gate: sharpe_p5 >= 0.5 and dd_p95 < 0.15
        if result.sharpe_p5 >= 0.5 and result.max_drawdown_p95 < 0.15:
            assert result.passed_gate is True
