"""Monte Carlo simulation for strategy validation.

Uses block bootstrap on trade returns to generate simulated equity curves.
Outputs confidence intervals for Sharpe ratio and max drawdown.

Block bootstrap preserves autocorrelation in trade sequences (e.g., streaks
of wins/losses that occur in trending regimes).
"""

import numpy as np
from loguru import logger

from gold_trading.models.strategy import MonteCarloResult


def block_bootstrap_trades(
    trade_pnls: np.ndarray,
    n_iterations: int = 1000,
    block_size: int | None = None,
) -> np.ndarray:
    """Generate bootstrapped trade sequences using block bootstrap.

    Args:
        trade_pnls: Array of trade P&L values (in USD).
        n_iterations: Number of Monte Carlo iterations.
        block_size: Size of blocks for bootstrap. If None, uses sqrt(n_trades).

    Returns:
        2D array of shape (n_iterations, n_trades) with resampled P&L sequences.
    """
    n_trades = len(trade_pnls)

    if block_size is None:
        block_size = max(2, int(np.sqrt(n_trades)))

    rng = np.random.default_rng(seed=42)
    simulations = np.zeros((n_iterations, n_trades))

    for i in range(n_iterations):
        # Sample random block start positions
        n_blocks = int(np.ceil(n_trades / block_size))
        starts = rng.integers(0, n_trades - block_size + 1, size=n_blocks)

        # Concatenate blocks and trim to n_trades
        resampled = np.concatenate([trade_pnls[s : s + block_size] for s in starts])
        simulations[i] = resampled[:n_trades]

    return simulations


def compute_equity_curves(
    simulations: np.ndarray,
    initial_capital: float = 50_000.0,
) -> np.ndarray:
    """Convert P&L simulations to equity curves.

    Args:
        simulations: 2D array (n_iterations, n_trades) of P&L values.
        initial_capital: Starting capital.

    Returns:
        2D array (n_iterations, n_trades+1) of equity values.
    """
    n_iter, n_trades = simulations.shape
    equity = np.zeros((n_iter, n_trades + 1))
    equity[:, 0] = initial_capital
    equity[:, 1:] = initial_capital + np.cumsum(simulations, axis=1)
    return equity


def compute_max_drawdowns(equity_curves: np.ndarray) -> np.ndarray:
    """Compute max drawdown for each equity curve.

    Args:
        equity_curves: 2D array (n_iterations, n_steps).

    Returns:
        1D array of max drawdown fractions (0.0 to 1.0).
    """
    running_max = np.maximum.accumulate(equity_curves, axis=1)
    drawdowns = (running_max - equity_curves) / running_max
    return np.max(drawdowns, axis=1)


def compute_sharpe_ratios(
    simulations: np.ndarray,
    trades_per_day: float = 2.0,
) -> np.ndarray:
    """Compute annualized Sharpe ratio for each simulation.

    Args:
        simulations: 2D array (n_iterations, n_trades) of P&L values.
        trades_per_day: Average number of trades per trading day.

    Returns:
        1D array of Sharpe ratios.
    """
    means = np.mean(simulations, axis=1)
    stds = np.std(simulations, axis=1)

    # Avoid division by zero
    stds = np.where(stds == 0, 1e-10, stds)

    # Annualize: assume 252 trading days
    annualization = np.sqrt(252 * trades_per_day)
    return (means / stds) * annualization


def run_monte_carlo(
    trade_pnls: list[float] | np.ndarray,
    strategy_id: str = "",
    n_iterations: int = 1000,
    initial_capital: float = 50_000.0,
    block_size: int | None = None,
    max_drawdown_threshold: float = 0.02,
) -> MonteCarloResult:
    """Run full Monte Carlo simulation on a strategy's trade history.

    Args:
        trade_pnls: List of P&L values per trade (in USD).
        strategy_id: Strategy identifier for the result.
        n_iterations: Number of simulated equity curves.
        initial_capital: Starting capital.
        block_size: Block size for bootstrap. None = auto (sqrt(n_trades)).
        max_drawdown_threshold: Drawdown threshold for ruin probability.

    Returns:
        MonteCarloResult with percentile statistics.
    """
    pnls = np.array(trade_pnls, dtype=float)

    if len(pnls) < 10:
        logger.warning(f"Only {len(pnls)} trades — Monte Carlo results unreliable")

    # Bootstrap
    simulations = block_bootstrap_trades(pnls, n_iterations, block_size)

    # Equity curves
    equity = compute_equity_curves(simulations, initial_capital)

    # Max drawdowns
    max_dds = compute_max_drawdowns(equity)

    # Sharpe ratios
    sharpes = compute_sharpe_ratios(simulations)

    # Ruin probability: fraction of simulations that breach the drawdown threshold
    ruin_prob = np.mean(max_dds >= max_drawdown_threshold)

    # Percentiles
    sharpe_p5 = float(np.percentile(sharpes, 5))
    sharpe_p50 = float(np.percentile(sharpes, 50))
    sharpe_p95 = float(np.percentile(sharpes, 95))
    dd_p5 = float(np.percentile(max_dds, 5))
    dd_p50 = float(np.percentile(max_dds, 50))
    dd_p95 = float(np.percentile(max_dds, 95))

    # TradingView deployment gate — strategy must survive worst-case scenarios
    passed = sharpe_p5 >= 0.5 and dd_p95 < 0.05 and ruin_prob < 0.05

    result = MonteCarloResult(
        strategy_id=strategy_id,
        iterations=n_iterations,
        sharpe_p5=round(sharpe_p5, 3),
        sharpe_p50=round(sharpe_p50, 3),
        sharpe_p95=round(sharpe_p95, 3),
        max_drawdown_p5=round(dd_p5, 4),
        max_drawdown_p50=round(dd_p50, 4),
        max_drawdown_p95=round(dd_p95, 4),
        ruin_probability=round(float(ruin_prob), 4),
        passed_gate=passed,
    )

    logger.info(
        f"Monte Carlo ({n_iterations} iterations): "
        f"Sharpe=[{sharpe_p5:.2f}, {sharpe_p50:.2f}, {sharpe_p95:.2f}], "
        f"MaxDD=[{dd_p5:.1%}, {dd_p50:.1%}, {dd_p95:.1%}], "
        f"Ruin={ruin_prob:.1%}, Gate={'PASS' if passed else 'FAIL'}"
    )

    return result
