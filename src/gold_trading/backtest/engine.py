"""vectorbt backtesting engine for gold futures strategies.

Translates strategy parameters into vectorbt signal arrays on 5m GC data.
Runs backtests with price="close" and 0.02% slippage to align with TradingView's
process_orders_on_close=true.

Key alignment settings (from research):
- vectorbt: price="close"
- Pine Script: process_orders_on_close=true
- Slippage: 0.02% both sides
- Fitness metric: Sharpe ratio + win rate (NOT max drawdown — structural divergence)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from gold_trading.models.strategy import BacktestResult

# Alignment with TradingView
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION_PER_CONTRACT = 2.05  # IBKR tiered pricing for GC


@dataclass
class StrategySignals:
    """Entry and exit signals for vectorbt."""

    entries: pd.Series  # Boolean series — True on entry bars
    exits: pd.Series  # Boolean series — True on exit bars
    direction: str = "long"  # 'long' or 'short'


def run_backtest(
    ohlcv: pd.DataFrame,
    signals: StrategySignals,
    instrument: str = "GC",
    initial_capital: float = 50_000.0,
    risk_per_trade: float = 0.005,
) -> BacktestResult:
    """Run a vectorbt backtest on 5m OHLCV data.

    Args:
        ohlcv: DataFrame with columns: timestamp, open, high, low, close, volume.
        signals: Entry/exit boolean series aligned with ohlcv index.
        instrument: 'GC' or 'MGC'.
        initial_capital: Starting capital in USD.
        risk_per_trade: Fraction of capital risked per trade.

    Returns:
        BacktestResult with Sharpe, win rate, expectancy, drawdown, trade count.
    """
    import vectorbt as vbt

    close = ohlcv["close"].astype(float)

    multiplier = 100.0 if instrument == "GC" else 10.0

    # Fixed size of 1 contract for simplicity in backtesting
    # Real position sizing happens in the risk calculator at execution time
    size = 1.0

    if signals.direction == "long":
        pf = vbt.Portfolio.from_signals(
            close,
            entries=signals.entries,
            exits=signals.exits,
            size=size,
            size_type="amount",
            init_cash=initial_capital,
            fees=COMMISSION_PER_CONTRACT / (close.mean() * multiplier),  # As fraction
            slippage=SLIPPAGE_PCT,
            freq="5min",
        )
    else:
        pf = vbt.Portfolio.from_signals(
            close,
            short_entries=signals.entries,
            short_exits=signals.exits,
            size=size,
            size_type="amount",
            init_cash=initial_capital,
            fees=COMMISSION_PER_CONTRACT / (close.mean() * multiplier),
            slippage=SLIPPAGE_PCT,
            freq="5min",
        )

    # Extract metrics
    stats = pf.stats()
    trades = pf.trades.records_readable

    total_trades = len(trades) if trades is not None else 0
    winning_trades = len(trades[trades["PnL"] > 0]) if total_trades > 0 else 0
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    sharpe = float(stats.get("Sharpe Ratio", 0.0))
    if np.isnan(sharpe) or np.isinf(sharpe):
        sharpe = 0.0

    max_dd = float(stats.get("Max Drawdown [%]", 0.0)) / 100.0
    if np.isnan(max_dd):
        max_dd = 0.0

    total_return = float(stats.get("Total Return [%]", 0.0))
    expectancy = (total_return / 100.0 * initial_capital) / max(total_trades, 1)

    avg_duration = None
    if total_trades > 0 and "Duration" in trades.columns:
        avg_duration = trades["Duration"].mean().total_seconds() / 60.0

    # Profit factor
    gross_profit = float(trades[trades["PnL"] > 0]["PnL"].sum()) if total_trades > 0 else 0.0
    gross_loss = abs(float(trades[trades["PnL"] < 0]["PnL"].sum())) if total_trades > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Validation gate
    passed = (
        sharpe >= 1.0
        and win_rate >= 0.45
        and total_trades >= 50
        and max_dd < 0.15  # vectorbt DD underestimates vs TradingView
    )

    result = BacktestResult(
        strategy_id="",  # Set by caller
        sharpe_ratio=round(sharpe, 3),
        win_rate=round(win_rate, 4),
        expectancy_usd=round(expectancy, 2),
        max_drawdown=round(max_dd, 4),
        total_trades=total_trades,
        profit_factor=round(profit_factor, 2) if not np.isinf(profit_factor) else None,
        avg_trade_duration_minutes=round(avg_duration, 1) if avg_duration else None,
        passed_gate=passed,
    )

    logger.info(
        f"Backtest: {total_trades} trades, Sharpe={sharpe:.2f}, "
        f"WR={win_rate:.1%}, DD={max_dd:.1%}, PF={profit_factor:.2f}, "
        f"Gate={'PASS' if passed else 'FAIL'}"
    )

    return result


def generate_breakout_signals(
    ohlcv: pd.DataFrame,
    lookback: int = 12,
    atr_period: int = 14,
    atr_multiplier: float = 1.5,
) -> StrategySignals:
    """Generate session breakout signals for backtesting.

    Simple breakout: enter when price breaks above the high of the last N bars
    with ATR confirmation. Exit when price drops below entry - ATR*multiplier.

    Args:
        ohlcv: DataFrame with OHLCV columns.
        lookback: Number of bars for the range calculation.
        atr_period: ATR period for volatility filter.
        atr_multiplier: ATR multiple for stop distance.

    Returns:
        StrategySignals with entry/exit boolean series.
    """
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)

    # Range high/low
    range_high = high.rolling(lookback).max().shift(1)

    # ATR
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    # Entry: close breaks above range high
    entries = (close > range_high) & (atr > atr.rolling(50).mean() * 0.8)

    # Exit: close drops below entry level - ATR * multiplier
    # Simplified: use range_low as exit
    exits = close < (range_high - atr * atr_multiplier)

    # Clean: no entries where already in a position (simple approach)
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    return StrategySignals(entries=entries, exits=exits, direction="long")


def generate_mean_reversion_signals(
    ohlcv: pd.DataFrame,
    vwap_period: int = 48,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    atr_period: int = 14,
) -> StrategySignals:
    """Generate mean-reversion signals for backtesting.

    Enter long when price is entry_threshold * ATR below VWAP.
    Exit when price returns to within exit_threshold * ATR of VWAP.

    Args:
        ohlcv: DataFrame with OHLCV columns.
        vwap_period: Rolling window for VWAP proxy (volume-weighted close).
        entry_threshold: ATR multiples below VWAP to enter.
        exit_threshold: ATR multiples from VWAP to exit.
        atr_period: ATR period.

    Returns:
        StrategySignals with entry/exit boolean series.
    """
    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    volume = ohlcv["volume"].astype(float).replace(0, 1)

    # VWAP proxy: volume-weighted rolling mean
    vwap = (close * volume).rolling(vwap_period).sum() / volume.rolling(vwap_period).sum()

    # ATR
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    # Distance from VWAP in ATR units
    distance = (close - vwap) / atr

    # Entry: price is far below VWAP (oversold)
    entries = distance < -entry_threshold

    # Exit: price returns near VWAP
    exits = distance > -exit_threshold

    entries = entries.fillna(False)
    exits = exits.fillna(False)

    return StrategySignals(entries=entries, exits=exits, direction="long")
