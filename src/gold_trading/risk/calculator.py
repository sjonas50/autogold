"""ATR-based position sizing and drawdown math for GC/MGC gold futures.

GC  = 100 oz contract, $100/point, $10/tick (tick = $0.10)
MGC = 10 oz contract,  $10/point,  $1/tick  (tick = $0.10)
"""

import os
from dataclasses import dataclass

# Contract multipliers (dollars per point of price movement per contract)
GC_MULTIPLIER = float(os.environ.get("GC_CONTRACT_MULTIPLIER", "100"))
MGC_MULTIPLIER = float(os.environ.get("MGC_CONTRACT_MULTIPLIER", "10"))

MULTIPLIERS = {
    "GC": GC_MULTIPLIER,
    "MGC": MGC_MULTIPLIER,
}


@dataclass
class PositionSize:
    """Result of position sizing calculation."""

    contracts: int
    risk_usd: float
    stop_distance_points: float
    instrument: str
    multiplier: float
    max_allowed_risk_usd: float


def get_multiplier(instrument: str) -> float:
    """Get the dollar-per-point multiplier for an instrument."""
    if instrument not in MULTIPLIERS:
        raise ValueError(f"Unknown instrument: {instrument}. Must be 'GC' or 'MGC'.")
    return MULTIPLIERS[instrument]


def calculate_position_size(
    account_equity: float,
    risk_per_trade: float,
    entry_price: float,
    stop_price: float,
    instrument: str = "GC",
) -> PositionSize:
    """Calculate the number of contracts to trade based on ATR-derived stop distance.

    Args:
        account_equity: Current account equity in USD.
        risk_per_trade: Fraction of account to risk (e.g., 0.005 for 0.5%).
        entry_price: Planned entry price.
        stop_price: Planned stop-loss price.
        instrument: 'GC' or 'MGC'.

    Returns:
        PositionSize with contracts (0 if risk is too large for even 1 contract).
    """
    if account_equity <= 0:
        raise ValueError("Account equity must be positive.")
    if risk_per_trade <= 0 or risk_per_trade > 1:
        raise ValueError("Risk per trade must be between 0 and 1.")
    if entry_price <= 0 or stop_price <= 0:
        raise ValueError("Prices must be positive.")

    multiplier = get_multiplier(instrument)
    stop_distance = abs(entry_price - stop_price)

    if stop_distance == 0:
        raise ValueError("Entry and stop price cannot be equal.")

    max_risk_usd = account_equity * risk_per_trade
    risk_per_contract = stop_distance * multiplier

    contracts = int(max_risk_usd / risk_per_contract)
    actual_risk = contracts * risk_per_contract

    return PositionSize(
        contracts=contracts,
        risk_usd=actual_risk,
        stop_distance_points=stop_distance,
        instrument=instrument,
        multiplier=multiplier,
        max_allowed_risk_usd=max_risk_usd,
    )


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    contracts: int,
    direction: str,
    instrument: str = "GC",
) -> float:
    """Calculate P&L in USD for a closed trade.

    Args:
        entry_price: Fill price at entry.
        exit_price: Fill price at exit.
        contracts: Number of contracts.
        direction: 'long' or 'short'.
        instrument: 'GC' or 'MGC'.

    Returns:
        P&L in USD (positive = profit, negative = loss).
    """
    multiplier = get_multiplier(instrument)
    price_diff = exit_price - entry_price

    if direction == "short":
        price_diff = -price_diff

    return price_diff * contracts * multiplier


def calculate_r_multiple(
    entry_price: float,
    exit_price: float,
    stop_price: float,
    direction: str,
) -> float:
    """Calculate the R-multiple of a trade (profit / initial risk).

    Args:
        entry_price: Fill price at entry.
        exit_price: Fill price at exit.
        stop_price: Initial stop-loss price.
        direction: 'long' or 'short'.

    Returns:
        R-multiple (e.g., 2.0 means profit was 2x the initial risk).
    """
    initial_risk = abs(entry_price - stop_price)
    if initial_risk == 0:
        return 0.0

    profit = exit_price - entry_price if direction == "long" else entry_price - exit_price
    return profit / initial_risk


def calculate_drawdown(
    peak_equity: float,
    current_equity: float,
) -> float:
    """Calculate current drawdown as a fraction of peak equity.

    Returns:
        Drawdown fraction (0.0 = no drawdown, 0.02 = 2% drawdown).
    """
    if peak_equity <= 0:
        return 0.0
    if current_equity >= peak_equity:
        return 0.0
    return (peak_equity - current_equity) / peak_equity


def apply_slippage(price: float, direction: str, slippage_pct: float = 0.0002) -> float:
    """Apply slippage to a fill price (adverse direction).

    Args:
        price: Raw fill price.
        direction: 'long' (buy) or 'short' (sell) — for entry. Reversed for exit.
        slippage_pct: Slippage as a fraction (0.0002 = 0.02%).

    Returns:
        Adjusted price with slippage applied against the trader.
    """
    if direction == "long":
        return price * (1 + slippage_pct)  # Buy higher
    else:
        return price * (1 - slippage_pct)  # Sell lower
