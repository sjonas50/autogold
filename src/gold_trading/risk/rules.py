"""Risk rule enforcement — deterministic, no LLM calls.

Hard rules:
- Max risk per trade: 0.5% of account
- Max drawdown before full halt: 2%
- Max concurrent positions: 1
- Daily loss limit: 1%
- Max trade duration: 120 minutes
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta

MAX_RISK_PER_TRADE = float(os.environ.get("MAX_RISK_PER_TRADE", "0.005"))
MAX_DRAWDOWN_LIMIT = float(os.environ.get("MAX_DRAWDOWN_LIMIT", "0.02"))
MAX_POSITIONS = int(os.environ.get("MAX_POSITIONS", "1"))
DAILY_LOSS_LIMIT = 0.01  # 1% daily
MAX_TRADE_DURATION_MINUTES = 120


@dataclass
class RiskCheckResult:
    """Result of a risk rule evaluation."""

    passed: bool
    violations: list[str]


def check_max_positions(open_position_count: int) -> RiskCheckResult:
    """Check if adding a new position would exceed the max."""
    if open_position_count >= MAX_POSITIONS:
        return RiskCheckResult(
            passed=False,
            violations=[
                f"Max positions exceeded: {open_position_count} open >= {MAX_POSITIONS} max"
            ],
        )
    return RiskCheckResult(passed=True, violations=[])


def check_drawdown(
    peak_equity: float,
    current_equity: float,
) -> RiskCheckResult:
    """Check if drawdown has breached the limit."""
    if peak_equity <= 0:
        return RiskCheckResult(passed=True, violations=[])

    drawdown = (peak_equity - current_equity) / peak_equity if current_equity < peak_equity else 0.0

    if drawdown >= MAX_DRAWDOWN_LIMIT:
        return RiskCheckResult(
            passed=False,
            violations=[
                f"Drawdown limit breached: {drawdown:.2%} >= {MAX_DRAWDOWN_LIMIT:.2%}. "
                "All strategies must halt."
            ],
        )
    return RiskCheckResult(passed=True, violations=[])


def check_daily_loss(
    daily_pnl: float,
    account_equity: float,
) -> RiskCheckResult:
    """Check if daily loss limit has been hit."""
    if account_equity <= 0:
        return RiskCheckResult(passed=True, violations=[])

    daily_loss_pct = abs(daily_pnl) / account_equity if daily_pnl < 0 else 0.0

    if daily_loss_pct >= DAILY_LOSS_LIMIT:
        return RiskCheckResult(
            passed=False,
            violations=[
                f"Daily loss limit hit: {daily_loss_pct:.2%} >= {DAILY_LOSS_LIMIT:.2%}. "
                "No more trades this session."
            ],
        )
    return RiskCheckResult(passed=True, violations=[])


def check_trade_duration(
    entry_time: datetime,
    current_time: datetime,
) -> RiskCheckResult:
    """Check if a trade has exceeded max duration."""
    duration = current_time - entry_time
    max_duration = timedelta(minutes=MAX_TRADE_DURATION_MINUTES)

    if duration >= max_duration:
        return RiskCheckResult(
            passed=False,
            violations=[
                f"Trade duration exceeded: {duration.total_seconds() / 60:.0f} min "
                f">= {MAX_TRADE_DURATION_MINUTES} min. Auto-exit required."
            ],
        )
    return RiskCheckResult(passed=True, violations=[])


def check_strategy_active(is_active: bool) -> RiskCheckResult:
    """Check if the strategy is currently active."""
    if not is_active:
        return RiskCheckResult(
            passed=False,
            violations=["Strategy is not active. Signal rejected."],
        )
    return RiskCheckResult(passed=True, violations=[])


def run_all_entry_checks(
    open_position_count: int,
    peak_equity: float,
    current_equity: float,
    daily_pnl: float,
    account_equity: float,
    strategy_is_active: bool,
) -> RiskCheckResult:
    """Run all risk checks required before opening a new position.

    Returns combined result — fails if ANY check fails.
    """
    checks = [
        check_max_positions(open_position_count),
        check_drawdown(peak_equity, current_equity),
        check_daily_loss(daily_pnl, account_equity),
        check_strategy_active(strategy_is_active),
    ]

    all_violations = []
    for check in checks:
        all_violations.extend(check.violations)

    return RiskCheckResult(
        passed=len(all_violations) == 0,
        violations=all_violations,
    )
