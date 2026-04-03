"""Pydantic models for trading strategies and backtest results."""

from datetime import datetime

from pydantic import BaseModel, Field


class Strategy(BaseModel):
    """A trading strategy definition with backtest metrics."""

    id: str
    name: str
    version: int = 1
    pine_script: str
    instrument: str = Field(default="GC", pattern=r"^(GC|MGC)$")
    timeframe: str = "5m"
    strategy_class: str | None = None
    vbt_sharpe: float | None = None
    vbt_win_rate: float | None = None
    vbt_expectancy: float | None = None
    vbt_max_drawdown: float | None = None
    vbt_total_trades: int | None = None
    vbt_profit_factor: float | None = None
    vbt_avg_duration_min: float | None = None
    backtest_params: dict | None = None
    mc_sharpe_p5: float | None = None
    mc_sharpe_p50: float | None = None
    status: str = Field(
        default="pending_deployment",
        pattern=r"^(pending_deployment|active|paused|retired)$",
    )
    is_active: bool = False
    cio_recommendation: str | None = None
    tradingview_alert_id: str | None = None
    created_at: datetime | None = None
    deployed_at: datetime | None = None
    retired_at: datetime | None = None


class BacktestResult(BaseModel):
    """Result of a vectorbt backtest run."""

    strategy_id: str
    sharpe_ratio: float
    win_rate: float = Field(ge=0.0, le=1.0)
    expectancy_usd: float
    max_drawdown: float = Field(ge=0.0, le=1.0)
    total_trades: int = Field(ge=0)
    profit_factor: float | None = None
    avg_trade_duration_minutes: float | None = None
    passed_gate: bool = False


class MonteCarloResult(BaseModel):
    """Result of Monte Carlo simulation on a strategy."""

    strategy_id: str
    iterations: int = 1000
    sharpe_p5: float
    sharpe_p50: float
    sharpe_p95: float
    max_drawdown_p5: float
    max_drawdown_p50: float
    max_drawdown_p95: float
    ruin_probability: float = Field(ge=0.0, le=1.0)
    passed_gate: bool = False
