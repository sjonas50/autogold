"""Walk-Forward Validation agent — tests strategies out-of-sample to detect overfitting.

Paperclip process adapter script. Runs on-demand (triggered by CIO or Quant Researcher).
Splits historical data into train/test windows, backtests on train, validates on test.
If Sharpe degrades >50% out-of-sample → rejects the strategy.

Run manually: uv run python scripts/walk_forward_validator.py
"""

import asyncio
import os

import asyncpg
import numpy as np
import pandas as pd
from loguru import logger

from gold_trading.backtest.engine import (
    generate_breakout_signals,
    generate_mean_reversion_signals,
    run_backtest,
)
from gold_trading.db.client import get_database_url
from gold_trading.db.queries.decisions import insert_decision
from gold_trading.db.queries.strategies import get_strategies_by_status, upsert_strategy
from gold_trading.models.lesson import DecisionLogEntry

# Walk-forward parameters
N_WINDOWS = 3  # Number of train/test splits
TRAIN_RATIO = 0.7  # 70% train, 30% test
OOS_DEGRADATION_THRESHOLD = 0.50  # Reject if OOS Sharpe < 50% of IS Sharpe

PAPERCLIP_URL = os.environ.get("PAPERCLIP_URL", "http://localhost:3100")
COMPANY_ID = os.environ.get("PAPERCLIP_COMPANY_ID", "3422f81a-8ca2-4ce1-aae5-5cf8ce34fa0e")
# Reports to Technical Analyst (Strategy Team Lead), escalates to CIO
STRATEGY_LEAD_ID = "e475c802-6bde-4d8e-bb43-602842ae5e7f"
CIO_AGENT_ID = "37bbe408-e573-4598-a374-cc369bad0258"


async def load_ohlcv(conn: asyncpg.Connection) -> pd.DataFrame | None:
    """Load all 5m GC OHLCV data."""
    rows = await conn.fetch(
        "SELECT timestamp, open, high, low, close, volume FROM ohlcv_5m "
        "WHERE instrument = 'GC' ORDER BY timestamp ASC"
    )
    if not rows:
        return None
    df = pd.DataFrame([dict(r) for r in rows])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(int)
    return df


def split_walk_forward(
    df: pd.DataFrame, n_windows: int = 3
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Split data into overlapping train/test windows.

    Returns list of (train_df, test_df) tuples.
    """
    total = len(df)
    window_size = total // n_windows
    splits = []

    for i in range(n_windows):
        start = i * (window_size // 2)  # Overlapping windows
        end = min(start + window_size, total)
        if end - start < 200:
            continue

        split_point = start + int((end - start) * TRAIN_RATIO)
        train = df.iloc[start:split_point].reset_index(drop=True)
        test = df.iloc[split_point:end].reset_index(drop=True)

        if len(train) >= 100 and len(test) >= 50:
            splits.append((train, test))

    return splits


def run_walk_forward_test(
    ohlcv: pd.DataFrame,
    strategy_class: str,
    params: dict,
) -> dict:
    """Run walk-forward validation for a strategy.

    Returns dict with IS and OOS results per window and overall assessment.
    """
    splits = split_walk_forward(ohlcv, N_WINDOWS)
    if not splits:
        return {"passed": False, "reason": "Insufficient data for walk-forward splits"}

    results = []

    for i, (train, test) in enumerate(splits):
        # Generate signals based on strategy class and params
        if params.get("type") == "breakout" or strategy_class in ("breakout", "momentum"):
            train_signals = generate_breakout_signals(
                train,
                lookback=params.get("lookback", 12),
                atr_multiplier=params.get("atr_multiplier", 1.5),
            )
            test_signals = generate_breakout_signals(
                test,
                lookback=params.get("lookback", 12),
                atr_multiplier=params.get("atr_multiplier", 1.5),
            )
        else:
            train_signals = generate_mean_reversion_signals(
                train,
                vwap_period=params.get("vwap_period", 48),
                entry_threshold=params.get("entry_threshold", 2.0),
            )
            test_signals = generate_mean_reversion_signals(
                test,
                vwap_period=params.get("vwap_period", 48),
                entry_threshold=params.get("entry_threshold", 2.0),
            )

        is_result = run_backtest(train, train_signals)
        oos_result = run_backtest(test, test_signals)

        results.append(
            {
                "window": i + 1,
                "train_bars": len(train),
                "test_bars": len(test),
                "is_sharpe": is_result.sharpe_ratio,
                "oos_sharpe": oos_result.sharpe_ratio,
                "is_win_rate": is_result.win_rate,
                "oos_win_rate": oos_result.win_rate,
                "is_trades": is_result.total_trades,
                "oos_trades": oos_result.total_trades,
                "degradation": (
                    1 - (oos_result.sharpe_ratio / is_result.sharpe_ratio)
                    if is_result.sharpe_ratio != 0
                    else 1.0
                ),
            }
        )

        logger.info(
            f"  Window {i + 1}: IS Sharpe={is_result.sharpe_ratio:.2f} → "
            f"OOS Sharpe={oos_result.sharpe_ratio:.2f} "
            f"(degradation: {results[-1]['degradation']:.1%})"
        )

    # Overall assessment
    avg_is_sharpe = np.mean([r["is_sharpe"] for r in results])
    avg_oos_sharpe = np.mean([r["oos_sharpe"] for r in results])
    avg_degradation = 1 - (avg_oos_sharpe / avg_is_sharpe) if avg_is_sharpe != 0 else 1.0

    # Pass if OOS doesn't degrade more than threshold
    passed = avg_degradation < OOS_DEGRADATION_THRESHOLD and avg_oos_sharpe > 0

    return {
        "passed": passed,
        "windows": results,
        "avg_is_sharpe": round(avg_is_sharpe, 3),
        "avg_oos_sharpe": round(avg_oos_sharpe, 3),
        "avg_degradation": round(avg_degradation, 3),
        "reason": (
            f"OOS Sharpe avg {avg_oos_sharpe:.2f} "
            f"({'passed' if passed else 'failed'}: "
            f"{avg_degradation:.1%} degradation vs {OOS_DEGRADATION_THRESHOLD:.0%} threshold)"
        ),
    }


async def create_cio_report(strategy_id: str, wf_result: dict) -> None:
    """Create a Paperclip task reporting walk-forward results to CIO."""
    import httpx

    status = "PASSED" if wf_result["passed"] else "FAILED"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{PAPERCLIP_URL}/api/companies/{COMPANY_ID}/issues",
                json={
                    "title": f"[WF-{status}] {strategy_id} walk-forward validation",
                    "description": (
                        f"Walk-forward validation for **{strategy_id}**: **{status}**\n\n"
                        f"- In-sample Sharpe: {wf_result['avg_is_sharpe']:.2f}\n"
                        f"- Out-of-sample Sharpe: {wf_result['avg_oos_sharpe']:.2f}\n"
                        f"- Degradation: {wf_result['avg_degradation']:.1%}\n"
                        f"- Threshold: {OOS_DEGRADATION_THRESHOLD:.0%}\n\n"
                        f"{'Strategy is robust — recommend proceeding to deployment.' if wf_result['passed'] else 'Strategy is overfit — recommend retiring.'}"
                    ),
                    "assigneeAgentId": STRATEGY_LEAD_ID,
                    "priority": "high" if not wf_result["passed"] else "normal",
                },
            )
    except Exception as e:
        logger.warning(f"Could not create Paperclip task: {e}")


async def main() -> None:
    """Main heartbeat: validate all pending_deployment strategies."""
    logger.info("Walk-Forward Validator heartbeat starting")

    conn = await asyncpg.connect(get_database_url())
    try:
        ohlcv = await load_ohlcv(conn)
        if ohlcv is None or len(ohlcv) < 1000:
            logger.warning("Insufficient data for walk-forward validation")
            return

        # Find strategies pending deployment
        pending = await get_strategies_by_status(conn, ["pending_deployment"])
        if not pending:
            logger.info("No strategies pending deployment — nothing to validate")
            return

        for strategy in pending:
            logger.info(f"Validating {strategy.id} ({strategy.strategy_class})...")

            params = strategy.backtest_params or {}
            wf_result = run_walk_forward_test(ohlcv, strategy.strategy_class or "breakout", params)

            # Log result
            await insert_decision(
                conn,
                DecisionLogEntry(
                    agent_name="walk_forward_validator",
                    decision_type="walk_forward_validation",
                    inputs_summary={
                        "strategy_id": strategy.id,
                        "params": params,
                        "n_windows": N_WINDOWS,
                    },
                    decision=f"{'PASS' if wf_result['passed'] else 'FAIL'}: {strategy.id}",
                    reasoning=wf_result["reason"],
                    confidence=0.8 if wf_result["passed"] else 0.9,
                ),
            )

            # Update strategy status
            if not wf_result["passed"]:
                strategy.status = "retired"
                await upsert_strategy(conn, strategy)
                logger.warning(f"{strategy.id} FAILED walk-forward — retired")
            else:
                logger.info(f"{strategy.id} PASSED walk-forward — remains pending_deployment")

            # Report to CIO
            await create_cio_report(strategy.id, wf_result)

        logger.info("Walk-Forward Validator heartbeat complete")

    finally:
        await conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
