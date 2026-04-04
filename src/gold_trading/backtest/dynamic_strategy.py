"""Dynamic strategy execution — run Claude-generated Python signal logic in vectorbt.

Instead of hardcoded signal generators, Claude writes the actual Python function
that produces entry/exit signals from OHLCV data. This module safely executes
that generated code and returns StrategySignals for backtesting.

The generated function must follow this interface:
    def generate_signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
        '''
        Args:
            df: DataFrame with columns: open, high, low, close, volume (float)
        Returns:
            entries: Boolean Series (True on entry bars)
            exits: Boolean Series (True on exit bars)
            direction: 'long' or 'short'
        '''
"""

import traceback

import numpy as np
import pandas as pd
from loguru import logger

from gold_trading.backtest.engine import StrategySignals


def execute_generated_strategy(
    code: str,
    ohlcv: pd.DataFrame,
) -> StrategySignals | None:
    """Safely execute Claude-generated signal logic against OHLCV data.

    Args:
        code: Python code string containing a `generate_signals(df)` function.
        ohlcv: DataFrame with open, high, low, close, volume columns.

    Returns:
        StrategySignals if execution succeeds, None on error.
    """
    # Prepare a safe execution namespace with allowed libraries
    namespace = {
        "pd": pd,
        "np": np,
        "pd_Series": pd.Series,
        "__builtins__": {
            "range": range,
            "len": len,
            "max": max,
            "min": min,
            "abs": abs,
            "int": int,
            "float": float,
            "bool": bool,
            "True": True,
            "False": False,
            "None": None,
            "print": logger.debug,
        },
    }

    try:
        # Execute the generated code to define the function
        exec(code, namespace)
    except Exception as e:
        logger.error(f"Strategy code compilation failed: {e}")
        logger.debug(f"Code:\n{code[:500]}")
        return None

    # Get the generate_signals function
    func = namespace.get("generate_signals")
    if func is None:
        logger.error("Generated code does not define 'generate_signals' function")
        return None

    try:
        # Execute with a copy of the data to prevent mutation
        df = ohlcv[["open", "high", "low", "close", "volume"]].copy().astype(float)
        result = func(df)

        if not isinstance(result, tuple) or len(result) != 3:
            logger.error(
                f"generate_signals must return (entries, exits, direction), got {type(result)}"
            )
            return None

        entries, exits, direction = result

        # Validate outputs
        if not isinstance(entries, pd.Series):
            entries = pd.Series(entries, index=ohlcv.index)
        if not isinstance(exits, pd.Series):
            exits = pd.Series(exits, index=ohlcv.index)

        entries = entries.fillna(False).astype(bool)
        exits = exits.fillna(False).astype(bool)

        if direction not in ("long", "short"):
            logger.warning(f"Invalid direction '{direction}', defaulting to 'long'")
            direction = "long"

        entry_count = entries.sum()
        exit_count = exits.sum()
        logger.info(
            f"Strategy generated {entry_count} entries, {exit_count} exits, direction={direction}"
        )

        if entry_count == 0:
            logger.warning("Strategy produced zero entries")

        return StrategySignals(entries=entries, exits=exits, direction=direction)

    except Exception as e:
        logger.error(f"Strategy execution failed: {e}")
        logger.debug(traceback.format_exc())
        return None


STRATEGY_CODE_PROMPT = """You are writing a Python trading strategy function for gold futures (GC) on 5-minute bars.

## Interface
Write a function with EXACTLY this signature:
```python
def generate_signals(df):
    # df has columns: open, high, low, close, volume (all float)
    # Must return: (entries, exits, direction)
    # entries: pd.Series of bool (True = enter position)
    # exits: pd.Series of bool (True = exit position)
    # direction: 'long' or 'short'
```

## Available Libraries
- `pd` (pandas) — for Series operations, rolling, ewm, etc.
- `np` (numpy) — for mathematical operations, where, etc.

## Rules
1. Use ONLY pd and np — no imports, no file I/O, no network calls
2. Return EXACTLY (entries_series, exits_series, direction_string)
3. Use .fillna(False) on boolean series to handle NaN
4. All indicators must be computed from the df columns — no external data
5. The function must be self-contained — no global variables

## Current Market Context
{context}

## Strategy Analyst Guidance
{guidance}

## Past Strategy Results (learn from these)
{past_results}

## Your Task
{task}

Respond with ONLY the Python code — no explanation, no markdown fences, just the function definition starting with `def generate_signals(df):`.
"""
