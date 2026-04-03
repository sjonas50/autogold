#!/bin/bash
cd /Users/sjonas/tradingview
exec /Users/sjonas/.local/bin/uv run python scripts/strategy_monitor.py "$@"
