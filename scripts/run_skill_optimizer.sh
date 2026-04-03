#!/bin/bash
cd /Users/sjonas/tradingview
exec /Users/sjonas/.local/bin/uv run python scripts/skill_optimizer.py "$@"
