#!/bin/bash
# Heartbeat scheduler for process adapter agents.
# Paperclip's built-in scheduler only auto-fires claude_local agents.
# This script triggers process agents on their intended intervals.
#
# Run this in the background:
#   nohup bash scripts/heartbeat_cron.sh &
#
# Or add individual crontab entries:
#   */15 * * * * curl -s -X POST http://localhost:3100/api/agents/80bab3f7-e5ef-40d3-8766-22f26bb394ec/heartbeat/invoke -H "Content-Type: application/json"

BASE="http://localhost:3100/api/agents"

trigger() {
    local agent_id="$1"
    local name="$2"
    result=$(curl -s -X POST "$BASE/$agent_id/heartbeat/invoke" -H "Content-Type: application/json" 2>/dev/null)
    status=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','error'))" 2>/dev/null)
    echo "[$(date '+%H:%M:%S')] $name → $status"
}

echo "AutoGoldFutures heartbeat scheduler started"
echo "Press Ctrl+C to stop"

iteration=0
while true; do
    iteration=$((iteration + 1))

    # Every 15 min: Risk Manager
    if [ $((iteration % 1)) -eq 0 ]; then
        trigger "80bab3f7-e5ef-40d3-8766-22f26bb394ec" "Risk Manager"
    fi

    # Every 30 min (2 iterations): Regime, Sentiment, Quant, WF, Strategy Monitor
    if [ $((iteration % 2)) -eq 0 ]; then
        trigger "47f7e6ee-4e47-4b71-919d-6de9923d1929" "Regime Analyst"
        trigger "fc065152-e720-4cdb-bed1-e8d1ae34d5f1" "Sentiment Analyst"
        trigger "881e708a-b4c2-472d-9c74-af7b515cac23" "Quant Researcher"
        trigger "cf64f80f-80e4-4d6f-828a-c96f6d34a4a6" "Walk-Forward Validator"
        trigger "3a01e631-8fbc-44fa-bb7b-b953b35d66e0" "Strategy Monitor"
    fi

    # Every 60 min (4 iterations): Macro, Data Pipeline, Econ Calendar, Technical Analyst
    if [ $((iteration % 4)) -eq 0 ]; then
        trigger "f3fcd8b4-ce00-4d10-b4a5-89719773d8ab" "Macro Analyst"
        trigger "b9c1a8e1-f7e5-408f-8311-161f769bb40a" "Data Pipeline"
        trigger "e10bb9f0-89b5-4d8a-a879-77aed8c6e184" "Economic Calendar"
    fi

    # Every 4 hours (16 iterations): Skill Optimizer
    if [ $((iteration % 16)) -eq 0 ]; then
        trigger "3201c323-a70d-4fdd-9e57-d771bdba8a61" "Skill Optimizer"
    fi

    # Sleep 15 minutes between iterations
    sleep 900
done
