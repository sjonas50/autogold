#!/bin/bash
# Heartbeat scheduler for process adapter agents.
# Triggers process agents on their intended intervals via Paperclip API.
#
# Run: nohup bash scripts/heartbeat_cron.sh &
# Stop: kill $(cat logs/heartbeat_cron.pid)

BASE="http://localhost:3100/api/agents"
echo $$ > logs/heartbeat_cron.pid

trigger() {
    local agent_id="$1"
    local name="$2"
    result=$(curl -s -X POST "$BASE/$agent_id/heartbeat/invoke" -H "Content-Type: application/json" 2>/dev/null)
    st=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','error'))" 2>/dev/null)
    echo "[$(date '+%H:%M:%S')] $name → $st"
}

echo "AutoGoldFutures heartbeat scheduler started (PID $$)"

iteration=0
while true; do
    iteration=$((iteration + 1))

    # Every 15 min: Risk Manager
    trigger "a68ef205-85f9-4c20-9d47-4ab76c864a47" "Risk Manager"

    # Every 30 min (2 iterations)
    if [ $((iteration % 2)) -eq 0 ]; then
        trigger "5c78fc6e-77d9-4310-85e4-3b923422e5df" "Regime Analyst"
        trigger "88622de5-1b07-4670-9417-363d5217b7e2" "Sentiment Analyst"
        trigger "8b85ccad-060e-4dd8-b007-4eced8098223" "Quant Researcher"
        trigger "3548ca9b-b975-40db-a55d-dec060ce79ee" "Walk-Forward Validator"
        trigger "2d9df290-a460-43f4-ac0c-233f61f65ec9" "Strategy Monitor"
    fi

    # Every 60 min (4 iterations)
    if [ $((iteration % 4)) -eq 0 ]; then
        trigger "06ba3cba-5814-4cb5-adfe-d7a63606c0c2" "Macro Analyst"
        trigger "e28bc80f-222f-44b3-bc91-a0c61dd269b1" "Data Pipeline"
        trigger "3467352e-e643-4285-a3ce-c04b105df898" "Economic Calendar"
    fi

    # Every 4 hours (16 iterations)
    if [ $((iteration % 16)) -eq 0 ]; then
        trigger "82221f8d-03de-425b-8862-6b21bf2fbf0c" "Skill Optimizer"
    fi

    sleep 900
done
