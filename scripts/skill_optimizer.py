"""Skill Optimizer agent — analyzes agent performance and proposes skill updates.

Paperclip process adapter script. Runs on heartbeat (every 4 hours).
Reviews decision_log accuracy for each agent, identifies systematic patterns,
and proposes SKILL.md updates via Paperclip tasks.

This is the meta-learning layer — agents improve their own operating instructions.

Run manually: uv run python scripts/skill_optimizer.py
"""

import asyncio
import json
import os

import anthropic
import asyncpg
import httpx
from loguru import logger

from gold_trading.db.client import get_database_url
from gold_trading.db.queries.decisions import insert_decision
from gold_trading.models.lesson import DecisionLogEntry

ANTHROPIC_MODEL = "claude-sonnet-4-6"
PAPERCLIP_URL = os.environ.get("PAPERCLIP_URL", "http://localhost:3100")
COMPANY_ID = os.environ.get("PAPERCLIP_COMPANY_ID", "3422f81a-8ca2-4ce1-aae5-5cf8ce34fa0e")
CIO_AGENT_ID = "37bbe408-e573-4598-a374-cc369bad0258"

# Minimum decisions needed before analyzing an agent
MIN_DECISIONS_FOR_ANALYSIS = 10


async def get_agent_accuracy(conn: asyncpg.Connection, agent_name: str) -> dict | None:
    """Calculate decision accuracy for an agent based on outcome tags."""
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE outcome_tag = 'correct') as correct,
            COUNT(*) FILTER (WHERE outcome_tag = 'incorrect') as incorrect,
            COUNT(*) FILTER (WHERE outcome_tag = 'pending') as pending
        FROM decision_log
        WHERE agent_name = $1
          AND created_at > NOW() - INTERVAL '7 days'
        """,
        agent_name,
    )
    if not row or row["total"] < MIN_DECISIONS_FOR_ANALYSIS:
        return None

    total = row["total"]
    tagged = row["correct"] + row["incorrect"]
    accuracy = row["correct"] / tagged if tagged > 0 else None

    return {
        "total_decisions": total,
        "correct": row["correct"],
        "incorrect": row["incorrect"],
        "pending": row["pending"],
        "accuracy": accuracy,
    }


async def get_recent_failures(conn: asyncpg.Connection, agent_name: str) -> list[dict]:
    """Get recent incorrect decisions for an agent."""
    rows = await conn.fetch(
        """
        SELECT decision_type, decision, reasoning, inputs_summary, created_at
        FROM decision_log
        WHERE agent_name = $1 AND outcome_tag = 'incorrect'
        ORDER BY created_at DESC
        LIMIT 10
        """,
        agent_name,
    )
    return [
        {
            "type": r["decision_type"],
            "decision": r["decision"],
            "reasoning": r["reasoning"],
            "inputs": json.loads(r["inputs_summary"])
            if isinstance(r["inputs_summary"], str)
            else r["inputs_summary"],
        }
        for r in rows
    ]


async def get_strategy_performance_patterns(conn: asyncpg.Connection) -> dict:
    """Get performance patterns across all strategies."""
    rows = await conn.fetch(
        """
        SELECT strategy_class,
               COUNT(*) as attempts,
               AVG(vbt_sharpe) as avg_sharpe,
               AVG(vbt_win_rate) as avg_win_rate,
               COUNT(*) FILTER (WHERE status = 'pending_deployment') as passed,
               COUNT(*) FILTER (WHERE status = 'retired') as failed
        FROM strategies
        WHERE strategy_class IS NOT NULL
        GROUP BY strategy_class
        ORDER BY avg_sharpe DESC
        """
    )
    return {
        r["strategy_class"]: {
            "attempts": r["attempts"],
            "avg_sharpe": float(r["avg_sharpe"]) if r["avg_sharpe"] else 0,
            "avg_win_rate": float(r["avg_win_rate"]) if r["avg_win_rate"] else 0,
            "passed": r["passed"],
            "failed": r["failed"],
            "success_rate": r["passed"] / r["attempts"] if r["attempts"] > 0 else 0,
        }
        for r in rows
    }


async def generate_skill_recommendations(
    accuracy_data: dict[str, dict | None],
    strategy_patterns: dict,
    failures: dict[str, list[dict]],
) -> list[dict]:
    """Use Claude to analyze patterns and recommend skill updates."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return []

    client = anthropic.AsyncAnthropic(api_key=api_key)

    prompt = f"""You are analyzing the performance of a gold trading AI system's agents.

## Agent Accuracy (last 7 days)
{json.dumps(accuracy_data, indent=2, default=str)}

## Strategy Performance by Class
{json.dumps(strategy_patterns, indent=2, default=str)}

## Recent Failures by Agent
{json.dumps(failures, indent=2, default=str)}

Based on this data, generate 1-3 specific, actionable skill update recommendations.
Each should improve a specific agent's decision-making.

Format as JSON array:
[{{
  "agent": "agent_name",
  "skill": "skill_name_to_update",
  "recommendation": "Specific change to make",
  "evidence": "What data pattern supports this",
  "priority": "high|medium|low"
}}]"""

    response = await client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse recommendations: {text[:200]}")
        return []


async def create_skill_update_task(recommendation: dict) -> None:
    """Create a Paperclip task for implementing a skill update."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{PAPERCLIP_URL}/api/companies/{COMPANY_ID}/issues",
                json={
                    "title": f"[SKILL] Update {recommendation['skill']} for {recommendation['agent']}",
                    "description": (
                        f"**Recommendation:** {recommendation['recommendation']}\n\n"
                        f"**Evidence:** {recommendation['evidence']}\n\n"
                        f"**Priority:** {recommendation['priority']}\n\n"
                        f"Review this recommendation and update the relevant skill or agent instructions."
                    ),
                    "assigneeAgentId": CIO_AGENT_ID,
                    "priority": recommendation.get("priority", "medium"),
                },
            )
            logger.info(
                f"Created skill update task: {recommendation['skill']} for {recommendation['agent']}"
            )
    except Exception as e:
        logger.warning(f"Could not create Paperclip task: {e}")


async def main() -> None:
    """Main heartbeat execution."""
    logger.info("Skill Optimizer heartbeat starting")

    conn = await asyncpg.connect(get_database_url())
    try:
        # Gather accuracy data for all agents
        agent_names = [
            "regime_analyst",
            "sentiment_analyst",
            "macro_analyst",
            "quant_researcher",
            "risk_manager",
            "cio",
        ]
        accuracy_data = {}
        failures = {}

        for agent in agent_names:
            accuracy_data[agent] = await get_agent_accuracy(conn, agent)
            agent_failures = await get_recent_failures(conn, agent)
            if agent_failures:
                failures[agent] = agent_failures

        # Get strategy patterns
        strategy_patterns = await get_strategy_performance_patterns(conn)

        # Only analyze if we have enough data
        has_data = any(v is not None for v in accuracy_data.values()) or strategy_patterns
        if not has_data:
            logger.info("Insufficient data for skill optimization — need more agent decisions")
            return

        # Generate recommendations
        recommendations = await generate_skill_recommendations(
            accuracy_data, strategy_patterns, failures
        )

        for rec in recommendations:
            logger.info(
                f"Recommendation: {rec.get('agent')}/{rec.get('skill')} — "
                f"{rec.get('recommendation', '')[:100]}"
            )
            await create_skill_update_task(rec)

        # Log decision
        await insert_decision(
            conn,
            DecisionLogEntry(
                agent_name="skill_optimizer",
                decision_type="skill_review",
                inputs_summary={
                    "agents_analyzed": len([v for v in accuracy_data.values() if v is not None]),
                    "strategy_classes": len(strategy_patterns),
                    "total_failures": sum(len(f) for f in failures.values()),
                },
                decision=f"Generated {len(recommendations)} skill update recommendations",
                reasoning="; ".join(
                    f"{r['agent']}/{r['skill']}: {r['recommendation'][:80]}"
                    for r in recommendations
                )
                or "No recommendations — insufficient data or no patterns detected.",
                confidence=0.6,
            ),
        )

        logger.info(f"Skill Optimizer complete. {len(recommendations)} recommendations.")

    finally:
        await conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
