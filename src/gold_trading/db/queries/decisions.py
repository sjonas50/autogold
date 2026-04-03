"""Database queries for decision_log table."""

import json
from uuid import UUID

import asyncpg

from gold_trading.models.lesson import DecisionLogEntry


async def insert_decision(conn: asyncpg.Connection, entry: DecisionLogEntry) -> UUID:
    """Insert a decision log entry."""
    row = await conn.fetchrow(
        """
        INSERT INTO decision_log
            (agent_name, decision_type, inputs_summary, decision,
             reasoning, confidence, outcome_tag, related_trade)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
        RETURNING id
        """,
        entry.agent_name,
        entry.decision_type,
        json.dumps(entry.inputs_summary),
        entry.decision,
        entry.reasoning,
        entry.confidence,
        entry.outcome_tag,
        entry.related_trade,
    )
    return row["id"]


async def get_agent_decisions(
    conn: asyncpg.Connection,
    agent_name: str,
    decision_type: str | None = None,
    hours: int = 24,
    limit: int = 10,
) -> list[DecisionLogEntry]:
    """Get recent decisions by a specific agent."""
    if decision_type:
        rows = await conn.fetch(
            """
            SELECT * FROM decision_log
            WHERE agent_name = $1 AND decision_type = $2
              AND created_at > NOW() - make_interval(hours => $3)
            ORDER BY created_at DESC LIMIT $4
            """,
            agent_name,
            decision_type,
            hours,
            limit,
        )
    else:
        rows = await conn.fetch(
            """
            SELECT * FROM decision_log
            WHERE agent_name = $1
              AND created_at > NOW() - make_interval(hours => $2)
            ORDER BY created_at DESC LIMIT $3
            """,
            agent_name,
            hours,
            limit,
        )
    return [
        DecisionLogEntry(**{**dict(r), "inputs_summary": json.loads(r["inputs_summary"])})
        for r in rows
    ]


async def tag_decision_outcome(conn: asyncpg.Connection, decision_id: UUID, outcome: str) -> None:
    """Tag a past decision as correct or incorrect."""
    await conn.execute(
        "UPDATE decision_log SET outcome_tag = $1 WHERE id = $2",
        outcome,
        decision_id,
    )
