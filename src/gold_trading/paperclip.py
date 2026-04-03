"""Shared Paperclip API helpers for agent task management.

Used by process adapter agents to:
- Check for tasks assigned to them by their manager
- Create sub-tasks for their direct reports
- Update task status after completion
"""

import os

import httpx
from loguru import logger

PAPERCLIP_URL = os.environ.get("PAPERCLIP_URL", "http://localhost:3100")
COMPANY_ID = os.environ.get("PAPERCLIP_COMPANY_ID", "3422f81a-8ca2-4ce1-aae5-5cf8ce34fa0e")

# Agent ID registry
AGENTS = {
    "cio": "37bbe408-e573-4598-a374-cc369bad0258",
    "technical_analyst": "e475c802-6bde-4d8e-bb43-602842ae5e7f",
    "regime_analyst": "47f7e6ee-4e47-4b71-919d-6de9923d1929",
    "data_pipeline": "b9c1a8e1-f7e5-408f-8311-161f769bb40a",
    "risk_manager": "80bab3f7-e5ef-40d3-8766-22f26bb394ec",
    "quant_researcher": "881e708a-b4c2-472d-9c74-af7b515cac23",
    "walk_forward_validator": "cf64f80f-80e4-4d6f-828a-c96f6d34a4a6",
    "strategy_monitor": "3a01e631-8fbc-44fa-bb7b-b953b35d66e0",
    "macro_analyst": "f3fcd8b4-ce00-4d10-b4a5-89719773d8ab",
    "sentiment_analyst": "fc065152-e720-4cdb-bed1-e8d1ae34d5f1",
    "economic_calendar": "e10bb9f0-89b5-4d8a-a879-77aed8c6e184",
    "skill_optimizer": "3201c323-a70d-4fdd-9e57-d771bdba8a61",
}


async def get_my_tasks(agent_name: str, status: str = "todo") -> list[dict]:
    """Get tasks assigned to this agent."""
    agent_id = AGENTS.get(agent_name)
    if not agent_id:
        return []

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{PAPERCLIP_URL}/api/companies/{COMPANY_ID}/issues",
                params={"assigneeAgentId": agent_id, "status": status},
            )
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.debug(f"Could not fetch tasks for {agent_name}: {e}")
    return []


async def create_task(
    title: str,
    description: str,
    assignee: str,
    priority: str = "normal",
    parent_id: str | None = None,
) -> str | None:
    """Create a Paperclip task assigned to an agent.

    Args:
        title: Task title.
        description: Task description with context.
        assignee: Agent name key from AGENTS dict.
        priority: 'low', 'normal', 'high', 'urgent'.
        parent_id: Optional parent issue ID for sub-tasks.

    Returns:
        Issue ID if created, None on failure.
    """
    agent_id = AGENTS.get(assignee)
    if not agent_id:
        logger.warning(f"Unknown agent: {assignee}")
        return None

    body: dict = {
        "title": title,
        "description": description,
        "assigneeAgentId": agent_id,
        "priority": priority,
    }
    if parent_id:
        body["parentId"] = parent_id

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{PAPERCLIP_URL}/api/companies/{COMPANY_ID}/issues",
                json=body,
            )
            if resp.status_code in (200, 201, 202):
                issue = resp.json()
                logger.info(f"Created task: {title} → {assignee}")
                return issue.get("id")
            else:
                logger.warning(f"Failed to create task: {resp.status_code}")
    except Exception as e:
        logger.warning(f"Could not create Paperclip task: {e}")
    return None


async def complete_task(issue_id: str, comment: str | None = None) -> None:
    """Mark a task as done with optional comment."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.patch(
                f"{PAPERCLIP_URL}/api/issues/{issue_id}",
                json={"status": "done"},
            )
            if comment:
                await client.post(
                    f"{PAPERCLIP_URL}/api/issues/{issue_id}/comments",
                    json={"body": comment},
                )
    except Exception as e:
        logger.debug(f"Could not complete task {issue_id}: {e}")
