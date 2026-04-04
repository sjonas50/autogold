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
    "cio": "2a732301-8c96-48da-8ae6-c9ef9a4fa517",
    "technical_analyst": "82c0b931-ac82-42fc-a64c-9b0e46af6770",
    "regime_analyst": "5c78fc6e-77d9-4310-85e4-3b923422e5df",
    "data_pipeline": "e28bc80f-222f-44b3-bc91-a0c61dd269b1",
    "risk_manager": "a68ef205-85f9-4c20-9d47-4ab76c864a47",
    "quant_researcher": "8b85ccad-060e-4dd8-b007-4eced8098223",
    "walk_forward_validator": "3548ca9b-b975-40db-a55d-dec060ce79ee",
    "strategy_monitor": "2d9df290-a460-43f4-ac0c-233f61f65ec9",
    "macro_analyst": "06ba3cba-5814-4cb5-adfe-d7a63606c0c2",
    "sentiment_analyst": "88622de5-1b07-4670-9417-363d5217b7e2",
    "economic_calendar": "3467352e-e643-4285-a3ce-c04b105df898",
    "skill_optimizer": "82221f8d-03de-425b-8862-6b21bf2fbf0c",
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
