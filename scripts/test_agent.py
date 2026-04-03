"""Minimal test script to verify Paperclip process adapter works.

Reads injected env vars, calls the Paperclip API to fetch agent identity,
and prints success. Run via Paperclip heartbeat or manually:
    PAPERCLIP_AGENT_ID=... PAPERCLIP_API_URL=... PAPERCLIP_API_KEY=... python scripts/test_agent.py
"""

import os

import httpx


def main() -> None:
    agent_id = os.environ.get("PAPERCLIP_AGENT_ID")
    api_url = os.environ.get("PAPERCLIP_API_URL", "http://localhost:3100")
    api_key = os.environ.get("PAPERCLIP_API_KEY", "")

    if not agent_id:
        print("PAPERCLIP_AGENT_ID not set — running outside Paperclip context")
        print("Checking Paperclip health instead...")
        resp = httpx.get(f"{api_url}/api/health")
        print(f"Paperclip status: {resp.json()['status']}")
        return

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = httpx.get(f"{api_url}/api/agents/me", headers=headers)
    if resp.status_code == 200:
        agent = resp.json()
        print(f"Agent identity confirmed: {agent['name']} (adapter={agent['adapterType']})")
    else:
        print(f"Agent identity check returned {resp.status_code}: {resp.text}")

    print("Test agent heartbeat complete.")


if __name__ == "__main__":
    main()
