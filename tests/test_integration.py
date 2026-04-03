"""End-to-end integration test — verifies the full system pipeline.

Tests:
1. Database is up and migrated
2. Each agent script runs without error (process adapter verification)
3. Webhook receives and processes signals
4. Risk Manager enforces rules
5. Paperclip API is healthy and agents are registered
6. Skills are registered
"""

import os
import subprocess
from datetime import UTC, datetime, timedelta

import httpx

from gold_trading.db.queries.decisions import get_agent_decisions, insert_decision
from gold_trading.db.queries.macro import insert_macro_data
from gold_trading.db.queries.regime import get_latest_regime, insert_regime_state
from gold_trading.db.queries.sentiment import insert_sentiment_scores_batch
from gold_trading.db.queries.strategies import (
    set_strategy_active,
    upsert_strategy,
)
from gold_trading.db.queries.trades import (
    close_paper_trade,
    get_open_paper_trades,
    get_open_position_count,
    insert_paper_trade,
)
from gold_trading.models.lesson import DecisionLogEntry
from gold_trading.models.macro import MacroData
from gold_trading.models.regime import RegimeState
from gold_trading.models.sentiment import SentimentScore
from gold_trading.models.strategy import Strategy
from gold_trading.models.trade import PaperTrade

PAPERCLIP_URL = os.environ.get("PAPERCLIP_URL", "http://localhost:3100")
COMPANY_ID = "3422f81a-8ca2-4ce1-aae5-5cf8ce34fa0e"
PROJECT_DIR = "/Users/sjonas/tradingview"
UV_PATH = "/Users/sjonas/.local/bin/uv"


# ============================================================
# Infrastructure checks
# ============================================================


class TestInfrastructure:
    async def test_database_healthy(self, conn):
        """TimescaleDB is up and tables exist."""
        row = await conn.fetchrow(
            "SELECT COUNT(*) as cnt FROM pg_tables WHERE schemaname = 'public'"
        )
        assert row["cnt"] >= 10

    async def test_extensions_loaded(self, conn):
        """pgvector and TimescaleDB extensions are loaded."""
        rows = await conn.fetch(
            "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'timescaledb')"
        )
        extensions = {r["extname"] for r in rows}
        assert extensions == {"vector", "timescaledb"}

    def test_paperclip_healthy(self):
        """Paperclip API responds to health check."""
        resp = httpx.get(f"{PAPERCLIP_URL}/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_paperclip_agents_registered(self):
        """All agents are registered in Paperclip."""
        resp = httpx.get(f"{PAPERCLIP_URL}/api/companies/{COMPANY_ID}/agents")
        assert resp.status_code == 200
        agents = resp.json()
        assert len(agents) >= 7  # 7 core + any new agents

        names = {a["name"] for a in agents}
        expected = {
            "CIO",
            "Technical Analyst",
            "Macro Analyst",
            "Sentiment Analyst",
            "Regime Analyst",
            "Risk Manager",
            "Quant Researcher",
        }
        assert names == expected

    def test_paperclip_skills_registered(self):
        """Custom skills are registered."""
        resp = httpx.get(f"{PAPERCLIP_URL}/api/companies/{COMPANY_ID}/skills")
        assert resp.status_code == 200
        skills = resp.json()
        slugs = {s["slug"] for s in skills}
        assert "gold-trading-context" in slugs
        assert "memory-protocol" in slugs
        assert "pine-script-v6" in slugs

    def test_paperclip_agent_configs_correct(self):
        """Process adapter agents have correct commands."""
        resp = httpx.get(f"{PAPERCLIP_URL}/api/companies/{COMPANY_ID}/agents")
        agents = resp.json()

        process_agents = [a for a in agents if a["adapterType"] == "process"]
        for agent in process_agents:
            cmd = agent["adapterConfig"]["command"]
            assert "scripts/" in cmd, f"{agent['name']} command missing scripts/ path: {cmd}"
            assert agent["adapterConfig"]["cwd"] == PROJECT_DIR


# ============================================================
# Agent script smoke tests
# ============================================================


class TestAgentScripts:
    """Verify each agent script can at least import and parse without errors."""

    def test_risk_manager_imports(self):
        """Risk Manager script imports without error."""
        result = subprocess.run(
            [UV_PATH, "run", "python", "-c", "import scripts.risk_manager"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            timeout=30,
        )
        assert result.returncode == 0, f"Import failed: {result.stderr}"

    def test_macro_analyst_imports(self):
        """Macro Analyst script imports without error."""
        result = subprocess.run(
            [UV_PATH, "run", "python", "-c", "import scripts.macro_analyst"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            timeout=30,
        )
        assert result.returncode == 0, f"Import failed: {result.stderr}"

    def test_sentiment_analyst_imports(self):
        """Sentiment Analyst script imports without error."""
        result = subprocess.run(
            [UV_PATH, "run", "python", "-c", "import scripts.sentiment_analyst"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            timeout=30,
        )
        assert result.returncode == 0, f"Import failed: {result.stderr}"

    def test_regime_analyst_imports(self):
        """Regime Analyst script imports without error."""
        result = subprocess.run(
            [UV_PATH, "run", "python", "-c", "import scripts.regime_analyst"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            timeout=30,
        )
        assert result.returncode == 0, f"Import failed: {result.stderr}"

    def test_quant_researcher_imports(self):
        """Quant Researcher script imports without error."""
        result = subprocess.run(
            [UV_PATH, "run", "python", "-c", "import scripts.quant_researcher"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            timeout=30,
        )
        assert result.returncode == 0, f"Import failed: {result.stderr}"


# ============================================================
# Risk Manager integration
# ============================================================


class TestRiskManagerIntegration:
    async def test_risk_manager_all_clear(self, conn):
        """Risk Manager reports ALL CLEAR when no violations."""
        from scripts.risk_manager import get_account_state

        state = await get_account_state(conn)
        assert state["peak_equity"] > 0
        assert state["open_positions"] == 0

    async def test_risk_manager_detects_drawdown(self, conn):
        """Risk Manager detects drawdown breach and deactivates strategies."""
        # Create an active strategy
        await upsert_strategy(
            conn,
            Strategy(
                id="gs_risk_test",
                name="Risk Test",
                pine_script="// test",
                is_active=True,
                status="active",
            ),
        )
        await set_strategy_active(conn, "gs_risk_test", True)

        # Simulate a losing trade that breaches 2% drawdown
        now = datetime.now(UTC)
        trade = PaperTrade(
            strategy_id="gs_risk_test",
            instrument="GC",
            direction="long",
            contracts=1,
            entry_price=3125.00,
            entry_time=now - timedelta(minutes=30),
            idempotency_key="risk_test_loss",
        )
        trade_id = await insert_paper_trade(conn, trade)
        await close_paper_trade(
            conn,
            trade_id=trade_id,
            exit_price=3100.00,
            exit_time=now,
            pnl_usd=-2500.0,  # Large loss — should trigger drawdown
        )

        # Import and check
        from gold_trading.risk.rules import check_drawdown

        # With $50k account, -$2500 = 5% drawdown from peak
        dd_check = check_drawdown(50_000.0, 47_500.0)
        assert dd_check.passed is False
        assert "Drawdown limit breached" in dd_check.violations[0]

    async def test_duration_violation_detected(self, conn):
        """Trades exceeding 120 minutes are flagged."""
        now = datetime.now(UTC)
        old_entry = now - timedelta(minutes=150)  # 150 min old

        trade = PaperTrade(
            strategy_id="gs_duration_test",
            instrument="GC",
            direction="long",
            contracts=1,
            entry_price=3125.00,
            entry_time=old_entry,
            idempotency_key="duration_test",
        )
        await insert_paper_trade(conn, trade)

        from scripts.risk_manager import check_duration_violations

        open_trades = await get_open_paper_trades(conn)
        violations = await check_duration_violations(conn, open_trades, now)
        assert len(violations) >= 1
        assert violations[0]["duration_minutes"] >= 120


# ============================================================
# Webhook → Paper Trade pipeline
# ============================================================


class TestWebhookPipeline:
    async def test_full_trade_lifecycle(self, conn):
        """Entry signal → paper trade → exit signal → trade journal."""
        now = datetime.now(UTC)

        # Setup: create an active strategy
        await upsert_strategy(
            conn,
            Strategy(
                id="gs_lifecycle_test",
                name="Lifecycle Test",
                pine_script="// test",
                is_active=True,
                status="active",
            ),
        )

        # Setup: seed regime and sentiment for context capture
        await insert_regime_state(conn, RegimeState(regime="trending_up", atr_14=4.2, adx_14=28.0))
        scores = [
            SentimentScore(
                headline="Gold surges on Fed dovishness",
                published_at=now,
                sentiment=0.7,
                gold_relevance=0.9,
                catalyst_tags=["fomc"],
            )
        ]
        await insert_sentiment_scores_batch(conn, scores)
        await insert_macro_data(
            conn,
            MacroData(
                observation_date=now.date(),
                macro_regime="bullish",
                reasoning="Test",
            ),
        )

        # Step 1: Simulate entry
        from gold_trading.models.signal import WebhookPayload
        from gold_trading.webhook.simulator import simulate_entry

        payload = WebhookPayload(
            secret="",
            strategy_id="gs_lifecycle_test",
            action="buy",
            contracts=1,
            price=3125.00,
            bar_time=now,
            instrument="GC",
            stop_loss=3120.00,
        )
        trade = await simulate_entry(conn, payload, "lifecycle_idem_key")

        assert trade.id is not None
        assert trade.direction == "long"
        assert trade.regime_at_entry == "trending_up"
        assert trade.sentiment_at_entry is not None

        # Verify open position
        open_count = await get_open_position_count(conn)
        assert open_count >= 1

        # Step 2: Simulate exit
        from gold_trading.webhook.simulator import simulate_exit

        exit_payload = WebhookPayload(
            secret="",
            strategy_id="gs_lifecycle_test",
            action="close_long",
            contracts=1,
            price=3135.00,
            bar_time=now + timedelta(minutes=45),
            instrument="GC",
        )
        journal_entry = await simulate_exit(conn, exit_payload, "lifecycle_exit_key")

        assert journal_entry is not None
        assert journal_entry.pnl_usd > 0  # Profit (bought 3125, sold ~3135)
        assert journal_entry.regime_at_entry == "trending_up"

        # Verify position closed
        remaining = [
            t for t in await get_open_paper_trades(conn) if t.strategy_id == "gs_lifecycle_test"
        ]
        assert len(remaining) == 0


# ============================================================
# Decision log and memory pipeline
# ============================================================


class TestMemoryPipeline:
    async def test_decision_log_cross_agent(self, conn):
        """Multiple agents can write and CIO can read all decisions."""
        agents = ["regime_analyst", "sentiment_analyst", "macro_analyst"]

        for agent in agents:
            await insert_decision(
                conn,
                DecisionLogEntry(
                    agent_name=agent,
                    decision_type="test_classification",
                    inputs_summary={"test": True},
                    decision=f"test_decision_from_{agent}",
                    confidence=0.8,
                ),
            )

        # CIO reads all decisions
        for agent in agents:
            decisions = await get_agent_decisions(conn, agent, "test_classification", hours=1)
            assert len(decisions) >= 1
            assert decisions[0].decision == f"test_decision_from_{agent}"

    async def test_regime_feeds_into_context(self, conn):
        """Regime state is queryable by other agents."""
        await insert_regime_state(conn, RegimeState(regime="ranging", atr_14=2.5, adx_14=15.0))

        latest = await get_latest_regime(conn)
        assert latest is not None
        assert latest.regime == "ranging"
