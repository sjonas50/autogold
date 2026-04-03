"""Tests for database migrations, CRUD operations, and pgvector similarity search.

All tests run against real TimescaleDB — no mocks.
"""

from datetime import date, timedelta

from gold_trading.db.queries.decisions import (
    get_agent_decisions,
    insert_decision,
    tag_decision_outcome,
)
from gold_trading.db.queries.lessons import (
    get_lesson_count,
    insert_lesson,
    search_similar_lessons,
)
from gold_trading.db.queries.macro import get_latest_macro, insert_macro_data
from gold_trading.db.queries.regime import (
    get_latest_regime,
    get_regime_history,
    insert_regime_state,
)
from gold_trading.db.queries.sentiment import (
    get_sentiment_summary,
    insert_sentiment_score,
    insert_sentiment_scores_batch,
)
from gold_trading.db.queries.strategies import (
    deactivate_all_strategies,
    get_active_strategies,
    get_strategy,
    is_strategy_active,
    set_strategy_active,
    upsert_strategy,
)
from gold_trading.db.queries.trades import (
    check_idempotency,
    close_paper_trade,
    get_open_paper_trades,
    get_open_position_count,
    get_recent_trades,
    insert_paper_trade,
    insert_trade_journal,
)
from gold_trading.models.lesson import DecisionLogEntry, Lesson
from gold_trading.models.macro import MacroData
from gold_trading.models.regime import RegimeState
from gold_trading.models.sentiment import SentimentScore
from gold_trading.models.strategy import Strategy
from gold_trading.models.trade import PaperTrade, TradeJournalEntry

# ============================================================
# Migration verification
# ============================================================


class TestMigrations:
    async def test_tables_exist(self, conn):
        """Verify all expected tables were created."""
        rows = await conn.fetch(
            """
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename
            """
        )
        tables = {r["tablename"] for r in rows}
        expected = {
            "trade_journal",
            "decision_log",
            "lessons",
            "regime_state",
            "sentiment_scores",
            "macro_data",
            "strategies",
            "paper_trades",
            "pinescript_corpus",
            "ohlcv_5m",
            "_migrations",
        }
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"

    async def test_extensions_enabled(self, conn):
        """Verify pgvector and timescaledb extensions."""
        rows = await conn.fetch(
            "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'timescaledb')"
        )
        extensions = {r["extname"] for r in rows}
        assert "vector" in extensions
        assert "timescaledb" in extensions

    async def test_hypertables_created(self, conn):
        """Verify hypertables are set up."""
        rows = await conn.fetch("SELECT hypertable_name FROM timescaledb_information.hypertables")
        hypertables = {r["hypertable_name"] for r in rows}
        expected = {
            "trade_journal",
            "decision_log",
            "regime_state",
            "sentiment_scores",
            "macro_data",
            "ohlcv_5m",
        }
        assert expected.issubset(hypertables), f"Missing hypertables: {expected - hypertables}"


# ============================================================
# Trade Journal
# ============================================================


class TestTradeJournal:
    async def test_insert_and_retrieve(self, conn, now):
        """Insert a trade and retrieve it."""
        trade = TradeJournalEntry(
            strategy_id="gs_v1_breakout",
            instrument="GC",
            direction="long",
            contracts=1,
            entry_price=3125.50,
            exit_price=3138.20,
            entry_time=now - timedelta(minutes=45),
            closed_at=now,
            pnl_usd=1270.0,
            r_multiple=2.1,
            regime_at_entry="trending_up",
            sentiment_score=0.45,
            macro_bias="bullish",
            session="london_ny_overlap",
            atr_at_entry=4.25,
        )
        trade_id = await insert_trade_journal(conn, trade)
        assert trade_id is not None

        trades = await get_recent_trades(conn, hours=1)
        assert len(trades) >= 1
        assert trades[0].strategy_id == "gs_v1_breakout"
        assert trades[0].pnl_usd == 1270.0


# ============================================================
# Paper Trades
# ============================================================


class TestPaperTrades:
    async def test_insert_and_get_open(self, conn, now):
        """Insert paper trade and verify it appears in open trades."""
        trade = PaperTrade(
            strategy_id="gs_v1_breakout",
            instrument="MGC",
            direction="long",
            contracts=2,
            entry_price=3120.00,
            entry_time=now,
            idempotency_key="test_key_001",
        )
        trade_id = await insert_paper_trade(conn, trade)
        assert trade_id is not None

        open_trades = await get_open_paper_trades(conn)
        assert len(open_trades) >= 1
        assert open_trades[0].idempotency_key == "test_key_001"

    async def test_idempotency_check(self, conn, now):
        """Verify idempotency key prevents duplicates."""
        trade = PaperTrade(
            strategy_id="gs_v1_breakout",
            instrument="GC",
            direction="long",
            contracts=1,
            entry_price=3120.00,
            entry_time=now,
            idempotency_key="unique_key_123",
        )
        await insert_paper_trade(conn, trade)

        assert await check_idempotency(conn, "unique_key_123") is True
        assert await check_idempotency(conn, "nonexistent_key") is False

    async def test_close_paper_trade(self, conn, now):
        """Insert, close, and verify paper trade lifecycle."""
        trade = PaperTrade(
            strategy_id="gs_v1_breakout",
            instrument="GC",
            direction="long",
            contracts=1,
            entry_price=3120.00,
            entry_time=now,
            idempotency_key="close_test_key",
        )
        trade_id = await insert_paper_trade(conn, trade)

        await close_paper_trade(
            conn,
            trade_id=trade_id,
            exit_price=3135.00,
            exit_time=now + timedelta(minutes=30),
            pnl_usd=1500.0,
            r_multiple=2.5,
        )

        # Should no longer appear in open trades
        open_trades = await get_open_paper_trades(conn)
        open_ids = [t.id for t in open_trades]
        assert trade_id not in open_ids

    async def test_open_position_count(self, conn, now):
        """Verify position counting."""
        initial_count = await get_open_position_count(conn)

        trade = PaperTrade(
            strategy_id="gs_v1_breakout",
            instrument="GC",
            direction="long",
            contracts=1,
            entry_price=3120.00,
            entry_time=now,
            idempotency_key="count_test_key",
        )
        await insert_paper_trade(conn, trade)

        new_count = await get_open_position_count(conn)
        assert new_count == initial_count + 1


# ============================================================
# Strategies
# ============================================================


class TestStrategies:
    async def test_upsert_and_get(self, conn):
        """Insert a strategy and retrieve it."""
        strategy = Strategy(
            id="gs_v1_breakout",
            name="Session Open Breakout v1",
            pine_script="//@version=6\nstrategy('Breakout')\n// ...",
            instrument="GC",
            strategy_class="breakout",
            vbt_sharpe=1.85,
            vbt_win_rate=0.52,
            vbt_expectancy=125.50,
            vbt_max_drawdown=0.015,
        )
        sid = await upsert_strategy(conn, strategy)
        assert sid == "gs_v1_breakout"

        retrieved = await get_strategy(conn, "gs_v1_breakout")
        assert retrieved is not None
        assert retrieved.vbt_sharpe == 1.85
        assert retrieved.strategy_class == "breakout"

    async def test_activate_deactivate(self, conn):
        """Test strategy activation and deactivation."""
        strategy = Strategy(
            id="gs_v2_revert",
            name="VWAP Reversion v2",
            pine_script="// pine",
            instrument="GC",
        )
        await upsert_strategy(conn, strategy)

        await set_strategy_active(conn, "gs_v2_revert", True)
        assert await is_strategy_active(conn, "gs_v2_revert") is True

        active = await get_active_strategies(conn)
        assert any(s.id == "gs_v2_revert" for s in active)

        await set_strategy_active(conn, "gs_v2_revert", False)
        assert await is_strategy_active(conn, "gs_v2_revert") is False

    async def test_deactivate_all(self, conn):
        """Test emergency deactivation of all strategies."""
        for i in range(3):
            s = Strategy(
                id=f"gs_deact_{i}",
                name=f"Strategy {i}",
                pine_script="// pine",
                is_active=True,
            )
            await upsert_strategy(conn, s)
            await set_strategy_active(conn, s.id, True)

        count = await deactivate_all_strategies(conn)
        assert count >= 3

        active = await get_active_strategies(conn)
        assert len(active) == 0


# ============================================================
# Decision Log
# ============================================================


class TestDecisionLog:
    async def test_insert_and_query(self, conn):
        """Insert decisions and query by agent."""
        entry = DecisionLogEntry(
            agent_name="cio",
            decision_type="strategy_review",
            inputs_summary={"regime": "trending_up", "macro": "bullish"},
            decision="Activate gs_v1_breakout",
            reasoning="Regime and macro aligned.",
            confidence=0.8,
        )
        decision_id = await insert_decision(conn, entry)
        assert decision_id is not None

        decisions = await get_agent_decisions(conn, "cio", "strategy_review")
        assert len(decisions) >= 1
        assert decisions[0].decision == "Activate gs_v1_breakout"

    async def test_tag_outcome(self, conn):
        """Tag a decision as correct/incorrect."""
        entry = DecisionLogEntry(
            agent_name="regime_analyst",
            decision_type="regime_classification",
            inputs_summary={"atr": 4.2, "adx": 28.5},
            decision="trending_up",
            confidence=0.85,
        )
        decision_id = await insert_decision(conn, entry)
        await tag_decision_outcome(conn, decision_id, "correct")

        decisions = await get_agent_decisions(conn, "regime_analyst")
        tagged = [d for d in decisions if d.id == decision_id]
        assert len(tagged) == 1
        assert tagged[0].outcome_tag == "correct"


# ============================================================
# Regime State
# ============================================================


class TestRegimeState:
    async def test_insert_and_get_latest(self, conn):
        """Insert regime state and retrieve latest."""
        state = RegimeState(
            regime="trending_up",
            hmm_state=0,
            hmm_confidence=0.87,
            atr_14=4.25,
            adx_14=28.5,
        )
        rid = await insert_regime_state(conn, state)
        assert rid is not None

        latest = await get_latest_regime(conn)
        assert latest is not None
        assert latest.regime == "trending_up"
        assert latest.hmm_confidence == 0.87

    async def test_regime_history(self, conn):
        """Insert multiple and verify we get all back."""
        for regime in ["trending_up", "ranging", "volatile"]:
            await insert_regime_state(
                conn,
                RegimeState(regime=regime, atr_14=4.0, adx_14=20.0),
            )

        history = await get_regime_history(conn, hours=1)
        assert len(history) >= 3
        regimes = {h.regime for h in history}
        assert regimes == {"trending_up", "ranging", "volatile"}


# ============================================================
# Sentiment
# ============================================================


class TestSentiment:
    async def test_insert_and_summary(self, conn, now):
        """Insert sentiment scores and get summary."""
        scores = [
            SentimentScore(
                headline="Gold rallies on dovish Fed comments",
                source="Reuters",
                published_at=now - timedelta(minutes=i * 30),
                sentiment=0.6 - (i * 0.1),
                gold_relevance=0.9,
                catalyst_tags=["fomc", "usd_weakness"],
            )
            for i in range(5)
        ]
        count = await insert_sentiment_scores_batch(conn, scores)
        assert count == 5

        summary = await get_sentiment_summary(conn, hours=4.0)
        assert summary.headline_count >= 5
        assert -1.0 <= summary.avg_sentiment <= 1.0

    async def test_single_insert(self, conn, now):
        """Insert a single sentiment score."""
        score = SentimentScore(
            headline="CPI comes in hot",
            source="Bloomberg",
            published_at=now,
            sentiment=-0.4,
            gold_relevance=0.8,
            catalyst_tags=["cpi"],
        )
        sid = await insert_sentiment_score(conn, score)
        assert sid is not None


# ============================================================
# Macro Data
# ============================================================


class TestMacroData:
    async def test_insert_and_get_latest(self, conn):
        """Insert macro data and retrieve latest."""
        data = MacroData(
            observation_date=date.today(),
            dxy=104.25,
            real_yield_10y=1.85,
            cpi_yoy=3.2,
            breakeven_10y=2.35,
            oil_wti=78.50,
            gold_fix_pm=3120.00,
            macro_regime="bullish",
            reasoning="Real yields falling, DXY weakening, supportive for gold.",
        )
        mid = await insert_macro_data(conn, data)
        assert mid is not None

        latest = await get_latest_macro(conn)
        assert latest is not None
        assert latest.macro_regime == "bullish"
        assert latest.dxy == 104.25


# ============================================================
# Lessons Store (pgvector)
# ============================================================


class TestLessonsStore:
    async def test_insert_lesson(self, conn):
        """Insert a lesson with embedding."""
        # Create a dummy 1536-dim embedding
        embedding = [0.01 * i for i in range(1536)]
        lesson = Lesson(
            content="Session open breakouts fail during NFP week pre-positioning.",
            embedding=embedding,
            regime_tags=["trending_up"],
            strategy_class="breakout",
            macro_context="neutral",
            confidence=0.8,
        )
        lid = await insert_lesson(conn, lesson)
        assert lid is not None

        count = await get_lesson_count(conn)
        assert count >= 1

    async def test_similarity_search(self, conn):
        """Insert lessons with different embeddings and verify ordering."""
        # Insert two lessons with distinct embeddings
        emb_a = [1.0] + [0.0] * 1535  # Points in dimension 0
        emb_b = [0.0] + [1.0] + [0.0] * 1534  # Points in dimension 1

        await insert_lesson(
            conn,
            Lesson(
                content="Lesson A: Breakout works in trending.",
                embedding=emb_a,
                regime_tags=["trending_up"],
                strategy_class="breakout",
                confidence=0.9,
            ),
        )
        await insert_lesson(
            conn,
            Lesson(
                content="Lesson B: Mean reversion works in ranging.",
                embedding=emb_b,
                regime_tags=["ranging"],
                strategy_class="mean_reversion",
                confidence=0.85,
            ),
        )

        # Search with a query similar to emb_a
        query = [0.9] + [0.1] + [0.0] * 1534
        results = await search_similar_lessons(conn, embedding=query, limit=2)

        assert len(results) >= 2
        # Lesson A should be more similar to our query
        assert results[0].content.startswith("Lesson A")

    async def test_similarity_search_with_filter(self, conn):
        """Test filtered similarity search."""
        emb = [0.5] * 1536
        await insert_lesson(
            conn,
            Lesson(
                content="Filtered lesson for ranging",
                embedding=emb,
                regime_tags=["ranging"],
                strategy_class="mean_reversion",
                confidence=0.7,
            ),
        )

        results = await search_similar_lessons(
            conn,
            embedding=emb,
            limit=5,
            regime_filter="ranging",
        )
        assert all("ranging" in r.regime_tags for r in results)
