"""AutoGoldFutures status dashboard — shows all agent results at a glance.

Usage: uv run python scripts/status.py
"""

import asyncio

import asyncpg
from loguru import logger

from gold_trading.db.client import get_database_url


async def main() -> None:
    conn = await asyncpg.connect(get_database_url())
    try:
        print("=" * 70)
        print("  AutoGoldFutures — System Status")
        print("=" * 70)

        # Market State
        print("\n--- MARKET STATE ---")
        regime = await conn.fetchrow(
            "SELECT regime, hmm_confidence, atr_14, adx_14, created_at "
            "FROM regime_state ORDER BY created_at DESC LIMIT 1"
        )
        if regime:
            print(
                f"  Regime:     {regime['regime']} (conf {float(regime['hmm_confidence'] or 0):.0%})"
            )
            print(f"  ATR(14):    {float(regime['atr_14'] or 0):.4f}")
            print(f"  ADX(14):    {float(regime['adx_14'] or 0):.1f}")
            print(f"  Updated:    {regime['created_at']}")

        macro = await conn.fetchrow(
            "SELECT macro_regime, dxy, real_yield_10y, created_at "
            "FROM macro_data ORDER BY created_at DESC LIMIT 1"
        )
        if macro:
            print(
                f"  Macro:      {macro['macro_regime']}"
                f" (DXY={float(macro['dxy'] or 0):.1f}, yield={float(macro['real_yield_10y'] or 0):.2f}%)"
            )

        sent = await conn.fetchrow(
            "SELECT AVG(sentiment) as avg, COUNT(*) as cnt "
            "FROM sentiment_scores WHERE ingested_at > NOW() - INTERVAL '4 hours'"
        )
        if sent:
            print(
                f"  Sentiment:  {float(sent['avg'] or 0):+.3f} ({sent['cnt']} headlines, 4h window)"
            )

        # Data Health
        print("\n--- DATA HEALTH ---")
        for table, tf in [
            ("ohlcv_5m", "5m"),
            ("ohlcv_15m", "15m"),
            ("ohlcv_1h", "1H"),
            ("ohlcv_daily", "Daily"),
        ]:
            row = await conn.fetchrow(
                f"SELECT COUNT(*) as cnt, MAX(timestamp) as latest FROM {table} WHERE instrument = 'GC'"
            )
            latest = str(row["latest"])[:19] if row["latest"] else "none"
            print(f"  {tf:>5s}: {row['cnt']:>7,} bars  (latest: {latest})")

        # Risk Status
        print("\n--- RISK STATUS ---")
        risk = await conn.fetchrow(
            "SELECT decision, inputs_summary FROM decision_log "
            "WHERE agent_name = 'risk_manager' ORDER BY created_at DESC LIMIT 1"
        )
        if risk:
            import json

            inputs = (
                json.loads(risk["inputs_summary"])
                if isinstance(risk["inputs_summary"], str)
                else risk["inputs_summary"]
            )
            dd = inputs.get("drawdown_pct", 0)
            print(f"  Status:     {risk['decision']}")
            print(f"  Drawdown:   {float(dd):.2%}")
            print(f"  Equity:     ${float(inputs.get('current_equity', 0)):,.2f}")
            print(f"  Daily P&L:  ${float(inputs.get('daily_pnl', 0)):,.2f}")
            print(f"  Positions:  {inputs.get('open_positions', 0)}")

        # Strategies
        print("\n--- STRATEGIES ---")
        strategies = await conn.fetch(
            "SELECT id, strategy_class, vbt_sharpe, vbt_win_rate, vbt_total_trades, "
            "vbt_profit_factor, mc_sharpe_p5, status, backtest_params "
            "FROM strategies ORDER BY created_at DESC LIMIT 10"
        )
        if strategies:
            print(
                f"  {'ID':<28s} {'Class':<15s} {'Sharpe':>7s} {'WR':>6s} {'Trades':>6s} {'PF':>5s} {'MC p5':>6s} {'Status'}"
            )
            print(
                f"  {'-' * 28} {'-' * 15} {'-' * 7} {'-' * 6} {'-' * 6} {'-' * 5} {'-' * 6} {'-' * 10}"
            )
            for s in strategies:
                sharpe = f"{float(s['vbt_sharpe']):.2f}" if s["vbt_sharpe"] else "  -"
                wr = f"{float(s['vbt_win_rate']) * 100:.0f}%" if s["vbt_win_rate"] else "  -"
                trades = str(s["vbt_total_trades"]) if s["vbt_total_trades"] else " -"
                pf = f"{float(s['vbt_profit_factor']):.1f}" if s["vbt_profit_factor"] else " -"
                mc = f"{float(s['mc_sharpe_p5']):.2f}" if s["mc_sharpe_p5"] else "  -"
                print(
                    f"  {s['id']:<28s} {(s['strategy_class'] or '-'):<15s} {sharpe:>7s} {wr:>6s} {trades:>6s} {pf:>5s} {mc:>6s} {s['status']}"
                )
        else:
            print("  No strategies yet")

        # Recent Agent Decisions
        print("\n--- LATEST AGENT DECISIONS ---")
        decisions = await conn.fetch(
            "SELECT DISTINCT ON (agent_name) agent_name, decision_type, "
            "LEFT(decision, 60) as decision, confidence, created_at "
            "FROM decision_log ORDER BY agent_name, created_at DESC"
        )
        if decisions:
            print(f"  {'Agent':<22s} {'Decision':<50s} {'Conf':>5s} {'When'}")
            print(f"  {'-' * 22} {'-' * 50} {'-' * 5} {'-' * 20}")
            for d in sorted(decisions, key=lambda x: x["agent_name"]):
                conf = f"{float(d['confidence']):.0%}" if d["confidence"] else " -"
                when = str(d["created_at"])[:19]
                print(f"  {d['agent_name']:<22s} {d['decision']:<50s} {conf:>5s} {when}")

        # Paper Trades
        print("\n--- PAPER TRADES ---")
        trades = await conn.fetch(
            "SELECT strategy_id, direction, pnl_usd, r_multiple, regime_at_entry, "
            "closed_at FROM trade_journal ORDER BY closed_at DESC LIMIT 5"
        )
        if trades:
            for t in trades:
                pnl = f"${float(t['pnl_usd']):+,.0f}" if t["pnl_usd"] else "open"
                r = f"{float(t['r_multiple']):+.1f}R" if t["r_multiple"] else ""
                print(
                    f"  {t['strategy_id']:<25s} {t['direction']:<5s} {pnl:>8s} {r:>6s}  regime={t['regime_at_entry']}"
                )
        else:
            print("  No paper trades yet")

        # Lessons
        print("\n--- LESSONS LEARNED ---")
        lesson_count = await conn.fetchrow("SELECT COUNT(*) as cnt FROM lessons")
        print(f"  Total lessons: {lesson_count['cnt']}")
        if lesson_count["cnt"] > 0:
            recent = await conn.fetch(
                "SELECT LEFT(content, 80) as content, confidence "
                "FROM lessons ORDER BY created_at DESC LIMIT 3"
            )
            for l in recent:
                print(f"  • {l['content']}")

        print("\n" + "=" * 70)

    finally:
        await conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
