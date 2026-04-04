"""Strategy Analyst agent — diagnoses why strategies fail and directs the Quant Researcher.

Paperclip process adapter script. Runs on heartbeat (every 30 minutes).
Analyzes the pattern of strategy failures, identifies what's working vs not,
and creates targeted Paperclip tasks for the Quant Researcher with specific
parameter guidance based on market conditions.

This is the bridge between "generate random strategies" and "generate
strategies informed by what we've learned."

Run manually: uv run python scripts/strategy_analyst.py
"""

import asyncio
import json
import os

import anthropic
import asyncpg
from loguru import logger

from gold_trading.db.client import get_database_url
from gold_trading.db.queries.decisions import insert_decision
from gold_trading.db.queries.regime import get_latest_regime
from gold_trading.models.lesson import DecisionLogEntry
from gold_trading.paperclip import create_task

ANTHROPIC_MODEL = "claude-sonnet-4-6"


async def get_strategy_history(conn: asyncpg.Connection) -> list[dict]:
    """Get all strategies with their performance metrics."""
    rows = await conn.fetch(
        """
        SELECT id, strategy_class, vbt_sharpe, vbt_win_rate, vbt_total_trades,
               vbt_profit_factor, vbt_max_drawdown, mc_sharpe_p5, mc_sharpe_p50,
               backtest_params, status, created_at
        FROM strategies
        ORDER BY created_at DESC
        LIMIT 20
        """
    )
    results = []
    for r in rows:
        params = r["backtest_params"]
        if isinstance(params, str):
            params = json.loads(params)
        results.append(
            {
                "id": r["id"],
                "class": r["strategy_class"],
                "sharpe": float(r["vbt_sharpe"]) if r["vbt_sharpe"] else None,
                "win_rate": float(r["vbt_win_rate"]) if r["vbt_win_rate"] else None,
                "trades": r["vbt_total_trades"],
                "profit_factor": float(r["vbt_profit_factor"]) if r["vbt_profit_factor"] else None,
                "max_dd": float(r["vbt_max_drawdown"]) if r["vbt_max_drawdown"] else None,
                "mc_p5": float(r["mc_sharpe_p5"]) if r["mc_sharpe_p5"] else None,
                "params": params,
                "status": r["status"],
            }
        )
    return results


async def analyze_and_recommend(
    strategies: list[dict],
    regime: str,
    regime_confidence: float,
    adx: float,
) -> dict:
    """Use Claude to analyze strategy failures and recommend next steps."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Build analysis summary
    by_class = {}
    for s in strategies:
        cls = s["class"] or "unknown"
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(s)

    class_summary = ""
    for cls, strats in by_class.items():
        sharpes = [s["sharpe"] for s in strats if s["sharpe"] is not None]
        avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0
        best = max(strats, key=lambda x: x["sharpe"] or -999)
        class_summary += (
            f"\n{cls}: {len(strats)} attempts, avg Sharpe={avg_sharpe:.2f}, "
            f"best={best['id']} (Sharpe={best['sharpe']:.2f}, WR={best.get('win_rate', 0):.0%})"
        )
        if best.get("params"):
            class_summary += f"\n  Best params: {json.dumps(best['params'])}"

    prompt = f"""You are a quantitative strategy analyst reviewing the performance of an automated gold futures (GC) trading system.

## Current Market Regime
- Regime: {regime} (confidence: {regime_confidence:.0%})
- ADX: {adx:.1f} (>25 = strong trend, <20 = ranging)

## Strategy Performance History
{class_summary}

## Full Strategy List (most recent first)
{
        json.dumps(
            [
                {
                    "id": s["id"],
                    "class": s["class"],
                    "sharpe": s["sharpe"],
                    "win_rate": s["win_rate"],
                    "trades": s["trades"],
                    "profit_factor": s["profit_factor"],
                    "params": s["params"],
                    "status": s["status"],
                }
                for s in strategies[:15]
            ],
            indent=2,
        )
    }

## Deployment Gates (BOTH must pass)
- Backtest: Sharpe >= 1.5, WR >= 50%, PF >= 1.3, DD < 5%, trades >= 50
- Monte Carlo: 5th percentile Sharpe >= 0.5, 95th percentile DD < 5%, ruin < 5%

## Your Task
1. Identify the ROOT CAUSE of why strategies keep failing
2. Based on the current regime ({regime}, ADX={adx:.1f}), what strategy CLASS should we try next?
3. Give SPECIFIC parameter recommendations (not ranges — exact values to try)
4. If mean-reversion keeps failing in a trending market, say so explicitly

Respond as JSON:
{{
    "diagnosis": "Why strategies are failing (2-3 sentences)",
    "recommended_class": "breakout|mean_reversion|momentum|short_breakout",
    "recommended_params": {{"type": "...", "lookback": N, "atr_multiplier": N.N}},
    "task_title": "Short actionable title for the Quant Researcher",
    "task_description": "Detailed instructions with specific parameters and rationale",
    "confidence": 0.0-1.0
}}"""

    response = await client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Extract JSON from various response formats
    import re

    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try stripping code fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith("{"):
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue

    logger.error(f"Failed to parse analysis: {text[:300]}")
    return {}


async def main() -> None:
    """Main heartbeat execution."""
    logger.info("Strategy Analyst heartbeat starting")

    conn = await asyncpg.connect(get_database_url())
    try:
        # Get strategy history
        strategies = await get_strategy_history(conn)
        if len(strategies) < 3:
            logger.info("Not enough strategies to analyze (need 3+)")
            return

        # Get current regime
        regime_state = await get_latest_regime(conn)
        regime = regime_state.regime if regime_state else "unknown"
        regime_conf = (
            float(regime_state.hmm_confidence)
            if regime_state and regime_state.hmm_confidence
            else 0.5
        )
        adx = float(regime_state.adx_14) if regime_state and regime_state.adx_14 else 0

        # Analyze
        analysis = await analyze_and_recommend(strategies, regime, regime_conf, adx)

        if not analysis:
            logger.error("Analysis failed")
            return

        logger.info(f"Diagnosis: {analysis.get('diagnosis', '?')}")
        logger.info(
            f"Recommended: {analysis.get('recommended_class', '?')} with {analysis.get('recommended_params', {})}"
        )

        # Create task for Quant Researcher via Technical Analyst (team lead)
        task_title = analysis.get("task_title", "Try new strategy approach")
        task_desc = analysis.get("task_description", "")

        # Add the specific params as structured data
        params = analysis.get("recommended_params", {})
        if params:
            task_desc += f"\n\nRecommended parameters: {json.dumps(params)}"
            task_desc += f"\nRecommended class: {analysis.get('recommended_class', 'breakout')}"
            task_desc += f"\nDiagnosis: {analysis.get('diagnosis', '')}"

        await create_task(
            title=f"[Strategy Analyst] {task_title}",
            description=task_desc,
            assignee="quant_researcher",
            priority="high",
        )

        # Log decision
        await insert_decision(
            conn,
            DecisionLogEntry(
                agent_name="strategy_analyst",
                decision_type="strategy_analysis",
                inputs_summary={
                    "strategies_analyzed": len(strategies),
                    "regime": regime,
                    "adx": adx,
                    "classes_tried": list({s["class"] for s in strategies if s["class"]}),
                    "avg_sharpe": sum(s["sharpe"] for s in strategies if s["sharpe"])
                    / max(len(strategies), 1),
                },
                decision=f"Recommend {analysis.get('recommended_class', '?')}: {task_title}",
                reasoning=analysis.get("diagnosis", ""),
                confidence=analysis.get("confidence", 0.5),
            ),
        )

        logger.info("Strategy Analyst complete. Task created for Quant Researcher.")

    finally:
        await conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
