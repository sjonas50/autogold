"""Strategy tournament — evolutionary approach to strategy development.

Maintains a leaderboard of strategies ranked by a composite fitness score.
Drives the explore/exploit balance: 30% new strategies, 70% refinements
of top performers.

Fitness score = weighted combination of:
- Sharpe ratio (40%)
- Win rate (25%)
- Profit factor (20%)
- Low drawdown bonus (15%)
"""

import json

import asyncpg


def calculate_fitness(
    sharpe: float | None,
    win_rate: float | None,
    profit_factor: float | None,
    max_drawdown: float | None,
    total_trades: int | None,
) -> float:
    """Calculate composite fitness score for a strategy.

    Returns a score from -10 to +10 where higher is better.
    """
    if sharpe is None or total_trades is None or total_trades < 10:
        return -10.0

    s = sharpe or 0
    wr = (win_rate or 0) * 100  # Convert to percentage
    pf = profit_factor or 0
    dd = max_drawdown or 0

    # Weighted composite
    fitness = (
        s * 0.40  # Sharpe (most important)
        + (wr - 50) * 0.05  # Win rate bonus/penalty (centered at 50%)
        + (pf - 1.0) * 2.0 * 0.20  # Profit factor bonus (centered at 1.0)
        + (0.05 - dd) * 20.0 * 0.15  # Low drawdown bonus (centered at 5%)
    )

    # Penalty for too few trades (statistical significance)
    if total_trades < 50:
        fitness *= total_trades / 50

    return round(fitness, 3)


async def get_leaderboard(conn: asyncpg.Connection, limit: int = 10) -> list[dict]:
    """Get the strategy leaderboard ranked by fitness score."""
    rows = await conn.fetch(
        """
        SELECT id, strategy_class, pine_script, vbt_sharpe, vbt_win_rate,
               vbt_profit_factor, vbt_max_drawdown, vbt_total_trades,
               mc_sharpe_p5, backtest_params, status
        FROM strategies
        WHERE vbt_sharpe IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 50
        """
    )

    strategies = []
    for r in rows:
        params = r["backtest_params"]
        if isinstance(params, str):
            params = json.loads(params)

        fitness = calculate_fitness(
            float(r["vbt_sharpe"]) if r["vbt_sharpe"] else None,
            float(r["vbt_win_rate"]) if r["vbt_win_rate"] else None,
            float(r["vbt_profit_factor"]) if r["vbt_profit_factor"] else None,
            float(r["vbt_max_drawdown"]) if r["vbt_max_drawdown"] else None,
            r["vbt_total_trades"],
        )

        strategies.append(
            {
                "id": r["id"],
                "class": r["strategy_class"],
                "code": r["pine_script"],  # Actually Python code for dynamic strategies
                "sharpe": float(r["vbt_sharpe"]) if r["vbt_sharpe"] else None,
                "win_rate": float(r["vbt_win_rate"]) if r["vbt_win_rate"] else None,
                "profit_factor": float(r["vbt_profit_factor"]) if r["vbt_profit_factor"] else None,
                "max_drawdown": float(r["vbt_max_drawdown"]) if r["vbt_max_drawdown"] else None,
                "trades": r["vbt_total_trades"],
                "mc_p5": float(r["mc_sharpe_p5"]) if r["mc_sharpe_p5"] else None,
                "fitness": fitness,
                "status": r["status"],
                "params": params,
            }
        )

    # Sort by fitness (best first)
    strategies.sort(key=lambda x: x["fitness"], reverse=True)
    return strategies[:limit]


def should_explore(leaderboard: list[dict]) -> bool:
    """Decide whether to explore (new strategy) or exploit (refine existing).

    - If <5 strategies: always explore (need diversity)
    - If best fitness > 0: 30% explore, 70% exploit
    - If all negative: 50/50
    """
    import random

    if len(leaderboard) < 5:
        return True

    best_fitness = leaderboard[0]["fitness"] if leaderboard else -10
    if best_fitness > 0:
        return random.random() < 0.30  # 30% explore
    else:
        return random.random() < 0.50  # 50% explore


def format_leaderboard_for_prompt(leaderboard: list[dict], top_n: int = 5) -> str:
    """Format leaderboard as text for Claude prompt."""
    if not leaderboard:
        return "No strategies yet — this is the first attempt."

    lines = ["Strategy Leaderboard (ranked by fitness):"]
    lines.append(
        f"{'Rank':<5} {'ID':<20} {'Fitness':>8} {'Sharpe':>7} {'WR':>6} {'PF':>5} {'Trades':>6} {'Status'}"
    )
    lines.append("-" * 80)

    for i, s in enumerate(leaderboard[:top_n]):
        wr = f"{s['win_rate'] * 100:.0f}%" if s["win_rate"] else " -"
        pf = f"{s['profit_factor']:.2f}" if s["profit_factor"] else " -"
        lines.append(
            f"#{i + 1:<4} {s['id']:<20} {s['fitness']:>+8.3f} {s['sharpe'] or 0:>+7.2f} "
            f"{wr:>6} {pf:>5} {s['trades'] or 0:>6} {s['status']}"
        )

    return "\n".join(lines)
