"""Performance metrics.  Unavailable metrics are ``None`` (never a fake 0)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_metrics(trades: pd.DataFrame, equity: pd.DataFrame, initial_capital: float) -> dict:
    n = len(trades)
    net = trades["net_pnl"] if n else pd.Series(dtype=float)
    wins, losses = int((net > 0).sum()), int((net < 0).sum())
    gross_profit = float(net[net > 0].sum()) if n else 0.0
    gross_loss = float(-net[net < 0].sum()) if n else 0.0
    final = float(equity["equity"].iloc[-1]) if len(equity) else initial_capital
    dd_abs = dd_pct = None
    if len(equity):
        dd_abs = float(equity["drawdown"].min())
        dd_pct = float(equity["drawdown_pct"].min())
    streak = best = 0
    for v in net:
        streak = streak + 1 if v < 0 else 0
        best = max(best, streak)
    return {
        "initial_capital": float(initial_capital),
        "final_capital": final,
        "net_pnl": final - initial_capital,
        "return_pct": (final / initial_capital - 1) * 100,
        "total_trades": n,
        "wins": wins,
        "losses": losses,
        "breakeven": n - wins - losses,
        "win_rate_pct": wins / n * 100 if n else None,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,   # undefined with no losing trades
        "expectancy": float(net.mean()) if n else None,
        "avg_r": float(trades["r_multiple"].mean()) if n else None,
        "avg_win": float(net[net > 0].mean()) if wins else None,
        "avg_loss": float(net[net < 0].mean()) if losses else None,
        "gross_pnl": float(trades["gross_pnl"].sum()) if n else None,
        "total_costs": float(trades["costs"].sum()) if n else None,
        "max_drawdown": dd_abs,
        "max_drawdown_pct": dd_pct,
        "max_consecutive_losses": best if n else None,
    }


def monthly_results(trades: pd.DataFrame, equity: pd.DataFrame, initial_capital: float,
                    start, end) -> pd.DataFrame:
    """One row per calendar month in [start, end]; zero-trade months show genuine zeros."""
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    rows = []
    eq = equity.copy()
    if len(eq):
        eq["month"] = eq["timestamp"].dt.tz_localize(None).dt.to_period("M")
    tr = trades.copy()
    if len(tr):
        tr["month"] = tr["exit_time"].dt.tz_localize(None).dt.to_period("M")
    prev_equity = initial_capital
    for m in months:
        t = tr[tr["month"] == m] if len(tr) else tr
        e = eq[eq["month"] == m] if len(eq) else eq
        end_equity = float(e["equity"].iloc[-1]) if len(e) else prev_equity
        pnl = end_equity - prev_equity
        rows.append({"month": str(m), "trades": len(t), "wins": int((t["net_pnl"] > 0).sum()) if len(t) else 0,
                     "net_pnl": pnl, "return_pct": pnl / prev_equity * 100 if prev_equity else None,
                     "end_equity": end_equity})
        prev_equity = end_equity
    return pd.DataFrame(rows)


def stability_warnings(metrics: dict, min_trades: int) -> list[str]:
    out = []
    n = metrics["total_trades"]
    if n == 0:
        out.append("No trades were taken: every performance metric is unavailable, not zero.")
    elif n < min_trades:
        out.append(f"Only {n} trades (< {min_trades}): results are statistically unreliable.")
    if n and metrics["profit_factor"] is None:
        out.append("Profit factor is undefined (no losing trades) -- usually a sign of too few trades.")
    return out
