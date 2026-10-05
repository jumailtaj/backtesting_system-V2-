"""Comparison and robustness tools (bounded -- no brute-force optimisation)."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from src.engine.runner import BacktestRunner, PreparedData
from src.reporting.metrics import monthly_results
from src.strategies.registry import REGISTRY, get_strategy

MAX_VARIANTS = 25
DISCLAIMER = ("The best historical variant is NOT expected to be the best going forward. Differences between "
              "variants are often noise, especially with few trades. Do not treat any ranking as a guarantee.")


@dataclass
class Variant:
    label: str
    strategy_key: str
    overrides: dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    table: pd.DataFrame
    monthly: pd.DataFrame
    configs: dict
    warnings: list
    start: date
    end: date
    split_date: Optional[date] = None
    cost_table: Optional[pd.DataFrame] = None


def split_period(start: date, end: date, fraction: float) -> date:
    """Last day of the in-sample window; out-of-sample starts the next day."""
    days = (end - start).days
    return start + timedelta(days=max(int(days * fraction), 0))


def _row(label: str, m: dict) -> dict:
    return {"trades": m["total_trades"], "net_pnl": m["net_pnl"], "return_pct": m["return_pct"],
            "win_rate_pct": m["win_rate_pct"], "profit_factor": m["profit_factor"],
            "expectancy": m["expectancy"], "max_drawdown_pct": m["max_drawdown_pct"],
            "total_costs": m["total_costs"]}


def compare_variants(runner: BacktestRunner, variants: list, instruments: list, start: date, end: date,
                     in_out_of_sample: bool = True, require_complete: bool = True) -> ComparisonResult:
    if not variants:
        raise ValueError("No variants selected.")
    if len(variants) > MAX_VARIANTS:
        raise ValueError(f"At most {MAX_VARIANTS} variants per comparison (avoids over-fitting by brute force).")
    s = runner.settings
    split = split_period(start, end, s.in_sample_fraction) if in_out_of_sample else None
    prepared_cache: dict = {}
    rows, monthly_rows, configs, warnings = [], [], {}, []
    for v in variants:
        strat = get_strategy(v.strategy_key, v.overrides)
        key = (strat.config.timeframe, strat.config.intraday)
        if key not in prepared_cache:
            prepared_cache[key] = runner.prepare(strat, instruments, start, end, require_complete)
        prepared: PreparedData = prepared_cache[key]
        signals = runner.generate_signals(strat, prepared)
        full = runner.execute(strat, prepared, signals)
        row = {"variant": v.label, "strategy": strat.label, **_row(v.label, full.metrics)}
        if split:
            ins = runner.execute(strat, prepared, signals, start, split)
            oos = runner.execute(strat, prepared, signals, split + timedelta(days=1), end)
            row.update({"IS_trades": ins.metrics["total_trades"], "IS_return_pct": ins.metrics["return_pct"],
                        "IS_profit_factor": ins.metrics["profit_factor"],
                        "OOS_trades": oos.metrics["total_trades"], "OOS_return_pct": oos.metrics["return_pct"],
                        "OOS_profit_factor": oos.metrics["profit_factor"]})
            warnings += _oos_warnings(v.label, row, s.min_trades_warning)
        if full.metrics["total_trades"] < s.min_trades_warning:
            warnings.append(f"[{v.label}] only {full.metrics['total_trades']} trades "
                            f"(< {s.min_trades_warning}): result is unstable.")
        for w in prepared.warnings:
            warnings.append(f"[{v.label}] {w}")
        rows.append(row)
        mr = monthly_results(full.result.trades, full.result.equity, s.initial_capital, start, end)
        for r in mr.itertuples():
            monthly_rows.append({"variant": v.label, "month": r.month, "net_pnl": r.net_pnl, "trades": r.trades})
        configs[v.label] = {"strategy": v.strategy_key, **strat.config.to_dict()}
    table = pd.DataFrame(rows)
    warnings.append(DISCLAIMER)
    return ComparisonResult(table, pd.DataFrame(monthly_rows), configs, list(dict.fromkeys(warnings)),
                            start, end, split)


def _oos_warnings(label: str, row: dict, min_trades: int) -> list:
    out = []
    if row["OOS_trades"] < max(min_trades // 3, 5):
        out.append(f"[{label}] out-of-sample has only {row['OOS_trades']} trades: too few to judge.")
    if (row["IS_return_pct"] or 0) > 0 and row["OOS_return_pct"] is not None and row["OOS_return_pct"] < 0:
        out.append(f"[{label}] profitable in-sample but LOSING out-of-sample: possible over-fitting.")
    pf_is, pf_oos = row["IS_profit_factor"], row["OOS_profit_factor"]
    if pf_is and pf_oos is not None and pf_oos < 0.6 * pf_is:
        out.append(f"[{label}] profit factor fell from {pf_is:.2f} (IS) to {pf_oos:.2f} (OOS): unstable.")
    return out


def sensitivity_variants(strategy_key: str, base_overrides: Optional[dict] = None, grid: Optional[dict] = None,
                         full_grid: bool = False) -> tuple[list, list]:
    """Bounded parameter-sensitivity design.

    Default: ONE-AT-A-TIME variation around the base configuration (cheap, hard to over-fit).
    ``full_grid=True``: Cartesian product, refused if it exceeds MAX_VARIANTS.
    Returns (variants, notes) -- invalid combinations are skipped with a note.
    """
    cls = REGISTRY[strategy_key]
    grid = grid or cls.sensitivity_grid
    base = cls.load(base_overrides)
    notes, variants = [], [Variant("base", strategy_key, dict(base_overrides or {}))]
    combos = []
    if full_grid:
        keys = list(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            combos.append(dict(zip(keys, values)))
    else:
        for k, values in grid.items():
            for val in values:
                if getattr(base.config, k) != val:
                    combos.append({k: val})
    if len(combos) + 1 > MAX_VARIANTS:
        raise ValueError(f"Grid would create {len(combos) + 1} variants (> {MAX_VARIANTS}). Shrink the grid.")
    for combo in combos:
        overrides = {**(base_overrides or {}), **combo}
        try:
            base.with_overrides(combo)
        except ValueError as exc:
            notes.append(f"skipped {combo}: {exc}")
            continue
        variants.append(Variant(", ".join(f"{k}={v}" for k, v in combo.items()), strategy_key, overrides))
    return variants, notes


def cost_sensitivity(runner: BacktestRunner, strategy_key: str, instruments: list, start: date, end: date,
                     multipliers=(0.0, 1.0, 2.0, 3.0), overrides: Optional[dict] = None,
                     require_complete: bool = True) -> pd.DataFrame:
    strat = get_strategy(strategy_key, overrides)
    prepared = runner.prepare(strat, instruments, start, end, require_complete)
    signals = runner.generate_signals(strat, prepared)
    rows = []
    for mult in multipliers:
        m = runner.execute(strat, prepared, signals, cost_multiplier=mult).metrics
        rows.append({"cost_multiplier": mult, **_row(strat.label, m)})
    return pd.DataFrame(rows)
