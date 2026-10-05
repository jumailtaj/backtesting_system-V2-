"""BacktestRunner: glue between data, strategies, the shared engine and reports."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import REPORT_DIR, AppSettings, parse_hhmm
from src.data.manager import DataManager
from src.data.ranges import intersect_ranges
from src.data.validation import check_warmup
from src.engine.backtester import BacktestEngine, BacktestResult, EngineConfig
from src.engine.costs import CostModel
from src.reporting.excel_report import write_run_report
from src.reporting.metrics import compute_metrics
from src.risk.manager import RiskConfig
from src.strategies.base import Strategy

ASSUMPTIONS = [
    "Signals are evaluated on completed candles; entries fill at the OPEN of the next candle.",
    "Candle timestamps are candle START times (IST); a candle closes at start + timeframe.",
    "Market fills (entry, stop, forced exit) include adverse slippage and adverse tick rounding.",
    "Target exits are limit orders filled at the target price when the candle touches it (optimistic).",
    "If one candle touches both stop and target, the stop is assumed to be hit first.",
    "If a candle opens beyond the stop, the exit is at the open price (gap), not the stop price.",
    "Intraday: no entries at/after the force-exit time; positions close at the force-exit candle's open "
    "and never carry overnight. Swing: overnight holding allowed.",
    "Transaction costs apply to both legs using the configured intraday/delivery rate tables.",
    "No leverage: total position value is capped at max_capital_allocation_pct of current equity.",
    "Position size = floor(equity x risk% / |entry - stop|), then capped by capital and position limits.",
    "One position per instrument; no pyramiding; no partial exits; no trailing stops.",
    "No exchange-holiday calendar is assumed; holidays appear as weekdays with no candles.",
    "Signals use only data available at the signal candle's close (no look-ahead).",
]


class DataIncompleteError(Exception):
    """Cached data does not cover the requested trading period."""


@dataclass
class PreparedData:
    candles: dict
    data_info: list
    warnings: list
    timeframe: str
    start: date
    end: date
    intraday: bool


@dataclass
class RunResult:
    result: BacktestResult
    metrics: dict
    start: date
    end: date


class BacktestRunner:
    def __init__(self, settings: AppSettings, data_manager: DataManager, report_dir: Path = REPORT_DIR,
                 sector_map: Optional[dict] = None) -> None:
        self.settings = settings
        self.dm = data_manager
        self.report_dir = Path(report_dir)
        self.sector_map = sector_map or {}

    # ------------------------------------------------------------- data
    def prepare(self, strategy: Strategy, instruments: list, start: date, end: date,
                require_complete: bool = True) -> PreparedData:
        cfg = strategy.config
        warm_days = (self.settings.intraday_warmup_calendar_days if cfg.intraday
                     else self.settings.swing_warmup_calendar_days)
        load_start = start - timedelta(days=warm_days)
        candles, infos, warnings = {}, [], []
        for inst in instruments:
            loaded = self.dm.load_candles(inst, cfg.timeframe, load_start, end)
            info = loaded.info
            infos.append(info)
            trading_missing = []
            for ms, me in info["missing_ranges"]:
                trading_missing += intersect_ranges((start, end), [(ms, me)])
            if trading_missing and require_complete:
                raise DataIncompleteError(
                    f"{inst} [{cfg.timeframe}]: cache does not cover "
                    + ", ".join(f"{a}..{b}" for a, b in trading_missing)
                    + ". Run 'Download / recover missing data' first (or allow an incomplete run).")
            df = loaded.candles
            d = df["timestamp"].dt.date if len(df) else pd.Series([], dtype=object)
            if not len(df) or not ((d >= start) & (d <= end)).any():
                raise DataIncompleteError(f"{inst} [{cfg.timeframe}]: no candles in {start}..{end}. "
                                          "Download data first.")
            w = check_warmup(df, pd.Timestamp(start), strategy.warmup_bars())
            if w:
                info["warnings"].append(w)
            warnings += [f"{inst}: {x}" for x in info["warnings"]]
            candles[inst] = df
        return PreparedData(candles, infos, warnings, cfg.timeframe, start, end, cfg.intraday)

    # ----------------------------------------------------------- engine
    def generate_signals(self, strategy: Strategy, prepared: PreparedData) -> dict:
        return {inst: strategy.generate_signals(df) for inst, df in prepared.candles.items()}

    def engine_config(self, strategy: Strategy, cost_multiplier: Optional[float] = None) -> EngineConfig:
        s, cfg = self.settings, strategy.config
        costs = CostModel(s.intraday_costs, s.delivery_costs,
                          s.cost_multiplier if cost_multiplier is None else cost_multiplier)
        return EngineConfig(
            initial_capital=s.initial_capital, slippage_pct=s.slippage_pct, tick_size=s.tick_size, costs=costs,
            intraday=cfg.intraday, force_exit=parse_hhmm(cfg.force_exit_time), allow_short=cfg.allow_short,
            risk=RiskConfig.from_strategy_config(cfg), max_holding_days=cfg.max_holding_days,
            sector_map=self.sector_map, strategy_name=strategy.label)

    def execute(self, strategy: Strategy, prepared: PreparedData, signals: dict, start: Optional[date] = None,
                end: Optional[date] = None, cost_multiplier: Optional[float] = None) -> RunResult:
        start, end = start or prepared.start, end or prepared.end
        engine = BacktestEngine(self.engine_config(strategy, cost_multiplier))
        result = engine.run(prepared.candles, signals, start, end)
        return RunResult(result, compute_metrics(result.trades, result.equity, result.initial_capital), start, end)

    def run(self, strategy: Strategy, instruments: list, start: date, end: date,
            require_complete: bool = True) -> tuple[PreparedData, RunResult]:
        prepared = self.prepare(strategy, instruments, start, end, require_complete)
        return prepared, self.execute(strategy, prepared, self.generate_signals(strategy, prepared))

    # ----------------------------------------------------------- report
    def build_meta(self, strategy: Strategy, prepared: PreparedData, start: date, end: date) -> dict:
        s = self.settings
        return {
            "strategy_label": strategy.label, "strategy_key": strategy.key,
            "instruments": list(prepared.candles), "timeframe": prepared.timeframe,
            "start": start.isoformat(), "end": end.isoformat(), "min_trades_warning": s.min_trades_warning,
            "config": {
                "Strategy parameters": strategy.config.to_dict(),
                "Capital and execution": {"initial_capital": s.initial_capital, "slippage_pct": s.slippage_pct,
                                          "tick_size": s.tick_size, "cost_multiplier": s.cost_multiplier},
                "Intraday cost rates": s.intraday_costs, "Delivery cost rates": s.delivery_costs,
            },
            "assumptions": ASSUMPTIONS, "data_info": prepared.data_info, "extra_warnings": prepared.warnings,
        }

    def write_report(self, strategy: Strategy, prepared: PreparedData, run: RunResult) -> Path:
        return write_run_report(run.result, self.build_meta(strategy, prepared, run.start, run.end),
                                self.report_dir)
