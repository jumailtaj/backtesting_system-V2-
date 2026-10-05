"""Shared backtesting engine used by ALL strategies.

Execution rules (also written into every report's Configuration sheet)
---------------------------------------------------------------------
1.  A signal is evaluated when its candle has CLOSED; the entry fills at the
    OPEN of the next candle (same session for intraday strategies).
2.  Market fills (entries, stop exits, forced exits) suffer adverse slippage
    (``slippage_pct``) and are rounded adversely to the tick size.  Target exits
    are limit orders: filled at the target price with no slippage.  A target is
    assumed filled when the candle's high/low merely touches it (optimistic).
3.  Stops/targets are checked on every candle from the entry candle onward.
    If one candle touches BOTH stop and target the stop is assumed hit first.
4.  Gaps: if a candle opens beyond the stop the exit is at the OPEN (worse than
    the stop); if it opens beyond the target the exit is at the open.
5.  Intraday positions are force-closed at the open of the first candle at or
    after ``force_exit_time``; no entries at/after that time; any leftover
    position closes at the session's last close.  Swing positions may be held
    overnight.  Anything still open when the backtest ends is closed at the last
    close ("end_of_backtest").
6.  Costs are charged on both legs (entry + exit).  Equity = initial capital +
    realised net P&L + open P&L (marked at candle closes, net of entry costs).
7.  One position per instrument.  Entries on the same candle across instruments
    are processed in alphabetical order.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Optional

import numpy as np
import pandas as pd

from src.engine.costs import CostModel
from src.risk.manager import PortfolioSnapshot, RiskConfig, RiskManager

REJ_NO_NEXT = "no_next_candle_in_session"
REJ_AFTER_FORCE = "entry_at_or_after_force_exit_time"
REJ_OPEN_POS = "position_already_open"
REJ_SHORT_OFF = "short_selling_disabled"
REJ_RR = "reward_to_risk_below_minimum"
REJ_TARGET = "invalid_target"


@dataclass
class EngineConfig:
    initial_capital: float
    slippage_pct: float
    tick_size: float
    costs: CostModel
    intraday: bool
    force_exit: time
    allow_short: bool
    risk: RiskConfig
    max_holding_days: Optional[int] = None
    sector_map: dict = field(default_factory=dict)
    strategy_name: str = ""


@dataclass
class _Position:
    instrument: str
    direction: int
    qty: int
    entry_price: float
    stop: float
    target: float
    entry_time: pd.Timestamp
    signal_time: pd.Timestamp
    entry_costs: float
    risk_amount: float
    setup: str
    last_close: float


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    rejections: pd.DataFrame
    warnings: list
    signals_generated: int
    initial_capital: float


class _Series:
    """Arrays for one instrument inside the trading window."""

    def __init__(self, df: pd.DataFrame, sg: pd.DataFrame) -> None:
        self.ts = list(df["timestamp"])
        self.date = [t.date() for t in self.ts]
        self.o, self.h, self.l, self.c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.dir = sg["direction"].to_numpy(int)
        self.stop = sg["stop_price"].to_numpy(float)
        self.rr = sg["target_rr"].to_numpy(float)
        self.tgt = sg["target_price"].to_numpy(float)
        self.min_rr = sg["min_rr"].to_numpy(float)
        self.reason = sg["reason"].to_numpy(object)
        n = len(self.ts)
        self.last_of_session = np.array([i == n - 1 or self.date[i + 1] != self.date[i] for i in range(n)])
        self.index = {t: i for i, t in enumerate(self.ts)}


class BacktestEngine:
    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self.risk = RiskManager(cfg.risk, cfg.sector_map)

    # --------------------------------------------------------------- public
    def run(self, candles: dict, signals: dict, start: date, end: date) -> BacktestResult:
        cfg = self.cfg
        self.positions: dict[str, _Position] = {}
        self.pending: dict[str, dict] = {}
        self.trades: list[dict] = []
        self.rejections: list[dict] = []
        self.warnings: list[str] = []
        self.realized = 0.0
        self.day_realized = 0.0
        self.day_start_equity = cfg.initial_capital
        self.cur_date: Optional[date] = None
        self.entries_today = 0
        self.trades_today: dict[str, int] = {}
        self.equity_rows: list[tuple] = []
        self.signal_count = 0
        self.series: dict[str, _Series] = {}

        for inst in sorted(candles):
            df, sg = candles[inst], signals[inst]
            if len(df) != len(sg):
                raise ValueError(f"{inst}: signals and candles have different lengths.")
            d = df["timestamp"].dt.date
            mask = ((d >= start) & (d <= end)).to_numpy()
            if not mask.any():
                self.warnings.append(f"{inst}: no candles inside {start}..{end}; instrument skipped.")
                continue
            self.series[inst] = _Series(df[mask].reset_index(drop=True), sg[mask].reset_index(drop=True))
        if cfg.risk.sector_limit and not cfg.sector_map:
            self.warnings.append("Sector limit configured but no sector data supplied (config/sectors.csv): "
                                 "the sector limit was NOT enforced.")

        timeline = sorted({t for s in self.series.values() for t in s.ts})
        for T in timeline:
            self._process_bar(T)
        self._close_remaining()
        return self._result()

    # ----------------------------------------------------------- bar loop
    def _process_bar(self, T: pd.Timestamp) -> None:
        cfg = self.cfg
        d = T.date()
        if d != self.cur_date:
            self.cur_date, self.entries_today, self.trades_today = d, 0, {}
            self.day_realized = 0.0
            self.day_start_equity = self._equity()
        present = [(inst, s, s.index[T]) for inst, s in self.series.items() if T in s.index]

        for inst, s, i in present:                                   # 1) events at the open
            pos = self.positions.get(inst)
            if pos is not None:
                self._open_events(pos, s, i, T)
        for inst, s, i in present:                                   # 2) entries at the open
            pend = self.pending.pop(inst, None)
            if pend is not None:
                self._try_enter(inst, s, i, T, pend)
        for inst, s, i in present:                                   # 3) stop / target inside the candle
            pos = self.positions.get(inst)
            if pos is not None:
                self._intrabar(pos, s, i, T)
        for inst, s, i in present:                                   # 4) end-of-session flatten
            pos = self.positions.get(inst)
            if pos is not None and cfg.intraday and s.last_of_session[i]:
                self._exit(pos, s.c[i], T, "end_of_session", market=True)
        for inst, s, i in present:                                   # 5) mark to market
            pos = self.positions.get(inst)
            if pos is not None:
                pos.last_close = s.c[i]
        self.equity_rows.append((T, self._equity()))
        for inst, s, i in present:                                   # 6) new signals -> pending
            if s.dir[i] == 0:
                continue
            self.signal_count += 1
            if cfg.intraday and s.last_of_session[i]:
                self._reject(T, inst, REJ_NO_NEXT)
                continue
            self.pending[inst] = {"i": i, "T": T, "dir": int(s.dir[i]), "stop": s.stop[i], "rr": s.rr[i],
                                  "tgt": s.tgt[i], "min_rr": s.min_rr[i], "reason": s.reason[i]}

    # ------------------------------------------------------------ helpers
    def _tick(self, price: float, mode: str) -> float:
        t = self.cfg.tick_size
        if not t or t <= 0:
            return price
        q = price / t
        q = {"up": math.ceil(q - 1e-9), "down": math.floor(q + 1e-9), "nearest": round(q)}[mode]
        return round(q * t, 6)

    def _fill(self, price: float, buying: bool) -> float:
        slip = self.cfg.slippage_pct / 100
        return self._tick(price * (1 + slip) if buying else price * (1 - slip), "up" if buying else "down")

    def _equity(self) -> float:
        eq = self.cfg.initial_capital + self.realized
        for pos in self.positions.values():
            eq += pos.direction * (pos.last_close - pos.entry_price) * pos.qty - pos.entry_costs
        return eq

    def _snapshot(self, inst: str) -> PortfolioSnapshot:
        sectors: dict[str, int] = {}
        for p in self.positions.values():
            sec = self.cfg.sector_map.get(p.instrument)
            if sec:
                sectors[sec] = sectors.get(sec, 0) + 1
        return PortfolioSnapshot(
            equity=self._equity(),
            allocated_notional=sum(p.qty * p.entry_price for p in self.positions.values()),
            open_risk=sum(p.risk_amount for p in self.positions.values()),
            open_positions=len(self.positions), sector_counts=sectors,
            trades_today_instrument=self.trades_today.get(inst, 0), entries_today_total=self.entries_today,
            day_realized=self.day_realized, day_start_equity=self.day_start_equity)

    def _reject(self, T, inst: str, reason: str, detail: str = "") -> None:
        self.rejections.append({"time": T, "instrument": inst, "reason": reason, "detail": detail})

    # -------------------------------------------------------------- entries
    def _try_enter(self, inst: str, s: _Series, i: int, T: pd.Timestamp, p: dict) -> None:
        cfg = self.cfg
        if cfg.intraday and s.date[i] != s.date[p["i"]]:
            return self._reject(T, inst, REJ_NO_NEXT)
        if cfg.intraday and T.time() >= cfg.force_exit:
            return self._reject(T, inst, REJ_AFTER_FORCE)
        if inst in self.positions:
            return self._reject(T, inst, REJ_OPEN_POS)
        direction = p["dir"]
        if direction == -1 and not cfg.allow_short:
            return self._reject(T, inst, REJ_SHORT_OFF)
        entry = self._fill(s.o[i], buying=direction == 1)
        stop = self._tick(p["stop"], "nearest") if np.isfinite(p["stop"]) else float("nan")
        decision = self.risk.evaluate(self._snapshot(inst), inst, direction, entry, stop)
        if not decision.accepted:
            return self._reject(T, inst, decision.reason, decision.detail)
        risk_per_share = abs(entry - stop)
        if np.isfinite(p["tgt"]):
            target = self._tick(p["tgt"], "nearest")
            if (target - entry) * direction <= 0:
                return self._reject(T, inst, REJ_TARGET, f"target {target:.2f} vs entry {entry:.2f}")
        elif np.isfinite(p["rr"]):
            target = self._tick(entry + direction * p["rr"] * risk_per_share, "nearest")
        else:
            target = float("nan")
        if np.isfinite(p["min_rr"]) and np.isfinite(target):
            rr = abs(target - entry) / risk_per_share
            if rr < p["min_rr"] - 1e-9:
                return self._reject(T, inst, REJ_RR, f"{rr:.2f} < {p['min_rr']:.2f}")
        qty = decision.quantity
        costs = cfg.costs.leg(qty * entry, "buy" if direction == 1 else "sell", cfg.intraday)
        self.positions[inst] = _Position(inst, direction, qty, entry, stop, target, T, p["T"], costs,
                                         decision.risk_amount, str(p["reason"]), entry)
        self.entries_today += 1
        self.trades_today[inst] = self.trades_today.get(inst, 0) + 1

    # ---------------------------------------------------------------- exits
    def _open_events(self, pos: _Position, s: _Series, i: int, T: pd.Timestamp) -> None:
        cfg, o = self.cfg, s.o[i]
        if cfg.intraday and T.time() >= cfg.force_exit:
            return self._exit(pos, o, T, "forced_exit", market=True)
        if cfg.max_holding_days is not None and (T.date() - pos.entry_time.date()).days >= cfg.max_holding_days:
            return self._exit(pos, o, T, "max_holding_days", market=True)
        if pos.direction == 1:
            if o <= pos.stop:
                return self._exit(pos, o, T, "stop_loss_gap", market=True)
            if o >= pos.target:
                return self._exit(pos, o, T, "target_gap", market=False)
        else:
            if o >= pos.stop:
                return self._exit(pos, o, T, "stop_loss_gap", market=True)
            if o <= pos.target:
                return self._exit(pos, o, T, "target_gap", market=False)

    def _intrabar(self, pos: _Position, s: _Series, i: int, T: pd.Timestamp) -> None:
        h, l = s.h[i], s.l[i]
        if pos.direction == 1:
            stop_hit, tgt_hit = l <= pos.stop, h >= pos.target
        else:
            stop_hit, tgt_hit = h >= pos.stop, l <= pos.target
        if stop_hit:
            self._exit(pos, pos.stop, T, "stop_loss", market=True, ambiguous=bool(tgt_hit))
        elif tgt_hit:
            self._exit(pos, pos.target, T, "target", market=False)

    def _exit(self, pos: _Position, price: float, T: pd.Timestamp, reason: str, market: bool,
              ambiguous: bool = False) -> None:
        cfg = self.cfg
        selling = pos.direction == 1                 # closing a long = sell
        fill = self._fill(price, buying=not selling) if market else price
        exit_costs = cfg.costs.leg(pos.qty * fill, "sell" if selling else "buy", cfg.intraday)
        gross = pos.direction * (fill - pos.entry_price) * pos.qty
        costs = pos.entry_costs + exit_costs
        net = gross - costs
        self.realized += net
        self.day_realized += net
        risk = pos.risk_amount
        self.trades.append({
            "instrument": pos.instrument, "strategy": cfg.strategy_name,
            "direction": "LONG" if pos.direction == 1 else "SHORT",
            "signal_time": pos.signal_time, "entry_time": pos.entry_time, "exit_time": T,
            "quantity": pos.qty, "entry_price": pos.entry_price, "exit_price": fill,
            "stop_price": pos.stop, "target_price": pos.target, "gross_pnl": gross, "costs": costs,
            "net_pnl": net, "r_multiple": net / risk if risk > 0 else np.nan,
            "holding_time": T - pos.entry_time, "exit_reason": reason, "setup": pos.setup,
            "stop_and_target_same_candle": ambiguous,
        })
        del self.positions[pos.instrument]

    def _close_remaining(self) -> None:
        for inst, pos in list(self.positions.items()):
            s = self.series[inst]
            self._exit(pos, pos.last_close, s.ts[-1], "end_of_backtest", market=True)
            self.warnings.append(f"{inst}: position still open at the end of the data was closed at the last close.")
        if self.equity_rows:       # reflect the final closing costs in the last equity point
            self.equity_rows[-1] = (self.equity_rows[-1][0], self.cfg.initial_capital + self.realized)

    # --------------------------------------------------------------- result
    def _result(self) -> BacktestResult:
        trade_cols = ["instrument", "strategy", "direction", "signal_time", "entry_time", "exit_time", "quantity",
                      "entry_price", "exit_price", "stop_price", "target_price", "gross_pnl", "costs", "net_pnl",
                      "r_multiple", "holding_time", "exit_reason", "setup", "stop_and_target_same_candle"]
        trades = pd.DataFrame(self.trades, columns=trade_cols)
        equity = pd.DataFrame(self.equity_rows, columns=["timestamp", "equity"])
        if not equity.empty:
            peak = np.maximum(equity["equity"].cummax(), self.cfg.initial_capital)
            equity["drawdown"] = equity["equity"] - peak
            equity["drawdown_pct"] = equity["drawdown"] / peak * 100
        else:
            equity["drawdown"] = []
            equity["drawdown_pct"] = []
        rej = pd.DataFrame(self.rejections, columns=["time", "instrument", "reason", "detail"])
        return BacktestResult(trades, equity, rej, self.warnings, self.signal_count, self.cfg.initial_capital)
