"""Risk manager: decides whether a trade is allowed and how many shares.

Sizing (spec):   risk budget = current equity x risk %
                 quantity    = floor(risk budget / |entry - stop|)
then the quantity is capped by available capital (no leverage assumed) and
the optional per-position value cap.  Every refusal returns a reason string.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# --- rejection reasons (strings appear in the Excel report) -------------------
DAILY_LOSS_LIMIT = "daily_loss_limit_reached"
MAX_TRADES_INSTRUMENT_DAY = "max_trades_per_instrument_per_day"
MAX_ENTRIES_DAY = "max_new_entries_per_day"
MAX_POSITIONS = "max_simultaneous_positions"
SECTOR_LIMIT = "sector_limit"
INVALID_STOP = "invalid_stop_distance"
STOP_TOO_TIGHT = "stop_distance_below_minimum"
ZERO_QTY = "zero_quantity"
NO_CAPITAL = "insufficient_capital_allocation"
OPEN_RISK = "max_portfolio_open_risk"


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.25
    max_trades_per_instrument_per_day: Optional[int] = None
    max_simultaneous_positions: Optional[int] = None
    max_new_entries_per_day: Optional[int] = None
    max_portfolio_open_risk_pct: Optional[float] = None
    max_capital_allocation_pct: float = 100.0
    max_position_value_pct: Optional[float] = None
    daily_loss_limit_pct: Optional[float] = None
    sector_limit: Optional[int] = None
    min_stop_distance_pct: float = 0.05

    @classmethod
    def from_strategy_config(cls, cfg) -> "RiskConfig":
        return cls(**{k: getattr(cfg, k) for k in cls.__dataclass_fields__})


@dataclass
class PortfolioSnapshot:
    equity: float
    allocated_notional: float = 0.0
    open_risk: float = 0.0
    open_positions: int = 0
    sector_counts: dict = field(default_factory=dict)
    trades_today_instrument: int = 0
    entries_today_total: int = 0
    day_realized: float = 0.0
    day_start_equity: float = 0.0


@dataclass
class Decision:
    accepted: bool
    quantity: int = 0
    reason: str = ""
    detail: str = ""
    risk_amount: float = 0.0


class RiskManager:
    def __init__(self, cfg: RiskConfig, sector_map: Optional[dict] = None) -> None:
        self.cfg = cfg
        self.sector_map = sector_map or {}

    @property
    def sector_limit_active(self) -> bool:
        return bool(self.cfg.sector_limit and self.sector_map)

    def evaluate(self, snap: PortfolioSnapshot, instrument: str, direction: int,
                 entry_price: float, stop_price: float) -> Decision:
        c = self.cfg
        if c.daily_loss_limit_pct is not None and snap.day_start_equity > 0:
            limit = snap.day_start_equity * c.daily_loss_limit_pct / 100
            if snap.day_realized <= -limit:
                return Decision(False, reason=DAILY_LOSS_LIMIT,
                                detail=f"day P&L {snap.day_realized:.2f} <= -{limit:.2f}")
        if c.max_trades_per_instrument_per_day is not None and \
                snap.trades_today_instrument >= c.max_trades_per_instrument_per_day:
            return Decision(False, reason=MAX_TRADES_INSTRUMENT_DAY)
        if c.max_new_entries_per_day is not None and snap.entries_today_total >= c.max_new_entries_per_day:
            return Decision(False, reason=MAX_ENTRIES_DAY)
        if c.max_simultaneous_positions is not None and snap.open_positions >= c.max_simultaneous_positions:
            return Decision(False, reason=MAX_POSITIONS)
        if self.sector_limit_active:
            sector = self.sector_map.get(instrument)
            if sector and snap.sector_counts.get(sector, 0) >= c.sector_limit:
                return Decision(False, reason=SECTOR_LIMIT, detail=f"sector {sector}")

        # stop must be on the protective side of the entry and not absurdly tight
        distance = (entry_price - stop_price) * direction
        if not (math.isfinite(distance) and distance > 0):
            return Decision(False, reason=INVALID_STOP, detail=f"entry {entry_price:.2f}, stop {stop_price:.2f}")
        if distance / entry_price * 100 < c.min_stop_distance_pct:
            return Decision(False, reason=STOP_TOO_TIGHT, detail=f"{distance / entry_price * 100:.3f}%")

        budget = snap.equity * c.risk_per_trade_pct / 100
        qty = math.floor(budget / distance)
        allowed_notional = snap.equity * c.max_capital_allocation_pct / 100 - snap.allocated_notional
        qty = min(qty, math.floor(max(allowed_notional, 0) / entry_price))
        if c.max_position_value_pct is not None:
            qty = min(qty, math.floor(snap.equity * c.max_position_value_pct / 100 / entry_price))
        if qty < 1:
            reason = (ZERO_QTY if math.floor(budget / distance) < 1 else NO_CAPITAL)
            return Decision(False, reason=reason, detail=f"budget {budget:.2f}, stop distance {distance:.2f}")
        risk_amount = qty * distance
        if c.max_portfolio_open_risk_pct is not None:
            cap = snap.equity * c.max_portfolio_open_risk_pct / 100
            if snap.open_risk + risk_amount > cap + 1e-9:
                return Decision(False, reason=OPEN_RISK,
                                detail=f"open risk {snap.open_risk + risk_amount:.2f} > cap {cap:.2f}")
        return Decision(True, quantity=int(qty), risk_amount=float(risk_amount))
