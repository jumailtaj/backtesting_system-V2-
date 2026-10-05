"""D. Swing support/resistance rejection on daily candles (long only)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.indicators.core import atr, build_zones, ema, find_pivots, relative_volume_daily
from src.strategies.base import Strategy, StrategyConfig, make_signals


@dataclass
class SwingSrConfig(StrategyConfig):
    timeframe: str = "1d"
    intraday: bool = False
    risk_per_trade_pct: float = 0.5
    max_simultaneous_positions: Optional[int] = 5
    max_portfolio_open_risk_pct: Optional[float] = 2.0
    max_capital_allocation_pct: float = 80.0
    pivot_left: int = 3
    pivot_right: int = 3                 # pivots are usable only after these candles have completed
    lookback_sessions: int = 60
    min_touches: int = 2
    zone_width_atr: float = 0.25         # zone width = this x ATR(atr_period)
    atr_period: int = 14
    max_penetration_zones: float = 1.0   # low may undercut the zone by at most this many zone-widths
    rvol_min: float = 1.2
    rvol_period: int = 20
    ema_period: int = 50
    stop_buffer_atr: float = 0.25        # stop = min(zone low, signal low) - buffer x ATR
    min_reward_to_risk: float = 2.0
    target_method: str = "fixed_rr"      # "fixed_rr" (target = min_reward_to_risk) or "resistance"

    def validate(self) -> None:
        super().validate()
        if self.allow_short:
            raise ValueError("Swing Support/Resistance does not support short selling.")
        if self.target_method not in ("fixed_rr", "resistance"):
            raise ValueError("target_method must be fixed_rr or resistance.")
        if self.pivot_left < 1 or self.pivot_right < 1 or self.min_touches < 1:
            raise ValueError("pivot_left, pivot_right and min_touches must be >= 1.")


class SwingSupportResistance(Strategy):
    key = "swing_sr"
    label = "Swing Support and Resistance"
    ConfigClass = SwingSrConfig
    sensitivity_grid = {"min_touches": [2, 3], "zone_width_atr": [0.15, 0.25, 0.40], "rvol_min": [1.0, 1.2, 1.5]}

    def warmup_bars(self) -> int:
        c = self.config
        return max(c.lookback_sessions + c.pivot_right, c.ema_period * 2)

    def generate_signals(self, candles):
        c = self.config
        sig = make_signals(candles.index)
        n = len(candles)
        if n == 0:
            return sig
        high, low, close = (candles[k].to_numpy(float) for k in ("high", "low", "close"))
        atr_ = atr(candles["high"], candles["low"], candles["close"], c.atr_period).to_numpy(float)
        ema_ = ema(candles["close"], c.ema_period).to_numpy(float)
        rvol = relative_volume_daily(candles["volume"], c.rvol_period).to_numpy(float)
        ph, pl = find_pivots(high, low, c.pivot_left, c.pivot_right)
        pivot_low_idx, pivot_high_idx = np.flatnonzero(pl), np.flatnonzero(ph)

        for t in range(n):
            if not (np.isfinite(atr_[t]) and np.isfinite(ema_[t]) and np.isfinite(rvol[t])):
                continue
            if rvol[t] < c.rvol_min or close[t] <= ema_[t]:
                continue
            # A pivot at index i is usable only at t >= i + pivot_right (right-side candles completed).
            usable = pivot_low_idx[(pivot_low_idx + c.pivot_right <= t) & (pivot_low_idx >= t - c.lookback_sessions + 1)]
            if len(usable) < c.min_touches:
                continue
            width = c.zone_width_atr * atr_[t]
            zones = build_zones(list(low[usable]), width, c.min_touches)
            midpoint = (high[t] + low[t]) / 2
            for zone in sorted(zones, key=lambda z: -z.upper):          # nearest zone below price first
                touched = low[t] <= zone.upper
                not_blown = low[t] >= zone.lower - c.max_penetration_zones * width
                rejected = close[t] > zone.upper and close[t] > midpoint
                if not (touched and not_blown and rejected):
                    continue
                stop = min(zone.lower, low[t]) - c.stop_buffer_atr * atr_[t]
                if not stop < close[t]:
                    continue
                target_price = np.nan
                if c.target_method == "resistance":
                    res_idx = pivot_high_idx[(pivot_high_idx + c.pivot_right <= t)
                                             & (pivot_high_idx >= t - c.lookback_sessions + 1)]
                    res_zones = [z for z in build_zones(list(high[res_idx]), width, c.min_touches)
                                 if z.lower > close[t]]
                    if not res_zones:
                        continue
                    target_price = min(z.lower for z in res_zones)
                sig.iat[t, sig.columns.get_loc("direction")] = 1
                sig.iat[t, sig.columns.get_loc("stop_price")] = stop
                if c.target_method == "fixed_rr":
                    sig.iat[t, sig.columns.get_loc("target_rr")] = c.min_reward_to_risk
                else:
                    sig.iat[t, sig.columns.get_loc("target_price")] = target_price
                sig.iat[t, sig.columns.get_loc("min_rr")] = c.min_reward_to_risk
                sig.iat[t, sig.columns.get_loc("reason")] = f"support rejection ({zone.touches} touches)"
                break
        return sig
