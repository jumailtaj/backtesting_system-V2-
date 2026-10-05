"""B. Intraday Opening Range Breakout (long; optional mirrored shorts)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config.settings import minutes_of_day
from src.data.manager import timeframe_minutes
from src.indicators.core import adx, relative_volume_tod, rsi, vwap
from src.strategies.base import Strategy, StrategyConfig, make_signals


@dataclass
class OrbConfig(StrategyConfig):
    timeframe: str = "5m"
    risk_per_trade_pct: float = 0.25
    max_trades_per_instrument_per_day: Optional[int] = 1
    daily_loss_limit_pct: Optional[float] = 0.75
    opening_range_start: str = "09:15"
    opening_range_end: str = "09:30"
    entry_start: str = "09:30"
    entry_end: str = "11:30"
    breakout_buffer_pct: float = 0.05
    min_range_width_pct: float = 0.20     # (high-low) / midpoint of the opening range, in %
    max_range_width_pct: float = 1.00
    reward_to_risk: float = 2.0
    allow_repeat_breakouts: bool = False  # True: re-arm after price closes back inside the range
    vwap_filter: bool = True
    rvol_filter: bool = True
    rvol_min: float = 1.2
    rvol_lookback: int = 10
    rsi_filter: bool = False
    rsi_period: int = 14
    rsi_long_min: float = 50.0
    rsi_short_max: float = 50.0
    adx_filter: bool = False
    adx_period: int = 14
    adx_min: float = 20.0

    def validate(self) -> None:
        super().validate()
        if self.min_range_width_pct > self.max_range_width_pct:
            raise ValueError("min_range_width_pct cannot exceed max_range_width_pct.")
        if minutes_of_day(self.opening_range_end) <= minutes_of_day(self.opening_range_start):
            raise ValueError("opening_range_end must be after opening_range_start.")


class OpeningRangeBreakout(Strategy):
    key = "orb"
    label = "Intraday Opening Range Breakout"
    ConfigClass = OrbConfig
    sensitivity_grid = {"breakout_buffer_pct": [0.0, 0.05, 0.10], "rvol_min": [1.0, 1.2, 1.5],
                        "reward_to_risk": [1.5, 2.0, 3.0]}

    def warmup_bars(self) -> int:
        return self.config.rvol_lookback * 75 if self.config.rvol_filter else 30

    def generate_signals(self, candles):
        c = self.config
        tf = timeframe_minutes(c.timeframe)
        sig = make_signals(candles.index)
        if candles.empty:
            return sig
        ts = candles["timestamp"]
        tmin = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
        day = ts.dt.normalize().to_numpy()
        close = candles["close"].to_numpy(float)
        high = candles["high"].to_numpy(float)
        low = candles["low"].to_numpy(float)
        v = vwap(candles).to_numpy(float) if c.vwap_filter else None
        rv = relative_volume_tod(candles, c.rvol_lookback).to_numpy(float) if c.rvol_filter else None
        rs = rsi(candles["close"], c.rsi_period).to_numpy(float) if c.rsi_filter else None
        ad = adx(candles["high"], candles["low"], candles["close"], c.adx_period).to_numpy(float) \
            if c.adx_filter else None

        or_start, or_end = minutes_of_day(c.opening_range_start), minutes_of_day(c.opening_range_end)
        e_start, e_end = minutes_of_day(c.entry_start), minutes_of_day(c.entry_end)
        expected_or_bars = max((or_end - or_start) // tf, 1)
        buf = c.breakout_buffer_pct / 100.0

        direction = np.zeros(len(candles), dtype=np.int8)
        stop = np.full(len(candles), np.nan)
        reason = np.full(len(candles), "", dtype=object)

        # session boundaries (candles are sorted)
        boundaries = np.flatnonzero(np.r_[True, day[1:] != day[:-1], True])
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            idx = np.arange(s, e)
            in_or = (tmin[idx] >= or_start) & (tmin[idx] < or_end)
            if in_or.sum() < expected_or_bars:
                continue                       # opening range incomplete/missing -> skip the day
            or_idx = idx[in_or]
            or_hi, or_lo = high[or_idx].max(), low[or_idx].min()
            width_pct = (or_hi - or_lo) / ((or_hi + or_lo) / 2) * 100
            if not (c.min_range_width_pct <= width_pct <= c.max_range_width_pct):
                continue
            long_trigger, short_trigger = or_hi * (1 + buf), or_lo * (1 - buf)
            armed_long = armed_short = True
            for i in idx[tmin[idx] >= or_end]:        # only candles AFTER the range has completed
                entry_min = tmin[i] + tf
                if entry_min > e_end:
                    break
                if close[i] <= long_trigger and c.allow_repeat_breakouts:
                    armed_long = True
                if close[i] >= short_trigger and c.allow_repeat_breakouts:
                    armed_short = True
                if entry_min < e_start:
                    continue
                if armed_long and close[i] > long_trigger and self._filters(i, 1, close, v, rv, rs, ad):
                    direction[i], stop[i], reason[i] = 1, or_lo, "ORB breakout above range high"
                    armed_long = False
                elif c.allow_short and armed_short and close[i] < short_trigger \
                        and self._filters(i, -1, close, v, rv, rs, ad):
                    direction[i], stop[i], reason[i] = -1, or_hi, "ORB breakdown below range low"
                    armed_short = False
        sig["direction"] = direction
        sig["stop_price"] = stop
        sig["reason"] = reason
        sig.loc[direction != 0, "target_rr"] = c.reward_to_risk
        return sig

    def _filters(self, i, side, close, v, rv, rs, ad) -> bool:
        c = self.config
        if c.vwap_filter and not (close[i] > v[i] if side == 1 else close[i] < v[i]):
            return False          # NaN comparisons are False -> filter fails
        if c.rvol_filter and not rv[i] >= c.rvol_min:
            return False
        if c.rsi_filter and not (rs[i] > c.rsi_long_min if side == 1 else rs[i] < c.rsi_short_max):
            return False
        if c.adx_filter and not ad[i] >= c.adx_min:
            return False
        return True
