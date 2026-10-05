"""A. Intraday EMA crossover (long; optional mirrored shorts)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.data.manager import timeframe_minutes
from src.indicators.core import adx, atr, ema, relative_volume_tod, rolling_high, rolling_low, rsi, vwap
from src.strategies.base import Strategy, StrategyConfig, entry_window_mask, make_signals


@dataclass
class EmaCrossoverConfig(StrategyConfig):
    timeframe: str = "5m"
    risk_per_trade_pct: float = 0.25
    max_trades_per_instrument_per_day: Optional[int] = 2
    fast_period: int = 9
    slow_period: int = 21
    entry_start: str = "09:30"
    entry_end: str = "14:30"
    reward_to_risk: float = 2.0
    # stop: "lowest_low" (N completed candles of the same session, incl. the signal candle),
    #       "atr" (close -/+ atr_stop_mult * ATR) or "signal_candle" (low/high of the signal candle)
    stop_method: str = "lowest_low"
    stop_lookback: int = 5
    atr_period: int = 14
    atr_stop_mult: float = 1.5
    # filters
    vwap_filter: bool = True
    rsi_filter: bool = False
    rsi_period: int = 14
    rsi_long_min: float = 50.0
    rsi_short_max: float = 50.0
    adx_filter: bool = False
    adx_period: int = 14
    adx_min: float = 20.0
    volume_filter: bool = False
    rvol_lookback: int = 10
    rvol_min: float = 1.0
    atr_filter: bool = False
    atr_min_pct: float = 0.05     # ATR as % of close must be >= this
    atr_max_pct: float = 2.0      # ... and <= this
    trend_filter: bool = False
    trend_period: int = 50        # long only above this EMA, short only below

    def validate(self) -> None:
        super().validate()
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be smaller than slow_period.")
        if self.stop_method not in ("lowest_low", "atr", "signal_candle"):
            raise ValueError("stop_method must be lowest_low, atr or signal_candle.")
        if self.reward_to_risk <= 0:
            raise ValueError("reward_to_risk must be positive.")


class EmaCrossover(Strategy):
    key = "ema_crossover"
    label = "Intraday EMA Crossover"
    ConfigClass = EmaCrossoverConfig
    sensitivity_grid = {"fast_period": [5, 9, 12], "slow_period": [13, 21, 26], "reward_to_risk": [1.5, 2.0, 3.0]}

    def warmup_bars(self) -> int:
        c = self.config
        return max(c.slow_period * 3, c.trend_period if c.trend_filter else 0, 40)

    def generate_signals(self, candles):
        c = self.config
        tf = timeframe_minutes(c.timeframe)
        sig = make_signals(candles.index)
        if candles.empty:
            return sig
        close = candles["close"]
        day = candles["timestamp"].dt.normalize()
        fast, slow = ema(close, c.fast_period), ema(close, c.slow_period)
        same_session = day == day.shift(1)    # crossover must happen between candles of ONE session
        up = (fast.shift(1) <= slow.shift(1)) & (fast > slow) & same_session
        down = (fast.shift(1) >= slow.shift(1)) & (fast < slow) & same_session

        window = entry_window_mask(candles, tf, c.entry_start, c.entry_end)
        long_ok, short_ok = up & window, down & window & c.allow_short
        if c.vwap_filter:
            v = vwap(candles)
            long_ok &= close > v
            short_ok &= close < v
        if c.rsi_filter:
            r = rsi(close, c.rsi_period)
            long_ok &= r > c.rsi_long_min
            short_ok &= r < c.rsi_short_max
        if c.adx_filter:
            a = adx(candles["high"], candles["low"], close, c.adx_period)
            long_ok &= a >= c.adx_min
            short_ok &= a >= c.adx_min
        if c.volume_filter:
            rv = relative_volume_tod(candles, c.rvol_lookback)
            long_ok &= rv >= c.rvol_min
            short_ok &= rv >= c.rvol_min
        atr_series = atr(candles["high"], candles["low"], close, c.atr_period)
        if c.atr_filter:
            pct = atr_series / close * 100
            ok = (pct >= c.atr_min_pct) & (pct <= c.atr_max_pct)
            long_ok &= ok
            short_ok &= ok
        if c.trend_filter:
            t = ema(close, c.trend_period)
            long_ok &= close > t
            short_ok &= close < t

        if c.stop_method == "lowest_low":
            stop_long = rolling_low(candles["low"], c.stop_lookback, day)
            stop_short = rolling_high(candles["high"], c.stop_lookback, day)
        elif c.stop_method == "atr":
            stop_long, stop_short = close - c.atr_stop_mult * atr_series, close + c.atr_stop_mult * atr_series
        else:
            stop_long, stop_short = candles["low"], candles["high"]
        long_ok &= stop_long.notna() & (stop_long < close)
        short_ok &= stop_short.notna() & (stop_short > close)

        sig.loc[long_ok, "direction"] = 1
        sig.loc[long_ok, "stop_price"] = stop_long[long_ok]
        sig.loc[short_ok, "direction"] = -1
        sig.loc[short_ok, "stop_price"] = stop_short[short_ok]
        active = sig["direction"] != 0
        sig.loc[active, "target_rr"] = c.reward_to_risk
        sig.loc[active, "reason"] = f"EMA{c.fast_period}/EMA{c.slow_period} crossover"
        return sig
