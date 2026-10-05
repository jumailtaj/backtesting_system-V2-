"""C. Swing breakout on daily candles (long only)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.indicators.core import atr, ema, previous_high, relative_volume_daily, rolling_low
from src.strategies.base import Strategy, StrategyConfig, make_signals


@dataclass
class SwingBreakoutConfig(StrategyConfig):
    timeframe: str = "1d"
    intraday: bool = False
    risk_per_trade_pct: float = 0.5
    max_simultaneous_positions: Optional[int] = 5
    max_new_entries_per_day: Optional[int] = 3
    max_portfolio_open_risk_pct: Optional[float] = 2.0
    max_capital_allocation_pct: float = 80.0
    sector_limit: Optional[int] = 2
    breakout_lookback: int = 20          # previous completed sessions (current candle excluded)
    breakout_buffer_pct: float = 0.25
    max_extension_pct: float = 3.0       # close may be at most this % above the resistance level
    rvol_min: float = 1.5
    rvol_period: int = 20
    ema_period: int = 50
    reward_to_risk: float = 3.0
    stop_method: str = "lowest_low"      # "lowest_low" (N sessions incl. signal candle) or "atr"
    stop_lookback: int = 10
    atr_period: int = 14
    atr_stop_mult: float = 2.0

    def validate(self) -> None:
        super().validate()
        if self.allow_short:
            raise ValueError("Swing Breakout does not support short selling.")
        if self.stop_method not in ("lowest_low", "atr"):
            raise ValueError("stop_method must be lowest_low or atr.")
        if self.reward_to_risk <= 0:
            raise ValueError("reward_to_risk must be positive.")


class SwingBreakout(Strategy):
    key = "swing_breakout"
    label = "Swing Breakout"
    ConfigClass = SwingBreakoutConfig
    sensitivity_grid = {"breakout_lookback": [15, 20, 30], "rvol_min": [1.2, 1.5, 2.0],
                        "reward_to_risk": [2.0, 3.0, 4.0]}

    def warmup_bars(self) -> int:
        c = self.config
        return max(c.ema_period * 2, c.breakout_lookback + 1, c.rvol_period + 1)

    def generate_signals(self, candles):
        c = self.config
        sig = make_signals(candles.index)
        if candles.empty:
            return sig
        close = candles["close"]
        resistance = previous_high(candles["high"], c.breakout_lookback)      # excludes today's candle
        trigger = resistance * (1 + c.breakout_buffer_pct / 100)
        cond = (close > trigger) & (close <= resistance * (1 + c.max_extension_pct / 100))
        cond &= relative_volume_daily(candles["volume"], c.rvol_period) >= c.rvol_min
        cond &= close > ema(close, c.ema_period)
        if c.stop_method == "lowest_low":
            stop = rolling_low(candles["low"], c.stop_lookback)
        else:
            stop = close - c.atr_stop_mult * atr(candles["high"], candles["low"], close, c.atr_period)
        cond &= stop.notna() & (stop < close)
        sig.loc[cond, "direction"] = 1
        sig.loc[cond, "stop_price"] = stop[cond]
        sig.loc[cond, "target_rr"] = c.reward_to_risk
        sig.loc[cond, "reason"] = f"close above {c.breakout_lookback}-session high"
        return sig
