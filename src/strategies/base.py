"""Strategy base classes.

A strategy ONLY decides *when* a trade idea exists.  It never sizes positions,
fills orders, or tracks P&L -- the shared engine does all of that.

Signal contract (``generate_signals``)
--------------------------------------
Input : candles DataFrame (timestamp [tz-aware IST], open, high, low, close, volume),
        sorted, default RangeIndex.  Warm-up candles are included.
Output: DataFrame with the SAME index and columns
        direction     1 = long entry, -1 = short entry, 0 = nothing
        stop_price    stop-loss price (set when direction != 0)
        target_rr     reward:risk multiple (target = entry +/- rr * |entry - stop|), or NaN
        target_price  absolute target price, or NaN (used instead of target_rr)
        min_rr        engine rejects the trade if the real reward:risk is lower, or NaN
        reason        short text describing the setup
The row for candle ``t`` may use data from candles <= t ONLY.  The engine
executes the signal at the OPEN of the next candle.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import STRATEGY_CONFIG_DIR, minutes_of_day, parse_hhmm
from src.data.manager import timeframe_minutes


@dataclass
class StrategyConfig:
    # ---- execution -------------------------------------------------------
    timeframe: str = "5m"                  # "5m", "15m", ... or "1d"
    intraday: bool = True                  # True: no overnight, forced exit; False: swing
    force_exit_time: str = "15:15"         # intraday only (HH:MM IST)
    allow_short: bool = False              # short selling is OFF unless you enable it
    # ---- risk ------------------------------------------------------------
    risk_per_trade_pct: float = 0.25       # % of current equity risked per trade
    max_trades_per_instrument_per_day: Optional[int] = None
    max_simultaneous_positions: Optional[int] = None
    max_new_entries_per_day: Optional[int] = None
    max_portfolio_open_risk_pct: Optional[float] = None
    max_capital_allocation_pct: float = 100.0   # no leverage: total notional <= this % of equity
    max_position_value_pct: Optional[float] = None
    daily_loss_limit_pct: Optional[float] = None
    sector_limit: Optional[int] = None
    min_stop_distance_pct: float = 0.05    # reject trades whose stop is closer than this % of entry
    max_holding_days: Optional[int] = None  # swing only: time-stop (calendar days)

    # ------------------------------------------------------------------
    def validate(self) -> None:
        timeframe_minutes(self.timeframe)
        parse_hhmm(self.force_exit_time)
        if self.risk_per_trade_pct <= 0 or self.risk_per_trade_pct > 5:
            raise ValueError("risk_per_trade_pct must be in (0, 5].")
        if not 0 < self.max_capital_allocation_pct <= 100:
            raise ValueError("max_capital_allocation_pct must be in (0, 100] (no leverage is assumed).")
        if self.min_stop_distance_pct < 0:
            raise ValueError("min_stop_distance_pct cannot be negative.")
        if self.intraday and self.timeframe.endswith("d"):
            raise ValueError("An intraday strategy cannot use a daily timeframe.")
        if not self.intraday and not self.timeframe.endswith("d"):
            raise ValueError("A swing strategy must use the daily timeframe (1d).")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        valid = {f.name: f for f in fields(cls)}
        unknown = set(data) - set(valid)
        if unknown:
            raise ValueError(f"Unknown parameter(s) {sorted(unknown)} for {cls.__name__}. "
                             f"Valid parameters: {sorted(valid)}")
        base = cls()
        values = base.to_dict()
        for key, raw in data.items():
            values[key] = _coerce(key, raw, getattr(base, key), "Optional" in str(valid[key].type))
        cfg = cls(**values)
        cfg.validate()
        return cfg


def _coerce(key: str, raw, default, optional: bool):
    if raw is None:
        if optional or default is None:
            return None
        raise ValueError(f"'{key}' cannot be null.")
    if isinstance(default, bool):
        if isinstance(raw, str):
            low = raw.strip().lower()
            if low in ("true", "yes", "1", "on"):
                return True
            if low in ("false", "no", "0", "off"):
                return False
        if isinstance(raw, bool):
            return raw
        raise ValueError(f"'{key}' must be true or false.")
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            f = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be a whole number.") from None
        if f != int(f):
            raise ValueError(f"'{key}' must be a whole number.")
        return int(f)
    if isinstance(default, float):
        try:
            return float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be a number.") from None
    if default is None:  # Optional numeric with default None
        try:
            return float(raw) if "." in str(raw) else int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be a number or null.") from None
    return str(raw)


def make_signals(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({
        "direction": np.zeros(len(index), dtype=np.int8),
        "stop_price": np.nan, "target_rr": np.nan, "target_price": np.nan, "min_rr": np.nan,
        "reason": "",
    }, index=index)


def entry_window_mask(candles: pd.DataFrame, tf_minutes: int, start: str, end: str) -> pd.Series:
    """True where the ENTRY time (signal candle close = next candle open) is inside [start, end]."""
    ts = candles["timestamp"]
    entry_min = ts.dt.hour * 60 + ts.dt.minute + tf_minutes
    return (entry_min >= minutes_of_day(start)) & (entry_min <= minutes_of_day(end))


class Strategy(ABC):
    key: str = ""
    label: str = ""
    ConfigClass = StrategyConfig
    # Small, bounded grid used by the parameter-sensitivity analysis: {param: [values...]}
    sensitivity_grid: dict = {}

    def __init__(self, config: Optional[StrategyConfig] = None) -> None:
        self.config = config or self.ConfigClass()
        self.config.validate()

    @abstractmethod
    def generate_signals(self, candles: pd.DataFrame) -> pd.DataFrame: ...

    def warmup_bars(self) -> int:
        """Approximate number of candles needed before indicators are valid."""
        return 50

    # ---- config files ---------------------------------------------------
    @classmethod
    def config_path(cls) -> Path:
        return STRATEGY_CONFIG_DIR / f"{cls.key}.json"

    @classmethod
    def load(cls, overrides: Optional[dict] = None, path: Optional[Path] = None) -> "Strategy":
        path = path or cls.config_path()
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        if overrides:
            data = {**data, **overrides}
        return cls(cls.ConfigClass.from_dict(data))

    def save_config(self, path: Optional[Path] = None) -> None:
        (path or self.config_path()).write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")

    def with_overrides(self, overrides: dict) -> "Strategy":
        return type(self)(self.config.from_dict({**self.config.to_dict(), **overrides}))
