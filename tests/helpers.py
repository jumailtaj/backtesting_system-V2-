"""Deterministic synthetic OHLCV builders (no network, no randomness)."""
from __future__ import annotations

from datetime import date, time, timedelta

import numpy as np
import pandas as pd

from config.settings import IST, AppSettings
from src.engine.backtester import EngineConfig
from src.engine.costs import CostModel
from src.risk.manager import RiskConfig
from src.strategies.base import make_signals

ZERO_COST = {"brokerage_pct": 0, "brokerage_cap": 0, "stt_buy_pct": 0, "stt_sell_pct": 0, "exchange_txn_pct": 0,
             "sebi_pct": 0, "stamp_buy_pct": 0, "gst_pct": 0, "dp_charge_per_sell": 0}


def ts(day: date, hh: int, mm: int) -> pd.Timestamp:
    return pd.Timestamp(day.year, day.month, day.day, hh, mm, tz=IST)


def minute_day(day: date, closes=None, base=100.0, volume=1000.0) -> pd.DataFrame:
    """375 one-minute candles for one session. ``closes``: array of 375 closes (default flat+tiny wiggle)."""
    n = 375
    if closes is None:
        closes = base + 0.01 * (np.arange(n) % 3)
    closes = np.asarray(closes, float)
    opens = np.r_[closes[0], closes[:-1]]
    high = np.maximum(opens, closes) + 0.05
    low = np.minimum(opens, closes) - 0.05
    stamps = [ts(day, 9, 15) + pd.Timedelta(minutes=i) for i in range(n)]
    return pd.DataFrame({"timestamp": stamps, "open": opens, "high": high, "low": low, "close": closes,
                         "volume": np.full(n, volume)})


def weekdays(start: date, count: int):
    out, d = [], start
    while len(out) < count:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def bars(rows, day=date(2026, 1, 5), start=(9, 15), step=5) -> pd.DataFrame:
    """rows: list of (open, high, low, close[, volume]) -> consecutive intraday candles."""
    stamps, data = [], []
    t0 = ts(day, *start)
    for i, r in enumerate(rows):
        v = r[4] if len(r) > 4 else 1000.0
        stamps.append(t0 + pd.Timedelta(minutes=i * step))
        data.append((r[0], r[1], r[2], r[3], v))
    df = pd.DataFrame(data, columns=["open", "high", "low", "close", "volume"])
    df.insert(0, "timestamp", stamps)
    return df


def daily(rows, start=date(2026, 1, 1)) -> pd.DataFrame:
    days = weekdays(start, len(rows))
    df = pd.DataFrame([(r[0], r[1], r[2], r[3], r[4] if len(r) > 4 else 1000.0) for r in rows],
                      columns=["open", "high", "low", "close", "volume"])
    df.insert(0, "timestamp", [pd.Timestamp(d.year, d.month, d.day, tz=IST) for d in days])
    return df


def engine_cfg(intraday=True, capital=100000.0, risk_pct=1.0, slippage=0.0, tick=0.0, allow_short=False,
               force="15:15", **risk_kw) -> EngineConfig:
    risk = RiskConfig(risk_per_trade_pct=risk_pct, min_stop_distance_pct=0.0, **risk_kw)
    h, m = force.split(":")
    return EngineConfig(initial_capital=capital, slippage_pct=slippage, tick_size=tick,
                        costs=CostModel(ZERO_COST, ZERO_COST), intraday=intraday, force_exit=time(int(h), int(m)),
                        allow_short=allow_short, risk=risk, strategy_name="test")


def signal_at(df, index, direction=1, stop=None, rr=2.0, target_price=np.nan, min_rr=np.nan):
    sg = make_signals(df.index)
    sg.loc[index, "direction"] = direction
    sg.loc[index, "stop_price"] = stop
    sg.loc[index, "target_rr"] = rr if not np.isfinite(target_price) else np.nan
    sg.loc[index, "target_price"] = target_price
    sg.loc[index, "min_rr"] = min_rr
    sg.loc[index, "reason"] = "test"
    return sg


def settings_for(tmp_path) -> AppSettings:
    return AppSettings(start_date="2026-01-05", end_date="2026-01-30")
