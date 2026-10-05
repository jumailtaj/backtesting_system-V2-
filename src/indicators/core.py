"""Indicator library.  Every function uses ONLY current and past candles.

Conventions
-----------
* Unavailable values (warm-up) are NaN -- never 0.  Any comparison with NaN is
  False, so a NaN indicator can never enable a trade.
* ``ema``  : span=N, ``adjust=False`` (recursive, seeded with the first value),
             values withheld until N observations exist.  Continuous across sessions.
* ``rsi/atr/adx`` : Wilder smoothing (alpha = 1/N), values withheld until N observations.
* ``vwap`` : session VWAP using typical price (H+L+C)/3, reset every trading day.
* ``relative_volume_tod``  : volume / mean(volume of the SAME time-of-day slot over the previous N sessions).
* ``relative_volume_daily``: volume / mean(volume of the previous N sessions) (current day excluded).
* Swing pivots are only *available* ``right`` candles after they occur.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    day = df["timestamp"].dt.normalize()
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_pv = (typical * df["volume"]).groupby(day).cumsum()
    cum_v = df["volume"].groupby(day).cumsum()
    return cum_pv / cum_v.where(cum_v > 0)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.where(avg_loss > 0)
    out = 100 - 100 / (1 + rs)
    out = out.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)   # only gains -> 100
    return out                                                     # flat price -> NaN (undefined)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index).where(up.notna())
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index).where(down.notna())
    wilder = dict(alpha=1 / period, adjust=False, min_periods=period)
    atr_ = true_range(high, low, close).ewm(**wilder).mean()
    plus_di = 100 * plus_dm.ewm(**wilder).mean() / atr_
    minus_di = 100 * minus_dm.ewm(**wilder).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).where((plus_di + minus_di) > 0)
    return dx.ewm(**wilder).mean()


def relative_volume_tod(df: pd.DataFrame, lookback_sessions: int = 10) -> pd.Series:
    slot = df["timestamp"].dt.strftime("%H:%M")
    vol = df["volume"].astype(float)
    base = vol.groupby(slot).transform(
        lambda s: s.shift(1).rolling(lookback_sessions, min_periods=lookback_sessions).mean())
    return vol / base.where(base > 0)


def relative_volume_daily(volume: pd.Series, period: int = 20) -> pd.Series:
    base = volume.shift(1).rolling(period, min_periods=period).mean()
    return volume / base.where(base > 0)


def previous_high(high: pd.Series, lookback: int) -> pd.Series:
    """Highest high of the previous ``lookback`` candles, EXCLUDING the current one."""
    return high.shift(1).rolling(lookback, min_periods=lookback).max()


def previous_low(low: pd.Series, lookback: int) -> pd.Series:
    return low.shift(1).rolling(lookback, min_periods=lookback).min()


def rolling_low(low: pd.Series, n: int, group: pd.Series | None = None) -> pd.Series:
    """Lowest low of the last ``n`` candles INCLUDING the current (completed) one."""
    if group is None:
        return low.rolling(n, min_periods=n).min()
    return low.groupby(group).transform(lambda s: s.rolling(n, min_periods=n).min())


def rolling_high(high: pd.Series, n: int, group: pd.Series | None = None) -> pd.Series:
    if group is None:
        return high.rolling(n, min_periods=n).max()
    return high.groupby(group).transform(lambda s: s.rolling(n, min_periods=n).max())


def find_pivots(high: np.ndarray, low: np.ndarray, left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean arrays marking pivot highs / lows by the candle's OWN index ``i``.

    Pivot low at i:  low[i] < every low in [i-left, i-1]  AND  low[i] <= every low in [i+1, i+right].
    (Strict on the left, non-strict on the right => exactly one pivot per run of equal lows.)
    A pivot at ``i`` only becomes usable at candle ``i + right`` (see ``pivot_available_index``);
    callers MUST NOT use it earlier.
    """
    n = len(high)
    ph = np.zeros(n, dtype=bool)
    pl = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        lw = low[i - left:i]
        rw = low[i + 1:i + right + 1]
        if low[i] < lw.min() and low[i] <= rw.min():
            pl[i] = True
        hl = high[i - left:i]
        hr = high[i + 1:i + right + 1]
        if high[i] > hl.max() and high[i] >= hr.max():
            ph[i] = True
    return ph, pl


def pivot_available_index(i: int, right: int) -> int:
    return i + right


@dataclass
class Zone:
    lower: float
    upper: float
    center: float
    touches: int


def build_zones(levels: list[float], width: float, min_touches: int) -> list[Zone]:
    """Cluster pivot prices into zones of total ``width``; keep zones with >= min_touches pivots."""
    if width <= 0 or not levels:
        return []
    clusters: list[list[float]] = []
    for level in sorted(levels):
        if clusters and level - clusters[-1][0] <= width:
            clusters[-1].append(level)
        else:
            clusters.append([level])
    zones = []
    for cl in clusters:
        if len(cl) >= min_touches:
            center = float(np.mean(cl))
            zones.append(Zone(center - width / 2, center + width / 2, center, len(cl)))
    return zones
