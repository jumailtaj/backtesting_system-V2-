"""Build 3/5/10/15/30-minute candles from 1-minute candles.

Rules
-----
* Buckets are anchored at the session open (09:15 IST), never at the clock hour.
* Buckets never span two trading sessions.
* A candle is COMPLETE only if every 1-minute candle inside its window is
  present.  The final bucket of a session may be shorter than N minutes when N
  does not divide 375 (e.g. the 10-minute 15:25 candle covers 15:25-15:29); it is
  complete if all minutes up to the session close exist.
* Incomplete buckets are dropped and counted, never filled in.
* Candle timestamp = candle START time (same convention as Kite).  A candle
  "closes" at timestamp + N minutes.
"""
from __future__ import annotations

import pandas as pd

from config.settings import IST, SESSION_MINUTES
from src.data.validation import REQUIRED_COLUMNS, SESSION_START_MIN, empty_candles


def resample_intraday(minute: pd.DataFrame, minutes: int, require_full_bucket: bool = True
                      ) -> tuple[pd.DataFrame, dict]:
    info = {"minutes": minutes, "input_rows": len(minute), "buckets_total": 0, "incomplete_dropped": 0,
            "out_of_session_dropped": 0}
    if minute.empty:
        return empty_candles(), info
    df = minute[REQUIRED_COLUMNS].sort_values("timestamp")
    ts = df["timestamp"].dt.tz_convert(IST)
    rel = ts.dt.hour * 60 + ts.dt.minute - SESSION_START_MIN
    in_session = (rel >= 0) & (rel < SESSION_MINUTES)
    info["out_of_session_dropped"] = int((~in_session).sum())
    df, ts, rel = df[in_session], ts[in_session], rel[in_session]
    if df.empty:
        return empty_candles(), info
    if minutes == 1:
        info["buckets_total"] = len(df)
        return df.reset_index(drop=True), info

    day = ts.dt.normalize().dt.tz_localize(None)     # local wall-clock midnight (naive)
    bucket = rel // minutes
    grouped = df.assign(_day=day.values, _b=bucket.values).groupby(["_day", "_b"], sort=True)
    agg = grouped.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                      close=("close", "last"), volume=("volume", "sum"), n=("open", "size"))
    agg = agg.reset_index()
    info["buckets_total"] = len(agg)
    expected = (SESSION_MINUTES - agg["_b"] * minutes).clip(upper=minutes)
    complete = agg["n"] == expected
    if require_full_bucket:
        info["incomplete_dropped"] = int((~complete).sum())
        agg = agg[complete]
    agg["timestamp"] = agg["_day"] + pd.to_timedelta(SESSION_START_MIN + agg["_b"] * minutes, unit="m")
    out = agg[REQUIRED_COLUMNS].reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.tz_localize(IST)
    return out, info
