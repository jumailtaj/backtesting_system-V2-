from datetime import date

import numpy as np
import pandas as pd
import pytest

from config.settings import IST
from src.data.resample import resample_intraday
from src.data.validation import validate_candles
from tests.helpers import minute_day, ts, weekdays


def test_5min_aggregation_values():
    day = date(2026, 1, 5)
    closes = np.arange(375, dtype=float) + 100
    df = minute_day(day, closes)
    out, info = resample_intraday(df, 5)
    assert len(out) == 75 and info["incomplete_dropped"] == 0
    first = df.iloc[:5]
    r = out.iloc[0]
    assert r.timestamp == ts(day, 9, 15)
    assert r.open == first.open.iloc[0] and r.close == first.close.iloc[-1]
    assert r.high == first.high.max() and r.low == first.low.min() and r.volume == first.volume.sum()
    assert out.iloc[-1].timestamp == ts(day, 15, 25)


def test_missing_minute_drops_bucket_not_filled():
    df = minute_day(date(2026, 1, 5))
    df = df.drop(index=7).reset_index(drop=True)          # 09:22 missing -> 09:20 bucket incomplete
    out, info = resample_intraday(df, 5)
    assert len(out) == 74 and info["incomplete_dropped"] == 1
    assert ts(date(2026, 1, 5), 9, 20) not in set(out.timestamp)


@pytest.mark.parametrize("minutes,expected", [(10, 38), (30, 13), (15, 25), (3, 125)])
def test_buckets_respect_session_end(minutes, expected):
    out, info = resample_intraday(minute_day(date(2026, 1, 5)), minutes)
    assert len(out) == expected and info["incomplete_dropped"] == 0
    assert out.timestamp.dt.time.max() < pd.Timestamp("15:30").time()


def test_never_combines_sessions():
    d1, d2 = weekdays(date(2026, 1, 5), 2)
    df = pd.concat([minute_day(d1), minute_day(d2)], ignore_index=True)
    out, _ = resample_intraday(df, 30)
    assert len(out) == 26
    per_day = out.groupby(out.timestamp.dt.date).size()
    assert (per_day == 13).all()


def test_out_of_session_minutes_dropped():
    df = minute_day(date(2026, 1, 5))
    extra = df.iloc[[0]].copy()
    extra["timestamp"] = ts(date(2026, 1, 5), 18, 0)
    out, info = resample_intraday(pd.concat([df, extra], ignore_index=True), 5)
    assert info["out_of_session_dropped"] == 1 and len(out) == 75


def test_validation_rejects_bad_rows_and_reports():
    df = minute_day(date(2026, 1, 5)).iloc[:10].copy()
    df.loc[1, "high"] = df.loc[1, "low"] - 1          # ohlc inconsistent
    df.loc[2, "volume"] = -5                           # negative volume
    df.loc[3, "close"] = np.nan                        # missing value
    df.loc[4, "open"] = 0                              # non positive
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)   # duplicate timestamp
    df = df.iloc[::-1].reset_index(drop=True)          # unordered
    clean, rep = validate_candles(df, "minute")
    assert rep.invalid == {"missing_values": 1, "non_positive_price": 1, "ohlc_inconsistent": 1,
                           "negative_volume": 1}
    assert rep.duplicates_removed == 1 and len(clean) == 6
    assert clean.timestamp.is_monotonic_increasing and any("sorted" in w for w in rep.warnings)


def test_validation_missing_columns_raises():
    with pytest.raises(ValueError):
        validate_candles(pd.DataFrame({"timestamp": []}), "minute")


def test_validation_out_of_session_minute():
    df = minute_day(date(2026, 1, 5)).iloc[:3].copy()
    df.loc[0, "timestamp"] = ts(date(2026, 1, 5), 9, 0)
    clean, rep = validate_candles(df, "minute")
    assert rep.invalid.get("out_of_session") == 1 and len(clean) == 2
