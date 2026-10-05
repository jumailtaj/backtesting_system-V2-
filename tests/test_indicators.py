import numpy as np
import pandas as pd

from src.indicators import core as ind
from tests.helpers import bars, daily, minute_day
from datetime import date


def test_ema_matches_manual_recursion_and_warmup_nan():
    s = pd.Series([10, 11, 12, 11, 13, 14, 13, 15.0])
    out = ind.ema(s, 3)
    alpha = 2 / 4
    manual = [s[0]]
    for v in s[1:]:
        manual.append(alpha * v + (1 - alpha) * manual[-1])
    assert out.iloc[:2].isna().all() and not out.iloc[2:].isna().any()
    np.testing.assert_allclose(out.iloc[2:], manual[2:])


def test_vwap_resets_each_session():
    d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
    df = pd.concat([minute_day(d1, base=100), minute_day(d2, base=200)], ignore_index=True)
    v = ind.vwap(df)
    assert abs(v.iloc[375] - (df.high.iloc[375] + df.low.iloc[375] + df.close.iloc[375]) / 3) < 1e-9
    assert 199 < v.iloc[400] < 201


def test_rsi_extremes_and_nan_warmup():
    up = pd.Series(np.arange(1, 40, dtype=float))
    r = ind.rsi(up, 14)
    assert r.iloc[:14].isna().all() and r.iloc[-1] == 100.0
    down = ind.rsi(-up + 100, 14)
    assert down.iloc[-1] < 1e-9
    assert ind.rsi(pd.Series([5.0] * 30), 14).isna().all()      # flat -> undefined, not 0 or 50


def test_atr_constant_range():
    n = 40
    h, l, c = pd.Series([11.0] * n), pd.Series([9.0] * n), pd.Series([10.0] * n)
    a = ind.atr(h, l, c, 14)
    assert a.iloc[:13].isna().all() and abs(a.iloc[-1] - 2.0) < 1e-9


def test_adx_trending_vs_flat():
    n = 120
    trend = pd.Series(np.arange(n, dtype=float))
    adx_t = ind.adx(trend + 1, trend - 1, trend, 14)
    assert adx_t.iloc[:20].isna().all() and adx_t.iloc[-1] > 60
    rng = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    flat = pd.Series(100 + rng)
    assert ind.adx(flat + 0.5, flat - 0.5, flat, 14).iloc[-1] < 25


def test_relative_volume_excludes_current_and_nan_warmup():
    vol = pd.Series([100.0] * 20 + [300.0])
    r = ind.relative_volume_daily(vol, 20)
    assert r.iloc[:20].isna().all() and r.iloc[20] == 3.0


def test_relative_volume_time_of_day():
    days = [date(2026, 1, 5 + i) for i in range(5)]
    frames = []
    for i, d in enumerate(days):
        f = minute_day(d, volume=100.0 * (i + 1))
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    r = ind.relative_volume_tod(df, 3)
    last_day = r.iloc[-375:]
    assert abs(last_day.iloc[10] - 500 / ((100 + 200 + 300 + 0) / 3 if False else (200 + 300 + 400) / 3)) < 1e-9
    assert r.iloc[:3 * 375].isna().all()


def test_previous_high_excludes_current_candle():
    h = pd.Series([1, 2, 3, 4, 100.0])
    assert ind.previous_high(h, 3).iloc[4] == 4          # not 100
    assert np.isnan(ind.previous_high(h, 3).iloc[2])


def test_pivots_are_available_only_after_right_candles():
    low = np.array([5, 4, 3, 2, 1, 2, 3, 4, 5, 6.0])
    high = low + 1
    ph, pl = ind.find_pivots(high, low, 3, 3)
    assert pl.nonzero()[0].tolist() == [4]
    assert ind.pivot_available_index(4, 3) == 7
    # truncating the series before the right candles complete must not create the pivot
    ph2, pl2 = ind.find_pivots(high[:7], low[:7], 3, 3)
    assert not pl2.any()


def test_equal_lows_give_single_pivot():
    low = np.array([5, 4, 3, 1, 1, 3, 4, 5.0])
    _, pl = ind.find_pivots(low + 1, low, 3, 3)
    assert pl.sum() == 1


def test_zone_clustering():
    zones = ind.build_zones([100.0, 100.4, 110.0], width=1.0, min_touches=2)
    assert len(zones) == 1 and zones[0].touches == 2 and abs(zones[0].center - 100.2) < 1e-9
    assert ind.build_zones([100.0], 1.0, 2) == [] and ind.build_zones([100, 101], 0, 1) == []
