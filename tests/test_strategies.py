from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.indicators.core import ema
from src.strategies.base import StrategyConfig
from src.strategies.ema_crossover import EmaCrossover
from src.strategies.orb import OpeningRangeBreakout
from src.strategies.registry import REGISTRY, get_strategy
from src.strategies.swing_breakout import SwingBreakout
from src.strategies.swing_sr import SwingSupportResistance
from tests.helpers import bars, daily, weekdays


def path_bars(closes, day=date(2026, 1, 5), volume=1000.0):
    vols = np.broadcast_to(np.asarray(volume, float), (len(closes),))
    rows, prev = [], closes[0]
    for c, v in zip(closes, vols):
        rows.append((prev, max(prev, c) + 0.1, min(prev, c) - 0.1, c, v))
        prev = c
    return bars(rows, day=day)


# ------------------------------------------------------------------ EMA crossover
def test_ema_signals_only_on_actual_cross_not_while_above():
    closes = list(np.linspace(100, 90, 30)) + list(np.linspace(90, 110, 40))
    df = path_bars(closes)
    strat = EmaCrossover(EmaCrossover.ConfigClass(vwap_filter=False, entry_start="09:15", entry_end="15:00"))
    sg = strat.generate_signals(df)
    idx = sg.index[sg.direction == 1].tolist()
    assert len(idx) == 1                                           # one cross, not one per candle above
    i = idx[0]
    f, s = ema(df.close, 9), ema(df.close, 21)
    assert f[i] > s[i] and f[i - 1] <= s[i - 1]
    assert sg.stop_price[i] == df.low[i - 4:i + 1].min() and sg.target_rr[i] == 2.0
    assert (sg.stop_price[i] < df.close[i])


def test_ema_no_signal_when_already_above_from_the_start():
    df = path_bars(list(np.linspace(100, 130, 70)))
    sg = EmaCrossover(EmaCrossover.ConfigClass(vwap_filter=False, entry_start="09:15", entry_end="15:00")
                      ).generate_signals(df)
    assert (sg.direction == 0).all()


def test_ema_entry_window_and_vwap_filter():
    closes = list(np.linspace(100, 90, 30)) + list(np.linspace(90, 110, 40))
    df = path_bars(closes)
    base = dict(vwap_filter=False)
    # window that excludes the crossover -> no signal
    sg = EmaCrossover(EmaCrossover.ConfigClass(entry_start="14:00", entry_end="14:30", **base)).generate_signals(df)
    assert (sg.direction == 0).all()
    # VWAP filter: heavy early volume at high prices keeps VWAP (~99) above the cross price (~94.6) -> filtered
    heavy = path_bars(closes, volume=np.where(np.arange(len(closes)) < 10, 100000.0, 1000.0))
    off = EmaCrossover(EmaCrossover.ConfigClass(vwap_filter=False, entry_start="09:15", entry_end="15:00"))
    on = EmaCrossover(EmaCrossover.ConfigClass(entry_start="09:15", entry_end="15:00"))
    assert (off.generate_signals(heavy).direction == 1).sum() == 1
    assert (on.generate_signals(heavy).direction == 0).all()
    assert (on.generate_signals(df).direction == 1).sum() == 1          # plain data: close > VWAP, signal kept


def test_ema_shorts_only_when_enabled():
    closes = list(np.linspace(100, 110, 30)) + list(np.linspace(110, 90, 40))
    df = path_bars(closes)
    kw = dict(vwap_filter=False, entry_start="09:15", entry_end="15:00")
    assert (EmaCrossover(EmaCrossover.ConfigClass(**kw)).generate_signals(df).direction == 0).all()
    sg = EmaCrossover(EmaCrossover.ConfigClass(allow_short=True, **kw)).generate_signals(df)
    assert (sg.direction == -1).sum() == 1 and sg.stop_price[sg.direction == -1].iloc[0] > 0


# ------------------------------------------------------------------ ORB
def orb_day(day, overrides=None, vol=1000.0):
    rows = [(100, 100.2, 99.9, 100, vol)] * 75
    rows = list(rows)
    rows[0] = (100, 100.5, 99.8, 100.2, vol)
    rows[1] = (100.2, 100.4, 99.9, 100.3, vol)
    rows[2] = (100.3, 100.4, 99.9, 100.0, vol)
    for k, v in (overrides or {}).items():
        rows[k] = v
    return bars(rows, day=day)


def orb_dataset(overrides=None, test_vol_overrides=None):
    prior = [orb_day(d) for d in weekdays(date(2026, 1, 1), 3)]
    test = orb_day(date(2026, 1, 6), overrides)
    return pd.concat(prior + [test], ignore_index=True)


ORB_CFG = dict(rvol_lookback=3)
BREAKOUT = {4: (100.3, 101.2, 100.3, 101.0, 3000)}


def test_orb_breakout_signal_rules():
    df = orb_dataset(BREAKOUT)
    sg = OpeningRangeBreakout(OpeningRangeBreakout.ConfigClass(**ORB_CFG)).generate_signals(df)
    idx = sg.index[sg.direction == 1].tolist()
    assert idx == [3 * 75 + 4]
    i = idx[0]
    assert sg.stop_price[i] == 99.8 and sg.target_rr[i] == 2.0              # opposite range boundary, 2R
    assert df.timestamp[i].time().isoformat() == "09:35:00"


def test_orb_requires_completed_range_and_confirmed_close():
    # intrabar poke above the range high but close back inside: no signal
    df = orb_dataset({4: (100.3, 101.5, 100.3, 100.4, 3000)})
    sg = OpeningRangeBreakout(OpeningRangeBreakout.ConfigClass(**ORB_CFG)).generate_signals(df)
    assert (sg.direction == 0).all()
    # breakout candle inside the opening range window itself never signals
    df = orb_dataset({2: (100.3, 102, 99.9, 101.9, 9000)})
    sg = OpeningRangeBreakout(OpeningRangeBreakout.ConfigClass(**ORB_CFG)).generate_signals(df)
    assert (sg.direction == 0).all()


def test_orb_filters_width_volume_vwap_and_window():
    cfg = OpeningRangeBreakout.ConfigClass(**ORB_CFG)
    narrow = {0: (100, 100.1, 99.99, 100.0), 1: (100, 100.1, 99.99, 100.0), 2: (100, 100.1, 99.99, 100.0),
              4: (100.0, 100.8, 100.0, 100.7, 3000)}
    assert (OpeningRangeBreakout(cfg).generate_signals(orb_dataset(narrow)).direction == 0).all()
    wide = {0: (100, 103, 99, 101), 4: (101, 104, 101, 103.9, 3000)}
    assert (OpeningRangeBreakout(cfg).generate_signals(orb_dataset(wide)).direction == 0).all()
    low_vol = {4: (100.3, 101.2, 100.3, 101.0, 1100)}
    assert (OpeningRangeBreakout(cfg).generate_signals(orb_dataset(low_vol)).direction == 0).all()
    late = {40: (100.0, 101.2, 100.0, 101.0, 3000)}          # 12:15 > 11:30
    assert (OpeningRangeBreakout(cfg).generate_signals(orb_dataset(late)).direction == 0).all()
    no_filters = OpeningRangeBreakout.ConfigClass(rvol_filter=False, vwap_filter=False, **ORB_CFG)
    assert (OpeningRangeBreakout(no_filters).generate_signals(orb_dataset(low_vol)).direction == 1).sum() == 1


def test_orb_no_repeat_entries_from_same_breakout():
    ov = {**BREAKOUT, 5: (101.0, 101.6, 100.9, 101.4, 3000), 6: (101.4, 101.9, 101.3, 101.8, 3000)}
    df = orb_dataset(ov)
    sg = OpeningRangeBreakout(OpeningRangeBreakout.ConfigClass(**ORB_CFG)).generate_signals(df)
    assert (sg.direction == 1).sum() == 1
    ov2 = {**BREAKOUT, 5: (101.0, 101.0, 100.0, 100.2, 3000), 6: (100.2, 101.6, 100.2, 101.5, 3000)}
    sg2 = OpeningRangeBreakout(OpeningRangeBreakout.ConfigClass(allow_repeat_breakouts=True, **ORB_CFG)
                               ).generate_signals(orb_dataset(ov2))
    assert (sg2.direction == 1).sum() == 2
    sg3 = OpeningRangeBreakout(OpeningRangeBreakout.ConfigClass(**ORB_CFG)).generate_signals(orb_dataset(ov2))
    assert (sg3.direction == 1).sum() == 1


def test_orb_short_only_when_enabled():
    ov = {4: (100.0, 100.0, 98.9, 99.0, 3000)}
    df = orb_dataset(ov)
    assert (OpeningRangeBreakout(OpeningRangeBreakout.ConfigClass(**ORB_CFG)).generate_signals(df).direction == 0).all()
    sg = OpeningRangeBreakout(OpeningRangeBreakout.ConfigClass(allow_short=True, **ORB_CFG)).generate_signals(df)
    assert (sg.direction == -1).sum() == 1 and sg.stop_price[sg.direction == -1].iloc[0] == 100.5


# ------------------------------------------------------------------ swing breakout
def uptrend(n=80, **last):
    rows = [(100 + 0.1 * i, 100.5 + 0.1 * i, 99.5 + 0.1 * i, 100 + 0.1 * i, 1000) for i in range(n)]
    return rows


def sb_signals(last_row, **cfg):
    rows = uptrend() + [last_row]
    df = daily(rows)
    return df, SwingBreakout(SwingBreakout.ConfigClass(**cfg)).generate_signals(df)


def test_swing_breakout_uses_only_prior_sessions_for_the_level():
    res = max(r[1] for r in uptrend()[-20:])
    close = res * 1.01
    df, sg = sb_signals((close - 0.5, 120.0, close - 1.0, close, 2000))     # spike high must NOT lift its own level
    assert sg.index[sg.direction == 1].tolist() == [80]
    assert sg.target_rr[80] == 3.0 and sg.stop_price[80] == df.low[71:81].min()


@pytest.mark.parametrize("close_mult,volume", [(1.001, 2000), (1.05, 2000), (1.01, 1200)])
def test_swing_breakout_filters(close_mult, volume):
    res = max(r[1] for r in uptrend()[-20:])
    close = res * close_mult                                  # below buffer / over-extended / low volume
    _, sg = sb_signals((close - 0.5, close + 0.2, close - 1.0, close, volume))
    assert (sg.direction == 0).all()


def test_swing_breakout_requires_close_above_ema50():
    rows = [(200 - 0.5 * i, 200.5 - 0.5 * i, 199.5 - 0.5 * i, 200 - 0.5 * i, 1000) for i in range(80)]
    res = max(r[1] for r in rows[-20:])
    close = res * 1.01
    rows.append((close - 0.5, close + 0.2, close - 1, close, 2000))
    sg = SwingBreakout(SwingBreakout.ConfigClass()).generate_signals(daily(rows))
    assert (sg.direction == 0).all()                           # downtrend: close < EMA50


# ------------------------------------------------------------------ swing support / resistance
def sr_rows(reject_at):
    n = 45
    rows = [(108, 110, 105, 108, 1000)] * n
    rows = list(rows)
    rows[10] = (108, 110, 100.0, 108, 1000)
    rows[25] = (108, 110, 100.1, 108, 1000)
    rows[reject_at] = (108, 109, 100.3, 108.5, 2000)
    return rows


SR_CFG = dict(ema_period=5, rvol_period=5)


def test_sr_rejection_signal_and_stop_below_zone():
    df = daily(sr_rows(40))
    sg = SwingSupportResistance(SwingSupportResistance.ConfigClass(**SR_CFG)).generate_signals(df)
    assert sg.index[sg.direction == 1].tolist() == [40]
    assert sg.stop_price[40] < 99.425 and sg.target_rr[40] == 2.0 and "2 touches" in sg.reason[40]


def test_sr_pivot_not_usable_before_right_candles_complete():
    cfg = SwingSupportResistance.ConfigClass(**SR_CFG)
    early = SwingSupportResistance(cfg).generate_signals(daily(sr_rows(27)))      # pivot@25 usable only at t>=28
    assert (early.direction == 0).all()
    ok = SwingSupportResistance(cfg).generate_signals(daily(sr_rows(28)))
    assert ok.index[ok.direction == 1].tolist() == [28]


def test_sr_needs_min_touches_and_volume_and_ema():
    cfg3 = SwingSupportResistance.ConfigClass(min_touches=3, **SR_CFG)
    assert (SwingSupportResistance(cfg3).generate_signals(daily(sr_rows(40))).direction == 0).all()
    rows = sr_rows(40)
    rows[40] = (108, 109, 100.3, 108.5, 1000)                                    # no volume expansion
    assert (SwingSupportResistance(SwingSupportResistance.ConfigClass(**SR_CFG)).generate_signals(
        daily(rows)).direction == 0).all()


def test_sr_resistance_target_mode_rejects_when_no_resistance():
    cfg = SwingSupportResistance.ConfigClass(target_method="resistance", **SR_CFG)
    sg = SwingSupportResistance(cfg).generate_signals(daily(sr_rows(40)))
    assert (sg.direction == 0).all()          # no confirmed resistance zone above price


# ------------------------------------------------------------------ no look-ahead (all strategies)
def noisy_intraday(days=12):
    frames = []
    for k, d in enumerate(weekdays(date(2026, 1, 1), days)):
        i = np.arange(75)
        close = 100 + 3 * np.sin((i + 7 * k) / 6.0) + 0.15 * k + 0.5 * np.sin(i * 1.3)
        vol = 1000 + 900 * (np.sin(i / 4.0 + k) ** 2)
        rows, prev = [], close[0]
        for c, v in zip(close, vol):
            rows.append((prev, max(prev, c) + 0.2, min(prev, c) - 0.2, c, v))
            prev = c
        frames.append(bars(rows, day=d))
    return pd.concat(frames, ignore_index=True)


def noisy_daily(n=260, trend=0.3, amp=4.0):
    i = np.arange(n)
    close = 100 + amp * np.sin(i / 9.0) + trend * i
    high = close + 1.0 + 0.4 * np.abs(np.sin(i))
    low = close - 1.0 - 0.4 * np.abs(np.cos(i * 1.7))
    vol = 1000 + 800 * (np.sin(i / 3.0) ** 2)
    return daily(list(zip(close, high, low, close, vol)))


def assert_no_lookahead(strategy, candles, cuts):
    full = strategy.generate_signals(candles)
    assert (full.direction != 0).sum() > 0, "fixture should produce signals for this test to be meaningful"
    cols = ["direction", "stop_price", "target_rr", "target_price", "min_rr"]
    for k in cuts:
        part = strategy.generate_signals(candles.iloc[:k].reset_index(drop=True))
        a, b = full.iloc[:k][cols].reset_index(drop=True), part[cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_no_lookahead_ema():
    strat = EmaCrossover(EmaCrossover.ConfigClass(vwap_filter=False))
    assert_no_lookahead(strat, noisy_intraday(), [100, 233, 400, 611, 700])


def test_no_lookahead_orb():
    strat = OpeningRangeBreakout(OpeningRangeBreakout.ConfigClass(rvol_lookback=3, rvol_min=0.9, vwap_filter=False,
                                                                  min_range_width_pct=0.05, max_range_width_pct=5))
    assert_no_lookahead(strat, noisy_intraday(), [160, 300, 377, 520, 777])


def test_no_lookahead_swing_breakout():
    res = max(r[1] for r in uptrend()[-20:])
    close = res * 1.01
    rows = uptrend() + [(close - 0.5, close + 0.2, close - 1.0, close, 2000)] + \
        [(close, close + 0.5, close - 0.5, close + 0.1 * k, 1000) for k in range(1, 6)]
    assert_no_lookahead(SwingBreakout(SwingBreakout.ConfigClass()), daily(rows), [60, 79, 80, 81, 85])


def test_no_lookahead_swing_sr():
    strat = SwingSupportResistance(SwingSupportResistance.ConfigClass(**SR_CFG))
    assert_no_lookahead(strat, daily(sr_rows(40)), [26, 29, 39, 40, 41, 44])


# ------------------------------------------------------------------ config handling
def test_all_four_strategies_registered_and_default_configs_roundtrip():
    assert set(REGISTRY) == {"ema_crossover", "orb", "swing_breakout", "swing_sr"}
    for key, cls in REGISTRY.items():
        s = get_strategy(key)
        assert type(s.config).from_dict(s.config.to_dict()) == s.config


def test_spec_defaults():
    e, o = EmaCrossover.ConfigClass(), OpeningRangeBreakout.ConfigClass()
    assert (e.fast_period, e.slow_period, e.timeframe, e.vwap_filter, e.stop_lookback, e.reward_to_risk,
            e.risk_per_trade_pct, e.max_trades_per_instrument_per_day, e.entry_start, e.entry_end,
            e.force_exit_time) == (9, 21, "5m", True, 5, 2.0, 0.25, 2, "09:30", "14:30", "15:15")
    assert (o.breakout_buffer_pct, o.min_range_width_pct, o.max_range_width_pct, o.rvol_min,
            o.max_trades_per_instrument_per_day, o.daily_loss_limit_pct, o.entry_end) == \
        (0.05, 0.20, 1.00, 1.2, 1, 0.75, "11:30")
    b, s = SwingBreakout.ConfigClass(), SwingSupportResistance.ConfigClass()
    assert (b.breakout_lookback, b.breakout_buffer_pct, b.rvol_min, b.ema_period, b.max_extension_pct,
            b.stop_lookback, b.reward_to_risk, b.risk_per_trade_pct, b.max_simultaneous_positions,
            b.max_new_entries_per_day, b.max_portfolio_open_risk_pct, b.max_capital_allocation_pct,
            b.sector_limit) == (20, 0.25, 1.5, 50, 3.0, 10, 3.0, 0.5, 5, 3, 2.0, 80.0, 2)
    assert (s.pivot_left, s.pivot_right, s.lookback_sessions, s.min_touches, s.zone_width_atr, s.rvol_min,
            s.ema_period, s.min_reward_to_risk, s.risk_per_trade_pct, s.max_simultaneous_positions,
            s.max_portfolio_open_risk_pct) == (3, 3, 60, 2, 0.25, 1.2, 50, 2.0, 0.5, 5, 2.0)
    assert e.allow_short is False and b.intraday is False and e.intraday is True


def test_config_validation_errors_are_clear():
    with pytest.raises(ValueError, match="Unknown parameter"):
        EmaCrossover.ConfigClass.from_dict({"fast_perod": 5})
    with pytest.raises(ValueError, match="fast_period"):
        EmaCrossover.ConfigClass.from_dict({"fast_period": 30})
    with pytest.raises(ValueError, match="short"):
        SwingBreakout.ConfigClass.from_dict({"allow_short": True})
    with pytest.raises(ValueError, match="whole number"):
        EmaCrossover.ConfigClass.from_dict({"fast_period": "abc"})
    with pytest.raises(ValueError, match="leverage"):
        EmaCrossover.ConfigClass.from_dict({"max_capital_allocation_pct": 250})
    assert EmaCrossover.ConfigClass.from_dict({"max_trades_per_instrument_per_day": None}
                                              ).max_trades_per_instrument_per_day is None


def test_with_overrides_does_not_mutate_original():
    s = get_strategy("ema_crossover")
    s2 = s.with_overrides({"reward_to_risk": 3.0})
    assert s.config.reward_to_risk == 2.0 and s2.config.reward_to_risk == 3.0
