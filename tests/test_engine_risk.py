from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.engine.backtester import BacktestEngine
from src.engine.costs import CostModel
from config.settings import default_delivery_costs, default_intraday_costs
from tests.helpers import bars, daily, engine_cfg, signal_at, ts

D = date(2026, 1, 5)


def run(df, sg, cfg=None, inst="AAA"):
    cfg = cfg or engine_cfg()
    return BacktestEngine(cfg).run({inst: df}, {inst: sg}, D, D + timedelta(days=60))


FLAT = (100, 100.5, 99.5, 100)


def test_entry_is_next_candle_open_not_signal_close():
    df = bars([FLAT, FLAT, (100, 100.5, 99.5, 100.2), (101, 101.5, 100.5, 101.2), FLAT])
    r = run(df, signal_at(df, 2, stop=98.0))
    t = r.trades.iloc[0]
    assert t.entry_time == df.timestamp[3] and t.entry_price == 101 and t.signal_time == df.timestamp[2]


def test_position_size_is_floor_of_risk_over_distance():
    df = bars([FLAT, FLAT, FLAT, FLAT, FLAT])
    r = run(df, signal_at(df, 1, stop=97.7), engine_cfg(capital=100000, risk_pct=1.0))
    assert r.trades.iloc[0].quantity == int(1000 // 2.3)        # 434


def test_stop_assumed_first_when_both_touched_in_same_candle():
    df = bars([FLAT, FLAT, (100, 105, 97, 100), FLAT])
    r = run(df, signal_at(df, 0, stop=98.0))                     # entry candle (idx1) is flat, idx2 spans both
    t = r.trades.iloc[0]
    assert t.exit_reason == "stop_loss" and t.exit_price == 98.0 and bool(t.stop_and_target_same_candle)


def test_target_hit_and_r_multiple():
    df = bars([FLAT, FLAT, (100, 104.5, 99.5, 104), FLAT])
    r = run(df, signal_at(df, 0, stop=98.0, rr=2.0))
    t = r.trades.iloc[0]
    assert t.exit_reason == "target" and t.exit_price == 104.0
    assert t.quantity == 500 and abs(t.net_pnl - 2000) < 1e-9 and abs(t.r_multiple - 2.0) < 1e-9


def test_stop_can_trigger_on_entry_candle():
    df = bars([FLAT, (100, 100.5, 97.0, 98), FLAT])
    r = run(df, signal_at(df, 0, stop=98.0))
    assert r.trades.iloc[0].exit_reason == "stop_loss" and r.trades.iloc[0].exit_time == df.timestamp[1]


def test_gap_through_stop_exits_at_open():
    df = bars([FLAT, FLAT, (96, 96.5, 95, 96), FLAT])
    r = run(df, signal_at(df, 0, stop=98.0))
    t = r.trades.iloc[0]
    assert t.exit_reason == "stop_loss_gap" and t.exit_price == 96 and t.net_pnl < -1000   # worse than 1R


def test_gap_entry_beyond_stop_is_rejected():
    df = bars([FLAT, (97, 97.5, 96.5, 97), FLAT])
    r = run(df, signal_at(df, 0, stop=98.0))
    assert r.trades.empty and r.rejections.reason.tolist() == ["invalid_stop_distance"]


def test_slippage_is_adverse_on_both_sides():
    df = bars([FLAT, FLAT, (100, 104.5, 99.5, 104), FLAT, (100, 100.5, 94, 95)])
    cfg = engine_cfg(slippage=0.1)
    r = run(df, signal_at(df, 0, stop=98.0), cfg)
    t = r.trades.iloc[0]
    assert t.entry_price > 100 and abs(t.entry_price - 100.1) < 1e-9
    assert t.exit_price == t.target_price                          # limit exit: no slippage
    df2 = bars([FLAT, FLAT, (100, 100.2, 97.5, 98)])
    t2 = run(df2, signal_at(df2, 0, stop=98.0), cfg).trades.iloc[0]
    assert t2.exit_reason == "stop_loss" and t2.exit_price < 98.0


def test_forced_exit_at_1515_open_and_no_overnight():
    rows = [FLAT] * 8
    df = bars(rows, start=(14, 50))                    # 14:50 .. 15:25
    r = run(df, signal_at(df, 0, stop=90.0))
    t = r.trades.iloc[0]
    assert t.exit_reason == "forced_exit" and t.exit_time.time().isoformat() == "15:15:00"


def test_leftover_intraday_position_closes_at_session_end_and_never_carries():
    d1 = bars([FLAT, FLAT, FLAT], day=D)
    d2 = bars([FLAT, FLAT], day=D + timedelta(days=1))
    df = pd.concat([d1, d2], ignore_index=True)
    r = run(df, signal_at(df, 0, stop=90.0))
    t = r.trades.iloc[0]
    assert t.exit_reason == "end_of_session" and t.exit_time.date() == D


def test_entry_at_or_after_force_exit_rejected():
    df = bars([FLAT] * 6, start=(15, 0))               # 15:00 .. 15:25
    r = run(df, signal_at(df, 2, stop=90.0))           # signal 15:10 -> entry 15:15
    assert r.trades.empty and r.rejections.reason.tolist() == ["entry_at_or_after_force_exit_time"]


def test_signal_on_last_candle_of_session_cannot_enter_next_day():
    d1 = bars([FLAT, FLAT], day=D)
    d2 = bars([FLAT, FLAT], day=D + timedelta(days=1))
    df = pd.concat([d1, d2], ignore_index=True)
    r = run(df, signal_at(df, 1, stop=90.0))
    assert r.trades.empty and r.rejections.reason.tolist() == ["no_next_candle_in_session"]


def test_max_trades_per_instrument_per_day():
    rows = [FLAT, FLAT, (100, 100.5, 97, 98), FLAT, FLAT, FLAT]
    df = bars(rows)
    sg = signal_at(df, 0, stop=98.0)
    sg.loc[3, ["direction", "stop_price", "target_rr"]] = [1, 98.0, 2.0]
    r = run(df, sg, engine_cfg(max_trades_per_instrument_per_day=1))
    assert len(r.trades) == 1 and "max_trades_per_instrument_per_day" in r.rejections.reason.tolist()
    r2 = run(df, sg, engine_cfg(max_trades_per_instrument_per_day=2))
    assert len(r2.trades) == 2


def test_daily_loss_limit_blocks_new_entries():
    rows = [FLAT, FLAT, (100, 100.5, 97, 98), FLAT, FLAT, FLAT]
    df = bars(rows)
    sg = signal_at(df, 0, stop=98.0)
    sg.loc[3, ["direction", "stop_price", "target_rr"]] = [1, 98.0, 2.0]
    r = run(df, sg, engine_cfg(daily_loss_limit_pct=0.5))        # 1R loss = 1% > 0.5%
    assert len(r.trades) == 1 and "daily_loss_limit_reached" in r.rejections.reason.tolist()


def test_zero_quantity_and_capital_cap():
    df = bars([FLAT] * 4)
    r = run(df, signal_at(df, 0, stop=80.0), engine_cfg(capital=1000, risk_pct=1.0))   # budget 10 < distance 20
    assert r.rejections.reason.tolist() == ["zero_quantity"]
    r = run(df, signal_at(df, 0, stop=99.5), engine_cfg(capital=10000, risk_pct=5.0))
    assert r.trades.iloc[0].quantity == 100                       # capped by capital, NOT 1000 (no leverage)


def test_invalid_stop_and_nan_stop_rejected():
    df = bars([FLAT] * 4)
    assert run(df, signal_at(df, 0, stop=105.0)).rejections.reason.tolist() == ["invalid_stop_distance"]
    assert run(df, signal_at(df, 0, stop=np.nan)).rejections.reason.tolist() == ["invalid_stop_distance"]


def test_short_selling_off_by_default_and_symmetric_when_on():
    df = bars([FLAT, FLAT, (100, 100.5, 95.5, 96), FLAT])
    sg = signal_at(df, 0, direction=-1, stop=102.0)
    assert run(df, sg).rejections.reason.tolist() == ["short_selling_disabled"]
    t = run(df, sg, engine_cfg(allow_short=True)).trades.iloc[0]
    assert t.direction == "SHORT" and t.exit_reason == "target" and t.exit_price == 96.0 and t.net_pnl > 0


def test_cost_model_numbers():
    cm = CostModel(default_intraday_costs(), default_delivery_costs())
    assert abs(cm.leg(100000, "buy", True) - 30.2226) < 1e-4
    assert abs(cm.leg(100000, "sell", True) - 52.2226) < 1e-4
    assert cm.leg(100000, "sell", False) > cm.leg(100000, "sell", True) - 30       # delivery STT + DP charge
    assert cm.with_multiplier(0).leg(100000, "buy", True) == 0


def test_costs_reduce_pnl_and_are_reported():
    df = bars([FLAT, FLAT, (100, 104.5, 99.5, 104), FLAT])
    cfg = engine_cfg()
    cfg.costs = CostModel(default_intraday_costs(), default_delivery_costs())
    t = run(df, signal_at(df, 0, stop=98.0), cfg).trades.iloc[0]
    assert t.costs > 0 and abs(t.net_pnl - (t.gross_pnl - t.costs)) < 1e-9


def test_equity_curve_reconciles_with_trades_and_drawdown_nonpositive():
    df = bars([FLAT, FLAT, (100, 100.5, 97, 98), FLAT, FLAT])
    r = run(df, signal_at(df, 0, stop=98.0))
    assert abs(r.equity.equity.iloc[-1] - (100000 + r.trades.net_pnl.sum())) < 1e-6
    assert (r.equity.drawdown <= 1e-12).all() and r.equity.drawdown.min() < 0


def test_min_rr_rejection_for_absolute_targets():
    df = bars([FLAT] * 4)
    sg = signal_at(df, 0, stop=98.0, target_price=101.0, min_rr=2.0)
    assert run(df, sg).rejections.reason.tolist() == ["reward_to_risk_below_minimum"]


def test_signals_outside_trading_window_are_ignored():
    df = bars([FLAT] * 4)
    sg = signal_at(df, 1, stop=98.0)
    r = BacktestEngine(engine_cfg()).run({"A": df}, {"A": sg}, D + timedelta(days=5), D + timedelta(days=9))
    assert r.trades.empty and any("no candles inside" in w for w in r.warnings)


# --- swing / multi-instrument portfolio rules ---------------------------------
def swing_frames():
    rows = [(100, 101, 99, 100)] * 6
    return daily(rows, start=D)


def test_swing_position_held_overnight_until_end_of_data():
    df = swing_frames()
    r = run(df, signal_at(df, 0, stop=90.0), engine_cfg(intraday=False))
    t = r.trades.iloc[0]
    assert t.exit_reason == "end_of_backtest" and t.exit_time > t.entry_time + pd.Timedelta(days=2)


def test_max_simultaneous_positions_and_open_risk_across_instruments():
    df = swing_frames()
    sg = signal_at(df, 0, stop=98.0)
    cfg = engine_cfg(intraday=False, max_simultaneous_positions=1)
    r = BacktestEngine(cfg).run({"A": df, "B": df}, {"A": sg, "B": sg}, D, D + timedelta(days=30))
    assert len(r.trades) == 1 and r.rejections.reason.tolist() == ["max_simultaneous_positions"]
    cfg = engine_cfg(intraday=False, max_portfolio_open_risk_pct=1.5)
    r = BacktestEngine(cfg).run({"A": df, "B": df}, {"A": sg, "B": sg}, D, D + timedelta(days=30))
    assert len(r.trades) == 1 and r.rejections.reason.tolist() == ["max_portfolio_open_risk"]


def test_max_new_entries_per_day_and_sector_limit():
    df = swing_frames()
    sg = signal_at(df, 0, stop=98.0)
    cfg = engine_cfg(intraday=False, max_new_entries_per_day=1)
    r = BacktestEngine(cfg).run({"A": df, "B": df}, {"A": sg, "B": sg}, D, D + timedelta(days=30))
    assert len(r.trades) == 1 and r.rejections.reason.tolist() == ["max_new_entries_per_day"]
    cfg = engine_cfg(intraday=False, sector_limit=1)
    cfg.sector_map = {"A": "IT", "B": "IT"}
    r = BacktestEngine(cfg).run({"A": df, "B": df}, {"A": sg, "B": sg}, D, D + timedelta(days=30))
    assert len(r.trades) == 1 and r.rejections.reason.tolist() == ["sector_limit"]


def test_sector_limit_without_data_is_flagged_not_silently_enforced():
    df = swing_frames()
    r = BacktestEngine(engine_cfg(intraday=False, sector_limit=2)).run(
        {"A": df}, {"A": signal_at(df, 0, stop=98.0)}, D, D + timedelta(days=30))
    assert any("Sector limit" in w for w in r.warnings)
