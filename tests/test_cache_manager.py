import os
from datetime import date, datetime

import pandas as pd
import pytest

from config.settings import IST, AppSettings
from src.api.kite_client import AuthenticationError, NetworkError
from src.data import excel_cache
from src.data.excel_cache import CacheError, ExcelCandleCache, merge_candles
from src.data.manager import CancelToken, DataManager
from src.data.ranges import chunk_range, merge_ranges, subtract_ranges, trim_weekends
from tests.helpers import minute_day, weekdays

INST = "NSE:TEST"


class FakeSource:
    """Offline stand-in for Kite: serves synthetic minute candles for weekdays (skips `holidays`)."""

    def __init__(self, holidays=(), fail=None, empty=False, bad=False):
        self.calls, self.holidays, self.fail, self.empty, self.bad = [], set(holidays), fail, empty, bad

    def fetch_candles(self, token, start, end, interval):
        self.calls.append((start, end, interval))
        if self.fail:
            raise self.fail
        if self.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        frames, d = [], start
        while d <= end:
            if d.weekday() < 5 and d not in self.holidays:
                frames.append(minute_day(d))
            d = d.fromordinal(d.toordinal() + 1)
        if not frames:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.concat(frames, ignore_index=True)
        if self.bad:
            df["high"] = df["low"] - 1
        return df


def manager(tmp_path, source, now=datetime(2026, 3, 1, 12, 0, tzinfo=IST)):
    cache = ExcelCandleCache(tmp_path)
    return DataManager(source, cache, AppSettings(), now_fn=lambda: now), cache


def test_range_helpers():
    a, b, c = date(2026, 1, 1), date(2026, 1, 10), date(2026, 1, 20)
    assert merge_ranges([(a, date(2026, 1, 5)), (date(2026, 1, 6), b)]) == [(a, b)]
    assert subtract_ranges((a, c), [(date(2026, 1, 5), date(2026, 1, 8))]) == \
        [(a, date(2026, 1, 4)), (date(2026, 1, 9), c)]
    assert trim_weekends([(date(2026, 1, 3), date(2026, 1, 4))]) == []           # Sat-Sun only
    assert trim_weekends([(date(2026, 1, 3), date(2026, 1, 7))]) == [(date(2026, 1, 5), date(2026, 1, 7))]
    assert chunk_range((a, date(2026, 3, 1)), 60) == [(a, date(2026, 3, 1))]
    assert len(chunk_range((a, date(2026, 6, 1)), 60)) == 3


def test_filename_is_predictable(tmp_path):
    assert ExcelCandleCache(tmp_path).path_for("NSE:RELIANCE", "minute").name == "NSE_RELIANCE_minute.xlsx"


def test_download_then_reuse_without_redownload(tmp_path):
    src = FakeSource()
    dm, cache = manager(tmp_path, src)
    rep = dm.ensure_data(INST, 123, "minute", date(2026, 1, 5), date(2026, 1, 16))
    assert rep.downloaded and not rep.failed and not rep.missing and len(src.calls) == 1
    assert len(cache.load(INST, "minute").candles) == 10 * 375
    rep2 = dm.ensure_data(INST, 123, "minute", date(2026, 1, 5), date(2026, 1, 16))
    assert len(src.calls) == 1 and rep2.cached and not rep2.downloaded      # nothing re-downloaded


def test_only_missing_ranges_are_downloaded(tmp_path):
    src = FakeSource()
    dm, _ = manager(tmp_path, src)
    dm.ensure_data(INST, 123, "minute", date(2026, 1, 12), date(2026, 1, 16))
    src.calls.clear()
    rep = dm.ensure_data(INST, 123, "minute", date(2026, 1, 5), date(2026, 1, 23))
    assert src.calls == [(date(2026, 1, 5), date(2026, 1, 11), "minute"),
                         (date(2026, 1, 17), date(2026, 1, 23), "minute")] or \
        [(c[0], c[1]) for c in src.calls] == [(date(2026, 1, 5), date(2026, 1, 9)), (date(2026, 1, 19), date(2026, 1, 23))]
    assert not rep.missing


def test_holiday_in_short_gap_is_recorded_not_fabricated(tmp_path):
    src = FakeSource(holidays=[date(2026, 1, 26)])
    dm, cache = manager(tmp_path, src)
    dm.ensure_data(INST, 1, "minute", date(2026, 1, 26), date(2026, 1, 26))
    c = cache.load(INST, "minute")
    assert c.candles.empty and (c.coverage.status == "no_data").all()
    loaded = dm.load_candles(INST, "5m", date(2026, 1, 26), date(2026, 1, 26))
    assert loaded.info["no_data_weekdays"] == [date(2026, 1, 26)] and loaded.candles.empty


def test_empty_response_for_long_range_is_failure_not_holidays(tmp_path):
    dm, cache = manager(tmp_path, FakeSource(empty=True))
    rep = dm.ensure_data(INST, 1, "minute", date(2026, 1, 5), date(2026, 1, 30))
    assert rep.failed and rep.missing and cache.load(INST, "minute").candles.empty


def test_all_invalid_rows_rejected(tmp_path):
    dm, cache = manager(tmp_path, FakeSource(bad=True))
    rep = dm.ensure_data(INST, 1, "minute", date(2026, 1, 5), date(2026, 1, 9))
    assert rep.failed and rep.invalid.get("ohlc_inconsistent") and rep.missing
    assert cache.load(INST, "minute").candles.empty


def test_auth_failure_aborts_without_data_loss(tmp_path):
    dm, cache = manager(tmp_path, FakeSource())
    dm.ensure_data(INST, 1, "minute", date(2026, 1, 5), date(2026, 1, 9))
    dm.source = FakeSource(fail=AuthenticationError("token expired"))
    rep = dm.ensure_data(INST, 1, "minute", date(2026, 1, 12), date(2026, 1, 16))
    assert rep.aborted and "expired" in rep.aborted
    assert len(cache.load(INST, "minute").candles) == 5 * 375             # earlier data preserved


def test_network_errors_stop_after_three_consecutive_failures(tmp_path):
    src = FakeSource(fail=NetworkError("boom"))
    dm, _ = manager(tmp_path, src)
    rep = dm.ensure_data(INST, 1, "minute", date(2025, 6, 2), date(2026, 1, 30))   # many 60-day chunks
    assert rep.aborted and len(src.calls) == 3 and len(rep.failed) == 3


def test_cancel_stops_before_next_chunk_and_keeps_progress(tmp_path):
    cancel = CancelToken()
    src = FakeSource()
    orig = src.fetch_candles

    def fetch(*a):
        out = orig(*a)
        cancel.cancel()
        return out

    src.fetch_candles = fetch
    dm, cache = manager(tmp_path, src)
    rep = dm.ensure_data(INST, 1, "minute", date(2025, 11, 3), date(2026, 2, 27), cancel=cancel)
    assert rep.cancelled and len(src.calls) == 1 and len(cache.load(INST, "minute").candles) > 0 and rep.missing


def test_todays_incomplete_session_is_not_marked_covered(tmp_path):
    now = datetime(2026, 1, 7, 11, 0, tzinfo=IST)           # market still open
    dm, cache = manager(tmp_path, FakeSource(), now)
    rep = dm.ensure_data(INST, 1, "minute", date(2026, 1, 5), date(2026, 1, 7))
    assert rep.not_yet_available == (date(2026, 1, 7), date(2026, 1, 7))
    assert cache.load(INST, "minute").candles.timestamp.dt.date.max() == date(2026, 1, 6)


def test_merge_dedupes_sorts_and_keeps_existing_on_conflict():
    d = date(2026, 1, 5)
    old = minute_day(d).iloc[:10]
    new = minute_day(d).iloc[5:20].copy()
    new.loc[new.index[0], "close"] += 1          # conflicting overlap
    merged, overlap, conflicts = merge_candles(old, new)
    assert len(merged) == 20 and overlap == 5 and conflicts == 1
    assert merged.timestamp.is_monotonic_increasing
    assert merged.loc[5, "close"] == old.loc[5, "close"]      # existing valid data preserved


def test_failed_save_keeps_previous_cache(tmp_path, monkeypatch):
    dm, cache = manager(tmp_path, FakeSource())
    dm.ensure_data(INST, 1, "minute", date(2026, 1, 5), date(2026, 1, 9))
    path = cache.path_for(INST, "minute")
    before = path.read_bytes()
    monkeypatch.setattr(excel_cache.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    dm2 = DataManager(FakeSource(), cache, AppSettings(), now_fn=lambda: datetime(2026, 3, 1, 12, tzinfo=IST))
    rep = dm2.ensure_data(INST, 1, "minute", date(2026, 1, 12), date(2026, 1, 16))
    assert rep.aborted and "Previous cache left intact" in rep.aborted
    assert path.read_bytes() == before and not list(tmp_path.glob("~tmp_*"))


def test_corrupt_cache_is_reported_and_not_modified(tmp_path):
    cache = ExcelCandleCache(tmp_path)
    p = cache.path_for(INST, "minute")
    p.write_bytes(b"not an excel file")
    with pytest.raises(CacheError):
        cache.load(INST, "minute")
    assert p.read_bytes() == b"not an excel file"


def test_token_mismatch_refuses_to_mix(tmp_path):
    dm, _ = manager(tmp_path, FakeSource())
    dm.ensure_data(INST, 111, "minute", date(2026, 1, 5), date(2026, 1, 9))
    with pytest.raises(CacheError):
        dm.ensure_data(INST, 222, "minute", date(2026, 1, 12), date(2026, 1, 16))


def test_load_candles_resamples_and_reports_missing(tmp_path):
    dm, _ = manager(tmp_path, FakeSource())
    dm.ensure_data(INST, 1, "minute", date(2026, 1, 5), date(2026, 1, 9))
    loaded = dm.load_candles(INST, "15m", date(2026, 1, 5), date(2026, 1, 16))
    assert len(loaded.candles) == 5 * 25
    assert loaded.info["missing_ranges"] == [(date(2026, 1, 12), date(2026, 1, 16))]
    assert any("uncovered" in w for w in loaded.info["warnings"])
