"""DataManager: coverage inspection, gap download/recovery, loading + resampling."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Callable, Optional, Protocol

import pandas as pd

from config.settings import IST, AppSettings
from src.api.kite_client import (MAX_DAYS_PER_REQUEST, AuthenticationError, InvalidInstrumentError,
                                 KiteAPIError)
from src.data.excel_cache import CacheContents, CacheError, ExcelCandleCache, merge_candles
from src.data.ranges import (Range, chunk_range, fmt, intersect_ranges, subtract_ranges, trim_weekends,
                             weekday_count)
from src.data.resample import resample_intraday
from src.data.validation import empty_candles, validate_candles

MAX_CONSECUTIVE_FAILURES = 3
SUSPICIOUS_EMPTY_WEEKDAYS = 4   # an empty response covering >= this many weekdays is NOT assumed to be holidays
DATA_FINAL_AFTER = time(15, 45)  # today's candles are treated as final only after this IST time


class HistoricalSource(Protocol):
    def fetch_candles(self, token: int, start: date, end: date, interval: str) -> pd.DataFrame: ...


class CancelToken:
    """Set ``cancelled`` (e.g. from a Ctrl+C handler) to stop after the current request."""

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


@dataclass
class DataReport:
    instrument: str = ""
    interval: str = ""
    requested: Optional[Range] = None
    cached: list = field(default_factory=list)          # ranges already covered
    downloaded: list = field(default_factory=list)      # ranges fetched this run
    missing: list = field(default_factory=list)         # ranges still missing afterwards
    failed: list = field(default_factory=list)          # (range, reason)
    not_yet_available: Optional[Range] = None
    rows_added: int = 0
    duplicates_removed: int = 0
    overlap_conflicts: int = 0
    invalid: dict = field(default_factory=dict)
    cancelled: bool = False
    aborted: Optional[str] = None
    notes: list = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        f = lambda rs: ", ".join(fmt(r) for r in rs) or "none"  # noqa: E731
        lines = [f"Instrument {self.instrument} [{self.interval}]  requested: "
                 f"{fmt(self.requested) if self.requested else '-'}",
                 f"  cached (already covered): {f(self.cached)}",
                 f"  downloaded this run:      {f(self.downloaded)}",
                 f"  still missing:            {f(self.missing)}"]
        if self.failed:
            lines.append("  failed: " + "; ".join(f"{fmt(r)} ({why})" for r, why in self.failed))
        if self.invalid:
            lines.append(f"  invalid rows rejected: {self.invalid}")
        if self.duplicates_removed or self.overlap_conflicts:
            lines.append(f"  duplicates removed: {self.duplicates_removed}; "
                         f"conflicting overlaps (kept existing): {self.overlap_conflicts}")
        if self.not_yet_available:
            lines.append(f"  not yet final (today's session not complete): {fmt(self.not_yet_available)}")
        if self.cancelled:
            lines.append("  CANCELLED by user (progress so far was saved).")
        if self.aborted:
            lines.append(f"  ABORTED: {self.aborted}")
        lines += [f"  note: {n}" for n in self.notes]
        return lines


@dataclass
class LoadedData:
    candles: pd.DataFrame
    info: dict


def base_interval_for(timeframe: str) -> str:
    return "day" if timeframe in ("1d", "day", "d") else "minute"


def timeframe_minutes(timeframe: str) -> Optional[int]:
    if base_interval_for(timeframe) == "day":
        return None
    tf = timeframe.strip().lower()
    if not tf.endswith("m") or not tf[:-1].isdigit():
        raise ValueError(f"Invalid timeframe '{timeframe}'. Use e.g. 5m, 15m or 1d.")
    return int(tf[:-1])


class DataManager:
    def __init__(self, source: Optional[HistoricalSource], cache: ExcelCandleCache, settings: AppSettings,
                 now_fn: Callable[[], datetime] = lambda: datetime.now(IST)) -> None:
        self.source = source
        self.cache = cache
        self.settings = settings
        self._now = now_fn

    # ------------------------------------------------------------ helpers
    def coverable_end(self) -> date:
        now = self._now()
        return now.date() if now.time() >= DATA_FINAL_AFTER else now.date() - timedelta(days=1)

    def _missing(self, contents: CacheContents, requested: Range) -> tuple[list, list, Optional[Range]]:
        eff_end = min(requested[1], self.coverable_end())
        not_yet = (max(requested[0], eff_end + timedelta(days=1)), requested[1]) if requested[1] > eff_end else None
        if eff_end < requested[0]:
            return [], [], not_yet
        eff = (requested[0], eff_end)
        covered = contents.covered_ranges()
        cached = intersect_ranges(eff, covered)
        missing = trim_weekends(subtract_ranges(eff, covered))
        return cached, missing, not_yet

    # ------------------------------------------------------------- status
    def status(self, instrument: str, interval: str, start: date, end: date) -> DataReport:
        contents = self.cache.load(instrument, interval)
        cached, missing, not_yet = self._missing(contents, (start, end))
        rep = DataReport(instrument=instrument, interval=interval, requested=(start, end), cached=cached,
                         missing=missing, not_yet_available=not_yet)
        c = contents.candles
        rep.notes.append(f"cache rows: {len(c)}" + (
            f"; first {c['timestamp'].iloc[0]:%Y-%m-%d %H:%M}, last {c['timestamp'].iloc[-1]:%Y-%m-%d %H:%M}"
            if len(c) else ""))
        return rep

    # ----------------------------------------------------------- download
    def ensure_data(self, instrument: str, token: int, interval: str, start: date, end: date,
                    progress: Callable[[str], None] = lambda m: None,
                    cancel: Optional[CancelToken] = None) -> DataReport:
        """Download ONLY the missing parts of [start, end] and merge them into the cache."""
        if self.source is None:
            raise KiteAPIError("No historical data source is configured.")
        if interval not in MAX_DAYS_PER_REQUEST:
            raise ValueError(f"Unsupported base interval '{interval}'.")
        contents = self.cache.load(instrument, interval)
        cached_token = contents.meta.get("token")
        if cached_token not in (None, "", "nan") and str(cached_token) != str(token):
            raise CacheError(f"Cache for {instrument} was built with token {cached_token}, but token {token} was "
                             "supplied. Refusing to mix data; move the old cache file if the token legitimately changed.")
        contents.meta.update({"instrument": instrument, "interval": interval, "token": str(token)})
        contents.meta.setdefault("created_at", datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"))

        cached, missing, not_yet = self._missing(contents, (start, end))
        rep = DataReport(instrument=instrument, interval=interval, requested=(start, end), cached=cached,
                         not_yet_available=not_yet)
        if not missing:
            progress("Nothing to download: requested range is already cached.")
            return rep

        chunks = [c for r in missing for c in chunk_range(r, MAX_DAYS_PER_REQUEST[interval])]
        progress(f"{len(chunks)} request(s) needed for {len(missing)} missing range(s).")
        consecutive_failures = 0
        for i, (cs, ce) in enumerate(chunks, 1):
            if cancel and cancel.cancelled:
                rep.cancelled = True
                break
            progress(f"[{i}/{len(chunks)}] downloading {fmt((cs, ce))} ...")
            try:
                raw = self.source.fetch_candles(token, cs, ce, interval)
            except (AuthenticationError, InvalidInstrumentError) as exc:
                rep.aborted = str(exc)
                contents.add_log("ERROR", f"aborted: {exc}")
                self._try_save(instrument, interval, contents, rep)
                break
            except KiteAPIError as exc:
                rep.failed.append(((cs, ce), str(exc)))
                contents.add_log("ERROR", f"{fmt((cs, ce))}: {exc}")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    rep.aborted = f"{MAX_CONSECUTIVE_FAILURES} consecutive request failures; stopping."
                    self._try_save(instrument, interval, contents, rep)
                    break
                continue
            consecutive_failures = 0

            if raw is None or raw.empty:
                if weekday_count((cs, ce)) >= SUSPICIOUS_EMPTY_WEEKDAYS:
                    why = ("Kite returned no candles for a range with several weekdays; NOT treated as holidays. "
                           "Possible causes: instrument inactive, outside Kite's history, or no Historical API access.")
                    rep.failed.append(((cs, ce), why))
                    contents.add_log("WARN", f"{fmt((cs, ce))}: empty response")
                else:
                    contents.add_coverage(cs, ce, "no_data", 0, "empty response; treated as non-trading days")
                    contents.add_log("INFO", f"{fmt((cs, ce))}: no data (likely holiday)")
                    rep.downloaded.append((cs, ce))
                    self._try_save(instrument, interval, contents, rep)
                continue

            clean, vrep = validate_candles(raw, interval)
            ts_dates = clean["timestamp"].dt.date
            outside = ~((ts_dates >= cs) & (ts_dates <= ce))
            if outside.any():
                vrep.invalid["outside_requested_range"] = int(outside.sum())
                clean = clean[~outside].reset_index(drop=True)
            for reason, n in vrep.invalid.items():
                rep.invalid[reason] = rep.invalid.get(reason, 0) + n
            rep.duplicates_removed += vrep.duplicates_removed
            if clean.empty:
                rep.failed.append(((cs, ce), f"all {vrep.input_rows} returned rows were invalid {vrep.invalid}"))
                contents.add_log("ERROR", f"{fmt((cs, ce))}: all rows invalid {vrep.invalid}")
                continue
            merged, overlaps, conflicts = merge_candles(contents.candles, clean)
            rep.overlap_conflicts += conflicts
            rep.rows_added += len(merged) - len(contents.candles)
            contents.candles = merged
            note = f"invalid rows dropped: {vrep.invalid}" if vrep.invalid else ""
            contents.add_coverage(cs, ce, "downloaded", len(clean), note)
            contents.add_log("INFO", f"{fmt((cs, ce))}: +{len(clean)} rows {note}")
            if self._try_save(instrument, interval, contents, rep):
                rep.downloaded.append((cs, ce))
        # Anything not successfully saved/covered remains in 'missing'.  If a save failed the in-memory
        # state is ahead of the file, so re-read the file; otherwise memory and file agree.
        path_exists = self.cache.path_for(instrument, interval).exists()
        after = self.cache.load(instrument, interval) if (rep.aborted and path_exists) else contents
        if rep.aborted and not path_exists:
            after = CacheContents()
        _, rep.missing, _ = self._missing(after, (start, end))
        return rep

    def _try_save(self, instrument: str, interval: str, contents: CacheContents, rep: DataReport) -> bool:
        try:
            self.cache.save(instrument, interval, contents)
            return True
        except Exception as exc:  # noqa: BLE001  (disk full, file open in Excel, ...)
            rep.aborted = f"Could not save cache ({type(exc).__name__}: {exc}). Previous cache left intact."
            return False

    # ------------------------------------------------------------- loading
    def load_candles(self, instrument: str, timeframe: str, start: date, end: date) -> LoadedData:
        """Read candles for [start, end] from the cache (resampling 1-minute data if needed)."""
        base = base_interval_for(timeframe)
        contents = self.cache.load(instrument, base)
        raw = contents.candles
        d = raw["timestamp"].dt.date if len(raw) else pd.Series([], dtype=object)
        sel = raw[(d >= start) & (d <= end)] if len(raw) else raw
        sel = sel.reset_index(drop=True)
        cached, missing, not_yet = self._missing(contents, (start, end))
        info = {"instrument": instrument, "timeframe": timeframe, "base_interval": base,
                "source_file": str(self.cache.path_for(instrument, base).name),
                "requested": (start, end), "covered_ranges": contents.covered_ranges(), "missing_ranges": missing,
                "not_yet_available": not_yet, "raw_rows": len(sel), "resample": None, "warnings": []}
        # Re-validate on load (cheap; catches hand-edited workbooks).
        sel, vrep = validate_candles(sel, base)
        info.update(duplicates_removed=vrep.duplicates_removed, invalid_rows=dict(vrep.invalid))
        if vrep.invalid_total or vrep.duplicates_removed:
            info["warnings"].append(f"cache contained invalid/duplicate rows that were ignored: {vrep.invalid}, "
                                    f"duplicates={vrep.duplicates_removed}")
        # Weekdays inside covered ranges without any candle (likely exchange holidays).
        have = set(sel["timestamp"].dt.date) if len(sel) else set()
        nodata = []
        for cs, ce in intersect_ranges((start, end), contents.covered_ranges()):
            day = cs
            while day <= ce:
                if day.weekday() < 5 and day not in have:
                    nodata.append(day)
                day += timedelta(days=1)
        info["no_data_weekdays"] = nodata
        if base == "minute" and len(sel):
            counts = sel.groupby(sel["timestamp"].dt.date).size()
            info["partial_sessions"] = {str(k): int(v) for k, v in counts.items() if v < 375}
        else:
            info["partial_sessions"] = {}
        if base == "minute":
            minutes = timeframe_minutes(timeframe)
            out, rinfo = resample_intraday(sel, minutes, self.settings.resample_require_full_bucket)
            info["resample"] = rinfo
            if rinfo["incomplete_dropped"]:
                info["warnings"].append(f"{rinfo['incomplete_dropped']} incomplete {minutes}-minute candle(s) dropped")
        else:
            out = sel
        if missing:
            info["warnings"].append("requested period has uncovered date ranges: " + ", ".join(fmt(r) for r in missing))
        info["rows"] = len(out)
        return LoadedData(out.reset_index(drop=True) if len(out) else empty_candles(), info)
