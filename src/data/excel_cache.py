"""Excel OHLCV cache: one workbook per instrument + base interval.

File name:   ``<EXCHANGE>_<SYMBOL>_<interval>.xlsx``   e.g. ``NSE_RELIANCE_minute.xlsx``
Sheets:
    Candles   timestamp (naive IST), open, high, low, close, volume
    Coverage  date ranges that were successfully fetched (including ranges with
              no data, such as holidays) -- this is what "cached" means
    Meta      instrument, token, interval, timezone, created/updated times
    Log       the most recent download events and warnings

Saving is atomic: data is written to a temporary workbook, re-read to verify
it, and only then moved over the real file with ``os.replace``.  A failure at
any step leaves the previous cache untouched.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd

from config.settings import CACHE_DIR, IST, TIMEZONE
from src.data.ranges import Range, merge_ranges
from src.data.validation import REQUIRED_COLUMNS, empty_candles, normalise_timestamps

COVERAGE_COLUMNS = ["start", "end", "status", "rows", "downloaded_at", "note"]
MAX_LOG_ROWS = 200


class CacheError(Exception):
    """The cache file is unreadable or inconsistent.  It is never auto-deleted."""


@dataclass
class CacheContents:
    candles: pd.DataFrame = field(default_factory=empty_candles)
    coverage: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=COVERAGE_COLUMNS))
    meta: dict = field(default_factory=dict)
    log: list = field(default_factory=list)

    def covered_ranges(self) -> list[Range]:
        rows = [(date.fromisoformat(str(r.start)[:10]), date.fromisoformat(str(r.end)[:10]))
                for r in self.coverage.itertuples()]
        return merge_ranges(rows)

    def add_coverage(self, start: date, end: date, status: str, rows: int, note: str = "") -> None:
        row = {"start": start.isoformat(), "end": end.isoformat(), "status": status, "rows": int(rows),
               "downloaded_at": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"), "note": note}
        self.coverage = pd.concat([self.coverage, pd.DataFrame([row])], ignore_index=True)

    def add_log(self, level: str, message: str) -> None:
        self.log.append({"time": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"), "level": level,
                         "message": message})
        self.log = self.log[-MAX_LOG_ROWS:]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", text)


def merge_candles(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Merge candle frames. Existing rows win on duplicate timestamps.

    Returns (merged, overlapping_timestamps, conflicting_overlaps) where a conflict is an
    overlap whose OHLCV values differ from the existing row.
    """
    if old.empty:
        return new.sort_values("timestamp").reset_index(drop=True), 0, 0
    if new.empty:
        return old, 0, 0
    overlap = new[new["timestamp"].isin(old["timestamp"])]
    conflicts = 0
    if not overlap.empty:
        a = old.set_index("timestamp").loc[overlap["timestamp"], REQUIRED_COLUMNS[1:]].to_numpy(dtype=float)
        b = overlap[REQUIRED_COLUMNS[1:]].to_numpy(dtype=float)
        conflicts = int((abs(a - b) > 1e-9).any(axis=1).sum())
    merged = pd.concat([old, new], ignore_index=True).drop_duplicates(subset="timestamp", keep="first")
    return merged.sort_values("timestamp").reset_index(drop=True), len(overlap), conflicts


class ExcelCandleCache:
    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)

    def path_for(self, instrument: str, interval: str) -> Path:
        exchange, _, symbol = instrument.partition(":")
        return self.cache_dir / f"{safe_name(exchange)}_{safe_name(symbol)}_{interval}.xlsx"

    # ------------------------------------------------------------ load
    def load(self, instrument: str, interval: str) -> CacheContents:
        path = self.path_for(instrument, interval)
        if not path.exists():
            return CacheContents(meta={"instrument": instrument, "interval": interval})
        try:
            sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        except (BadZipFile, OSError, ValueError, KeyError) as exc:
            raise CacheError(
                f"Cache file {path.name} could not be read ({type(exc).__name__}: {exc}). "
                "It has NOT been modified. Move or repair it, then retry.") from exc
        if "Candles" not in sheets or "Coverage" not in sheets:
            raise CacheError(f"Cache file {path.name} is missing its Candles/Coverage sheet. "
                             "It has NOT been modified.")
        candles = sheets["Candles"]
        if candles.empty:
            candles = empty_candles()
        else:
            missing = [c for c in REQUIRED_COLUMNS if c not in candles.columns]
            if missing:
                raise CacheError(f"Cache file {path.name} lacks columns {missing}. It has NOT been modified.")
            candles = candles[REQUIRED_COLUMNS].copy()
            candles["timestamp"] = normalise_timestamps(candles["timestamp"])
            candles = candles.sort_values("timestamp").reset_index(drop=True)
        coverage = sheets["Coverage"]
        for col in COVERAGE_COLUMNS:
            if col not in coverage.columns:
                coverage[col] = ""
        meta_df = sheets.get("Meta")
        meta = {}
        if meta_df is not None and not meta_df.empty and {"key", "value"} <= set(meta_df.columns):
            meta = {str(k): v for k, v in zip(meta_df["key"], meta_df["value"])}
        log_df = sheets.get("Log")
        log = log_df.fillna("").to_dict("records") if log_df is not None else []
        return CacheContents(candles, coverage[COVERAGE_COLUMNS].copy(), meta, log)

    # ------------------------------------------------------------ save
    def _write_workbook(self, tmp: Path, contents: CacheContents) -> None:
        candles = contents.candles.copy()
        candles["timestamp"] = candles["timestamp"].dt.tz_convert(IST).dt.tz_localize(None)
        meta = dict(contents.meta)
        meta.update({"timezone": TIMEZONE, "updated_at": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                     "rows": len(candles), "source": "Kite Connect historical_data (via this tool)"})
        meta_df = pd.DataFrame({"key": list(meta.keys()), "value": [str(v) for v in meta.values()]})
        log_df = pd.DataFrame(contents.log, columns=["time", "level", "message"])
        with pd.ExcelWriter(tmp, engine="openpyxl", datetime_format="yyyy-mm-dd hh:mm:ss") as writer:
            candles.to_excel(writer, sheet_name="Candles", index=False)
            contents.coverage.to_excel(writer, sheet_name="Coverage", index=False)
            meta_df.to_excel(writer, sheet_name="Meta", index=False)
            log_df.to_excel(writer, sheet_name="Log", index=False)
            ws = writer.sheets["Candles"]
            ws.column_dimensions["A"].width = 20
            ws.freeze_panes = "A2"

    def save(self, instrument: str, interval: str, contents: CacheContents) -> Path:
        path = self.path_for(instrument, interval)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name("~tmp_" + path.name)
        try:
            self._write_workbook(tmp, contents)
            check = pd.read_excel(tmp, sheet_name="Candles", engine="openpyxl")
            if len(check) != len(contents.candles):
                raise CacheError("Verification of the temporary workbook failed (row count mismatch).")
            os.replace(tmp, path)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            raise
        return path
