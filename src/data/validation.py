"""Candle validation.  Invalid rows are REMOVED and counted, never repaired."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import IST, SESSION_MINUTES

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
SESSION_START_MIN = 9 * 60 + 15


@dataclass
class ValidationReport:
    input_rows: int = 0
    accepted_rows: int = 0
    duplicates_removed: int = 0
    invalid: dict = field(default_factory=dict)       # reason -> row count
    warnings: list = field(default_factory=list)

    @property
    def invalid_total(self) -> int:
        return int(sum(self.invalid.values()))


def empty_candles() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.Series(dtype=f"datetime64[ns, {IST.key}]"),
        **{c: pd.Series(dtype="float64") for c in REQUIRED_COLUMNS[1:]},
    })


def normalise_timestamps(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if ts.dt.tz is None:
        return ts.dt.tz_localize(IST)
    return ts.dt.tz_convert(IST)


def validate_candles(df: pd.DataFrame, interval: str) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate raw candles for a base interval (``minute`` or ``day``).

    Checks: required columns, timestamps, missing values, non-positive prices,
    OHLC consistency, negative volume, out-of-session minute candles,
    duplicate timestamps and ordering.
    """
    rep = ValidationReport(input_rows=len(df))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Candle data is missing required columns: {missing}")
    if df.empty:
        return empty_candles(), rep

    work = df[REQUIRED_COLUMNS].copy()
    work["timestamp"] = normalise_timestamps(work["timestamp"])
    for col in REQUIRED_COLUMNS[1:]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    def drop(mask: pd.Series, reason: str) -> None:
        nonlocal work
        n = int(mask.sum())
        if n:
            rep.invalid[reason] = rep.invalid.get(reason, 0) + n
            work = work.loc[~mask]

    drop(work["timestamp"].isna(), "bad_timestamp")
    drop(work[REQUIRED_COLUMNS].isna().any(axis=1), "missing_values")
    drop(~np.isfinite(work[REQUIRED_COLUMNS[1:]].to_numpy(dtype=float)).all(axis=1), "non_finite")
    drop((work[["open", "high", "low", "close"]] <= 0).any(axis=1), "non_positive_price")
    o, h, l, c = (work[k] for k in ("open", "high", "low", "close"))
    drop((h < np.maximum.reduce([o, c, l])) | (l > np.minimum.reduce([o, c, h])), "ohlc_inconsistent")
    drop(work["volume"] < 0, "negative_volume")
    if interval == "minute":
        tmin = work["timestamp"].dt.hour * 60 + work["timestamp"].dt.minute
        drop((tmin < SESSION_START_MIN) | (tmin >= SESSION_START_MIN + SESSION_MINUTES), "out_of_session")

    n_before = len(work)
    work = work.drop_duplicates(subset="timestamp", keep="first")
    rep.duplicates_removed = n_before - len(work)
    if not work["timestamp"].is_monotonic_increasing:
        rep.warnings.append("timestamps were not sorted; sorted ascending")
        work = work.sort_values("timestamp")
    work = work.reset_index(drop=True)
    rep.accepted_rows = len(work)
    return work, rep


def check_warmup(candles: pd.DataFrame, trading_start, min_bars: int) -> str | None:
    """Return a warning if fewer than ``min_bars`` candles precede ``trading_start``."""
    if candles.empty:
        return "no candles available"
    start_ts = pd.Timestamp(trading_start).tz_localize(IST) if pd.Timestamp(trading_start).tzinfo is None \
        else pd.Timestamp(trading_start)
    n = int((candles["timestamp"] < start_ts).sum())
    if n < min_bars:
        return (f"only {n} warm-up candles precede the start date (strategy wants ~{min_bars}); "
                "early signals are suppressed (indicators stay NaN) but results may differ from a fully warmed-up run")
    return None
