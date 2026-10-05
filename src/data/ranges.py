"""Date-range helpers for cache coverage (inclusive ``(start, end)`` date tuples)."""
from __future__ import annotations

from datetime import date, timedelta

Range = tuple[date, date]


def merge_ranges(ranges: list[Range]) -> list[Range]:
    """Merge overlapping or touching ranges (end+1 == next start)."""
    out: list[Range] = []
    for start, end in sorted(r for r in ranges if r[0] <= r[1]):
        if out and start <= out[-1][1] + timedelta(days=1):
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def subtract_ranges(requested: Range, covered: list[Range]) -> list[Range]:
    """Parts of ``requested`` NOT inside ``covered``."""
    missing: list[Range] = []
    cursor = requested[0]
    for c_start, c_end in merge_ranges(covered):
        if c_end < cursor:
            continue
        if c_start > requested[1]:
            break
        if c_start > cursor:
            missing.append((cursor, min(c_start - timedelta(days=1), requested[1])))
        cursor = max(cursor, c_end + timedelta(days=1))
        if cursor > requested[1]:
            break
    if cursor <= requested[1]:
        missing.append((cursor, requested[1]))
    return missing


def intersect_ranges(requested: Range, covered: list[Range]) -> list[Range]:
    out = []
    for c_start, c_end in merge_ranges(covered):
        s, e = max(c_start, requested[0]), min(c_end, requested[1])
        if s <= e:
            out.append((s, e))
    return out


def trim_weekends(ranges: list[Range]) -> list[Range]:
    """Drop leading/trailing Saturdays/Sundays; drop ranges that are weekend-only.

    No exchange-holiday calendar is assumed (none is fabricated); holidays are
    discovered when Kite returns no candles for a weekday.
    """
    out = []
    for start, end in ranges:
        while start <= end and start.weekday() >= 5:
            start += timedelta(days=1)
        while end >= start and end.weekday() >= 5:
            end -= timedelta(days=1)
        if start <= end:
            out.append((start, end))
    return out


def chunk_range(r: Range, max_days: int) -> list[Range]:
    out, cursor = [], r[0]
    while cursor <= r[1]:
        end = min(cursor + timedelta(days=max_days - 1), r[1])
        out.append((cursor, end))
        cursor = end + timedelta(days=1)
    return out


def weekday_count(r: Range) -> int:
    n, d = 0, r[0]
    while d <= r[1]:
        n += d.weekday() < 5
        d += timedelta(days=1)
    return n


def fmt(r: Range) -> str:
    return f"{r[0].isoformat()} -> {r[1].isoformat()}"
