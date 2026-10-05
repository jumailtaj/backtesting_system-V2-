"""Global application settings.

Everything a user may want to change without touching the engine lives here
or in ``config/strategies/*.json``.  Settings are plain dataclasses; the CLI
saves your edits to ``config/user_settings.json`` (git-ignored).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
STRATEGY_CONFIG_DIR = CONFIG_DIR / "strategies"
USER_SETTINGS_PATH = CONFIG_DIR / "user_settings.json"
SECTORS_PATH = CONFIG_DIR / "sectors.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
ENV_PATH = PROJECT_ROOT / ".env"

TIMEZONE = "Asia/Kolkata"
IST = ZoneInfo(TIMEZONE)
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)
SESSION_MINUTES = 375  # 09:15 -> 15:30
SUPPORTED_INTRADAY_MINUTES = (1, 3, 5, 10, 15, 30)

INSTRUMENT_RE = re.compile(r"^[A-Z]{2,5}:[A-Za-z0-9&_\-]+$")


def now_ist() -> datetime:
    return datetime.now(IST)


def today_ist() -> date:
    return now_ist().date()


def default_backtest_period(today: Optional[date] = None) -> tuple[date, date]:
    """The previous six *full calendar months* in market-local (IST) dates.

    Example: on 2026-10-04 this returns 2026-04-01 .. 2026-09-30.
    """
    today = today or today_ist()
    first_of_this_month = today.replace(day=1)
    end = first_of_this_month - timedelta(days=1)
    year, month = first_of_this_month.year, first_of_this_month.month - 6
    while month <= 0:
        month += 12
        year -= 1
    return date(year, month, 1), end


def parse_hhmm(value: str) -> time:
    try:
        hh, mm = value.strip().split(":")
        return time(int(hh), int(mm))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid time '{value}'. Use HH:MM, e.g. 09:30.") from exc


def minutes_of_day(value: str) -> int:
    t = parse_hhmm(value)
    return t.hour * 60 + t.minute


# --------------------------------------------------------------------------
# Transaction costs.  Defaults model a Zerodha-style NSE equity schedule.
# THESE RATES CHANGE: verify them against your broker's current charge sheet
# and edit them in config/user_settings.json ("intraday_costs"/"delivery_costs").
# --------------------------------------------------------------------------
def default_intraday_costs() -> dict:
    return {
        "brokerage_pct": 0.03,       # % of turnover per executed order ...
        "brokerage_cap": 20.0,       # ... capped at this many rupees per order
        "stt_buy_pct": 0.0,
        "stt_sell_pct": 0.025,
        "exchange_txn_pct": 0.00297,
        "sebi_pct": 0.0001,
        "stamp_buy_pct": 0.003,
        "gst_pct": 18.0,             # on brokerage + exchange + SEBI charges
        "dp_charge_per_sell": 0.0,
    }


def default_delivery_costs() -> dict:
    return {
        "brokerage_pct": 0.0,
        "brokerage_cap": 0.0,
        "stt_buy_pct": 0.1,
        "stt_sell_pct": 0.1,
        "exchange_txn_pct": 0.00297,
        "sebi_pct": 0.0001,
        "stamp_buy_pct": 0.015,
        "gst_pct": 18.0,
        "dp_charge_per_sell": 15.93,  # depository charge incl. GST, per sell
    }


@dataclass
class AppSettings:
    instruments: list = field(default_factory=lambda: ["NSE:RELIANCE"])
    start_date: Optional[str] = None      # ISO date; None -> previous six calendar months
    end_date: Optional[str] = None
    intraday_minutes: int = 5
    swing_interval: str = "day"
    initial_capital: float = 500000.0
    slippage_pct: float = 0.02            # adverse slippage per market fill, in %
    tick_size: float = 0.05               # 0 disables price rounding
    cost_multiplier: float = 1.0
    intraday_costs: dict = field(default_factory=default_intraday_costs)
    delivery_costs: dict = field(default_factory=default_delivery_costs)
    intraday_warmup_calendar_days: int = 21
    swing_warmup_calendar_days: int = 180
    resample_require_full_bucket: bool = True
    in_sample_fraction: float = 0.7
    min_trades_warning: int = 30
    active_strategies: list = field(
        default_factory=lambda: ["ema_crossover", "orb", "swing_breakout", "swing_sr"]
    )

    # -- period ---------------------------------------------------------
    def period(self, today: Optional[date] = None) -> tuple[date, date]:
        d_start, d_end = default_backtest_period(today)
        start = date.fromisoformat(self.start_date) if self.start_date else d_start
        end = date.fromisoformat(self.end_date) if self.end_date else d_end
        return start, end

    # -- validation -----------------------------------------------------
    def validate(self) -> None:
        if not self.instruments:
            raise ValueError("At least one instrument is required.")
        for inst in self.instruments:
            if not INSTRUMENT_RE.match(inst):
                raise ValueError(f"Instrument '{inst}' must look like EXCHANGE:SYMBOL, e.g. NSE:RELIANCE.")
        start, end = self.period()
        if start > end:
            raise ValueError(f"Start date {start} is after end date {end}.")
        if self.intraday_minutes not in SUPPORTED_INTRADAY_MINUTES:
            raise ValueError(f"intraday_minutes must be one of {SUPPORTED_INTRADAY_MINUTES}.")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if self.slippage_pct < 0 or self.cost_multiplier < 0 or self.tick_size < 0:
            raise ValueError("slippage_pct, cost_multiplier and tick_size cannot be negative.")
        if not 0.1 <= self.in_sample_fraction <= 0.9:
            raise ValueError("in_sample_fraction must be between 0.1 and 0.9.")

    # -- persistence ----------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path = USER_SETTINGS_PATH) -> None:
        self.validate()
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "AppSettings":
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"Unknown setting(s): {sorted(unknown)}. Valid: {sorted(valid)}")
        base = cls()
        merged = base.to_dict()
        for key, value in data.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        settings = cls(**merged)
        settings.validate()
        return settings

    @classmethod
    def load(cls, path: Path = USER_SETTINGS_PATH) -> "AppSettings":
        if not path.exists():
            settings = cls()
            settings.validate()
            return settings
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        return cls.from_dict(data)


def load_sector_map(path: Path = SECTORS_PATH) -> dict:
    """Optional ``config/sectors.csv`` (columns: instrument,sector).

    Kite's instrument master has no sector field, so the sector limit is only
    enforced when *you* provide a mapping.  Returns {} when the file is absent.
    """
    if not path.exists():
        return {}
    import pandas as pd

    df = pd.read_csv(path)
    if not {"instrument", "sector"} <= set(df.columns):
        raise ValueError(f"{path} must have the columns: instrument,sector")
    return {str(r.instrument).strip(): str(r.sector).strip() for r in df.itertuples()}
