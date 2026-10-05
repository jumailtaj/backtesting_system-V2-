"""Kite Connect access: authentication, instrument master, historical candles.

Only the official ``kiteconnect`` client is used, and ONLY read-only calls are
made (profile, instruments, historical_data, login URL, session creation).
There is deliberately no order-placement code anywhere in this project.

UNVERIFIED AGAINST THE LIVE API: this module was written from Kite's public
documentation and tested only with mocks.  See README "Known limitations".
"""
from __future__ import annotations

import logging
import os
import time as time_module
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from dotenv import load_dotenv

from config.settings import ENV_PATH, IST

try:  # the package is a hard requirement, but keep import errors readable
    from kiteconnect import KiteConnect
    from kiteconnect import exceptions as kex
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install dependencies first:  pip install -r requirements.txt") from exc

log = logging.getLogger(__name__)

CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# Documented maximum span of one historical_data request (calendar days).
MAX_DAYS_PER_REQUEST = {"minute": 60, "day": 2000}
KITE_INTERVAL = {"minute": "minute", "day": "day"}


# ----------------------------------------------------------------- errors
class KiteAPIError(Exception):
    """Base class for all errors raised by this module (messages are redacted)."""


class AuthenticationError(KiteAPIError):
    """Missing/invalid/expired credentials, or the account lacks API permission."""


class InvalidInstrumentError(KiteAPIError):
    """Instrument or request parameters rejected by Kite."""


class RateLimitError(KiteAPIError):
    """Kite rate limit hit and retries exhausted."""


class NetworkError(KiteAPIError):
    """Connectivity / server problem and retries exhausted."""


# ----------------------------------------------------------------- secrets
SECRET_ENV_VARS = ("KITE_API_KEY", "KITE_API_SECRET", "KITE_ACCESS_TOKEN")


def redact(text: str, extra: tuple = ()) -> str:
    """Replace any known secret value inside ``text`` with ``***``."""
    out = str(text)
    candidates = [os.environ.get(v, "") for v in SECRET_ENV_VARS] + list(extra)
    for secret in candidates:
        if secret and len(secret) >= 6:
            out = out.replace(secret, "***")
    return out


def _mask(value: str) -> str:
    return "(not set)" if not value else f"set ({len(value)} chars)"


@dataclass
class InstrumentInfo:
    instrument: str            # "NSE:RELIANCE"
    exchange: str
    tradingsymbol: str
    token: int
    name: str
    tick_size: float
    lot_size: int
    instrument_type: str
    segment: str


class KiteSession:
    """Thin, defensive wrapper around :class:`kiteconnect.KiteConnect`."""

    def __init__(
        self,
        kite_factory: Callable = KiteConnect,
        env_path: Path = ENV_PATH,
        sleep: Callable[[float], None] = time_module.sleep,
        min_request_interval: float = 0.4,   # Kite allows ~3 historical requests/second
        max_retries: int = 4,
    ) -> None:
        load_dotenv(env_path, override=False)
        self._factory = kite_factory
        self._env_path = Path(env_path)
        self._sleep = sleep
        self._min_interval = min_request_interval
        self._max_retries = max_retries
        self._last_request = 0.0
        self._kite = None

    # ------------------------------------------------------------ config
    @property
    def api_key(self) -> str:
        return os.environ.get("KITE_API_KEY", "").strip()

    @property
    def api_secret(self) -> str:
        return os.environ.get("KITE_API_SECRET", "").strip()

    @property
    def access_token(self) -> str:
        return os.environ.get("KITE_ACCESS_TOKEN", "").strip()

    def credential_status(self) -> dict:
        """Safe-to-print status; never includes secret values."""
        return {
            "KITE_API_KEY": _mask(self.api_key),
            "KITE_API_SECRET": _mask(self.api_secret),
            "KITE_ACCESS_TOKEN": _mask(self.access_token),
        }

    def _client(self, need_token: bool = True):
        if not self.api_key:
            raise AuthenticationError("KITE_API_KEY is not set. Copy .env.example to .env and fill it in.")
        if self._kite is None:
            self._kite = self._factory(api_key=self.api_key)
        if need_token:
            if not self.access_token:
                raise AuthenticationError(
                    "KITE_ACCESS_TOKEN is not set. Use the CLI's login option (or see README 'Daily login')."
                )
            self._kite.set_access_token(self.access_token)
        return self._kite

    # -------------------------------------------------------------- login
    def login_url(self) -> str:
        return self._client(need_token=False).login_url()

    def create_session(self, request_token: str) -> None:
        """Exchange the one-time ``request_token`` for an access token.

        The token is stored in this process' environment (and NOT printed).
        Call :meth:`save_access_token` to persist it into ``.env``.
        """
        if not self.api_secret:
            raise AuthenticationError("KITE_API_SECRET is not set in .env.")
        request_token = request_token.strip()
        if not request_token:
            raise AuthenticationError("request_token is empty.")
        try:
            data = self._client(need_token=False).generate_session(request_token, api_secret=self.api_secret)
        except kex.TokenException as exc:
            raise AuthenticationError(
                redact(f"Login failed ({exc}). A request_token is single-use and expires within minutes; "
                       "get a fresh one.", (request_token,))
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise self._map_error(exc, (request_token,)) from exc
        token = data.get("access_token") if isinstance(data, dict) else None
        if not token:
            raise AuthenticationError("Kite did not return an access token.")
        os.environ["KITE_ACCESS_TOKEN"] = token
        if self._kite is not None:
            self._kite.set_access_token(token)

    def save_access_token(self) -> None:
        """Write the current access token into ``.env`` (replacing any old one)."""
        token = self.access_token
        if not token:
            raise AuthenticationError("No access token to save.")
        lines = self._env_path.read_text(encoding="utf-8").splitlines() if self._env_path.exists() else []
        out, replaced = [], False
        for line in lines:
            if line.strip().startswith("KITE_ACCESS_TOKEN="):
                out.append(f"KITE_ACCESS_TOKEN={token}")
                replaced = True
            else:
                out.append(line)
        if not replaced:
            out.append(f"KITE_ACCESS_TOKEN={token}")
        tmp = self._env_path.with_name(self._env_path.name + ".tmp")
        tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:  # pragma: no cover (Windows)
            pass
        os.replace(tmp, self._env_path)

    def check_connection(self) -> dict:
        """Call ``profile()`` to prove the token works. Returns non-secret fields."""
        try:
            profile = self._client().profile()
        except Exception as exc:  # noqa: BLE001
            raise self._map_error(exc) from exc
        return {"user_name": profile.get("user_name"), "broker": profile.get("broker"),
                "user_type": profile.get("user_type")}

    # --------------------------------------------------------- error map
    def _map_error(self, exc: Exception, extra: tuple = ()) -> KiteAPIError:
        if isinstance(exc, KiteAPIError):
            return exc
        msg = redact(f"{type(exc).__name__}: {exc}", extra)
        if isinstance(exc, kex.TokenException):
            return AuthenticationError(
                f"{msg} -- your access token is missing, invalid or expired (tokens expire daily). "
                "Log in again to get a new one.")
        if isinstance(exc, kex.PermissionException):
            return AuthenticationError(
                f"{msg} -- your Kite app/account lacks permission for this call "
                "(historical data needs the paid Historical API add-on).")
        if isinstance(exc, kex.InputException):
            return InvalidInstrumentError(f"{msg} -- check the instrument token, interval and date range.")
        if "too many requests" in msg.lower() or "429" in msg:
            return RateLimitError(msg)
        return NetworkError(msg)

    # ------------------------------------------------------ instruments
    def instruments(self, exchange: str = "NSE") -> pd.DataFrame:
        try:
            rows = self._client().instruments(exchange)
        except Exception as exc:  # noqa: BLE001
            raise self._map_error(exc) from exc
        df = pd.DataFrame(rows)
        if df.empty:
            raise KiteAPIError(f"Kite returned an empty instrument master for {exchange}.")
        return df

    def search_instruments(self, query: str, exchange: str = "NSE", limit: int = 20) -> pd.DataFrame:
        df = self.instruments(exchange)
        q = query.strip().upper()
        if not q:
            return df.head(0)
        names = df["name"] if "name" in df.columns else pd.Series("", index=df.index)
        mask = df["tradingsymbol"].astype(str).str.upper().str.contains(q, regex=False) | \
            names.astype(str).str.upper().str.contains(q, regex=False)
        cols = [c for c in ("exchange", "tradingsymbol", "name", "instrument_token", "instrument_type",
                            "segment", "tick_size", "lot_size") if c in df.columns]
        return df.loc[mask, cols].head(limit).reset_index(drop=True)

    def verify_instrument(self, instrument: str) -> InstrumentInfo:
        """Look ``EXCHANGE:SYMBOL`` up in Kite's official instrument master.

        The token is NEVER assumed: it is read from the master on every
        verification.  Cash equities (instrument_type == EQ) only.
        """
        if ":" not in instrument:
            raise InvalidInstrumentError("Use EXCHANGE:SYMBOL, e.g. NSE:RELIANCE.")
        exchange, symbol = instrument.split(":", 1)
        df = self.instruments(exchange)
        match = df[(df["tradingsymbol"].astype(str) == symbol) & (df["exchange"].astype(str) == exchange)]
        if "instrument_type" in match.columns:
            match = match[match["instrument_type"].astype(str) == "EQ"]
        if match.empty:
            raise InvalidInstrumentError(
                f"{instrument} was not found as an EQ instrument in Kite's {exchange} instrument master.")
        row = match.iloc[0]
        return InstrumentInfo(
            instrument=instrument, exchange=exchange, tradingsymbol=symbol,
            token=int(row["instrument_token"]), name=str(row.get("name", "") or ""),
            tick_size=float(row.get("tick_size", 0.05) or 0.05), lot_size=int(row.get("lot_size", 1) or 1),
            instrument_type=str(row.get("instrument_type", "")), segment=str(row.get("segment", "")),
        )

    # ------------------------------------------------------- historical
    def fetch_candles(self, token: int, start: date, end: date, interval: str) -> pd.DataFrame:
        """Fetch ONE request's worth of candles (caller must respect MAX_DAYS_PER_REQUEST).

        Returns a DataFrame with CANDLE_COLUMNS and tz-aware IST timestamps.
        An empty frame means Kite returned no rows (not an error by itself).
        """
        if interval not in KITE_INTERVAL:
            raise InvalidInstrumentError(f"Unsupported download interval '{interval}'.")
        span = (end - start).days + 1
        if span > MAX_DAYS_PER_REQUEST[interval]:
            raise InvalidInstrumentError(
                f"{span} days exceeds Kite's {MAX_DAYS_PER_REQUEST[interval]}-day limit for '{interval}' candles.")
        if interval == "minute":
            from_dt = datetime.combine(start, time(9, 0))
            to_dt = datetime.combine(end, time(15, 30))
        else:
            from_dt = datetime.combine(start, time(0, 0))
            to_dt = datetime.combine(end, time(23, 59, 59))

        last_error: Optional[KiteAPIError] = None
        for attempt in range(self._max_retries):
            wait = self._min_interval - (time_module.monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            self._last_request = time_module.monotonic()
            try:
                rows = self._client().historical_data(int(token), from_dt, to_dt, KITE_INTERVAL[interval])
                return self._rows_to_frame(rows)
            except Exception as exc:  # noqa: BLE001
                mapped = self._map_error(exc)
                if isinstance(mapped, (AuthenticationError, InvalidInstrumentError)):
                    raise mapped from exc
                last_error = mapped
                log.warning("Historical request failed (attempt %d/%d): %s", attempt + 1, self._max_retries, mapped)
                self._sleep(min(2 ** attempt, 8))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _rows_to_frame(rows) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=CANDLE_COLUMNS)
        df = pd.DataFrame(rows)
        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        missing = [c for c in CANDLE_COLUMNS if c not in df.columns]
        if missing:
            raise KiteAPIError(f"Kite response is missing columns {missing}.")
        ts = pd.to_datetime(df["timestamp"])
        ts = ts.dt.tz_localize(IST) if ts.dt.tz is None else ts.dt.tz_convert(IST)
        df["timestamp"] = ts
        return df[CANDLE_COLUMNS]
