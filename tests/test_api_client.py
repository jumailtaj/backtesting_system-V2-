import os
from datetime import date

import pandas as pd
import pytest
from kiteconnect import exceptions as kex

from src.api import kite_client as kc
from src.api.kite_client import (AuthenticationError, InvalidInstrumentError, KiteSession, NetworkError,
                                 RateLimitError, redact)

SECRET = "SUPERSECRETTOKEN123"


class FakeKite:
    script = []          # list of exceptions / results consumed by historical_data
    master = []
    instances = []

    def __init__(self, api_key):
        self.api_key, self.token, self.hist_calls = api_key, None, 0
        FakeKite.instances.append(self)

    def set_access_token(self, t):
        self.token = t

    def login_url(self):
        return "https://kite.example/connect/login?api_key=" + self.api_key

    def generate_session(self, request_token, api_secret):
        if request_token == "bad":
            raise kex.TokenException("Token is invalid or has expired")
        return {"access_token": "NEWTOKEN_ABCDEF"}

    def profile(self):
        return {"user_name": "Test User", "broker": "ZERODHA", "user_type": "individual", "email": "x@y.z"}

    def instruments(self, exchange):
        return self.master

    def historical_data(self, token, from_dt, to_dt, interval):
        FakeKite.last = (token, from_dt, to_dt, interval)
        self.hist_calls += 1
        item = FakeKite.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def env(monkeypatch, tmp_path):
    for k in ("KITE_API_KEY", "KITE_API_SECRET", "KITE_ACCESS_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    FakeKite.script, FakeKite.master, FakeKite.instances = [], [], []
    return tmp_path / ".env"


def session(env_path, sleeps=None, **kw):
    return KiteSession(kite_factory=FakeKite, env_path=env_path, sleep=(sleeps.append if sleeps is not None
                                                                         else (lambda s: None)), **kw)


def set_env(monkeypatch, token=SECRET):
    monkeypatch.setenv("KITE_API_KEY", "key12345")
    monkeypatch.setenv("KITE_API_SECRET", "secret12345")
    monkeypatch.setenv("KITE_ACCESS_TOKEN", token)


ROWS = [{"date": pd.Timestamp("2026-01-05 09:15:00+05:30").to_pydatetime(), "open": 1, "high": 2, "low": 0.5,
         "close": 1.5, "volume": 10}]


def test_missing_credentials_are_reported_clearly(env):
    with pytest.raises(AuthenticationError, match="KITE_API_KEY"):
        session(env).check_connection()


def test_missing_access_token(env, monkeypatch):
    monkeypatch.setenv("KITE_API_KEY", "key12345")
    with pytest.raises(AuthenticationError, match="KITE_ACCESS_TOKEN"):
        session(env).check_connection()


def test_credential_status_never_contains_secret_values(env, monkeypatch):
    set_env(monkeypatch)
    status = session(env).credential_status()
    assert SECRET not in str(status) and "secret12345" not in str(status) and "key12345" not in str(status)


def test_expired_token_mapped_and_secret_redacted(env, monkeypatch):
    set_env(monkeypatch)
    FakeKite.script = [kex.TokenException(f"Incorrect api_key or access_token {SECRET}")]
    with pytest.raises(AuthenticationError) as ei:
        session(env).fetch_candles(1, date(2026, 1, 5), date(2026, 1, 9), "minute")
    assert SECRET not in str(ei.value) and "expire" in str(ei.value).lower()


def test_permission_error_mentions_historical_addon(env, monkeypatch):
    set_env(monkeypatch)
    FakeKite.script = [kex.PermissionException("Insufficient permission")]
    with pytest.raises(AuthenticationError, match="Historical"):
        session(env).fetch_candles(1, date(2026, 1, 5), date(2026, 1, 9), "minute")


def test_rate_limit_is_retried_with_backoff_then_succeeds(env, monkeypatch):
    set_env(monkeypatch)
    FakeKite.script = [kex.NetworkException("Too many requests"), kex.NetworkException("Too many requests"), ROWS]
    sleeps = []
    df = session(env, sleeps).fetch_candles(1, date(2026, 1, 5), date(2026, 1, 9), "minute")
    assert len(df) == 1 and str(df.timestamp.dt.tz) == "Asia/Kolkata"
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert sum(1 for s in sleeps if s >= 1) >= 2           # backoff sleeps happened


def test_retries_exhausted_raise_rate_limit_and_network_errors(env, monkeypatch):
    set_env(monkeypatch)
    FakeKite.script = [kex.NetworkException("Too many requests")] * 4
    with pytest.raises(RateLimitError):
        session(env).fetch_candles(1, date(2026, 1, 5), date(2026, 1, 9), "minute")
    FakeKite.script = [ConnectionError("down")] * 4
    with pytest.raises(NetworkError):
        session(env).fetch_candles(1, date(2026, 1, 5), date(2026, 1, 9), "minute")


def test_invalid_instrument_is_not_retried(env, monkeypatch):
    set_env(monkeypatch)
    FakeKite.script = [kex.InputException("invalid token")]
    s = session(env)
    with pytest.raises(InvalidInstrumentError):
        s.fetch_candles(999, date(2026, 1, 5), date(2026, 1, 9), "minute")
    assert FakeKite.instances[-1].hist_calls == 1


def test_empty_response_returns_empty_frame_not_fabricated_data(env, monkeypatch):
    set_env(monkeypatch)
    FakeKite.script = [[]]
    df = session(env).fetch_candles(1, date(2026, 1, 5), date(2026, 1, 9), "minute")
    assert df.empty and list(df.columns) == kc.CANDLE_COLUMNS


def test_request_span_limits_enforced(env, monkeypatch):
    set_env(monkeypatch)
    with pytest.raises(InvalidInstrumentError, match="60-day"):
        session(env).fetch_candles(1, date(2026, 1, 1), date(2026, 3, 31), "minute")
    with pytest.raises(InvalidInstrumentError):
        session(env).fetch_candles(1, date(2026, 1, 1), date(2026, 1, 5), "5minute")
    assert kc.MAX_DAYS_PER_REQUEST == {"minute": 60, "day": 2000}


def test_minute_request_uses_session_hours(env, monkeypatch):
    set_env(monkeypatch)
    FakeKite.script = [ROWS]
    session(env).fetch_candles(55, date(2026, 1, 5), date(2026, 1, 9), "minute")
    token, f, t, interval = FakeKite.last
    assert token == 55 and interval == "minute" and f.hour == 9 and t.hour == 15 and t.minute == 30


def test_verify_instrument_reads_token_from_master_only(env, monkeypatch):
    set_env(monkeypatch)
    FakeKite.master = [
        {"instrument_token": 111, "tradingsymbol": "ABC", "name": "ABC LTD", "exchange": "NSE",
         "instrument_type": "EQ", "segment": "NSE", "tick_size": 0.05, "lot_size": 1},
        {"instrument_token": 222, "tradingsymbol": "ABC", "name": "ABC BE", "exchange": "NSE",
         "instrument_type": "BE", "segment": "NSE", "tick_size": 0.05, "lot_size": 1}]
    info = session(env).verify_instrument("NSE:ABC")
    assert info.token == 111 and info.tick_size == 0.05
    with pytest.raises(InvalidInstrumentError, match="not found"):
        session(env).verify_instrument("NSE:NOPE")
    with pytest.raises(InvalidInstrumentError):
        session(env).verify_instrument("ABC")
    assert len(session(env).search_instruments("abc")) == 2


def test_login_flow_and_env_persistence(env, monkeypatch):
    monkeypatch.setenv("KITE_API_KEY", "key12345")
    monkeypatch.setenv("KITE_API_SECRET", "secret12345")
    env.write_text("KITE_API_KEY=key12345\nKITE_ACCESS_TOKEN=OLD\n")
    s = session(env)
    assert "kite.example" in s.login_url()
    with pytest.raises(AuthenticationError, match="single-use"):
        s.create_session("bad")
    s.create_session("good")
    assert s.access_token == "NEWTOKEN_ABCDEF"
    s.save_access_token()
    text = env.read_text()
    assert "KITE_ACCESS_TOKEN=NEWTOKEN_ABCDEF" in text and "OLD" not in text and "KITE_API_KEY=key12345" in text
    assert s.check_connection() == {"user_name": "Test User", "broker": "ZERODHA", "user_type": "individual"}


def test_redact_helper(monkeypatch):
    monkeypatch.setenv("KITE_ACCESS_TOKEN", SECRET)
    assert SECRET not in redact(f"oops {SECRET}")


def test_no_order_placement_code_exists():
    import pathlib
    src = pathlib.Path(kc.__file__).parents[2]
    banned = ("place_order", "modify_order", "cancel_order", "place_gtt")
    for path in src.rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text()
        assert not any(b in text for b in banned), f"{path} references order placement"
