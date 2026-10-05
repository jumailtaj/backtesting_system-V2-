from datetime import date

import pytest

from config.settings import AppSettings
from src.cli.app import CLI, Cancelled


def make_cli(inputs, settings=None):
    it = iter(inputs)
    out = []
    cli = CLI(settings=settings or AppSettings(), input_fn=lambda prompt: next(it), print_fn=lambda *a: out.append(" ".join(map(str, a))))
    return cli, out


def test_menu_exit_and_invalid_choice_handling():
    cli, out = make_cli(["x", "42", "0"])
    cli.run()
    text = "\n".join(out)
    assert text.count("Please enter a number") == 2 and "Goodbye" in text and "no orders" in text


def test_download_never_starts_without_confirmation(tmp_path):
    cli, out = make_cli(["5", "n", "0"])         # menu 5 -> decline the confirmation -> exit
    cli.cache.cache_dir = tmp_path
    calls = []
    cli.session.fetch_candles = lambda *a, **k: calls.append(a)
    cli.run()
    assert calls == [] and any("Cancelled" in l for l in out) and not list(tmp_path.glob("*.xlsx"))


def test_ask_date_validation_and_cancel():
    cli, out = make_cli(["not-a-date", "2026-02-30", "2026-02-01"])
    assert cli.ask_date("Start", date(2026, 1, 1)) == date(2026, 2, 1)
    cli2, _ = make_cli(["q"])
    with pytest.raises(Cancelled):
        cli2.ask("anything")


def test_settings_menu_validates_and_does_not_save_without_consent(tmp_path, monkeypatch):
    import config.settings as cs
    monkeypatch.setattr("src.cli.app.USER_SETTINGS_PATH", tmp_path / "us.json")
    cli, out = make_cli(["NSE:ABC, NSE:XYZ", "2026-01-01", "2026-01-31", "15", "250000", "0.02", "1", "n"])
    cli.menu_settings()
    assert cli.settings.instruments == ["NSE:ABC", "NSE:XYZ"] and cli.settings.intraday_minutes == 15
    assert cli.settings.initial_capital == 250000 and not (tmp_path / "us.json").exists()
