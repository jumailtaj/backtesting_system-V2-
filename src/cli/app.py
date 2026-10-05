"""Menu-driven CLI.  Nothing downloads or backtests without an explicit confirmation."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from config.settings import (CACHE_DIR, REPORT_DIR, STRATEGY_CONFIG_DIR, USER_SETTINGS_PATH, AppSettings,
                             load_sector_map)
from src.analysis.compare import (Variant, compare_variants, cost_sensitivity, sensitivity_variants)
from src.api.kite_client import KiteAPIError, KiteSession
from src.data.excel_cache import CacheError, ExcelCandleCache
from src.data.manager import CancelToken, DataManager, base_interval_for
from src.data.ranges import fmt
from src.engine.runner import BacktestRunner, DataIncompleteError
from src.reporting.comparison_report import write_comparison_report
from src.strategies.registry import REGISTRY, get_strategy, write_default_configs

BANNER = """
==============================================================
  Kite Historical Backtester   (backtesting only - no orders)
==============================================================
"""


class Cancelled(Exception):
    pass


class CLI:
    def __init__(self, settings: Optional[AppSettings] = None, session: Optional[KiteSession] = None,
                 input_fn: Callable[[str], str] = input, print_fn: Callable[..., None] = print) -> None:
        self.settings = settings or AppSettings.load()
        self._session = session
        self.input = input_fn
        self.print = print_fn
        self.cache = ExcelCandleCache(CACHE_DIR)
        self.tokens: dict[str, int] = {}
        write_default_configs()

    # ------------------------------------------------------------ plumbing
    @property
    def session(self) -> KiteSession:
        if self._session is None:
            self._session = KiteSession()
        return self._session

    def data_manager(self, with_source: bool = False) -> DataManager:
        return DataManager(self.session if with_source else None, self.cache, self.settings)

    def runner(self) -> BacktestRunner:
        return BacktestRunner(self.settings, self.data_manager(), REPORT_DIR, load_sector_map())

    def ask(self, prompt: str, default: Optional[str] = None) -> str:
        suffix = f" [{default}]" if default not in (None, "") else ""
        try:
            raw = self.input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise Cancelled() from None
        if raw.lower() in ("q", "cancel"):
            raise Cancelled()
        return raw or (default or "")

    def confirm(self, prompt: str) -> bool:
        return self.ask(f"{prompt} (y/N)", "n").lower() in ("y", "yes")

    def ask_date(self, prompt: str, default: date) -> date:
        while True:
            raw = self.ask(f"{prompt} (YYYY-MM-DD)", default.isoformat())
            try:
                return date.fromisoformat(raw)
            except ValueError:
                self.print("  Invalid date. Example: 2026-04-01")

    def ask_number(self, prompt: str, default, cast=float, minimum=None):
        while True:
            raw = self.ask(prompt, str(default))
            try:
                v = cast(raw)
                if minimum is not None and v < minimum:
                    raise ValueError
                return v
            except ValueError:
                self.print(f"  Enter a valid number{'' if minimum is None else f' >= {minimum}'}.")

    @contextmanager
    def cancellable(self):
        """Ctrl+C during a long operation requests a safe stop instead of killing it mid-write."""
        token = CancelToken()
        previous = signal.getsignal(signal.SIGINT)

        def handler(signum, frame):  # noqa: ARG001
            token.cancel()
            self.print("\n  Cancel requested - finishing the current step safely...")

        try:
            signal.signal(signal.SIGINT, handler)
        except ValueError:      # not in main thread (tests)
            previous = None
        try:
            yield token
        finally:
            if previous is not None:
                signal.signal(signal.SIGINT, previous)

    def error(self, exc: Exception) -> None:
        self.print(f"\n  ERROR: {exc}")

    # ---------------------------------------------------------------- menu
    def run(self) -> None:
        self.print(BANNER)
        actions = {
            "1": ("Authentication & connection check", self.menu_auth),
            "2": ("Instrument search & token verification", self.menu_instruments),
            "3": ("Instrument & date-range settings", self.menu_settings),
            "4": ("Historical cache status", self.menu_cache_status),
            "5": ("Download / recover missing data", self.menu_download),
            "6": ("Select & configure strategies", self.menu_strategies),
            "7": ("Run a backtest", self.menu_backtest),
            "8": ("Review results / open Excel report", self.menu_results),
            "9": ("Compare strategies / parameter variants", self.menu_compare),
            "0": ("Exit", None),
        }
        while True:
            start, end = self.settings.period()
            self.print(f"\nInstruments: {', '.join(self.settings.instruments)} | {start} -> {end} | "
                       f"intraday {self.settings.intraday_minutes}m | capital {self.settings.initial_capital:,.0f}")
            for k, (label, _) in actions.items():
                self.print(f"  {k}. {label}")
            try:
                choice = self.ask("Choose (q = cancel any prompt)", "").strip()
            except Cancelled:
                continue
            if choice == "0":
                self.print("Goodbye.")
                return
            if choice not in actions:
                self.print("  Please enter a number from the menu.")
                continue
            try:
                actions[choice][1]()
            except Cancelled:
                self.print("  Cancelled.")
            except (KiteAPIError, CacheError, DataIncompleteError, ValueError) as exc:
                self.error(exc)

    # ---------------------------------------------------------------- 1
    def menu_auth(self) -> None:
        s = self.session
        self.print("\nCredential status (values are never shown):")
        for k, v in s.credential_status().items():
            self.print(f"  {k}: {v}")
        self.print("  a) check connection   b) daily login (generate access token)   c) back")
        c = self.ask("Choose", "a").lower()
        if c == "a":
            info = s.check_connection()
            self.print(f"  Connected. User: {info.get('user_name')}  Broker: {info.get('broker')}")
        elif c == "b":
            self.print("\n  1. Open this URL, log in to Kite, and approve the app:")
            self.print(f"     {s.login_url()}")
            self.print("  2. After login you are redirected to your app's redirect URL; copy the value of the")
            self.print("     'request_token' query parameter from the address bar (single use, short-lived).")
            token = self.ask("  Paste request_token")
            s.create_session(token)
            self.print("  Access token obtained (valid until it expires, roughly 06:00 IST next day).")
            if self.confirm("  Save it into .env so later runs can reuse it?"):
                s.save_access_token()
                self.print("  Saved to .env")
            info = s.check_connection()
            self.print(f"  Connection verified. User: {info.get('user_name')}")

    # ---------------------------------------------------------------- 2
    def menu_instruments(self) -> None:
        exchange = self.ask("Exchange", "NSE").upper()
        query = self.ask("Search text (symbol or name) or EXCHANGE:SYMBOL to verify")
        if ":" in query:
            info = self.session.verify_instrument(query.upper())
            self.tokens[info.instrument] = info.token
            self.print(f"  Verified in Kite's instrument master: {info.instrument} -> token {info.token}, "
                       f"{info.name}, tick {info.tick_size}, lot {info.lot_size}")
            return
        df = self.session.search_instruments(query, exchange)
        if df.empty:
            self.print("  No matches.")
            return
        self.print(df.to_string(index=False))
        self.print("  To use one, add it in menu 3 and verify it here with EXCHANGE:SYMBOL.")

    # ---------------------------------------------------------------- 3
    def menu_settings(self) -> None:
        st = self.settings
        start, end = st.period()
        self.print("\nCurrent settings (press Enter to keep a value):")
        raw = self.ask("Instruments, comma separated", ",".join(st.instruments))
        instruments = [x.strip().upper() for x in raw.split(",") if x.strip()]
        new_start = self.ask_date("Start date", start)
        new_end = self.ask_date("End date", end)
        minutes = self.ask_number("Intraday timeframe in minutes (1,3,5,10,15,30)", st.intraday_minutes, int, 1)
        capital = self.ask_number("Initial capital (INR)", st.initial_capital, float, 1)
        slippage = self.ask_number("Slippage per market fill (%)", st.slippage_pct, float, 0)
        mult = self.ask_number("Cost multiplier (1 = configured rates)", st.cost_multiplier, float, 0)
        candidate = AppSettings.from_dict({**st.to_dict(), "instruments": instruments,
                                           "start_date": new_start.isoformat(), "end_date": new_end.isoformat(),
                                           "intraday_minutes": minutes, "initial_capital": capital,
                                           "slippage_pct": slippage, "cost_multiplier": mult})
        self.settings = candidate
        # keep strategy timeframe in sync with the chosen intraday minutes
        for key, cls in REGISTRY.items():
            strat = cls.load()
            if strat.config.intraday and strat.config.timeframe != f"{minutes}m":
                strat.with_overrides({"timeframe": f"{minutes}m"}).save_config()
        if self.confirm("Save these settings for next time?"):
            candidate.save()
            self.print(f"  Saved to {USER_SETTINGS_PATH.name}")
        self.print("  (Costs and other advanced settings: edit config/user_settings.json - see README.)")

    # ---------------------------------------------------------------- 4/5
    def _needed_intervals(self) -> list[tuple[str, str]]:
        out = []
        for key in self.settings.active_strategies:
            strat = get_strategy(key)
            out.append((strat.config.timeframe, base_interval_for(strat.config.timeframe)))
        return out

    def _data_windows(self) -> dict:
        """base interval -> (load_start, end) including warm-up."""
        from datetime import timedelta
        start, end = self.settings.period()
        windows: dict = {}
        for key in self.settings.active_strategies:
            strat = get_strategy(key)
            base = base_interval_for(strat.config.timeframe)
            days = (self.settings.intraday_warmup_calendar_days if strat.config.intraday
                    else self.settings.swing_warmup_calendar_days)
            ls = start - timedelta(days=days)
            windows[base] = (min(ls, windows[base][0]) if base in windows else ls, end)
        return windows

    def menu_cache_status(self) -> None:
        dm = self.data_manager()
        for inst in self.settings.instruments:
            for base, (ls, end) in self._data_windows().items():
                rep = dm.status(inst, base, ls, end)
                self.print("")
                for line in rep.summary_lines():
                    self.print(line)

    def menu_download(self) -> None:
        dm = self.data_manager(with_source=True)
        plan = []
        for inst in self.settings.instruments:
            for base, (ls, end) in self._data_windows().items():
                rep = dm.status(inst, base, ls, end)
                plan.append((inst, base, ls, end, rep))
                self.print(f"\n{inst} [{base}] missing: {', '.join(fmt(r) for r in rep.missing) or 'nothing'}")
        if not any(p[4].missing for p in plan):
            self.print("\nNothing to download - the cache already covers everything requested.")
            return
        self.print("\nThis will call Kite's historical data API (needs the Historical API add-on) and "
                   "download only the missing ranges above.")
        if not self.confirm("Proceed with the download?"):
            raise Cancelled()
        with self.cancellable() as cancel:
            for inst, base, ls, end, rep in plan:
                if not rep.missing:
                    continue
                token = self.tokens.get(inst) or self.session.verify_instrument(inst).token
                self.tokens[inst] = token
                result = dm.ensure_data(inst, token, base, ls, end, progress=lambda m: self.print("  " + m),
                                        cancel=cancel)
                self.print("")
                for line in result.summary_lines():
                    self.print(line)
                if cancel.cancelled or result.aborted:
                    break

    # ---------------------------------------------------------------- 6
    def menu_strategies(self) -> None:
        st = self.settings
        self.print("\nStrategies (config files live in config/strategies/):")
        for k, cls in REGISTRY.items():
            self.print(f"  [{'x' if k in st.active_strategies else ' '}] {k:16s} {cls.label}")
        self.print("  t) toggle active   e) edit a parameter   v) view parameters   r) reset to defaults   b) back")
        c = self.ask("Choose", "b").lower()
        if c == "t":
            key = self.ask("Strategy key")
            if key not in REGISTRY:
                raise ValueError(f"Unknown strategy '{key}'.")
            active = [a for a in st.active_strategies if a != key] if key in st.active_strategies \
                else st.active_strategies + [key]
            self.settings = AppSettings.from_dict({**st.to_dict(), "active_strategies": active})
            self.print(f"  Active: {', '.join(active)}")
        elif c in ("e", "v"):
            key = self.ask("Strategy key")
            strat = get_strategy(key)
            if c == "v":
                self.print(json.dumps(strat.config.to_dict(), indent=2))
                return
            param = self.ask("Parameter name")
            value = self.ask("New value (use 'null' to disable an optional limit)")
            parsed = None if value.lower() == "null" else value
            updated = strat.with_overrides({param: parsed})      # validates the value
            updated.save_config()
            self.print(f"  Saved {param} = {getattr(updated.config, param)} to {updated.config_path().name}")
        elif c == "r":
            if self.confirm("Overwrite ALL strategy config files with the defaults?"):
                write_default_configs(overwrite=True)
                self.print("  Defaults restored.")

    # ---------------------------------------------------------------- 7
    def _pick_strategy(self) -> str:
        keys = self.settings.active_strategies
        for i, k in enumerate(keys, 1):
            self.print(f"  {i}. {REGISTRY[k].label} ({k})")
        raw = self.ask("Strategy number or key", "1")
        if raw.isdigit() and 1 <= int(raw) <= len(keys):
            return keys[int(raw) - 1]
        if raw in REGISTRY:
            return raw
        raise ValueError("Unknown strategy choice.")

    def menu_backtest(self) -> None:
        key = self._pick_strategy()
        strat = get_strategy(key)
        start, end = self.settings.period()
        runner = self.runner()
        try:
            prepared = runner.prepare(strat, self.settings.instruments, start, end)
        except DataIncompleteError as exc:
            self.print(f"\n  {exc}")
            if not self.confirm("Run anyway on the incomplete data (results will carry a warning)?"):
                raise Cancelled() from None
            prepared = runner.prepare(strat, self.settings.instruments, start, end, require_complete=False)
            prepared.warnings.append("RUN USED INCOMPLETE DATA: " + str(exc))
        self.print(f"\n  {strat.label}: {', '.join(self.settings.instruments)} {start} -> {end}, "
                   f"{strat.config.timeframe}")
        if not self.confirm("Start the backtest?"):
            raise Cancelled()
        run = runner.execute(strat, prepared, runner.generate_signals(strat, prepared))
        m = run.metrics
        self.print("\n  Result (historical simulation; not a prediction):")
        self.print(f"   trades {m['total_trades']}  net P&L {m['net_pnl']:,.2f}  return {m['return_pct']:.2f}%  "
                   f"win rate {_fmt(m['win_rate_pct'])}%  profit factor {_fmt(m['profit_factor'])}  "
                   f"max DD {_fmt(m['max_drawdown_pct'])}%")
        path = runner.write_report(strat, prepared, run)
        self.last_report = path
        self.print(f"  Report: {path}")

    # ---------------------------------------------------------------- 8
    def menu_results(self) -> None:
        files = sorted(REPORT_DIR.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)[:15]
        if not files:
            self.print("  No reports yet. Run a backtest first.")
            return
        for i, f in enumerate(files, 1):
            self.print(f"  {i}. {f.name}")
        raw = self.ask("Open which report? (number, Enter to skip)", "")
        if raw.isdigit() and 1 <= int(raw) <= len(files):
            open_file(files[int(raw) - 1], self.print)

    # ---------------------------------------------------------------- 9
    def menu_compare(self) -> None:
        self.print("\n  1) compare strategies   2) parameter sensitivity   3) cost sensitivity")
        mode = self.ask("Choose", "1")
        start, end = self.settings.period()
        instruments = self.settings.instruments
        runner = self.runner()
        if mode == "1":
            raw = self.ask("Strategy keys, comma separated", ",".join(self.settings.active_strategies))
            keys = [k.strip() for k in raw.split(",") if k.strip()]
            variants = [Variant(REGISTRY[k].label if k in REGISTRY else k, k) for k in keys]
            label = "strategies"
        elif mode == "2":
            key = self._pick_strategy()
            variants, notes = sensitivity_variants(key)
            for n in notes:
                self.print("  " + n)
            self.print(f"  {len(variants)} bounded one-at-a-time variants will run.")
            label = f"sensitivity-{key}"
        elif mode == "3":
            key = self._pick_strategy()
            if not self.confirm("Run the cost sensitivity analysis?"):
                raise Cancelled()
            table = cost_sensitivity(runner, key, instruments, start, end)
            self.print(table.to_string(index=False))
            return
        else:
            raise ValueError("Unknown choice.")
        if not self.confirm(f"Run {len(variants)} backtest(s) with in/out-of-sample split?"):
            raise Cancelled()
        comp = compare_variants(runner, variants, instruments, start, end)
        self.print("\n" + comp.table.to_string(index=False))
        for w in comp.warnings:
            self.print("  ! " + w)
        path = write_comparison_report(comp, label, instruments, REPORT_DIR)
        self.print(f"\n  Comparison report: {path}")


def _fmt(v) -> str:
    return "N/A" if v is None else f"{v:.2f}"


def open_file(path: Path, print_fn=print) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as exc:  # noqa: BLE001
        print_fn(f"  Could not open the file automatically ({exc}). Open it manually: {path}")
