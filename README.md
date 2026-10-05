# Kite Historical Backtester

A beginner-friendly, menu-driven **historical backtesting** CLI for NSE equities using Kite Connect data,
an Excel data cache, a shared backtesting engine, four strategies and Excel reports.

**Backtesting only.** There is no order placement, paper trading, web dashboard or database. Historical
results are simulations and are **not** a guarantee of future performance.

---------------------------------------------------------------------------------------------------
## 1. Installation

Requires Python 3.10+ (developed on 3.12).

```bash
cd kite_backtester
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # Windows: copy .env.example .env   (then edit .env)
python main.py
```

Dependencies: `kiteconnect` (official client), `pandas`, `numpy`, `openpyxl` (Excel), `python-dotenv`,
`pytest`.

## 2. Kite API credentials and the daily login

1. Create an app at <https://developers.kite.trade/>. Your account needs API access **and the paid Historical
   Data add-on** - without it historical requests fail with a permission error (you are responsible for valid
   API access; check Kite's current pricing/terms).
2. Put `KITE_API_KEY` and `KITE_API_SECRET` in `.env`. Never commit `.env` (it is git-ignored). The tool never
   prints these values; error messages are redacted.
3. **Access tokens expire every day** (Kite resets them early morning IST) and there is no refresh token. Each
   trading day you need a new one:
   - CLI menu **1 -> b**: it prints the Kite login URL, you log in, then copy the `request_token` parameter
     from the redirect URL in your browser's address bar and paste it. The `request_token` is single-use and
     expires within minutes.
   - The tool exchanges it for an access token and (if you agree) saves it as `KITE_ACCESS_TOKEN` in `.env`.
4. Menu **1 -> a** checks the connection (`profile()` call).

Kite API limits this tool respects: historical requests are capped per call (about **60 days for minute
candles, 2000 days for daily**) and rate-limited (about 3 requests/second). Downloads are chunked and throttled,
with retry/backoff. Cached data can be used offline; only downloading needs credentials.

## 3. Running the CLI

`python main.py` shows the menu:

| # | Menu | What it does |
|---|------|--------------|
| 1 | Authentication | credential status (masked), connection check, daily login |
| 2 | Instrument search | search Kite's instrument master; `NSE:RELIANCE` verifies and shows the **token read from the master** |
| 3 | Settings | instruments, dates, intraday timeframe, capital, slippage, cost multiplier |
| 4 | Cache status | per-instrument coverage and missing ranges (no network) |
| 5 | Download / recover | shows exactly what is missing, asks for confirmation, downloads only gaps |
| 6 | Strategies | toggle active, view/edit parameters, reset defaults |
| 7 | Run backtest | loads cache, asks for confirmation, writes an Excel report |
| 8 | Results | lists reports; opens one in Excel |
| 9 | Compare | strategies side by side, parameter sensitivity, cost sensitivity |
| 0 | Exit | |

Type `q` at any prompt to cancel. During a download, **Ctrl+C** finishes the current request, saves progress
and stops safely. Nothing downloads or backtests without a `y` confirmation.

## 4. Instruments and dates

* Default instrument `NSE:RELIANCE`. The instrument token is **never hard-coded**: menu 2 looks it up in Kite's
  instrument master (cash equities, `instrument_type == EQ`) and menu 5 verifies it before downloading. The token
  is stored in the cache workbook's `Meta` sheet, and a later download with a different token is refused.
* Default period: the **previous six full calendar months** in IST (e.g. on 2026-10-04 -> 2026-04-01 ..
  2026-09-30). Change in menu 3 or via `config/user_settings.json`.
* Timeframes: intraday default 5 minutes (any of 1/3/5/10/15/30), swing default daily.
* Extra history before the start date is downloaded for indicator warm-up
  (`intraday_warmup_calendar_days`=21, `swing_warmup_calendar_days`=180). Signals are suppressed while
  indicators are NaN; the report shows the warm-up window.

All settings live in `config/user_settings.json` (created when you save in menu 3). Example (only list what you
want to change):

```json
{
  "instruments": ["NSE:RELIANCE", "NSE:TCS"],
  "initial_capital": 500000,
  "slippage_pct": 0.02,
  "intraday_costs": {"brokerage_cap": 20.0}
}
```

## 5. Cache behaviour and missing-data recovery

* One workbook per instrument and **base** interval in `data/cache/`:
  `NSE_RELIANCE_minute.xlsx` (1-minute candles) and `NSE_RELIANCE_day.xlsx` (daily candles).
  Sheets: `Candles`, `Coverage` (date ranges successfully fetched), `Meta`, `Log`.
* Coverage, not candle gaps, decides what is "cached". Before every download the tool subtracts the covered
  ranges from the request and downloads only what is missing (weekend-only gaps are skipped).
* Intraday timeframes (3/5/10/15/30 min) are **built from 1-minute data**; daily candles come from Kite's daily
  data, never from partial intraday data.
* Every download is validated first: columns, timestamps, missing values, non-positive prices, OHLC consistency,
  negative volume, out-of-session minutes, duplicates, ordering. Invalid rows are rejected and counted. If every
  row in a chunk is invalid, the chunk fails and is **not** marked covered.
* An empty reply for a range with 4+ weekdays is treated as a **failure** (possible permissions/history limit),
  not as holidays. A short empty range (<4 weekdays) is recorded as "no data" (likely holiday). No exchange
  holiday calendar is assumed; holidays show up as weekdays with no candles in the *Data Validation* sheet.
* Today's candles are not marked covered until after 15:45 IST.
* Merging keeps existing valid rows on duplicate timestamps and warns if values conflict. **Candles are never
  fabricated or forward-filled.**
* Saves are atomic (temp workbook -> verify -> replace). A failed write leaves the old cache intact. An
  unreadable cache is reported and never modified or deleted; you decide what to do with it.
* If a download is interrupted, re-run menu 5: it resumes with the remaining gaps.

## 6. How a backtest works (assumptions and limitations)

Written into each report's *Configuration* sheet too.

* Signals are evaluated when a candle has **closed**; entry is at the **next candle's open** (same session for
  intraday). Candle timestamps are candle *start* times.
* Market fills (entry, stop, forced exit) get adverse slippage (`slippage_pct`) and adverse tick rounding.
  Targets are limit orders: filled at the target price when touched (optimistic - real fills may not occur on
  a mere touch).
* If one candle touches both stop and target, **the stop is assumed first**. A gap beyond the stop exits at the
  open (worse than the stop).
* Intraday: no entries at/after the force-exit time, forced exit at the open of the 15:15 candle, never overnight.
  Swing: held overnight; open positions at the end of data close at the last close (`end_of_backtest`).
* Costs on both legs from configurable rate tables (intraday vs delivery): brokerage (capped), STT, exchange
  charges, SEBI fee, stamp duty, GST, DP charge (delivery sells). **The default rates are a Zerodha-style
  schedule entered from memory and WILL go out of date - verify them against your broker's current charge sheet
  and edit `intraday_costs` / `delivery_costs`.**
* No leverage; one position per instrument; no pyramiding, partial exits or trailing stops; no short selling
  unless a strategy's `allow_short` is true (intraday strategies only).
* Equity is marked to market at candle closes; drawdown is measured on that curve.
* Limits of the data: Kite's historical availability varies by account and instrument; corporate actions and
  adjustments are whatever Kite returns; survivorship and liquidity effects are not modelled; slippage is a flat
  percentage.
* Relative volume definitions: intraday = candle volume / average volume of the **same time slot** over the
  previous N sessions (default 10); daily = volume / average of the previous N sessions (default 20).
* With only six months of daily data the swing strategies will take **very few trades**. Reports flag
  insufficient samples; do not draw conclusions from them.

## 7. Strategies and configuration

Each strategy has a JSON file in `config/strategies/` (created on first run). Edit it directly or via menu 6.
Set an optional limit to `null` to disable it. Unknown or invalid parameters are rejected with a clear error.

| Key | Strategy | Highlights (defaults) |
|-----|----------|-----------------------|
| `ema_crossover` | Intraday EMA Crossover | EMA 9/21 on 5m; real cross between completed candles; VWAP filter on; stop = lowest low of 5 completed candles of the session; 2R; risk 0.25%; max 2 trades/day; entries 09:30-14:30; exit 15:15. Optional RSI/ADX/volume/ATR/trend filters (off). |
| `orb` | Opening Range Breakout | range 09:15-09:30; entries 09:30-11:30; buffer 0.05%; width 0.20-1.00% of range midpoint; VWAP + rel. volume 1.2; stop = opposite boundary; 2R; risk 0.25%; 1 trade/day; daily loss limit 0.75%. |
| `swing_breakout` | Swing Breakout (daily) | close > prior-20-session high (current candle excluded) +0.25%, rel. volume 1.5, close > EMA50, <= 3% above the level; 10-day swing-low stop; 3R; risk 0.5%; <=5 positions, <=3 entries/day, <=2% open risk, <=80% capital, 2 per sector. |
| `swing_sr` | Swing Support/Resistance (daily) | pivots 3 left/3 right, 60-session lookback, >= 2 touches, zone = 0.25 x ATR(14); candle dips into zone and closes above the zone and above its midpoint; rel. volume 1.2; close > EMA50; stop below zone (+0.25 ATR buffer); min 2R; risk 0.5%. |

Key common risk parameters: `risk_per_trade_pct`, `max_trades_per_instrument_per_day`,
`max_simultaneous_positions`, `max_new_entries_per_day`, `max_portfolio_open_risk_pct`,
`max_capital_allocation_pct`, `max_position_value_pct`, `daily_loss_limit_pct`, `sector_limit`,
`min_stop_distance_pct`, `allow_short`, `force_exit_time`.

Behaviour notes you should know:

* Sizing: `quantity = floor(equity x risk% / |entry - stop|)`, then capped by available capital and position
  caps. Zero quantity and invalid stops are rejected with a reason (*Rejected Signals* sheet).
* `max_portfolio_open_risk_pct` is enforced by **rejecting** a trade that would exceed it (not by shrinking it).
  With the swing defaults (0.5% per trade, 2% cap) at most about four positions fit, so the "5 positions"
  limit rarely binds.
* Strategy-level details: the EMA stop uses candles of the same session only, so with a 5-candle lookback the
  first possible EMA signal appears after the 5th candle of the day. Pivot zones only use pivots confirmed
  `pivot_right` candles earlier (never back-dated).
* Sector limit: Kite has no sector field. Provide `config/sectors.csv` (`instrument,sector`) to enable it;
  otherwise the report states that it was not enforced.

### Adding a strategy (no engine changes)

1. Create `src/strategies/my_strategy.py` with a config dataclass (subclass `StrategyConfig`) and a
   `Strategy` subclass setting `key`, `label`, `ConfigClass`, and implementing
   `generate_signals(candles) -> DataFrame` (columns `direction, stop_price, target_rr, target_price, min_rr,
   reason`; row *t* may use candles <= *t* only - see `src/strategies/base.py`).
2. Add the class to `REGISTRY` in `src/strategies/registry.py`.
3. Run once (defaults JSON is created), add the key to `active_strategies`. Sizing, fills, risk, costs and
   reports come from the shared engine. Copy `assert_no_lookahead` from `tests/test_strategies.py` to prove the
   signals are free of look-ahead.

## 8. Excel reports

Reports go to `data/reports/` with unique names
(`<strategy>_<instrument>_<timeframe>_<start>_<end>_<timestamp>.xlsx`); existing files are never overwritten.

* **Summary** - strategy, instrument, timeframe, period, capital, net P&L, return, trades, wins/losses, win rate,
  profit factor, expectancy, average R, drawdown, costs, warnings. **`N/A` means unavailable/undefined (for
  example profit factor with no losing trades) - it is not zero.** Genuine zeros are shown as 0.
* **Trades** - signal/entry/exit times, direction, quantity, prices, stop, target, gross/costs/net P&L, R multiple,
  holding time, exit reason, and a flag when stop and target were both touched in one candle.
* **Equity Curve** - equity and drawdown per candle, with a chart. **Monthly Results**, **Configuration**
  (all parameters, costs, assumptions), **Data Validation** (source file, coverage, missing ranges, duplicates,
  invalid rows, dropped candles, warnings), **Rejected Signals** (reasons and counts).
* Comparison reports add in-sample vs out-of-sample columns (default 70/30 split by date), warnings for few
  trades or in-sample/out-of-sample divergence, a bounded one-at-a-time parameter sensitivity (max 25 variants -
  no brute-force optimisation) and a cost-sensitivity table (0x-3x costs).
  **The best historical variant is not a forecast. Treat rankings as noise unless trade counts are large.**

## 9. Tests

```bash
python -m pytest -q
```

All tests run **offline** with deterministic synthetic candles and a mocked Kite client; no credentials needed.
They cover resampling, validation, cache merge/recovery/atomic save, signal timing, next-candle entry, stop/target
rules, sizing, risk limits, no-look-ahead (truncation test for every strategy), Excel output and error handling.
They do **not** verify real Kite behaviour - see Known limitations.

## 10. Troubleshooting

| Message | Meaning / fix |
|---------|---------------|
| `KITE_API_KEY is not set` | copy `.env.example` to `.env` and fill it in |
| `access token is missing, invalid or expired` | tokens expire daily - menu 1 -> b |
| `lacks permission ... Historical API add-on` | your Kite account/app doesn't include historical data |
| `Too many requests` | handled with backoff; if it persists, wait and rerun menu 5 (it resumes) |
| `was not found as an EQ instrument` | wrong symbol; search in menu 2 |
| `Kite returned no candles for a range with several weekdays` | instrument inactive, outside Kite's history, or no API access; nothing was cached for that range |
| `cache does not cover ...` | run menu 5, or accept running on incomplete data (flagged in the report) |
| `Cache file ... could not be read` | file is corrupt or open in Excel; it was not modified - close/move it |
| `Could not save cache` | disk full or file locked; previous cache intact |
| `Unknown parameter(s)` | typo in a strategy JSON; the message lists the valid names |
| Report won't open | close any Excel window holding an old copy; reports are new files each run |

## 11. Known limitations

* **Not verified against the live Kite API** (no credentials were available when this was built): authentication,
  instrument-master parsing, `historical_data` behaviour (limits, rate-limit error type, empty responses,
  timestamp format) follow Kite's public documentation and were only tested with mocks. Run menu 1, 2, then a
  small download first and read any error closely.
* Default transaction-cost rates are from memory and must be verified (section 6).
* Intraday history availability for 1-minute data depends on Kite/your account.
* One position per instrument; flat slippage; touch-fills for targets; no corporate-action handling beyond
  Kite's data; no exchange-holiday calendar.
