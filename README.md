# Sentiment Analysis

Part 1 scans recent Twitter/X stock chatter, scores sentiment, filters for high-traction tickers, and checks whether those tickers have earnings coming up.

Part 2 can plug into the same output later with Unusual Whales data.

The project also includes an optional paper-first earnings options automation service. It turns saved dashboard recommendations into timed plans and can route approved plans through the official Webull OpenAPI.

## Why It Is Modular

- `providers/x_api.py`: live X recent-search collector.
- `providers/csv_posts.py`: local fixture collector for testing without API credentials.
- `sentiment/vader.py`: fast social-text sentiment analyzer.
- `providers/earnings.py`: earnings-date providers.
- `providers/political_activity.py`: cached public political-disclosure activity by ticker.
- `pipeline.py`: provider-agnostic orchestration and scoring.
- `storage.py`: SQLite scan history, signals, and matched posts.

The broad pipeline only requests earnings dates after ticker traction has been calculated. The dashboard's cheap mode flips that order: it gets the earnings-week candidate list first, screens for options-implied move, uses cheap counts to rank chatter, then fetches a small sample of actual posts.

For the live X path, `--x-query-mode discovery` uses broad finance queries plus `has:cashtags`, then validates extracted symbols against your ticker universe. `--x-query-mode watchlist` instead builds explicit `$AAPL OR $TSLA ...` style chunks from the ticker file.

The dashboard default is cheaper than broad discovery: it first finds companies reporting in the current/upcoming earnings week, keeps only tickers with an options-implied earnings move at or above the threshold, uses X recent counts to rank chatter, then spends recent-search Post reads only on the top counted tickers.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

For live X/Twitter collection, set `X_BEARER_TOKEN` in your shell or `.env`. The project uses X API v2 recent search, which retrieves matching public posts from the last 7 days.

For SEC insider-filing checks, optionally set a descriptive user agent:

```bash
export SEC_USER_AGENT="sentiment-analysis/0.1 your-email@example.com"
```

## Refresh The Stock Universe

```bash
sentiment-scan refresh-tickers --out data/tickers_us.csv
```

This downloads current US listed symbols from Nasdaq Trader symbol-directory files and excludes ETFs/test issues by default.
It also skips obvious non-earnings instruments such as rights, units, warrants, preferreds, and listed notes unless you pass `--include-special-securities`.

## Run The Local Demo

```bash
PYTHONPATH=src python -m sentiment_analysis run \
  --post-source csv \
  --posts-csv examples/sample_posts.csv \
  --earnings-source csv \
  --earnings-csv examples/sample_earnings.csv \
  --ticker-file data/tickers_sample.csv \
  --min-mentions 2 \
  --earnings-horizon-days 21 \
  --hours-back 240 \
  --out-csv output/demo_signals.csv \
  --out-json output/demo_signals.json
```

## Run Against X

```bash
export X_BEARER_TOKEN="your-token"

sentiment-scan run \
  --post-source x \
  --earnings-source yfinance \
  --ticker-file data/tickers_us.csv \
  --x-query-mode discovery \
  --use-trends \
  --trend-woeid 23424977 \
  --trend-woeid 1 \
  --hours-back 24 \
  --min-mentions 25 \
  --earnings-horizon-days 14 \
  --max-pages-per-query 25 \
  --x-max-workers 4 \
  --out-csv output/signals.csv \
  --out-json output/signals.json
```

## Run The Local Dashboard

```bash
source .venv/bin/activate
export X_BEARER_TOKEN="your-token"
sentiment-dashboard --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

Previous dashboard runs are saved in `data/scans.sqlite3`. After restarting the dashboard, use the Run History panel to load an older scan back into the metrics and table without spending new X API credits.

Cheap default settings:

- `Scan Mode`: Cheap Earnings Week.
- `Earnings`: yfinance Calendar.
- `Window`: Earnings Week.
- `Min Move %`: 5.
- `EM Checks`: 0, meaning all earnings candidates that survive junk-name screening.
- `EM Workers`: 8.
- `Max X Counts`: 150.
- `Skip Junk Names`: On.
- `Max Tickers`: 8.
- `Posts / Page`: 100.
- `Pages / Query`: 1.
- `Market Checks`: On.
- `Opt Exp`: 2.
- `Opt Vol`: 250.
- `Vol/OI`: 2.
- `Opt Premium`: 500000.
- `Max OTM %`: 35.
- `Insider Days`: 30.
- `Political Checks`: On.
- `Political Days`: 365.
- `Trend Pull`: Off.

This keeps the default run focused on companies reporting by the end of the current earnings week where options imply at least a 5% move. The dashboard screens earnings candidates before X counts, prioritizes the count budget toward liquid/recognizable names without using market cap as a hard filter, then spends recent-search Post reads only on the top counted tickers. After sentiment flags high-traction tickers, the dashboard runs cheap checks for options-chain anomalies, recent SEC Form 3/4/5 filings, and political disclosures. Results and matched posts are stored in `data/scans.sqlite3`.

The final watch table and automated planner enforce the gate again before any market or political enrichment: earnings must be inside the scan window, implied move must meet `Min Move %`, X preflight count must meet `Min Mentions`, and the fetched sample must also meet `Min Mentions`. Political activity can change direction or confidence only after those checks; it cannot add or rescue a ticker that failed them.

To pull more X posts for sentiment, increase `Pages / Query`. Estimated post reads are roughly:

```text
Max Tickers * Posts / Page * Pages / Query
```

For example, `8 * 100 * 5 = 4,000` possible post reads. X counts still run first, so post reads are only spent on the selected tickers.

## Output Columns

- `ticker`: stock symbol.
- `mention_count`: total posts mentioning the ticker.
- `unique_authors`: distinct authors in the sample.
- `total_engagement`: likes + weighted replies/reposts/quotes.
- `avg_sentiment`: average compound sentiment from -1.0 to 1.0.
- `crowd_bias`: bullish, bearish, or neutral.
- `earnings_date`: next known earnings date, when available.
- `expected_move_pct`: options-implied move percentage from the earnings-week options chain.
- `expected_move_abs`: estimated dollar move from the ATM straddle or IV fallback.
- `expected_move_source`: `atm_straddle` or `atm_iv`.
- `options_expiration`: option expiration used for the expected move estimate.
- `days_until_earnings`: distance from today.
- `contrarian_watch`: a simple flag when sentiment is strong and earnings are close.

## Dashboard Market Checks

The dashboard adds a `Market Evidence` panel for flagged tickers:

- `option_alert`: `unusual`, `active`, `quiet`, or `unknown`. `unusual` requires the option volume, volume/open-interest ratio, estimated premium, and max distance-from-spot thresholds to be met.
- `option_bias`: call/put volume tilt across checked expirations.
- `option_top_contract`: the highest-priority unusual contract from volume/open-interest screening.
- `insider_recent_count`: recent SEC Form 3/4/5 filing count inside the configured lookback window.
- `insider_latest_form` and `insider_latest_date`: newest insider filing metadata when found.
- `political_bias`: disclosed political activity over the configured lookback. A hold-proxy buyer is counted only when that filer's latest ticker transaction is a buy; the UI says `no later sale found`, not `still owns`.
- `political_strength`: directional score based on distinct hold-proxy filers, disclosed value range, and leadership weight. Three hold-proxy buyers or one high-profile hold-proxy buyer create a strong bullish signal even when unrelated filers sold. The President, congressional leaders, and a former Speaker receive more weight than a standard member.
- `political_net_strength`: the separate latest-side buyer-minus-seller score retained as context; it does not erase a qualifying hold cluster.
- `political_source_url`: the public ticker evidence page. The dashboard also stores the official House and Senate disclosure search links and caches ticker pages for 24 hours.
- `recommendation`: combined research label using contrarian X sentiment, political disclosures, options activity, and SEC insider filings.
- `recommendation_score`: 0-100 research score. A qualifying political hold cluster can override the X-only contrarian direction; unusual options are then compared with the final direction.

Political disclosures are delayed, reported in value ranges, and may belong to a spouse or child. `No later sale found` only describes the available dataset and does not prove that a position remains open. This is a cheap research screen, not a paid real-time options-flow or holdings feed. The providers remain modular so paid data can be added later without changing the sentiment pipeline.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Weekly Trade Automation

The automation service uses `America/New_York` and the NYSE holiday/early-close calendar. It scans once after 8:00 AM Eastern each Monday, then persists trade plans in `data/scans.sqlite3`.

- `AMC`: enter 15 minutes before that session's actual close and exit 15 minutes after the next trading session opens.
- `BMO`: enter 15 minutes before the prior trading session's actual close and exit 15 minutes after the earnings session opens.
- Unknown earnings timing, recommendations below the score floor, X-only recommendations without corroborating options/political data, and already-missed entries are skipped.
- A corroborated directional setup receives a 5% cash cap. A recommendation containing `options_confirm` receives a 7% cap.
- The defaults also limit aggregate open premium to 20%, one contract per ticker, and bid/ask spread to 12%.
- Contract selection targets roughly 0.80 absolute delta / 10% in-the-money, rejects contracts whose extrinsic premium requires more than a 2.5% underlying move to reach expiration breakeven, and requires at least 100 contracts of open interest.
- Expiration must be 14-28 calendar days beyond the planned exit. This and the deeper in-the-money rule reduce reliance on event premium, but they do not remove earnings volatility risk.
- At the opening bell, the service places a resting sell-to-close limit 5% above entry. It leaves that order unchanged until 15 minutes after the open, then reprices toward the executable market in 10% steps until filled.

A Monday morning scan cannot catch Monday premarket earnings because that entry belongs on the prior Friday. Use a Sunday scan time if those reports must be included.

The cost-controlled weekly settings are in `data/automation_scan_options.json`. To reset them to the documented higher-volume example:

```bash
cp data/automation_scan_options.json.example data/automation_scan_options.json
```

## Shadow Trade Research

The primary paper service can collect a forward-only research dataset alongside the real paper strategy. It never submits, modifies, or cancels an order. For each qualified ticker, it freezes the scan features available before earnings and schedules virtual calls and puts across:

- 0.50, 0.65, and 0.80 target absolute delta.
- Expirations approximately 7, 14, 21, and 28 days after the planned exit.
- Entries 60, 30, and 15 minutes before the session close.
- Exit observations at the open and 5, 15, and 30 minutes after the open.

The default grid creates up to 72 virtual positions per ticker, or 576 for an eight-ticker scan. Virtual entries use the displayed ask and exit returns use the displayed bid, with midpoint returns retained for comparison. Quote timestamps, spreads, volume, open interest, delta, vega, implied volatility, and underlying prices are stored with quality labels. Missing or stale quotes remain labeled as such instead of becoming synthetic fills.

Research tables live in the same SQLite file as their associated scanner but are isolated from trade plans and orders. The main service enables this collector with `SHADOW_RESEARCH_ENABLED=true`; `.env.small` disables it so the $10,000 sleeve does not duplicate Webull market-data requests.

View the collector in the dashboard's **Shadow Research** panel or from the shell:

```bash
sentiment-trader research-status
sentiment-trader research-status --json
```

Collection begins with the next qualifying weekly scan. The system intentionally does not backfill prior earnings because historical feature reconstruction would leak information that was unavailable at decision time. These observations are the dataset for later walk-forward policy evaluation; no model retraining or automatic policy promotion is enabled yet.

Paper mode is the default:

```bash
source .venv/bin/activate
pip install -e .
sentiment-trader plan --run-id latest
sentiment-trader status
sentiment-trader run-once --mode paper --skip-scan
sentiment-trader daemon --mode paper
```

`paper` uses delayed yfinance quotes and simulated immediate fills. It is useful for exercising scheduling and persistence, not for measuring realistic opening fills or slippage.

## Webull OpenAPI Setup

Use the official Webull OpenAPI, not an unofficial session-scraping package.

1. In Webull, open Developer Tool, then OpenAPI Management and apply for an individual Trading API application.
2. Create sandbox credentials first. Put only the App Key and App Secret in `.env`, then run `sentiment-trader verify-webull --environment sandbox`; it lists the available Account IDs without placing an order.
3. Enable options trading permission for the account.
4. Purchase/enable the OpenAPI OPRA real-time non-display entitlement. A quote subscription in the normal Webull app or desktop client is separate.
5. Put the option-enabled sandbox Account ID printed by the verification command in `WEBULL_SANDBOX_ACCOUNT_ID`, then run the command again to verify its balance.
6. Complete the SDK's Webull-app verification prompt if 2FA is enabled.
7. Verify read-only access, then run one sandbox iteration:

```bash
sentiment-trader verify-webull --environment sandbox
sentiment-trader run-once --mode sandbox --skip-scan
sentiment-trader status
```

The verifier is read-only. It lists accounts, checks the configured account balance, and requests a SPY option quote to confirm OpenAPI OPRA access; it never submits an order.

Keep sandbox mode through multiple full earnings cycles. Production orders require all of these values:

```dotenv
TRADING_MODE=live
WEBULL_LIVE_APP_KEY=your-production-app-key
WEBULL_LIVE_APP_SECRET=your-production-app-secret
WEBULL_LIVE_ACCOUNT_ID=your-option-account-id
ENABLE_LIVE_TRADING=I_UNDERSTAND_THE_RISK
```

Webull options orders are limit orders. The service previews each order, buys at the current ask, places a resting profit exit at the open, checks order status, and begins aggressive exit repricing after the configured hard-exit time. Manual Webull closes are reconciled from order history at their actual fill. An error state is persistent and visible in the dashboard; ambiguous submission failures are not blindly duplicated.

## Always-On VPS

The computer executing trades must remain powered on. To run while your laptop is off, install the project on an always-on Linux VPS under `~/sentiment_analysis`, then install the included user services:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/sentiment-trader.service deploy/sentiment-dashboard.service ~/.config/systemd/user/
sudo loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now sentiment-trader.service sentiment-dashboard.service
systemctl --user status sentiment-trader.service
```

The VPS dashboard remains bound to localhost. Reach it securely from your laptop with an SSH tunnel:

```bash
ssh -L 8000:127.0.0.1:8000 your-user@your-vps
```

Then open `http://127.0.0.1:8000`. Check service logs with:

```bash
journalctl --user -u sentiment-trader.service -f
```

### Additional $10,000 paper sleeve

The small-account service runs alongside the primary service. It uses a second Webull sandbox cash account, a separate SQLite database, and a virtual starting equity of $10,000. It mirrors the primary Monday scan instead of spending X credits twice. Realized gains and losses compound the virtual equity; the service never treats the sandbox account's displayed $1 million balance as available strategy capital.

Copy `.env.small.example` to `.env.small`, set the second sandbox Account ID, and install the additional units:

```bash
cp .env.small.example .env.small
chmod 600 .env.small
cp deploy/sentiment-trader-small.service deploy/sentiment-dashboard-small.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now sentiment-trader-small.service sentiment-dashboard-small.service
```

The primary dashboard remains on port 8000. Tunnel both dashboards from the VPS:

```bash
ssh -L 8000:127.0.0.1:8000 -L 8001:127.0.0.1:8001 your-user@your-vps
```

Open `http://127.0.0.1:8001` for the read-only small-account dashboard. Its logs are separate:

```bash
journalctl --user -u sentiment-trader-small.service -f
```
