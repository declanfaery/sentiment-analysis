from __future__ import annotations

import argparse
import json
import math
import threading
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from sentiment_analysis.config import AppConfig
from sentiment_analysis.io import write_signals_csv, write_signals_json
from sentiment_analysis.market_calendar import days_window, earnings_week_window
from sentiment_analysis.pipeline import PipelineOptions, PipelineResult, SentimentPipeline
from sentiment_analysis.providers.csv_posts import CsvPostProvider
from sentiment_analysis.providers.earnings import (
    CsvEarningsProvider,
    NullEarningsProvider,
    YFinanceEarningsProvider,
    YFinanceCalendarEarningsProvider,
)
from sentiment_analysis.providers.expected_move import YFinanceExpectedMoveProvider
from sentiment_analysis.providers.market_signals import build_market_signal_rows
from sentiment_analysis.providers.trends import XTrendsProvider
from sentiment_analysis.providers.x_api import XApiError, XRecentSearchProvider
from sentiment_analysis.recommendations import build_research_recommendation
from sentiment_analysis.screening import LOW_QUALITY_NAME_RE, CandidateScreen, screen_earnings_candidates
from sentiment_analysis.sentiment.vader import VaderSentimentAnalyzer
from sentiment_analysis.storage import load_scan_result, recent_runs, save_scan_result
from sentiment_analysis.tickers import build_trend_queries, load_ticker_universe
from sentiment_analysis.trading.storage import list_trade_plans
from sentiment_analysis.trading.research_storage import research_summary


DEFAULT_OPTIONS: dict[str, Any] = {
    "scan_mode": "cheap_earnings_week",
    "post_source": "x",
    "posts_csv": "examples/sample_posts.csv",
    "earnings_source": "yfinance_calendar",
    "earnings_csv": "examples/sample_earnings.csv",
    "ticker_file": "data/tickers_us.csv",
    "min_mentions": 5,
    "earnings_window_mode": "earnings_week",
    "earnings_horizon_days": 7,
    "hours_back": 24,
    "include_plain_tickers": False,
    "x_query_mode": "discovery",
    "x_max_results": 100,
    "max_pages_per_query": 1,
    "x_max_workers": 2,
    "enrich_market_signals": True,
    "option_activity_expirations": 2,
    "option_activity_min_volume": 250,
    "option_activity_min_vol_oi": 2.0,
    "option_activity_min_premium": 500000,
    "option_activity_max_moneyness_pct": 35,
    "insider_lookback_days": 30,
    "political_activity_enabled": True,
    "political_lookback_days": 365,
    "political_cache_hours": 24,
    "use_trends": False,
    "trend_woeids": "23424977,1",
    "trend_max_queries": 5,
    "max_tickers_to_fetch": 8,
    "max_candidates_to_count": 150,
    "expected_move_min_pct": 5,
    "max_expected_move_checks": 0,
    "expected_move_workers": 8,
    "skip_low_quality_names": True,
    "sqlite_path": "data/scans.sqlite3",
    "out_csv": "output/dashboard_signals.csv",
    "out_json": "output/dashboard_signals.json",
}


@dataclass
class DashboardState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    last_result: PipelineResult | None = None
    logs: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=lambda: DEFAULT_OPTIONS.copy())

    def append_log(self, message: str) -> None:
        stamped = f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC  {message}"
        with self.lock:
            self.logs.append(stamped)
            self.logs = self.logs[-160:]

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            result = self.last_result.as_dict() if self.last_result else None
            return {
                "running": self.running,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "error": self.error,
                "logs": list(self.logs),
                "options": dict(self.options),
                "result": result,
            }


STATE = DashboardState()
DASHBOARD_READ_ONLY = False
DASHBOARD_TITLE = "Sentiment Dashboard"


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "SentimentDashboard/0.1"

    def do_HEAD(self) -> None:
        if self.path in {"/", "/app.css", "/app.js", "/api/status", "/api/config", "/api/history", "/api/trade-plans", "/api/shadow-research"}:
            self.send_response(HTTPStatus.OK)
            self.end_headers()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_text(INDEX_HTML, "text/html; charset=utf-8")
        elif self.path == "/app.css":
            self._send_text(APP_CSS, "text/css; charset=utf-8")
        elif self.path == "/app.js":
            self._send_text(APP_JS, "text/javascript; charset=utf-8")
        elif self.path == "/api/status":
            self._send_json(STATE.snapshot())
        elif self.path == "/api/config":
            self._send_json(
                {
                    **DEFAULT_OPTIONS,
                    "dashboard_read_only": DASHBOARD_READ_ONLY,
                    "dashboard_title": DASHBOARD_TITLE,
                }
            )
        elif self.path == "/api/history":
            self._send_json(recent_runs(Path(str(DEFAULT_OPTIONS["sqlite_path"]))))
        elif self.path == "/api/trade-plans":
            plans = list_trade_plans(Path(str(DEFAULT_OPTIONS["sqlite_path"])), limit=40)
            self._send_json([plan.as_dict() for plan in plans])
        elif self.path == "/api/shadow-research":
            self._send_json(
                research_summary(
                    Path(str(DEFAULT_OPTIONS["sqlite_path"])),
                    limit=30,
                )
            )
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == "/api/load-run":
            self._handle_load_run()
            return
        if self.path != "/api/run":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if DASHBOARD_READ_ONLY:
            self._send_json(
                {"error": "this dashboard mirrors the primary scan and is read-only"},
                status=HTTPStatus.FORBIDDEN,
            )
            return

        try:
            body = self._read_json()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        with STATE.lock:
            if STATE.running:
                self._send_json({"error": "scan already running"}, status=HTTPStatus.CONFLICT)
                return
            options = {**DEFAULT_OPTIONS, **body}
            STATE.running = True
            STATE.started_at = datetime.now(timezone.utc).isoformat()
            STATE.finished_at = None
            STATE.error = None
            STATE.options = options
            STATE.logs = []

        thread = threading.Thread(target=_run_scan, args=(options,), daemon=True)
        thread.start()
        self._send_json({"ok": True})

    def _handle_load_run(self) -> None:
        try:
            body = self._read_json()
            run_id = str(body.get("run_id", "")).strip()
            if not run_id:
                raise ValueError("run_id is required")
            result, saved_options = load_scan_result(Path(str(DEFAULT_OPTIONS["sqlite_path"])), run_id)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        with STATE.lock:
            STATE.running = False
            STATE.started_at = result.started_at.isoformat()
            STATE.finished_at = result.finished_at.isoformat()
            STATE.error = None
            STATE.last_result = result
            STATE.options = {**DEFAULT_OPTIONS, **saved_options}
            STATE.logs = [f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC  Loaded saved scan {run_id}."]
            options = dict(STATE.options)
            logs = list(STATE.logs)

        self._send_json({"ok": True, "result": result.as_dict(), "options": options, "logs": logs})

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _send_text(self, text: str, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = text.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _run_scan(options: dict[str, Any]) -> None:
    try:
        result = run_dashboard_scan(options, STATE.append_log)
    except Exception as exc:
        STATE.append_log(f"Scan failed: {exc}")
        with STATE.lock:
            STATE.error = str(exc)
            STATE.running = False
            STATE.finished_at = datetime.now(timezone.utc).isoformat()
        return

    with STATE.lock:
        STATE.last_result = result
        STATE.running = False
        STATE.finished_at = datetime.now(timezone.utc).isoformat()
    STATE.append_log(f"Scan complete: {len(result.signals)} high-traction ticker(s).")


def run_dashboard_scan(options: dict[str, Any], log) -> PipelineResult:
    config = AppConfig.from_env()
    ticker_file = Path(str(options["ticker_file"]))
    ticker_universe = load_ticker_universe(ticker_file)
    log(f"Loaded {len(ticker_universe):,} ticker symbols from {ticker_file}.")

    if options.get("scan_mode") == "cheap_earnings_week" and options.get("post_source") == "x":
        result = _run_cheap_earnings_week_scan(options, config, ticker_universe, log)
        result = _with_market_signal_rows(result, options, config, log)
        _persist_outputs(result, options, log)
        return result

    post_provider = _build_post_provider(options, config)
    earnings_provider = _build_earnings_provider(options)
    trend_names, trend_queries = _load_trends(options, config, ticker_universe, log)

    pipeline_options = PipelineOptions(
        ticker_universe=ticker_universe,
        min_mentions=_as_int(options["min_mentions"], 25),
        earnings_horizon_days=_as_int(options["earnings_horizon_days"], 14),
        hours_back=_as_int(options["hours_back"], 24),
        include_plain_tickers=bool(options.get("include_plain_tickers")),
        query_mode=str(options.get("x_query_mode", "discovery")),
        x_query_max_chars=config.x_query_max_chars,
        extra_queries=tuple(trend_queries),
        trend_names=tuple(trend_names),
    )
    pipeline = SentimentPipeline(
        post_provider=post_provider,
        earnings_provider=earnings_provider,
        sentiment_analyzer=VaderSentimentAnalyzer(),
        options=pipeline_options,
    )

    preview_queries = pipeline._build_queries()
    log(
        f"Running {len(preview_queries)} query lane(s), "
        f"{_as_int(options['max_pages_per_query'], 10)} page(s) each, "
        f"max {_as_int(options['x_max_results'], 100)} posts/page."
    )
    result = _with_market_signal_rows(pipeline.run_with_metadata(), options, config, log)
    _persist_outputs(result, options, log)
    log(
        f"Scanned {result.raw_posts:,} raw post(s), matched {result.matched_posts:,}, "
        f"saw {result.tickers_seen:,} ticker(s)."
    )
    return result


def _run_cheap_earnings_week_scan(
    options: dict[str, Any],
    config: AppConfig,
    ticker_universe: set[str],
    log,
) -> PipelineResult:
    start_date, end_date = _earnings_window(options)
    log(f"Cheap mode window: {start_date.isoformat()} through {end_date.isoformat()}.")

    earnings_provider = _build_calendar_earnings_provider(options)
    earnings_events = _calendar_between(
        earnings_provider,
        start_date,
        end_date,
        ticker_universe=ticker_universe,
    )
    raw_candidate_count = len(earnings_events)
    expected_move_candidates = _expected_move_candidates(
        list(earnings_events.values()),
        options,
    )
    log(
        f"Found {raw_candidate_count} earnings candidate(s); "
        f"checking expected move for {len(expected_move_candidates)}."
    )
    enriched_events = YFinanceExpectedMoveProvider(
        max_workers=_as_int(options["expected_move_workers"], 8),
    ).enrich(expected_move_candidates)
    screened_events = screen_earnings_candidates(
        enriched_events.values(),
        screen=CandidateScreen(
            min_expected_move_pct=_as_float(options["expected_move_min_pct"], 5.0),
            require_expected_move=True,
            max_candidates=len(enriched_events),
            skip_low_quality_names=_as_bool(options.get("skip_low_quality_names"), True),
        ),
    )
    if not screened_events:
        if raw_candidate_count:
            log("No earnings candidates passed the expected-move gate.")
        else:
            log("No earnings candidates found for this window.")
        return PipelineResult(
            signals=[],
            queries=(),
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            raw_posts=0,
            duplicate_posts=0,
            matched_posts=0,
            tickers_seen=0,
            high_traction_tickers=0,
        )

    max_count_candidates = _as_int(options["max_candidates_to_count"], 150)
    count_events = _prioritize_for_x_counts(screened_events, max_count_candidates)
    candidate_tickers = [event.ticker for event in count_events]
    candidate_rows = _candidate_rows(screened_events, {})
    log(
        f"Expected-move gate found {len(screened_events)} ticker(s) "
        f"at or above {_as_float(options['expected_move_min_pct'], 5.0):.1f}%; "
        f"counting X chatter for {len(candidate_tickers)} prioritized ticker(s)."
    )
    if not candidate_tickers:
        log("Max X Counts is 0, so no live X counts were requested.")
        return _empty_result(candidate_rows=tuple(candidate_rows))

    count_queries = {ticker: f"${ticker} lang:en -is:retweet" for ticker in candidate_tickers}
    counter = XRecentSearchProvider(
        config.x_bearer_token or "",
        max_results=10,
        max_pages_per_query=1,
        max_workers=1,
    )
    try:
        counts_by_query = counter.recent_counts(
            list(count_queries.values()),
            start_time=datetime.now(timezone.utc).replace(microsecond=0)
            - timedelta(hours=_as_int(options["hours_back"], 24)),
        )
    except XApiError as exc:
        log(f"X counts unavailable: {exc}")
        return _empty_result(candidate_rows=tuple(candidate_rows))

    preflight_counts = {
        ticker: counts_by_query.get(query, 0)
        for ticker, query in count_queries.items()
    }
    candidate_rows = _candidate_rows(screened_events, preflight_counts)
    count_floor = _as_int(options["min_mentions"], 5)
    max_tickers = _as_int(options["max_tickers_to_fetch"], 8)
    selected_tickers = _select_tickers_for_posts(
        preflight_counts,
        min_mentions=count_floor,
        max_tickers=max_tickers,
    )

    log(
        f"Counts preflight kept {len(selected_tickers)} ticker(s): "
        f"{', '.join(selected_tickers) if selected_tickers else 'none'}."
    )
    if not selected_tickers:
        return _empty_result(
            preflight_counts=preflight_counts,
            candidate_rows=tuple(candidate_rows),
        )

    post_provider = XRecentSearchProvider(
        config.x_bearer_token or "",
        max_results=_as_int(options["x_max_results"], 100),
        max_pages_per_query=_as_int(options["max_pages_per_query"], 1),
        max_workers=_as_int(options["x_max_workers"], config.x_max_workers),
    )
    manual_queries = tuple(f"${ticker} lang:en -is:retweet" for ticker in selected_tickers)
    log(
        f"Fetching recent-search posts for {len(manual_queries)} query lane(s), "
        f"capped at about {_estimated_post_read_cap(options, len(manual_queries)):,} post read(s)."
    )
    pipeline = SentimentPipeline(
        post_provider=post_provider,
        earnings_provider=StaticEarningsProvider(enriched_events),
        sentiment_analyzer=VaderSentimentAnalyzer(),
        options=PipelineOptions(
            ticker_universe=set(selected_tickers),
            min_mentions=count_floor,
            earnings_horizon_days=_as_int(options["earnings_horizon_days"], 7),
            hours_back=_as_int(options["hours_back"], 24),
            query_mode="manual",
            extra_queries=manual_queries,
            preflight_counts=preflight_counts,
            selected_tickers=tuple(selected_tickers),
        ),
    )
    result = pipeline.run_with_metadata()
    result = replace(result, candidate_rows=tuple(candidate_rows))
    return _enforce_cheap_signal_gate(result, options, start_date, end_date, log)


def _with_market_signal_rows(
    result: PipelineResult,
    options: dict[str, Any],
    config: AppConfig,
    log,
) -> PipelineResult:
    if not _as_bool(options.get("enrich_market_signals"), True):
        return result
    tickers = [signal.ticker for signal in result.signals]
    if not tickers:
        return result

    log(
        f"Checking options, SEC insider filings, and political disclosures "
        f"for {len(tickers)} flagged ticker(s)."
    )
    market_rows = build_market_signal_rows(
        tickers,
        option_expirations=_as_int(options.get("option_activity_expirations"), 2),
        option_min_volume=_as_int(options.get("option_activity_min_volume"), 250),
        option_min_vol_oi_ratio=_as_float(options.get("option_activity_min_vol_oi"), 2.0),
        option_min_premium=_as_float(options.get("option_activity_min_premium"), 500000),
        option_max_moneyness_distance_pct=_as_float(
            options.get("option_activity_max_moneyness_pct"),
            35,
        ),
        insider_lookback_days=_as_int(options.get("insider_lookback_days"), 30),
        political_activity_enabled=_as_bool(options.get("political_activity_enabled"), True),
        political_lookback_days=_as_int(options.get("political_lookback_days"), 365),
        political_cache_hours=_as_int(options.get("political_cache_hours"), 24),
        sec_user_agent=config.sec_user_agent,
        max_workers=_as_int(options.get("x_max_workers"), config.x_max_workers),
    )
    market_by_ticker = {str(row.get("ticker")): row for row in market_rows}
    rows = [
        _with_recommendation(signal, market_by_ticker.get(signal.ticker, {"ticker": signal.ticker}))
        for signal in result.signals
    ]
    unusual_count = sum(1 for row in rows if row.get("option_alert") == "unusual")
    insider_count = sum(1 for row in rows if int(row.get("insider_recent_count") or 0) > 0)
    political_count = sum(
        1 for row in rows if row.get("political_bias") in {"bullish", "bearish"}
    )
    log(
        f"Market checks found {unusual_count} unusual option activity row(s) "
        f"{insider_count} recent insider filing row(s), and "
        f"{political_count} directional political-disclosure row(s)."
    )
    return replace(result, market_signal_rows=tuple(rows))


def _enforce_cheap_signal_gate(
    result: PipelineResult,
    options: dict[str, Any],
    start_date: date,
    end_date: date,
    log,
) -> PipelineResult:
    min_mentions = _as_int(options.get("min_mentions"), 5)
    min_expected_move = _as_float(options.get("expected_move_min_pct"), 5.0)
    eligible = []
    rejected = []
    for signal in result.signals:
        reason = _cheap_signal_rejection_reason(
            signal,
            min_mentions=min_mentions,
            min_expected_move=min_expected_move,
            start_date=start_date,
            end_date=end_date,
        )
        if reason:
            rejected.append(f"{signal.ticker} ({reason})")
        else:
            eligible.append(signal)

    if rejected:
        log(f"Final eligibility gate rejected {len(rejected)} ticker(s): {', '.join(rejected)}.")
    log(
        f"Final eligibility gate kept {len(eligible)} ticker(s) with earnings in-window, "
        f"expected move >= {min_expected_move:.1f}%, X count >= {min_mentions}, "
        f"and sampled mentions >= {min_mentions}."
    )
    return replace(
        result,
        signals=eligible,
        high_traction_tickers=len(eligible),
    )


def _cheap_signal_rejection_reason(
    signal,
    *,
    min_mentions: int,
    min_expected_move: float,
    start_date: date,
    end_date: date,
) -> str | None:
    if signal.earnings_date is None or not start_date <= signal.earnings_date <= end_date:
        return "earnings outside scan window"
    if signal.expected_move_pct is None or signal.expected_move_pct < min_expected_move:
        return f"expected move below {min_expected_move:.1f}%"
    if signal.preflight_count is None or signal.preflight_count < min_mentions:
        return f"X count below {min_mentions}"
    if signal.mention_count < min_mentions:
        return f"sample mentions below {min_mentions}"
    return None


def _with_recommendation(signal, market_row: dict[str, object]) -> dict[str, object]:
    recommendation = build_research_recommendation(
        crowd_bias=signal.crowd_bias,
        days_until_earnings=signal.days_until_earnings,
        option_alert=_as_optional_str(market_row.get("option_alert")),
        option_bias=_as_optional_str(market_row.get("option_bias")),
        insider_recent_count=_as_int(market_row.get("insider_recent_count"), 0),
        political_alert=_as_optional_str(market_row.get("political_alert")),
        political_bias=_as_optional_str(market_row.get("political_bias")),
        political_strength=_as_int(market_row.get("political_strength"), 0),
        political_buyers=_as_int(market_row.get("political_buyers"), 0),
        political_sellers=_as_int(market_row.get("political_sellers"), 0),
        political_high_profile_count=len(
            market_row.get("political_directional_high_profile_names") or []
        ),
    )
    return {**market_row, **recommendation.as_dict()}


def _candidate_rows(events, preflight_counts: dict[str, int]) -> list[dict[str, object]]:
    rows = []
    for event in events:
        rows.append(
            {
                "ticker": event.ticker,
                "company": event.company,
                "earnings_date": event.earnings_date.isoformat() if event.earnings_date else None,
                "timing": event.timing,
                "expected_move_pct": round(event.expected_move_pct, 4)
                if event.expected_move_pct is not None
                else None,
                "expected_move_abs": round(event.expected_move_abs, 4)
                if event.expected_move_abs is not None
                else None,
                "expected_move_source": event.expected_move_source,
                "options_expiration": event.options_expiration.isoformat()
                if event.options_expiration
                else None,
                "x_count": preflight_counts.get(event.ticker),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["x_count"] if isinstance(row["x_count"], int) else -1,
            row["expected_move_pct"] if isinstance(row["expected_move_pct"], float) else -1,
        ),
        reverse=True,
    )


def _prioritize_for_x_counts(events, max_candidates: int) -> list:
    if max_candidates <= 0:
        return []
    pool = list(events)
    selected: dict[str, object] = {}

    liquid_events = [event for event in pool if (event.market_cap or 0.0) > 0]
    reserve_slots = min(len(liquid_events), max_candidates // 2)
    for event in sorted(liquid_events, key=_liquidity_score, reverse=True)[:reserve_slots]:
        selected[event.ticker] = event

    remaining_slots = max_candidates - len(selected)
    for event in sorted(pool, key=_x_count_priority_score, reverse=True):
        if remaining_slots <= 0:
            break
        if event.ticker in selected:
            continue
        selected[event.ticker] = event
        remaining_slots -= 1

    return list(selected.values())


def _select_tickers_for_posts(
    preflight_counts: dict[str, int],
    *,
    min_mentions: int,
    max_tickers: int,
) -> list[str]:
    return [
        ticker
        for ticker, count in sorted(preflight_counts.items(), key=lambda item: item[1], reverse=True)
        if count >= min_mentions
    ][: max(0, max_tickers)]


def _liquidity_score(event) -> tuple[float, float, float]:
    return (
        event.market_cap or 0.0,
        event.expected_move_pct or 0.0,
        -len(event.ticker),
    )


def _x_count_priority_score(event) -> float:
    expected_move = event.expected_move_pct or 0.0
    market_cap = event.market_cap or 0.0
    liquidity_bonus = 0.0
    if market_cap > 0:
        liquidity_bonus = max(0.0, min(math.log10(market_cap), 13.0) - 8.5) * 2.0
    timing_bonus = 0.75 if event.timing in {"BMO", "AMC"} else 0.0
    ticker_bonus = max(0, 5 - len(event.ticker)) * 0.15
    return expected_move + liquidity_bonus + timing_bonus + ticker_bonus


def _empty_result(
    *,
    candidate_rows: tuple[dict[str, object], ...] = (),
    preflight_counts: dict[str, int] | None = None,
) -> PipelineResult:
    now = datetime.now(timezone.utc)
    return PipelineResult(
        signals=[],
        queries=(),
        started_at=now,
        finished_at=now,
        raw_posts=0,
        duplicate_posts=0,
        matched_posts=0,
        tickers_seen=0,
        high_traction_tickers=0,
        preflight_counts=preflight_counts or {},
        selected_tickers=(),
        candidate_rows=candidate_rows,
        market_signal_rows=(),
    )


class StaticEarningsProvider(NullEarningsProvider):
    def __init__(self, events: dict[str, object]) -> None:
        self.events = events

    def upcoming(self, tickers, *, horizon_days: int):
        del horizon_days
        return {ticker: self.events.get(ticker) for ticker in tickers}


def _persist_outputs(result: PipelineResult, options: dict[str, Any], log) -> None:
    write_signals_csv(result.signals, Path(str(options["out_csv"])))
    write_signals_json(result.signals, Path(str(options["out_json"])))
    run_id = save_scan_result(result, options, Path(str(options["sqlite_path"])))
    log(f"Saved scan to SQLite run {run_id}.")


def _build_post_provider(options: dict[str, Any], config: AppConfig):
    if options.get("post_source") == "csv":
        return CsvPostProvider(Path(str(options["posts_csv"])))
    return XRecentSearchProvider(
        config.x_bearer_token or "",
        max_results=_as_int(options["x_max_results"], 100),
        max_pages_per_query=_as_int(options["max_pages_per_query"], 10),
        max_workers=_as_int(options["x_max_workers"], config.x_max_workers),
    )


def _build_earnings_provider(options: dict[str, Any]):
    source = options.get("earnings_source")
    if source == "csv":
        return CsvEarningsProvider(Path(str(options["earnings_csv"])))
    if source == "yfinance_calendar":
        return YFinanceCalendarEarningsProvider()
    if source == "yfinance":
        return YFinanceEarningsProvider()
    return NullEarningsProvider()


def _build_calendar_earnings_provider(options: dict[str, Any]):
    source = options.get("earnings_source")
    if source == "csv":
        return CsvEarningsProvider(Path(str(options["earnings_csv"])))
    return YFinanceCalendarEarningsProvider()


def _calendar_between(
    provider,
    start_date: date,
    end_date: date,
    *,
    ticker_universe: set[str],
):
    if hasattr(provider, "between"):
        return provider.between(
            start_date,
            end_date,
            ticker_universe=ticker_universe,
        )
    horizon_days = (end_date - date.today()).days
    events = provider.upcoming(sorted(ticker_universe), horizon_days=max(0, horizon_days))
    return {
        ticker: event
        for ticker, event in events.items()
        if event and event.earnings_date and start_date <= event.earnings_date <= end_date
    }


def _earnings_window(options: dict[str, Any]) -> tuple[date, date]:
    if options.get("earnings_window_mode") == "days":
        return days_window(_as_int(options["earnings_horizon_days"], 7))
    return earnings_week_window()


def _expected_move_candidates(
    events: list,
    options: dict[str, Any],
):
    skip_low_quality = _as_bool(options.get("skip_low_quality_names"), True)
    candidates = [
        event
        for event in events
        if not (skip_low_quality and event.company and LOW_QUALITY_NAME_RE.search(event.company))
    ]
    candidates.sort(key=lambda event: (event.earnings_date or date.max, event.ticker))
    max_checks = _as_int(options.get("max_expected_move_checks"), 0)
    if max_checks > 0:
        return candidates[:max_checks]
    return candidates


def _estimated_post_read_cap(options: dict[str, Any], query_count: int) -> int:
    return (
        max(0, query_count)
        * _as_int(options.get("max_pages_per_query"), 1)
        * _as_int(options.get("x_max_results"), 100)
    )


def _load_trends(
    options: dict[str, Any],
    config: AppConfig,
    ticker_universe: set[str],
    log,
) -> tuple[list[str], list[str]]:
    if not options.get("use_trends"):
        return [], []
    provider = XTrendsProvider(config.x_bearer_token or "")
    woeids = _parse_woeids(str(options.get("trend_woeids", "23424977,1")))
    log(f"Pulling X trends for WOEID(s): {', '.join(str(value) for value in woeids)}.")
    trends = provider.current_trends(woeids)
    trends = sorted(trends, key=lambda trend: (-(trend.tweet_count or -1), trend.name))
    trend_names = [trend.name for trend in trends]
    trend_queries = build_trend_queries(
        trend_names,
        ticker_universe,
        max_queries=_as_int(options["trend_max_queries"], 25),
    )
    log(f"Pulled {len(trend_names)} trend(s), generated {len(trend_queries)} trend query lane(s).")
    return trend_names, trend_queries


def _parse_woeids(raw: str) -> list[int]:
    values: list[int] = []
    for chunk in raw.split(","):
        clean = chunk.strip()
        if clean:
            values.append(int(clean))
    return values or [23424977, 1]


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def main(argv: list[str] | None = None) -> int:
    global DASHBOARD_READ_ONLY, DASHBOARD_TITLE
    parser = argparse.ArgumentParser(prog="sentiment-dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--sqlite-path", type=Path, default=Path("data/scans.sqlite3"))
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--title", default="Sentiment Dashboard")
    args = parser.parse_args(argv)

    DEFAULT_OPTIONS["sqlite_path"] = str(args.sqlite_path)
    STATE.options["sqlite_path"] = str(args.sqlite_path)
    DASHBOARD_READ_ONLY = args.read_only
    DASHBOARD_TITLE = args.title

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    return 0


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sentiment Dashboard</title>
  <link rel="stylesheet" href="/app.css" />
</head>
<body>
  <main class="shell">
    <section class="topbar">
      <div>
        <h1 id="dashboardTitle">Sentiment Dashboard</h1>
        <p id="subhead">Recent-search stock sentiment scanner</p>
      </div>
      <button id="runButton" class="primary">Run Scan</button>
    </section>

    <section class="controls" aria-label="scan controls">
      <label>Scan Mode
        <select id="scan_mode">
          <option value="cheap_earnings_week">Cheap Earnings Week</option>
          <option value="broad_discovery">Broad Discovery</option>
        </select>
      </label>
      <label>Posts
        <select id="post_source">
          <option value="csv">CSV Demo</option>
          <option value="x">X Recent Search</option>
        </select>
      </label>
      <label>Earnings
        <select id="earnings_source">
          <option value="csv">CSV Demo</option>
          <option value="yfinance_calendar">yfinance Calendar</option>
          <option value="yfinance">yfinance</option>
          <option value="none">None</option>
        </select>
      </label>
      <label>Min Mentions
        <input id="min_mentions" type="number" min="1" max="500" step="1" />
      </label>
      <label>Hours Back
        <input id="hours_back" type="number" min="1" max="168" step="1" />
      </label>
      <label>Window
        <select id="earnings_window_mode">
          <option value="earnings_week">Earnings Week</option>
          <option value="days">Days</option>
        </select>
      </label>
      <label>Earnings Days
        <input id="earnings_horizon_days" type="number" min="1" max="90" step="1" />
      </label>
      <label>Max Tickers
        <input id="max_tickers_to_fetch" type="number" min="1" max="100" step="1" />
      </label>
      <label>Max X Counts
        <input id="max_candidates_to_count" type="number" min="1" max="300" step="1" />
      </label>
      <label>Min Move %
        <input id="expected_move_min_pct" type="number" min="0" max="100" step="0.5" />
      </label>
      <label>EM Checks
        <input id="max_expected_move_checks" type="number" min="0" max="1000" step="10" />
      </label>
      <label>EM Workers
        <input id="expected_move_workers" type="number" min="1" max="16" step="1" />
      </label>
      <label>Skip Junk Names
        <select id="skip_low_quality_names">
          <option value="true">On</option>
          <option value="false">Off</option>
        </select>
      </label>
      <label>Posts / Page
        <input id="x_max_results" type="number" min="10" max="100" step="10" />
      </label>
      <label>Pages / Query
        <input id="max_pages_per_query" type="number" min="1" max="100" step="1" />
      </label>
      <label>Workers
        <input id="x_max_workers" type="number" min="1" max="12" step="1" />
      </label>
      <label>Market Checks
        <select id="enrich_market_signals">
          <option value="true">On</option>
          <option value="false">Off</option>
        </select>
      </label>
      <label>Opt Exp
        <input id="option_activity_expirations" type="number" min="1" max="6" step="1" />
      </label>
      <label>Opt Vol
        <input id="option_activity_min_volume" type="number" min="1" max="10000" step="50" />
      </label>
      <label>Vol/OI
        <input id="option_activity_min_vol_oi" type="number" min="0" max="100" step="0.5" />
      </label>
      <label>Opt Premium
        <input id="option_activity_min_premium" type="number" min="0" max="10000000" step="50000" />
      </label>
      <label>Max OTM %
        <input id="option_activity_max_moneyness_pct" type="number" min="0" max="500" step="5" />
      </label>
      <label>Insider Days
        <input id="insider_lookback_days" type="number" min="1" max="365" step="1" />
      </label>
      <label>Political Checks
        <select id="political_activity_enabled">
          <option value="true">On</option>
          <option value="false">Off</option>
        </select>
      </label>
      <label>Political Days
        <input id="political_lookback_days" type="number" min="30" max="1095" step="30" />
      </label>
      <label>Trend Pull
        <select id="use_trends">
          <option value="false">Off</option>
          <option value="true">On</option>
        </select>
      </label>
      <label>WOEIDs
        <input id="trend_woeids" type="text" />
      </label>
      <label>Trend Queries
        <input id="trend_max_queries" type="number" min="0" max="100" step="1" />
      </label>
    </section>

    <section class="metrics" aria-label="scan metrics">
      <div class="metric"><span>Raw Posts</span><strong id="raw_posts">0</strong></div>
      <div class="metric"><span>Matched</span><strong id="matched_posts">0</strong></div>
      <div class="metric"><span>Tickers Seen</span><strong id="tickers_seen">0</strong></div>
      <div class="metric"><span>Signals</span><strong id="high_traction_tickers">0</strong></div>
      <div class="metric"><span>Queries</span><strong id="query_count">0</strong></div>
      <div class="metric"><span>Elapsed</span><strong id="elapsed">0s</strong></div>
    </section>

    <section class="split">
      <div class="panel table-panel">
        <div class="panel-head">
          <h2>High-Traction Earnings Watch</h2>
          <span id="statusPill" class="pill idle">Idle</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Exp Move</th>
                <th>X Count</th>
                <th>Mentions</th>
                <th>Authors</th>
                <th>Engagement</th>
                <th>Sentiment</th>
                <th>Bias</th>
                <th>Earnings</th>
                <th>Options</th>
                <th>Insider</th>
                <th>Politics</th>
                <th>Watch</th>
              </tr>
            </thead>
            <tbody id="signalsBody"></tbody>
          </table>
        </div>
      </div>

      <aside class="panel side">
        <div>
          <h2>Trend Feed</h2>
          <div id="trendList" class="chips"></div>
        </div>
        <div>
          <h2>Expected-Move Candidates</h2>
          <div id="candidateBox" class="candidate-list"></div>
        </div>
        <div>
          <h2>Market Evidence</h2>
          <div id="marketSignalsBox" class="market-signal-list"></div>
        </div>
        <div>
          <h2>Automated Trade Plans</h2>
          <div id="tradePlansBox" class="trade-plan-list"></div>
        </div>
        <div>
          <h2>Shadow Research</h2>
          <div id="shadowResearchBox" class="shadow-research-list"></div>
        </div>
        <div>
          <h2>Run Log</h2>
          <pre id="logBox"></pre>
        </div>
        <div>
          <h2>Run History</h2>
          <div id="historyBox" class="history"></div>
        </div>
      </aside>
    </section>
  </main>
  <script src="/app.js"></script>
</body>
</html>
"""


APP_CSS = """
:root {
  color-scheme: dark;
  --bg: #111318;
  --band: #181b22;
  --panel: #20242c;
  --line: #353b46;
  --text: #f2f4f8;
  --muted: #a8b0bd;
  --green: #4fd18b;
  --red: #ff6d70;
  --amber: #f4bd4f;
  --blue: #68a8ff;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.shell { min-height: 100vh; padding: 24px; }
.topbar {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: center;
  margin-bottom: 18px;
}
h1, h2, p { margin: 0; }
h1 { font-size: 28px; font-weight: 760; }
h2 { font-size: 15px; font-weight: 720; }
#subhead { margin-top: 4px; color: var(--muted); font-size: 14px; }
.primary {
  border: 1px solid #6fb1ff;
  background: #2d72d9;
  color: white;
  min-height: 42px;
  padding: 0 18px;
  border-radius: 8px;
  font-weight: 760;
  cursor: pointer;
}
.primary:disabled { opacity: .55; cursor: wait; }
.controls {
  display: grid;
  grid-template-columns: repeat(8, minmax(120px, 1fr));
  gap: 10px;
  padding: 14px;
  background: var(--band);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin-bottom: 14px;
}
label { color: var(--muted); font-size: 12px; display: grid; gap: 6px; }
input, select {
  width: 100%;
  min-height: 36px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #12151b;
  color: var(--text);
  padding: 0 10px;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 10px;
  margin-bottom: 14px;
}
.metric {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  min-height: 78px;
  padding: 14px;
  display: grid;
  align-content: space-between;
}
.metric span { color: var(--muted); font-size: 12px; }
.metric strong { font-size: 24px; }
.split {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 14px;
  min-width: 0;
}
.split > * { min-width: 0; }
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 16px;
  border-bottom: 1px solid var(--line);
}
.pill {
  border-radius: 999px;
  padding: 5px 9px;
  font-size: 12px;
  font-weight: 760;
}
.pill.idle { color: var(--muted); background: #2a2f39; }
.pill.running { color: #08150d; background: var(--green); }
.pill.error { color: #210608; background: var(--red); }
.table-wrap { overflow: auto; width: 100%; max-width: 100%; max-height: calc(100vh - 285px); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
th { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0; background: #1b1f27; position: sticky; top: 0; }
td.ticker { font-weight: 800; color: var(--blue); }
.bias-bullish { color: var(--green); }
.bias-bearish { color: var(--red); }
.bias-neutral { color: var(--amber); }
.side {
  padding: 14px;
  display: grid;
  gap: 18px;
  align-content: start;
}
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; max-height: 174px; overflow: auto; }
.chip {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 6px 9px;
  color: var(--text);
  background: #151922;
  font-size: 12px;
}
pre {
  white-space: pre-wrap;
  min-height: 240px;
  max-height: calc(100vh - 500px);
  overflow: auto;
  margin: 10px 0 0;
  color: #c8d0dc;
  font-size: 12px;
  line-height: 1.5;
}
.history { display: grid; gap: 8px; margin-top: 10px; }
.candidate-list { display: grid; gap: 8px; margin-top: 10px; max-height: 260px; overflow: auto; }
.market-signal-list { display: grid; gap: 8px; margin-top: 10px; max-height: 260px; overflow: auto; }
.trade-plan-list { display: grid; gap: 8px; margin-top: 10px; max-height: 300px; overflow: auto; }
.shadow-research-list { display: grid; gap: 8px; margin-top: 10px; max-height: 320px; overflow: auto; }
.candidate-row {
  display: grid;
  grid-template-columns: 52px 1fr auto;
  gap: 8px;
  align-items: center;
  padding: 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #151922;
}
.candidate-row strong { color: var(--blue); font-size: 12px; }
.candidate-row span { color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.candidate-row em { color: var(--green); font-style: normal; font-size: 12px; font-weight: 760; }
.market-row {
  display: grid;
  grid-template-columns: 52px 1fr auto;
  gap: 8px;
  align-items: center;
  padding: 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #151922;
}
.market-row strong { color: var(--blue); font-size: 12px; }
.market-row span { color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.market-row em { color: var(--amber); font-style: normal; font-size: 12px; font-weight: 760; }
.trade-plan-row {
  display: grid;
  grid-template-columns: 52px 1fr auto;
  gap: 8px;
  align-items: center;
  padding: 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #151922;
}
.trade-plan-row strong { color: var(--blue); font-size: 12px; }
.trade-plan-row span { color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.trade-plan-row em { color: var(--amber); font-style: normal; font-size: 12px; font-weight: 760; }
.trade-plan-row .status-open, .trade-plan-row .status-closed { color: var(--green); }
.trade-plan-row .status-error, .trade-plan-row .status-skipped { color: var(--red); }
.shadow-summary { color: var(--muted); font-size: 12px; line-height: 1.5; }
.shadow-row {
  display: grid;
  grid-template-columns: 52px 1fr auto;
  gap: 8px;
  align-items: center;
  padding: 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #151922;
}
.shadow-row strong { color: var(--blue); font-size: 12px; }
.shadow-row span { color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.shadow-row em { color: var(--amber); font-style: normal; font-size: 12px; font-weight: 760; }
.shadow-row em.positive { color: var(--green); }
.shadow-row em.negative { color: var(--red); }
  .alert-unusual, .alert-recent { color: var(--red); font-weight: 760; }
  .alert-active { color: var(--amber); font-weight: 760; }
  .alert-quiet, .alert-none, .alert-unknown, .alert-disabled { color: var(--muted); }
  .politics-bullish { color: var(--green); font-weight: 760; }
  .politics-bearish { color: var(--red); font-weight: 760; }
  .politics-mixed { color: var(--amber); font-weight: 760; }
  .politics-none, .politics-unknown { color: var(--muted); }
  .politics-link { color: inherit; text-decoration: none; }
  .politics-link:hover { text-decoration: underline; }
  .watch-confirm { color: var(--green); font-weight: 760; }
  .watch-conflict { color: var(--red); font-weight: 760; }
  .watch-override { color: var(--blue); font-weight: 800; }
.watch-base { color: var(--amber); font-weight: 760; }
.watch-skip { color: var(--muted); }
.history-row {
  display: grid;
  gap: 3px;
  padding: 9px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #151922;
  color: var(--text);
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.history-row:hover { border-color: var(--blue); }
.history-row strong { font-size: 12px; }
.history-row span, .muted { color: var(--muted); font-size: 12px; }
@media (max-width: 1100px) {
  .controls { grid-template-columns: repeat(4, minmax(120px, 1fr)); }
  .metrics { grid-template-columns: repeat(3, 1fr); }
  .split { grid-template-columns: 1fr; }
  .table-wrap { max-height: none; }
}
@media (max-width: 640px) {
  .shell { padding: 14px; }
  .topbar { align-items: stretch; flex-direction: column; }
  .controls { grid-template-columns: 1fr 1fr; }
  .metrics { grid-template-columns: 1fr 1fr; }
}
"""


APP_JS = """
const fields = [
  "scan_mode", "post_source", "earnings_source", "min_mentions", "hours_back",
  "earnings_window_mode", "earnings_horizon_days", "max_tickers_to_fetch",
  "max_candidates_to_count", "expected_move_min_pct", "max_expected_move_checks",
  "expected_move_workers", "skip_low_quality_names", "x_max_results", "max_pages_per_query",
  "x_max_workers", "enrich_market_signals", "option_activity_expirations",
  "option_activity_min_volume", "option_activity_min_vol_oi", "option_activity_min_premium",
  "option_activity_max_moneyness_pct", "insider_lookback_days", "political_activity_enabled",
  "political_lookback_days", "use_trends",
  "trend_woeids", "trend_max_queries"
];
let busy = false;

async function loadConfig() {
  const config = await fetch("/api/config").then(r => r.json());
  applyOptions(config);
  document.getElementById("dashboardTitle").textContent = config.dashboard_title || "Sentiment Dashboard";
  if (config.dashboard_read_only) document.getElementById("runButton").hidden = true;
}

function applyOptions(config) {
  for (const field of fields) {
    const el = document.getElementById(field);
    if (!el) continue;
    el.value = String(config[field] ?? "");
  }
}

function collectOptions() {
  const options = {};
  for (const field of fields) {
    const el = document.getElementById(field);
    if (!el) continue;
    options[field] = el.value;
  }
  options.use_trends = document.getElementById("use_trends").value === "true";
  options.skip_low_quality_names = document.getElementById("skip_low_quality_names").value === "true";
  options.enrich_market_signals = document.getElementById("enrich_market_signals").value === "true";
  options.political_activity_enabled = document.getElementById("political_activity_enabled").value === "true";
  return options;
}

async function runScan() {
  if (busy) return;
  busy = true;
  renderRunning(true);
  const response = await fetch("/api/run", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(collectOptions())
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    alert(payload.error || "Scan could not start");
    busy = false;
    renderRunning(false);
  }
}

async function loadRun(runId) {
  if (busy || !runId) return;
  const response = await fetch("/api/load-run", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ run_id: runId })
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    alert(payload.error || "Saved run could not be loaded");
    return;
  }
  applyOptions(payload.options || {});
  renderResult(payload.result);
  renderLogs(payload.logs || []);
}

async function poll() {
  const payload = await fetch("/api/status").then(r => r.json());
  busy = payload.running;
  renderRunning(payload.running, payload.error);
  renderResult(payload.result);
  renderLogs(payload.logs || []);
  renderHistory();
  renderTradePlans();
  renderShadowResearch();
  setTimeout(poll, payload.running ? 900 : 2000);
}

function renderRunning(running, error) {
  const button = document.getElementById("runButton");
  const pill = document.getElementById("statusPill");
  button.disabled = running;
  button.textContent = running ? "Running" : "Run Scan";
  pill.className = "pill " + (error ? "error" : running ? "running" : "idle");
  pill.textContent = error ? "Error" : running ? "Running" : "Idle";
}

function renderResult(result) {
  const zero = { raw_posts: 0, matched_posts: 0, tickers_seen: 0, high_traction_tickers: 0, queries: [], elapsed_seconds: 0, signals: [], trend_names: [], candidate_rows: [], market_signal_rows: [] };
  const data = result || zero;
  setMetric("raw_posts", data.raw_posts);
  setMetric("matched_posts", data.matched_posts);
  setMetric("tickers_seen", data.tickers_seen);
  setMetric("high_traction_tickers", data.high_traction_tickers);
  setMetric("query_count", data.queries.length);
  setMetric("elapsed", `${data.elapsed_seconds || 0}s`);
  renderSignals(data.signals || [], data.market_signal_rows || []);
  renderTrends(data.trend_names || []);
  renderCandidates(data.candidate_rows || []);
  renderMarketSignals(data.market_signal_rows || []);
}

function setMetric(id, value) {
  document.getElementById(id).textContent = Number.isFinite(value) ? value.toLocaleString() : value;
}

function renderSignals(signals, marketRows) {
  const body = document.getElementById("signalsBody");
  if (!signals.length) {
    body.innerHTML = `<tr><td colspan="13" class="empty">No high-traction tickers yet</td></tr>`;
    return;
  }
  const marketByTicker = Object.fromEntries((marketRows || []).map(row => [row.ticker, row]));
  body.innerHTML = signals.map(signal => `
    <tr>
      ${marketSignalCells(signal, marketByTicker[signal.ticker])}
    </tr>
  `).join("");
}

function marketSignalCells(signal, market) {
  const optionAlert = market?.option_alert || "unknown";
  const optionBias = market?.option_bias || "";
  const insiderAlert = market?.insider_alert || "unknown";
  const insiderText = market?.insider_recent_count ? `${formatNumber(market.insider_recent_count)} Form ${market.insider_latest_form || "filing"}` : insiderAlert;
  const politicalBias = market?.political_bias || "unknown";
  const politicalText = politicalLabel(market);
  const politicalTitle = politicalTooltip(market);
  const politicalUrl = safeExternalUrl(market?.political_source_url);
  const politicalContent = politicalUrl
    ? `<a class="politics-link" href="${escapeHtml(politicalUrl)}" target="_blank" rel="noreferrer" title="${escapeHtml(politicalTitle)}">${escapeHtml(politicalText)}</a>`
    : `<span title="${escapeHtml(politicalTitle)}">${escapeHtml(politicalText)}</span>`;
  const recommendation = market?.recommendation || signal.contrarian_watch;
  const score = market?.recommendation_score == null ? "" : ` ${market.recommendation_score}`;
  return `
      <td class="ticker">${escapeHtml(signal.ticker)}</td>
      <td>${signal.expected_move_pct == null ? "na" : `${Number(signal.expected_move_pct).toFixed(1)}%`}</td>
      <td>${signal.preflight_count == null ? "na" : formatNumber(signal.preflight_count)}</td>
      <td>${formatNumber(signal.mention_count)}</td>
      <td>${formatNumber(signal.unique_authors)}</td>
      <td>${formatNumber(signal.total_engagement)}</td>
      <td>${Number(signal.avg_sentiment).toFixed(3)}</td>
      <td class="bias-${escapeHtml(signal.crowd_bias)}">${escapeHtml(signal.crowd_bias)}</td>
      <td>${signal.earnings_date || "unknown"}</td>
      <td class="alert-${escapeHtml(optionAlert)}">${escapeHtml(optionLabel(optionAlert, optionBias))}</td>
      <td class="alert-${escapeHtml(insiderAlert)}">${escapeHtml(insiderText)}</td>
      <td class="politics-${escapeHtml(politicalBias)}">${politicalContent}</td>
      <td class="${recommendationClass(recommendation)}">${escapeHtml(recommendationLabel(recommendation))}${escapeHtml(score)}</td>
  `;
}

function politicalLabel(row) {
  const bias = row?.political_bias || "unknown";
  if (["unknown", "none"].includes(bias)) return row?.political_alert || bias;
  const prefix = row?.political_alert === "strong" ? "strong " : "";
  const hold = row?.political_signal_basis === "no_later_sale_cluster" ? "hold " : "";
  return `${prefix}${bias} ${hold}${formatNumber(row?.political_buyers)}B/${formatNumber(row?.political_sellers)}S`;
}

function politicalTooltip(row) {
  if (!row) return "No political-disclosure data";
  const buyers = (row.political_buyer_names || []).join(", ") || "none";
  const sellers = (row.political_seller_names || []).join(", ") || "none";
  const stale = row.political_stale ? " Cached data may be stale." : "";
  const net = row.political_net_strength == null ? "unknown" : Number(row.political_net_strength).toLocaleString();
  return `Hold-proxy buyers with no later sale found: ${buyers}. Latest-side sellers: ${sellers}. Latest-side net strength: ${net}. ${row.political_disclaimer || ""}${stale}`;
}

function optionLabel(alert, bias) {
  if (alert === "unusual" && bias && bias !== "quiet") return `unusual ${bias}`;
  if (alert === "active" && bias && bias !== "quiet") return `active ${bias}`;
  return alert || "unknown";
}

function recommendationLabel(value) {
  const text = String(value || "");
  if (!text) return "";
  if (text.includes("politics_override") && text.includes("options_confirm")) return `${directionLabel(text)} politics override + options confirm`;
  if (text.includes("politics_override") && text.includes("options_conflict")) return `${directionLabel(text)} politics override / options conflict`;
  if (text.includes("politics_override")) return `${directionLabel(text)} politics override`;
  if (text.includes("politics_lead")) return `${directionLabel(text)} politics lead`;
  if (text.includes("politics_confirm") && text.includes("options_confirm")) return `${directionLabel(text)} politics + options confirm`;
  if (text.includes("options_confirm")) return `${directionLabel(text)} options confirm`;
  if (text.includes("options_conflict")) return `${directionLabel(text)} options conflict`;
  if (text.includes("politics_confirm")) return `${directionLabel(text)} politics confirm`;
  if (text.includes("politics_conflict")) return `${directionLabel(text)} politics conflict`;
  if (text.includes("politics_support")) return `${directionLabel(text)} politics support`;
  if (text.includes("mixed_options")) return `${directionLabel(text)} mixed`;
  if (text.includes("neutral")) return "neutral skip";
  if (text.includes("skip")) return "skip";
  return directionLabel(text);
}

function directionLabel(value) {
  if (String(value).includes("bearish")) return "bearish";
  if (String(value).includes("bullish")) return "bullish";
  return "watch";
}

function recommendationClass(value) {
  const text = String(value || "");
  if (text.includes("politics_override") || text.includes("politics_lead")) return "watch-override";
  if (text.includes("confirm")) return "watch-confirm";
  if (text.includes("conflict")) return "watch-conflict";
  if (text.includes("skip") || text.includes("neutral")) return "watch-skip";
  return "watch-base";
}

function renderTrends(trends) {
  const box = document.getElementById("trendList");
  box.innerHTML = trends.slice(0, 30).map(name => `<span class="chip">${escapeHtml(name)}</span>`).join("") || `<span class="chip">No trends loaded</span>`;
}

function renderCandidates(candidates) {
  const box = document.getElementById("candidateBox");
  box.innerHTML = candidates.slice(0, 30).map(row => `
    <div class="candidate-row">
      <strong>${escapeHtml(row.ticker)}</strong>
      <span>${escapeHtml(row.company || row.earnings_date || "")}</span>
      <em>${row.expected_move_pct == null ? "na" : `${Number(row.expected_move_pct).toFixed(1)}%`} · X ${row.x_count ?? "na"}</em>
    </div>
  `).join("") || `<span class="muted">No expected-move candidates yet</span>`;
}

function renderMarketSignals(rows) {
  const box = document.getElementById("marketSignalsBox");
  box.innerHTML = rows.slice(0, 30).map(row => {
    const option = optionLabel(row.option_alert || "unknown", row.option_bias || "");
    const contract = row.option_top_contract ? `${row.option_top_contract} x${formatNumber(row.option_top_volume)} / OI ${formatNumber(row.option_top_open_interest)}` : "no top contract";
    const insider = row.insider_recent_count ? `${formatNumber(row.insider_recent_count)} insider filing(s)` : "no recent insider filings";
    const politics = politicalLabel(row);
    const notes = (row.recommendation_notes || []).join("; ");
    return `
      <div class="market-row">
        <strong>${escapeHtml(row.ticker)}</strong>
        <span>${escapeHtml(recommendationLabel(row.recommendation) || option)} · ${escapeHtml(contract)} · ${escapeHtml(notes)}</span>
        <em title="${escapeHtml(politicalTooltip(row))}">${escapeHtml(politics)} · ${escapeHtml(insider)}</em>
      </div>
    `;
  }).join("") || `<span class="muted">No market evidence checks yet</span>`;
}

function safeExternalUrl(value) {
  try {
    const url = new URL(String(value || ""));
    return url.protocol === "https:" ? url.href : "";
  } catch (_) {
    return "";
  }
}

async function renderTradePlans() {
  const box = document.getElementById("tradePlansBox");
  const plans = await fetch("/api/trade-plans").then(r => r.json()).catch(() => []);
  box.innerHTML = plans.slice(0, 20).map(plan => {
    const contract = plan.option_symbol || `${plan.option_type} pending`;
    const schedule = new Date(plan.entry_at).toLocaleString();
    const detail = plan.skip_reason || plan.last_error || `${contract} · ${schedule}`;
    return `
      <div class="trade-plan-row">
        <strong>${escapeHtml(plan.ticker)}</strong>
        <span title="${escapeHtml(detail)}">${escapeHtml(detail)}</span>
        <em class="status-${escapeHtml(plan.status)}">${escapeHtml(plan.status)} · ${Number(plan.allocation_pct * 100).toFixed(0)}%</em>
      </div>
    `;
  }).join("") || `<span class="muted">No automated trade plans yet</span>`;
}

async function renderShadowResearch() {
  const box = document.getElementById("shadowResearchBox");
  const data = await fetch("/api/shadow-research").then(r => r.json()).catch(() => null);
  if (!data) {
    box.innerHTML = `<span class="muted">Shadow research unavailable</span>`;
    return;
  }
  const statuses = Object.entries(data.statuses || {}).map(([name, count]) => `${name} ${formatNumber(count)}`).join(" · ");
  const rows = (data.recent || []).slice(0, 16).map(row => {
    const best = row.best_bid_return_pct == null ? null : Number(row.best_bid_return_pct);
    const result = best == null ? row.status : `${best >= 0 ? "+" : ""}${(best * 100).toFixed(1)}% best`;
    const resultClass = best == null ? "" : best >= 0 ? "positive" : "negative";
    const policy = `${row.option_type} · d${Number(row.target_delta).toFixed(2)} · +${row.target_expiration_days}d · -${row.entry_minutes_before_close}m`;
    return `
      <div class="shadow-row">
        <strong>${escapeHtml(row.ticker)}</strong>
        <span title="${escapeHtml(row.contract_symbol || row.last_error || policy)}">${escapeHtml(policy)}</span>
        <em class="${resultClass}">${escapeHtml(result)}</em>
      </div>
    `;
  }).join("");
  box.innerHTML = `
    <div class="shadow-summary">${formatNumber(data.feature_snapshots)} snapshots · ${formatNumber(data.shadow_positions)} positions · ${formatNumber(data.observations)} observations${statuses ? `<br>${escapeHtml(statuses)}` : ""}</div>
    ${rows || `<span class="muted">Waiting for the next automated scan</span>`}
  `;
}

function renderLogs(logs) {
  document.getElementById("logBox").textContent = logs.join("\\n");
}

async function renderHistory() {
  const box = document.getElementById("historyBox");
  const runs = await fetch("/api/history").then(r => r.json()).catch(() => []);
  box.innerHTML = runs.slice(0, 8).map(run => `
    <button type="button" class="history-row" data-run-id="${escapeHtml(run.id)}">
      <strong>${new Date(run.started_at).toLocaleString()}</strong>
      <span>${formatNumber(run.raw_posts)} posts · ${formatNumber(run.high_traction_tickers)} signals</span>
    </button>
  `).join("") || `<span class="muted">No saved scans yet</span>`;
}

function formatNumber(value) {
  return Number(value || 0).toLocaleString();
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[ch]));
}

document.getElementById("runButton").addEventListener("click", runScan);
document.getElementById("historyBox").addEventListener("click", event => {
  const row = event.target.closest("[data-run-id]");
  if (row) loadRun(row.dataset.runId);
});
loadConfig().then(poll);
"""


if __name__ == "__main__":
    raise SystemExit(main())
