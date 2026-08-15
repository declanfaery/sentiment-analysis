from __future__ import annotations

import argparse
from pathlib import Path

from sentiment_analysis.config import AppConfig
from sentiment_analysis.io import write_signals_csv, write_signals_json
from sentiment_analysis.pipeline import PipelineOptions, SentimentPipeline
from sentiment_analysis.providers.csv_posts import CsvPostProvider
from sentiment_analysis.providers.earnings import (
    CsvEarningsProvider,
    NullEarningsProvider,
    YFinanceEarningsProvider,
    YFinanceCalendarEarningsProvider,
)
from sentiment_analysis.providers.trends import XTrendsProvider
from sentiment_analysis.providers.x_api import XRecentSearchProvider
from sentiment_analysis.sentiment.vader import VaderSentimentAnalyzer
from sentiment_analysis.tickers import build_trend_queries, load_ticker_universe, refresh_us_ticker_universe


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_command(args)
    if args.command == "refresh-tickers":
        return refresh_tickers_command(args)
    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentiment-scan")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Run the Part 1 ticker sentiment scan.")
    run.add_argument("--post-source", choices=["csv", "x"], default="csv")
    run.add_argument("--posts-csv", type=Path, default=Path("examples/sample_posts.csv"))
    run.add_argument("--earnings-source", choices=["csv", "yfinance", "yfinance_calendar", "none"], default="csv")
    run.add_argument("--earnings-csv", type=Path, default=Path("examples/sample_earnings.csv"))
    run.add_argument("--ticker-file", type=Path, default=Path("data/tickers_sample.csv"))
    run.add_argument("--min-mentions", type=int)
    run.add_argument("--earnings-horizon-days", type=int)
    run.add_argument("--hours-back", type=int, default=24)
    run.add_argument("--include-plain-tickers", action="store_true")
    run.add_argument("--x-query-mode", choices=["discovery", "watchlist"], default="discovery")
    run.add_argument("--x-query-max-chars", type=int)
    run.add_argument("--x-max-results", type=int, default=100)
    run.add_argument("--max-pages-per-query", type=int, default=10)
    run.add_argument("--x-max-workers", type=int)
    run.add_argument("--use-trends", action="store_true")
    run.add_argument("--trend-woeid", type=int, action="append", default=[])
    run.add_argument("--trend-max-queries", type=int, default=25)
    run.add_argument("--out-csv", type=Path, default=Path("output/signals.csv"))
    run.add_argument("--out-json", type=Path, default=Path("output/signals.json"))

    refresh = subparsers.add_parser(
        "refresh-tickers",
        help="Download a fresh US listed-stock ticker universe.",
    )
    refresh.add_argument("--out", type=Path, default=Path("data/tickers_us.csv"))
    refresh.add_argument("--include-etfs", action="store_true")
    refresh.add_argument("--include-special-securities", action="store_true")
    return parser


def run_command(args: argparse.Namespace) -> int:
    config = AppConfig.from_env()
    ticker_universe = load_ticker_universe(args.ticker_file)

    post_provider = _build_post_provider(args, config)
    earnings_provider = _build_earnings_provider(args)
    trend_names, trend_queries = _build_trend_queries(args, config, ticker_universe)

    options = PipelineOptions(
        ticker_universe=ticker_universe,
        min_mentions=args.min_mentions or config.min_mentions,
        earnings_horizon_days=args.earnings_horizon_days or config.earnings_horizon_days,
        hours_back=args.hours_back,
        x_query_max_chars=args.x_query_max_chars or config.x_query_max_chars,
        include_plain_tickers=args.include_plain_tickers,
        query_mode=args.x_query_mode,
        extra_queries=tuple(trend_queries),
        trend_names=tuple(trend_names),
    )
    pipeline = SentimentPipeline(
        post_provider=post_provider,
        earnings_provider=earnings_provider,
        sentiment_analyzer=VaderSentimentAnalyzer(),
        options=options,
    )
    result = pipeline.run_with_metadata()
    signals = result.signals

    write_signals_csv(signals, args.out_csv)
    write_signals_json(signals, args.out_json)

    print(f"Found {len(signals)} high-traction ticker(s).")
    print(
        f"Scanned {result.raw_posts} post(s), matched {result.matched_posts}, "
        f"used {len(result.queries)} query lane(s) in {result.elapsed_seconds:.2f}s."
    )
    print(f"CSV: {args.out_csv}")
    print(f"JSON: {args.out_json}")
    for signal in signals[:10]:
        earnings = signal.earnings_date.isoformat() if signal.earnings_date else "unknown"
        print(
            f"{signal.ticker}: mentions={signal.mention_count} "
            f"bias={signal.crowd_bias} earnings={earnings} "
            f"watch={signal.contrarian_watch}"
        )
    return 0


def refresh_tickers_command(args: argparse.Namespace) -> int:
    count = refresh_us_ticker_universe(
        args.out,
        include_etfs=args.include_etfs,
        include_special_securities=args.include_special_securities,
    )
    print(f"Wrote {count} ticker(s) to {args.out}")
    return 0


def _build_post_provider(args: argparse.Namespace, config: AppConfig):
    if args.post_source == "csv":
        return CsvPostProvider(args.posts_csv)
    return XRecentSearchProvider(
        config.x_bearer_token or "",
        max_results=args.x_max_results,
        max_pages_per_query=args.max_pages_per_query,
        max_workers=args.x_max_workers or config.x_max_workers,
    )


def _build_earnings_provider(args: argparse.Namespace):
    if args.earnings_source == "csv":
        return CsvEarningsProvider(args.earnings_csv)
    if args.earnings_source == "yfinance":
        return YFinanceEarningsProvider()
    if args.earnings_source == "yfinance_calendar":
        return YFinanceCalendarEarningsProvider()
    return NullEarningsProvider()


def _build_trend_queries(
    args: argparse.Namespace,
    config: AppConfig,
    ticker_universe: set[str],
) -> tuple[list[str], list[str]]:
    if not args.use_trends:
        return [], []

    provider = XTrendsProvider(config.x_bearer_token or "")
    woeids = args.trend_woeid or [23424977, 1]
    trends = provider.current_trends(woeids)
    trend_names = [trend.name for trend in sorted(trends, key=_trend_sort_key)]
    trend_queries = build_trend_queries(
        trend_names,
        ticker_universe,
        max_queries=args.trend_max_queries,
    )
    return trend_names, trend_queries


def _trend_sort_key(trend) -> tuple[int, str]:
    tweet_count = trend.tweet_count if trend.tweet_count is not None else -1
    return (-tweet_count, trend.name)
