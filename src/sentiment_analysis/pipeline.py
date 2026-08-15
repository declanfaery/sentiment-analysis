from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Literal

from sentiment_analysis.models import EarningsEvent, SocialPost, TickerSignal
from sentiment_analysis.providers.base import EarningsProvider, PostProvider
from sentiment_analysis.sentiment.vader import VaderSentimentAnalyzer
from sentiment_analysis.tickers import build_cashtag_queries, build_discovery_queries, extract_tickers


@dataclass(frozen=True)
class PipelineOptions:
    ticker_universe: set[str]
    min_mentions: int = 25
    earnings_horizon_days: int = 14
    hours_back: int = 24
    x_query_max_chars: int = 512
    include_plain_tickers: bool = False
    sentiment_threshold: float = 0.15
    query_mode: Literal["watchlist", "discovery", "manual"] = "discovery"
    extra_queries: tuple[str, ...] = ()
    trend_names: tuple[str, ...] = ()
    preflight_counts: dict[str, int] = field(default_factory=dict)
    selected_tickers: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineResult:
    signals: list[TickerSignal]
    queries: tuple[str, ...]
    started_at: datetime
    finished_at: datetime
    raw_posts: int
    duplicate_posts: int
    matched_posts: int
    tickers_seen: int
    high_traction_tickers: int
    trend_names: tuple[str, ...] = ()
    matched_social_posts: tuple[SocialPost, ...] = ()
    preflight_counts: dict[str, int] = field(default_factory=dict)
    selected_tickers: tuple[str, ...] = ()
    candidate_rows: tuple[dict[str, object], ...] = ()
    market_signal_rows: tuple[dict[str, object], ...] = ()

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, (self.finished_at - self.started_at).total_seconds())

    def as_dict(self) -> dict[str, object]:
        return {
            "signals": [signal.as_dict() for signal in self.signals],
            "queries": list(self.queries),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "raw_posts": self.raw_posts,
            "duplicate_posts": self.duplicate_posts,
            "unique_posts": self.raw_posts - self.duplicate_posts,
            "matched_posts": self.matched_posts,
            "tickers_seen": self.tickers_seen,
            "high_traction_tickers": self.high_traction_tickers,
            "trend_names": list(self.trend_names),
            "saved_posts": len(self.matched_social_posts),
            "preflight_counts": self.preflight_counts,
            "selected_tickers": list(self.selected_tickers),
            "candidate_rows": list(self.candidate_rows),
            "market_signal_rows": list(self.market_signal_rows),
        }


@dataclass
class _TickerAccumulator:
    ticker: str
    authors: set[str] = field(default_factory=set)
    sentiments: list[float] = field(default_factory=list)
    positive_posts: int = 0
    neutral_posts: int = 0
    negative_posts: int = 0
    total_engagement: int = 0

    @property
    def mention_count(self) -> int:
        return len(self.sentiments)

    def add(self, post: SocialPost, sentiment: float, label: str) -> None:
        if post.author_id:
            self.authors.add(post.author_id)
        elif post.username:
            self.authors.add(post.username)
        self.sentiments.append(sentiment)
        self.total_engagement += engagement_score(post)
        if label == "positive":
            self.positive_posts += 1
        elif label == "negative":
            self.negative_posts += 1
        else:
            self.neutral_posts += 1


class SentimentPipeline:
    def __init__(
        self,
        *,
        post_provider: PostProvider,
        earnings_provider: EarningsProvider,
        sentiment_analyzer: VaderSentimentAnalyzer,
        options: PipelineOptions,
    ) -> None:
        self.post_provider = post_provider
        self.earnings_provider = earnings_provider
        self.sentiment_analyzer = sentiment_analyzer
        self.options = options

    def run(self) -> list[TickerSignal]:
        return self.run_with_metadata().signals

    def run_with_metadata(self) -> PipelineResult:
        started_at = datetime.now(timezone.utc)
        start_time = datetime.now(timezone.utc) - timedelta(hours=self.options.hours_back)
        queries = self._build_queries()

        accumulators: dict[str, _TickerAccumulator] = {}
        seen_post_ids: set[str] = set()
        raw_posts = 0
        duplicate_posts = 0
        matched_posts = 0
        matched_social_posts: list[SocialPost] = []

        for post in self.post_provider.recent_posts(queries, start_time=start_time):
            raw_posts += 1
            if post.id in seen_post_ids:
                duplicate_posts += 1
                continue
            seen_post_ids.add(post.id)

            tickers = extract_tickers(
                post.text,
                self.options.ticker_universe,
                include_plain_tickers=self.options.include_plain_tickers,
            )
            if not tickers:
                continue
            matched_posts += 1
            matched_social_posts.append(post)

            score = self.sentiment_analyzer.score(post.text)
            for ticker in tickers:
                accumulator = accumulators.setdefault(ticker, _TickerAccumulator(ticker=ticker))
                accumulator.add(post, score.compound, score.label)

        high_traction = [
            accumulator
            for accumulator in accumulators.values()
            if accumulator.mention_count >= self.options.min_mentions
        ]
        high_traction.sort(key=_raw_traction_score, reverse=True)

        earnings = self.earnings_provider.upcoming(
            [accumulator.ticker for accumulator in high_traction],
            horizon_days=self.options.earnings_horizon_days,
        )

        signals = [
            self._to_signal(accumulator, earnings.get(accumulator.ticker))
            for accumulator in high_traction
        ]
        sorted_signals = sorted(signals, key=lambda signal: signal.traction_score, reverse=True)
        return PipelineResult(
            signals=sorted_signals,
            queries=tuple(queries),
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            raw_posts=raw_posts,
            duplicate_posts=duplicate_posts,
            matched_posts=matched_posts,
            tickers_seen=len(accumulators),
            high_traction_tickers=len(sorted_signals),
            trend_names=self.options.trend_names,
            matched_social_posts=tuple(matched_social_posts),
            preflight_counts=dict(self.options.preflight_counts),
            selected_tickers=self.options.selected_tickers,
        )

    def _build_queries(self) -> list[str]:
        if self.options.query_mode == "manual":
            base_queries = []
        elif self.options.query_mode == "discovery":
            base_queries = build_discovery_queries()
        else:
            base_queries = build_cashtag_queries(
                self.options.ticker_universe,
                max_chars=self.options.x_query_max_chars,
            )
        return _dedupe_queries([*base_queries, *self.options.extra_queries])

    def _to_signal(
        self,
        accumulator: _TickerAccumulator,
        event: EarningsEvent | None,
    ) -> TickerSignal:
        avg_sentiment = statistics.fmean(accumulator.sentiments)
        sentiment_stddev = (
            statistics.pstdev(accumulator.sentiments)
            if len(accumulator.sentiments) > 1
            else 0.0
        )
        earnings_date = event.earnings_date if event else None
        days_until_earnings = _days_until(earnings_date)
        crowd_bias = _crowd_bias(avg_sentiment, self.options.sentiment_threshold)
        return TickerSignal(
            ticker=accumulator.ticker,
            mention_count=accumulator.mention_count,
            unique_authors=len(accumulator.authors),
            total_engagement=accumulator.total_engagement,
            avg_sentiment=avg_sentiment,
            sentiment_stddev=sentiment_stddev,
            positive_posts=accumulator.positive_posts,
            neutral_posts=accumulator.neutral_posts,
            negative_posts=accumulator.negative_posts,
            traction_score=_raw_traction_score(accumulator),
            crowd_bias=crowd_bias,
            earnings_date=earnings_date,
            days_until_earnings=days_until_earnings,
            contrarian_watch=_contrarian_watch(crowd_bias, days_until_earnings),
            preflight_count=self.options.preflight_counts.get(accumulator.ticker),
            expected_move_pct=event.expected_move_pct if event else None,
            expected_move_abs=event.expected_move_abs if event else None,
            expected_move_source=event.expected_move_source if event else None,
            options_expiration=event.options_expiration if event else None,
        )


def engagement_score(post: SocialPost) -> int:
    metrics = post.public_metrics
    return (
        int(metrics.get("like_count", 0))
        + int(metrics.get("reply_count", 0)) * 2
        + int(metrics.get("retweet_count", 0)) * 3
        + int(metrics.get("quote_count", 0)) * 2
    )


def _raw_traction_score(accumulator: _TickerAccumulator) -> float:
    return (
        accumulator.mention_count
        + len(accumulator.authors) * 1.5
        + math.log1p(accumulator.total_engagement) * 2.0
    )


def _crowd_bias(avg_sentiment: float, threshold: float) -> str:
    if avg_sentiment >= threshold:
        return "bullish"
    if avg_sentiment <= -threshold:
        return "bearish"
    return "neutral"


def _contrarian_watch(crowd_bias: str, days_until_earnings: int | None) -> str:
    if days_until_earnings is None:
        return "no_earnings_date"
    if days_until_earnings < 0:
        return "earnings_passed"
    if days_until_earnings > 14:
        return "earnings_not_close"
    if crowd_bias == "bullish":
        return "bearish_contrarian_watch"
    if crowd_bias == "bearish":
        return "bullish_contrarian_watch"
    return "neutral_watch"


def _days_until(earnings_date: date | None) -> int | None:
    if earnings_date is None:
        return None
    return (earnings_date - date.today()).days


def _dedupe_queries(queries: list[str]) -> list[str]:
    clean_queries: list[str] = []
    seen: set[str] = set()
    for query in queries:
        clean = query.strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        clean_queries.append(clean)
    return clean_queries
