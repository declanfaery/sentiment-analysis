from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Mapping


@dataclass(frozen=True)
class SocialPost:
    id: str
    text: str
    created_at: datetime
    author_id: str | None = None
    username: str | None = None
    public_metrics: Mapping[str, int] = field(default_factory=dict)
    source: str = "unknown"


@dataclass(frozen=True)
class SentimentScore:
    compound: float
    positive: float
    neutral: float
    negative: float
    label: str


@dataclass(frozen=True)
class EarningsEvent:
    ticker: str
    earnings_date: date | None
    source: str
    company: str | None = None
    market_cap: float | None = None
    timing: str | None = None
    expected_move_pct: float | None = None
    expected_move_abs: float | None = None
    expected_move_source: str | None = None
    options_expiration: date | None = None


@dataclass(frozen=True)
class TrendTopic:
    name: str
    tweet_count: int | None
    woeid: int
    source: str

    def as_dict(self) -> dict[str, str | int | None]:
        return {
            "name": self.name,
            "tweet_count": self.tweet_count,
            "woeid": self.woeid,
            "source": self.source,
        }


@dataclass(frozen=True)
class TickerSignal:
    ticker: str
    mention_count: int
    unique_authors: int
    total_engagement: int
    avg_sentiment: float
    sentiment_stddev: float
    positive_posts: int
    neutral_posts: int
    negative_posts: int
    traction_score: float
    crowd_bias: str
    earnings_date: date | None
    days_until_earnings: int | None
    contrarian_watch: str
    preflight_count: int | None = None
    expected_move_pct: float | None = None
    expected_move_abs: float | None = None
    expected_move_source: str | None = None
    options_expiration: date | None = None

    def as_dict(self) -> dict[str, str | int | float | None]:
        return {
            "ticker": self.ticker,
            "mention_count": self.mention_count,
            "unique_authors": self.unique_authors,
            "total_engagement": self.total_engagement,
            "avg_sentiment": round(self.avg_sentiment, 4),
            "sentiment_stddev": round(self.sentiment_stddev, 4),
            "positive_posts": self.positive_posts,
            "neutral_posts": self.neutral_posts,
            "negative_posts": self.negative_posts,
            "traction_score": round(self.traction_score, 4),
            "crowd_bias": self.crowd_bias,
            "earnings_date": self.earnings_date.isoformat() if self.earnings_date else None,
            "days_until_earnings": self.days_until_earnings,
            "contrarian_watch": self.contrarian_watch,
            "preflight_count": self.preflight_count,
            "expected_move_pct": round(self.expected_move_pct, 4) if self.expected_move_pct is not None else None,
            "expected_move_abs": round(self.expected_move_abs, 4) if self.expected_move_abs is not None else None,
            "expected_move_source": self.expected_move_source,
            "options_expiration": self.options_expiration.isoformat() if self.options_expiration else None,
        }
