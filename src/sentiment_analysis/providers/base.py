from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, Sequence

from sentiment_analysis.models import EarningsEvent, SocialPost, TrendTopic


class PostProvider(ABC):
    @abstractmethod
    def recent_posts(
        self,
        queries: Sequence[str],
        *,
        start_time: datetime | None = None,
    ) -> Iterable[SocialPost]:
        raise NotImplementedError


class EarningsProvider(ABC):
    @abstractmethod
    def upcoming(
        self,
        tickers: Sequence[str],
        *,
        horizon_days: int,
    ) -> dict[str, EarningsEvent]:
        raise NotImplementedError


class TrendProvider(ABC):
    @abstractmethod
    def current_trends(self, woeids: Sequence[int]) -> list[TrendTopic]:
        raise NotImplementedError
