from __future__ import annotations

from typing import Sequence

import httpx

from sentiment_analysis.models import TrendTopic
from sentiment_analysis.providers.base import TrendProvider


class NullTrendProvider(TrendProvider):
    def current_trends(self, woeids: Sequence[int]) -> list[TrendTopic]:
        del woeids
        return []


class XTrendsProvider(TrendProvider):
    endpoint_template = "https://api.x.com/2/trends/by/woeid/{woeid}"

    def __init__(
        self,
        bearer_token: str,
        *,
        max_trends: int = 50,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not bearer_token:
            raise ValueError("X_BEARER_TOKEN is required for X trends")
        self.bearer_token = bearer_token
        self.max_trends = max(1, min(max_trends, 50))
        self.timeout_seconds = timeout_seconds

    def current_trends(self, woeids: Sequence[int]) -> list[TrendTopic]:
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        topics: list[TrendTopic] = []

        with httpx.Client(headers=headers, timeout=self.timeout_seconds) as client:
            for woeid in dict.fromkeys(int(value) for value in woeids):
                response = client.get(
                    self.endpoint_template.format(woeid=woeid),
                    params={
                        "max_trends": self.max_trends,
                        "trend.fields": "trend_name,tweet_count",
                    },
                )
                if response.status_code >= 400:
                    continue
                payload = response.json()
                for item in payload.get("data", []):
                    name = item.get("trend_name") or item.get("name")
                    if not name:
                        continue
                    topics.append(
                        TrendTopic(
                            name=str(name),
                            tweet_count=_coerce_count(item.get("tweet_count")),
                            woeid=woeid,
                            source="x",
                        )
                    )

        return topics


def _coerce_count(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
