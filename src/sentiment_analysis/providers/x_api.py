from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable, Sequence

import httpx

from sentiment_analysis.models import SocialPost
from sentiment_analysis.providers.base import PostProvider


class XApiError(RuntimeError):
    pass


class XRecentSearchProvider(PostProvider):
    endpoint = "https://api.x.com/2/tweets/search/recent"

    def __init__(
        self,
        bearer_token: str,
        *,
        max_results: int = 100,
        max_pages_per_query: int = 10,
        page_delay_seconds: float = 0.25,
        max_workers: int = 4,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not bearer_token:
            raise ValueError("X_BEARER_TOKEN is required for post-source=x")
        self.bearer_token = bearer_token
        self.max_results = max(10, min(max_results, 100))
        self.max_pages_per_query = max(1, max_pages_per_query)
        self.page_delay_seconds = max(0.0, page_delay_seconds)
        self.max_workers = max(1, max_workers)
        self.timeout_seconds = timeout_seconds

    def recent_posts(
        self,
        queries: Sequence[str],
        *,
        start_time: datetime | None = None,
    ) -> Iterable[SocialPost]:
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        clean_queries = [query for query in queries if query.strip()]
        if self.max_workers == 1 or len(clean_queries) <= 1:
            with httpx.Client(headers=headers, timeout=self.timeout_seconds) as client:
                for query in clean_queries:
                    yield from self._fetch_query(client, query, start_time=start_time)
            return

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(clean_queries))) as executor:
            futures = {
                executor.submit(self._fetch_query_in_worker, query, start_time): query
                for query in clean_queries
            }
            for future in as_completed(futures):
                yield from future.result()

    def _fetch_query_in_worker(
        self,
        query: str,
        start_time: datetime | None,
    ) -> list[SocialPost]:
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        with httpx.Client(headers=headers, timeout=self.timeout_seconds) as client:
            return list(self._fetch_query(client, query, start_time=start_time))

    def recent_counts(
        self,
        queries: Sequence[str],
        *,
        start_time: datetime | None = None,
    ) -> dict[str, int]:
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        counts: dict[str, int] = {}
        with httpx.Client(headers=headers, timeout=self.timeout_seconds) as client:
            for query in queries:
                params = {"query": query}
                if start_time:
                    params["start_time"] = start_time.isoformat().replace("+00:00", "Z")
                response = client.get("https://api.x.com/2/tweets/counts/recent", params=params)
                if response.status_code == 429:
                    self._sleep_until_rate_limit_reset(response)
                    response = client.get("https://api.x.com/2/tweets/counts/recent", params=params)
                if response.status_code >= 400:
                    raise XApiError(
                        f"X recent counts request failed for {query!r}: "
                        f"{response.status_code} {response.text}"
                    )
                counts[query] = sum(int(row.get("tweet_count", 0)) for row in response.json().get("data", []))
        return counts

    def _fetch_query(
        self,
        client: httpx.Client,
        query: str,
        *,
        start_time: datetime | None,
    ) -> Iterable[SocialPost]:
        next_token: str | None = None

        for page_num in range(self.max_pages_per_query):
            params = {
                "query": query,
                "max_results": self.max_results,
                "tweet.fields": "author_id,created_at,public_metrics,lang,source",
                "expansions": "author_id",
                "user.fields": "username",
            }
            if start_time:
                params["start_time"] = start_time.isoformat().replace("+00:00", "Z")
            if next_token:
                params["next_token"] = next_token

            response = client.get(self.endpoint, params=params)
            if response.status_code == 429:
                self._sleep_until_rate_limit_reset(response)
                response = client.get(self.endpoint, params=params)
            if response.status_code >= 400:
                raise XApiError(f"X API request failed: {response.status_code} {response.text}")

            payload = response.json()
            users = {
                user["id"]: user.get("username")
                for user in payload.get("includes", {}).get("users", [])
            }

            for raw_post in payload.get("data", []):
                created_at = _parse_x_datetime(raw_post.get("created_at"))
                author_id = raw_post.get("author_id")
                yield SocialPost(
                    id=raw_post["id"],
                    text=raw_post["text"],
                    created_at=created_at,
                    author_id=author_id,
                    username=users.get(author_id),
                    public_metrics=raw_post.get("public_metrics", {}),
                    source="x",
                )

            next_token = payload.get("meta", {}).get("next_token")
            if not next_token or page_num == self.max_pages_per_query - 1:
                break
            if self.page_delay_seconds:
                time.sleep(self.page_delay_seconds)

    def _sleep_until_rate_limit_reset(self, response: httpx.Response) -> None:
        raw_reset = response.headers.get("x-rate-limit-reset")
        if not raw_reset:
            time.sleep(60)
            return
        try:
            reset_at = int(raw_reset)
        except ValueError:
            time.sleep(60)
            return
        sleep_for = max(1, min(reset_at - int(time.time()) + 1, 15 * 60))
        time.sleep(sleep_for)


def _parse_x_datetime(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(timezone.utc)
    if raw.endswith("Z"):
        return datetime.fromisoformat(f"{raw[:-1]}+00:00")
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return parsedate_to_datetime(raw)
