from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from sentiment_analysis.models import SocialPost
from sentiment_analysis.providers.base import PostProvider


def _parse_datetime(raw: str) -> datetime:
    value = raw.strip()
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _metric(row: dict[str, str], name: str) -> int:
    raw = row.get(name, "0") or "0"
    try:
        return int(raw)
    except ValueError:
        return 0


class CsvPostProvider(PostProvider):
    def __init__(self, path: Path) -> None:
        self.path = path

    def recent_posts(
        self,
        queries: Sequence[str],
        *,
        start_time: datetime | None = None,
    ) -> Iterable[SocialPost]:
        del queries
        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"id", "created_at", "text"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{self.path} is missing required columns: {sorted(missing)}")

            for row in reader:
                created_at = _parse_datetime(row["created_at"])
                if start_time and created_at < start_time:
                    continue
                yield SocialPost(
                    id=row["id"],
                    text=row["text"],
                    created_at=created_at,
                    author_id=row.get("author_id") or None,
                    username=row.get("username") or None,
                    public_metrics={
                        "like_count": _metric(row, "like_count"),
                        "reply_count": _metric(row, "reply_count"),
                        "retweet_count": _metric(row, "retweet_count"),
                        "quote_count": _metric(row, "quote_count"),
                    },
                    source="csv",
                )

