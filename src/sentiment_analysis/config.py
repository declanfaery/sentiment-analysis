from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def _int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class AppConfig:
    x_bearer_token: str | None
    sec_user_agent: str
    x_query_max_chars: int
    x_max_workers: int
    min_mentions: int
    earnings_horizon_days: int
    output_dir: Path

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_env_file()
        return cls(
            x_bearer_token=os.getenv("X_BEARER_TOKEN") or None,
            sec_user_agent=os.getenv(
                "SEC_USER_AGENT",
                "sentiment-analysis/0.1 local research contact@example.com",
            ),
            x_query_max_chars=_int_from_env("X_QUERY_MAX_CHARS", 512),
            x_max_workers=_int_from_env("X_MAX_WORKERS", 4),
            min_mentions=_int_from_env("MIN_MENTIONS", 25),
            earnings_horizon_days=_int_from_env("EARNINGS_HORIZON_DAYS", 14),
            output_dir=Path(os.getenv("OUTPUT_DIR", "output")),
        )
