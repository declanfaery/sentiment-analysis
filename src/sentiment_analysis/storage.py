from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

from sentiment_analysis.models import SocialPost, TickerSignal
from sentiment_analysis.pipeline import PipelineResult


SCHEMA = """
CREATE TABLE IF NOT EXISTS scan_runs (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    elapsed_seconds REAL NOT NULL,
    raw_posts INTEGER NOT NULL,
    duplicate_posts INTEGER NOT NULL,
    matched_posts INTEGER NOT NULL,
    tickers_seen INTEGER NOT NULL,
    high_traction_tickers INTEGER NOT NULL,
    options_json TEXT NOT NULL,
    queries_json TEXT NOT NULL,
    trend_names_json TEXT NOT NULL,
    preflight_counts_json TEXT NOT NULL,
    selected_tickers_json TEXT NOT NULL,
    candidate_rows_json TEXT NOT NULL DEFAULT '[]',
    market_signal_rows_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS scan_signals (
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    mention_count INTEGER NOT NULL,
    unique_authors INTEGER NOT NULL,
    total_engagement INTEGER NOT NULL,
    avg_sentiment REAL NOT NULL,
    sentiment_stddev REAL NOT NULL,
    positive_posts INTEGER NOT NULL,
    neutral_posts INTEGER NOT NULL,
    negative_posts INTEGER NOT NULL,
    traction_score REAL NOT NULL,
    crowd_bias TEXT NOT NULL,
    earnings_date TEXT,
    days_until_earnings INTEGER,
    contrarian_watch TEXT NOT NULL,
    preflight_count INTEGER,
    expected_move_pct REAL,
    expected_move_abs REAL,
    expected_move_source TEXT,
    options_expiration TEXT,
    PRIMARY KEY (run_id, ticker),
    FOREIGN KEY (run_id) REFERENCES scan_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scan_posts (
    run_id TEXT NOT NULL,
    post_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    author_id TEXT,
    username TEXT,
    source TEXT NOT NULL,
    text TEXT NOT NULL,
    like_count INTEGER NOT NULL,
    reply_count INTEGER NOT NULL,
    retweet_count INTEGER NOT NULL,
    quote_count INTEGER NOT NULL,
    PRIMARY KEY (run_id, post_id),
    FOREIGN KEY (run_id) REFERENCES scan_runs(id) ON DELETE CASCADE
);
"""


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as connection:
        connection.executescript(SCHEMA)
        _add_missing_columns(connection)


def save_scan_result(result: PipelineResult, options: dict[str, Any], path: Path) -> str:
    init_db(path)
    run_id = uuid.uuid4().hex
    safe_options = {key: value for key, value in options.items() if "token" not in key.lower()}

    with _connect(path) as connection:
        connection.execute(
            """
            INSERT INTO scan_runs (
                id, started_at, finished_at, elapsed_seconds, raw_posts, duplicate_posts,
                matched_posts, tickers_seen, high_traction_tickers, options_json,
                queries_json, trend_names_json, preflight_counts_json,
                selected_tickers_json, candidate_rows_json, market_signal_rows_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                result.started_at.isoformat(),
                result.finished_at.isoformat(),
                result.elapsed_seconds,
                result.raw_posts,
                result.duplicate_posts,
                result.matched_posts,
                result.tickers_seen,
                result.high_traction_tickers,
                json.dumps(safe_options, default=str, sort_keys=True),
                json.dumps(list(result.queries)),
                json.dumps(list(result.trend_names)),
                json.dumps(result.preflight_counts, sort_keys=True),
                json.dumps(list(result.selected_tickers)),
                json.dumps(list(result.candidate_rows), default=str),
                json.dumps(list(result.market_signal_rows), default=str),
            ),
        )
        connection.executemany(
            """
            INSERT INTO scan_signals (
                run_id, ticker, mention_count, unique_authors, total_engagement,
                avg_sentiment, sentiment_stddev, positive_posts, neutral_posts,
                negative_posts, traction_score, crowd_bias, earnings_date,
                days_until_earnings, contrarian_watch, preflight_count,
                expected_move_pct, expected_move_abs, expected_move_source, options_expiration
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_signal_row(run_id, signal) for signal in result.signals],
        )
        connection.executemany(
            """
            INSERT INTO scan_posts (
                run_id, post_id, created_at, author_id, username, source, text,
                like_count, reply_count, retweet_count, quote_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_post_row(run_id, post) for post in result.matched_social_posts],
        )

    return run_id


def recent_runs(path: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    init_db(path)
    with _connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, started_at, finished_at, elapsed_seconds, raw_posts,
                   matched_posts, tickers_seen, high_traction_tickers,
                   selected_tickers_json
            FROM scan_runs
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            **dict(row),
            "selected_tickers": json.loads(row["selected_tickers_json"]),
        }
        for row in rows
    ]


def load_scan_result(path: Path, run_id: str) -> tuple[PipelineResult, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    init_db(path)
    with _connect(path) as connection:
        connection.row_factory = sqlite3.Row
        run = connection.execute(
            "SELECT * FROM scan_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if run is None:
            raise KeyError(run_id)

        signal_rows = connection.execute(
            "SELECT * FROM scan_signals WHERE run_id = ? ORDER BY traction_score DESC",
            (run_id,),
        ).fetchall()
        post_rows = connection.execute(
            "SELECT * FROM scan_posts WHERE run_id = ? ORDER BY created_at DESC",
            (run_id,),
        ).fetchall()

    signals = [_signal_from_row(row) for row in signal_rows]
    candidate_rows = json.loads(run["candidate_rows_json"] or "[]")
    if not candidate_rows:
        candidate_rows = [_candidate_row_from_signal(signal) for signal in signals]
    market_signal_rows = json.loads(run["market_signal_rows_json"] or "[]")

    result = PipelineResult(
        signals=signals,
        queries=tuple(json.loads(run["queries_json"])),
        started_at=_parse_datetime(run["started_at"]),
        finished_at=_parse_datetime(run["finished_at"]),
        raw_posts=run["raw_posts"],
        duplicate_posts=run["duplicate_posts"],
        matched_posts=run["matched_posts"],
        tickers_seen=run["tickers_seen"],
        high_traction_tickers=run["high_traction_tickers"],
        trend_names=tuple(json.loads(run["trend_names_json"])),
        matched_social_posts=tuple(_post_from_row(row) for row in post_rows),
        preflight_counts=json.loads(run["preflight_counts_json"]),
        selected_tickers=tuple(json.loads(run["selected_tickers_json"])),
        candidate_rows=tuple(candidate_rows),
        market_signal_rows=tuple(market_signal_rows),
    )
    return result, json.loads(run["options_json"])


def _signal_row(run_id: str, signal: TickerSignal) -> tuple[Any, ...]:
    return (
        run_id,
        signal.ticker,
        signal.mention_count,
        signal.unique_authors,
        signal.total_engagement,
        signal.avg_sentiment,
        signal.sentiment_stddev,
        signal.positive_posts,
        signal.neutral_posts,
        signal.negative_posts,
        signal.traction_score,
        signal.crowd_bias,
        signal.earnings_date.isoformat() if signal.earnings_date else None,
        signal.days_until_earnings,
        signal.contrarian_watch,
        signal.preflight_count,
        signal.expected_move_pct,
        signal.expected_move_abs,
        signal.expected_move_source,
        signal.options_expiration.isoformat() if signal.options_expiration else None,
    )


def _signal_from_row(row: sqlite3.Row) -> TickerSignal:
    return TickerSignal(
        ticker=row["ticker"],
        mention_count=row["mention_count"],
        unique_authors=row["unique_authors"],
        total_engagement=row["total_engagement"],
        avg_sentiment=row["avg_sentiment"],
        sentiment_stddev=row["sentiment_stddev"],
        positive_posts=row["positive_posts"],
        neutral_posts=row["neutral_posts"],
        negative_posts=row["negative_posts"],
        traction_score=row["traction_score"],
        crowd_bias=row["crowd_bias"],
        earnings_date=_parse_optional_date(row["earnings_date"]),
        days_until_earnings=row["days_until_earnings"],
        contrarian_watch=row["contrarian_watch"],
        preflight_count=row["preflight_count"],
        expected_move_pct=row["expected_move_pct"],
        expected_move_abs=row["expected_move_abs"],
        expected_move_source=row["expected_move_source"],
        options_expiration=_parse_optional_date(row["options_expiration"]),
    )


def _candidate_row_from_signal(signal: TickerSignal) -> dict[str, object]:
    return {
        "ticker": signal.ticker,
        "company": None,
        "earnings_date": signal.earnings_date.isoformat() if signal.earnings_date else None,
        "timing": None,
        "expected_move_pct": signal.expected_move_pct,
        "expected_move_abs": signal.expected_move_abs,
        "expected_move_source": signal.expected_move_source,
        "options_expiration": signal.options_expiration.isoformat()
        if signal.options_expiration
        else None,
        "x_count": signal.preflight_count,
    }


def _post_row(run_id: str, post: SocialPost) -> tuple[Any, ...]:
    metrics = post.public_metrics
    return (
        run_id,
        post.id,
        post.created_at.isoformat(),
        post.author_id,
        post.username,
        post.source,
        post.text,
        int(metrics.get("like_count", 0)),
        int(metrics.get("reply_count", 0)),
        int(metrics.get("retweet_count", 0)),
        int(metrics.get("quote_count", 0)),
    )


def _post_from_row(row: sqlite3.Row) -> SocialPost:
    return SocialPost(
        id=row["post_id"],
        text=row["text"],
        created_at=_parse_datetime(row["created_at"]),
        author_id=row["author_id"],
        username=row["username"],
        source=row["source"],
        public_metrics={
            "like_count": row["like_count"],
            "reply_count": row["reply_count"],
            "retweet_count": row["retweet_count"],
            "quote_count": row["quote_count"],
        },
    )


def _add_missing_columns(connection: sqlite3.Connection) -> None:
    run_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(scan_runs)").fetchall()
    }
    run_additions = {
        "candidate_rows_json": "TEXT NOT NULL DEFAULT '[]'",
        "market_signal_rows_json": "TEXT NOT NULL DEFAULT '[]'",
    }
    for name, column_type in run_additions.items():
        if name not in run_columns:
            connection.execute(f"ALTER TABLE scan_runs ADD COLUMN {name} {column_type}")

    signal_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(scan_signals)").fetchall()
    }
    signal_additions = {
        "expected_move_pct": "REAL",
        "expected_move_abs": "REAL",
        "expected_move_source": "TEXT",
        "options_expiration": "TEXT",
    }
    for name, column_type in signal_additions.items():
        if name not in signal_columns:
            connection.execute(f"ALTER TABLE scan_signals ADD COLUMN {name} {column_type}")


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _parse_optional_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()
