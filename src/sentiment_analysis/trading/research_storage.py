from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

from sentiment_analysis.trading.models import OptionContract, ShadowPosition
from sentiment_analysis.trading.storage import init_trading_db


RESEARCH_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_feature_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_run_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    features_json TEXT NOT NULL,
    UNIQUE (scan_run_id, ticker)
);

CREATE TABLE IF NOT EXISTS shadow_positions (
    id TEXT PRIMARY KEY,
    scan_run_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    earnings_date TEXT NOT NULL,
    earnings_timing TEXT NOT NULL,
    option_type TEXT NOT NULL,
    target_delta REAL NOT NULL,
    target_expiration_days INTEGER NOT NULL,
    entry_minutes_before_close INTEGER NOT NULL,
    scheduled_entry_at TEXT NOT NULL,
    market_open_at TEXT NOT NULL,
    expiration_target TEXT NOT NULL,
    status TEXT NOT NULL,
    contract_symbol TEXT,
    contract_expiration TEXT,
    strike_price REAL,
    multiplier INTEGER NOT NULL DEFAULT 100,
    entry_observed_at TEXT,
    entry_bid REAL,
    entry_ask REAL,
    entry_underlying REAL,
    entry_delta REAL,
    entry_vega REAL,
    entry_iv REAL,
    entry_volume INTEGER NOT NULL DEFAULT 0,
    entry_open_interest INTEGER NOT NULL DEFAULT 0,
    entry_quote_time_ms INTEGER,
    entry_quote_quality TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (
        plan_id, option_type, target_delta, target_expiration_days,
        entry_minutes_before_close
    )
);

CREATE INDEX IF NOT EXISTS shadow_positions_entry_idx
ON shadow_positions(status, scheduled_entry_at, ticker);

CREATE TABLE IF NOT EXISTS shadow_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shadow_position_id TEXT NOT NULL,
    checkpoint_minutes INTEGER NOT NULL,
    scheduled_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    bid REAL,
    ask REAL,
    midpoint REAL,
    underlying_price REAL,
    implied_volatility REAL,
    delta REAL,
    quote_time_ms INTEGER,
    quote_quality TEXT NOT NULL,
    bid_return_pct REAL,
    midpoint_return_pct REAL,
    underlying_return_pct REAL,
    UNIQUE (shadow_position_id, checkpoint_minutes),
    FOREIGN KEY (shadow_position_id) REFERENCES shadow_positions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS shadow_observations_position_idx
ON shadow_observations(shadow_position_id, checkpoint_minutes);
"""


def init_research_db(path: Path) -> None:
    init_trading_db(path)
    with _connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(RESEARCH_SCHEMA)


def save_feature_snapshot(
    path: Path,
    *,
    scan_run_id: str,
    plan_id: str,
    ticker: str,
    captured_at: datetime,
    features: dict[str, object],
) -> bool:
    init_research_db(path)
    with _connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO research_feature_snapshots (
                scan_run_id, plan_id, ticker, captured_at, features_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                scan_run_id,
                plan_id,
                ticker,
                captured_at.isoformat(),
                json.dumps(features, default=str, sort_keys=True),
            ),
        )
        return cursor.rowcount > 0


def save_shadow_positions(path: Path, positions: Iterable[ShadowPosition]) -> int:
    init_research_db(path)
    rows = [_shadow_row(position) for position in positions]
    if not rows:
        return 0
    with _connect(path) as connection:
        cursor = connection.executemany(
            """
            INSERT OR IGNORE INTO shadow_positions (
                id, scan_run_id, plan_id, ticker, earnings_date,
                earnings_timing, option_type, target_delta,
                target_expiration_days, entry_minutes_before_close,
                scheduled_entry_at, market_open_at, expiration_target, status,
                contract_symbol, contract_expiration, strike_price, multiplier,
                entry_observed_at, entry_bid, entry_ask, entry_underlying,
                entry_delta, entry_vega, entry_iv, entry_volume,
                entry_open_interest, entry_quote_time_ms, entry_quote_quality,
                last_error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cursor.rowcount


def list_due_entry_group(
    path: Path,
    *,
    now: datetime,
    grace_minutes: int,
) -> list[ShadowPosition]:
    init_research_db(path)
    cutoff = now - timedelta(minutes=max(1, grace_minutes))
    with _connect(path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute(
            """
            UPDATE shadow_positions
            SET status='missed_entry',
                last_error='shadow entry window expired',
                updated_at=?
            WHERE status='scheduled' AND scheduled_entry_at < ?
            """,
            (now.isoformat(), cutoff.isoformat()),
        )
        first = connection.execute(
            """
            SELECT ticker, scheduled_entry_at
            FROM shadow_positions
            WHERE status='scheduled'
              AND scheduled_entry_at <= ?
              AND scheduled_entry_at >= ?
            ORDER BY scheduled_entry_at, ticker
            LIMIT 1
            """,
            (now.isoformat(), cutoff.isoformat()),
        ).fetchone()
        if first is None:
            return []
        rows = connection.execute(
            """
            SELECT * FROM shadow_positions
            WHERE status='scheduled' AND ticker=? AND scheduled_entry_at=?
            ORDER BY option_type, target_expiration_days, target_delta
            """,
            (first["ticker"], first["scheduled_entry_at"]),
        ).fetchall()
    return [_shadow_from_row(row) for row in rows]


def record_shadow_entry(
    path: Path,
    position_id: str,
    *,
    contract: OptionContract,
    observed_at: datetime,
    quote_quality: str,
) -> None:
    init_research_db(path)
    status = "entered" if quote_quality != "stale" else "invalid_entry_quote"
    error = None if status == "entered" else "entry option quote was stale"
    with _connect(path) as connection:
        connection.execute(
            """
            UPDATE shadow_positions SET
                status=?, contract_symbol=?, contract_expiration=?,
                strike_price=?, multiplier=?, entry_observed_at=?, entry_bid=?,
                entry_ask=?, entry_underlying=?, entry_delta=?, entry_vega=?,
                entry_iv=?, entry_volume=?, entry_open_interest=?,
                entry_quote_time_ms=?, entry_quote_quality=?, last_error=?,
                updated_at=?
            WHERE id=?
            """,
            (
                status,
                contract.symbol,
                contract.expiration.isoformat(),
                contract.strike,
                contract.multiplier,
                observed_at.isoformat(),
                contract.bid,
                contract.ask,
                contract.underlying_price,
                contract.delta,
                contract.vega,
                contract.implied_volatility,
                contract.volume,
                contract.open_interest,
                contract.quote_time_ms,
                quote_quality,
                error,
                observed_at.isoformat(),
                position_id,
            ),
        )


def record_shadow_failure(
    path: Path,
    position_id: str,
    *,
    status: str,
    message: str,
    observed_at: datetime,
) -> None:
    init_research_db(path)
    with _connect(path) as connection:
        connection.execute(
            """
            UPDATE shadow_positions
            SET status=?, last_error=?, updated_at=?
            WHERE id=?
            """,
            (status, message[:1000], observed_at.isoformat(), position_id),
        )


def list_shadow_positions(
    path: Path,
    *,
    statuses: tuple[str, ...] | None = None,
    limit: int = 5000,
) -> list[ShadowPosition]:
    init_research_db(path)
    where = ""
    params: list[object] = []
    if statuses:
        where = f"WHERE status IN ({','.join('?' for _ in statuses)})"
        params.extend(statuses)
    params.append(max(1, limit))
    with _connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"""
            SELECT * FROM shadow_positions {where}
            ORDER BY scheduled_entry_at, ticker
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [_shadow_from_row(row) for row in rows]


def observation_quality(
    path: Path,
    position_id: str,
    checkpoint_minutes: int,
) -> str | None:
    init_research_db(path)
    with _connect(path) as connection:
        row = connection.execute(
            """
            SELECT quote_quality FROM shadow_observations
            WHERE shadow_position_id=? AND checkpoint_minutes=?
            """,
            (position_id, checkpoint_minutes),
        ).fetchone()
    return str(row[0]) if row else None


def observation_qualities(path: Path) -> dict[tuple[str, int], str]:
    init_research_db(path)
    with _connect(path) as connection:
        rows = connection.execute(
            """
            SELECT shadow_position_id, checkpoint_minutes, quote_quality
            FROM shadow_observations
            """
        ).fetchall()
    return {(str(row[0]), int(row[1])): str(row[2]) for row in rows}


def save_shadow_observation(
    path: Path,
    *,
    position: ShadowPosition,
    checkpoint_minutes: int,
    scheduled_at: datetime,
    observed_at: datetime,
    quote: OptionContract,
    quote_quality: str,
) -> None:
    init_research_db(path)
    midpoint = (quote.bid + quote.ask) / 2 if quote.bid > 0 and quote.ask > 0 else None
    entry_ask = position.entry_ask or 0.0
    bid_return = (quote.bid / entry_ask - 1) if entry_ask > 0 and quote.bid > 0 else None
    midpoint_return = (
        midpoint / entry_ask - 1
        if entry_ask > 0 and midpoint is not None
        else None
    )
    underlying_return = (
        quote.underlying_price / position.entry_underlying - 1
        if position.entry_underlying and quote.underlying_price > 0
        else None
    )
    values = (
        position.id,
        checkpoint_minutes,
        scheduled_at.isoformat(),
        observed_at.isoformat(),
        quote.bid,
        quote.ask,
        midpoint,
        quote.underlying_price,
        quote.implied_volatility,
        quote.delta,
        quote.quote_time_ms,
        quote_quality,
        bid_return,
        midpoint_return,
        underlying_return,
    )
    with _connect(path) as connection:
        connection.execute(
            """
            INSERT INTO shadow_observations (
                shadow_position_id, checkpoint_minutes, scheduled_at,
                observed_at, bid, ask, midpoint, underlying_price,
                implied_volatility, delta, quote_time_ms, quote_quality,
                bid_return_pct, midpoint_return_pct, underlying_return_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(shadow_position_id, checkpoint_minutes) DO UPDATE SET
                observed_at=excluded.observed_at,
                bid=excluded.bid,
                ask=excluded.ask,
                midpoint=excluded.midpoint,
                underlying_price=excluded.underlying_price,
                implied_volatility=excluded.implied_volatility,
                delta=excluded.delta,
                quote_time_ms=excluded.quote_time_ms,
                quote_quality=excluded.quote_quality,
                bid_return_pct=excluded.bid_return_pct,
                midpoint_return_pct=excluded.midpoint_return_pct,
                underlying_return_pct=excluded.underlying_return_pct
            """,
            values,
        )


def save_missing_observation(
    path: Path,
    *,
    position_id: str,
    checkpoint_minutes: int,
    scheduled_at: datetime,
    observed_at: datetime,
) -> None:
    init_research_db(path)
    with _connect(path) as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO shadow_observations (
                shadow_position_id, checkpoint_minutes, scheduled_at,
                observed_at, quote_quality
            ) VALUES (?, ?, ?, ?, 'missed')
            """,
            (
                position_id,
                checkpoint_minutes,
                scheduled_at.isoformat(),
                observed_at.isoformat(),
            ),
        )


def mark_shadow_completed(path: Path, position_id: str, *, now: datetime) -> None:
    init_research_db(path)
    with _connect(path) as connection:
        connection.execute(
            """
            UPDATE shadow_positions
            SET status='completed', updated_at=?
            WHERE id=? AND status='entered'
            """,
            (now.isoformat(), position_id),
        )


def research_summary(path: Path, *, limit: int = 20) -> dict[str, object]:
    init_research_db(path)
    with _connect(path) as connection:
        connection.row_factory = sqlite3.Row
        feature_count = connection.execute(
            "SELECT COUNT(*) FROM research_feature_snapshots"
        ).fetchone()[0]
        observation_count = connection.execute(
            "SELECT COUNT(*) FROM shadow_observations"
        ).fetchone()[0]
        status_rows = connection.execute(
            "SELECT status, COUNT(*) count FROM shadow_positions GROUP BY status"
        ).fetchall()
        recent = connection.execute(
            """
            SELECT
                p.id, p.ticker, p.option_type, p.target_delta,
                p.target_expiration_days, p.entry_minutes_before_close,
                p.status, p.contract_symbol, p.entry_ask,
                p.entry_quote_quality, p.last_error,
                COUNT(o.id) AS observation_count,
                MAX(o.bid_return_pct) AS best_bid_return_pct,
                MIN(o.bid_return_pct) AS worst_bid_return_pct,
                MAX(o.checkpoint_minutes) AS latest_checkpoint
            FROM shadow_positions p
            LEFT JOIN shadow_observations o ON o.shadow_position_id=p.id
            GROUP BY p.id
            ORDER BY p.scheduled_entry_at DESC, p.ticker, p.option_type,
                     p.target_expiration_days, p.target_delta
            LIMIT ?
            """,
            (max(1, limit),),
        ).fetchall()
    statuses = {str(row["status"]): int(row["count"]) for row in status_rows}
    return {
        "feature_snapshots": int(feature_count),
        "shadow_positions": sum(statuses.values()),
        "observations": int(observation_count),
        "statuses": statuses,
        "recent": [dict(row) for row in recent],
    }


def _shadow_row(position: ShadowPosition) -> tuple[object, ...]:
    now = datetime.now(timezone.utc)
    created_at = position.created_at or now
    updated_at = position.updated_at or now
    return (
        position.id,
        position.scan_run_id,
        position.plan_id,
        position.ticker,
        position.earnings_date.isoformat(),
        position.earnings_timing,
        position.option_type,
        position.target_delta,
        position.target_expiration_days,
        position.entry_minutes_before_close,
        position.scheduled_entry_at.isoformat(),
        position.market_open_at.isoformat(),
        position.expiration_target.isoformat(),
        position.status,
        position.contract_symbol,
        position.contract_expiration.isoformat()
        if position.contract_expiration
        else None,
        position.strike_price,
        position.multiplier,
        position.entry_observed_at.isoformat()
        if position.entry_observed_at
        else None,
        position.entry_bid,
        position.entry_ask,
        position.entry_underlying,
        position.entry_delta,
        position.entry_vega,
        position.entry_iv,
        position.entry_volume,
        position.entry_open_interest,
        position.entry_quote_time_ms,
        position.entry_quote_quality,
        position.last_error,
        created_at.isoformat(),
        updated_at.isoformat(),
    )


def _shadow_from_row(row: sqlite3.Row) -> ShadowPosition:
    return ShadowPosition(
        id=row["id"],
        scan_run_id=row["scan_run_id"],
        plan_id=row["plan_id"],
        ticker=row["ticker"],
        earnings_date=date.fromisoformat(row["earnings_date"]),
        earnings_timing=row["earnings_timing"],
        option_type=row["option_type"],
        target_delta=row["target_delta"],
        target_expiration_days=row["target_expiration_days"],
        entry_minutes_before_close=row["entry_minutes_before_close"],
        scheduled_entry_at=datetime.fromisoformat(row["scheduled_entry_at"]),
        market_open_at=datetime.fromisoformat(row["market_open_at"]),
        expiration_target=date.fromisoformat(row["expiration_target"]),
        status=row["status"],
        contract_symbol=row["contract_symbol"],
        contract_expiration=(
            date.fromisoformat(row["contract_expiration"])
            if row["contract_expiration"]
            else None
        ),
        strike_price=row["strike_price"],
        multiplier=row["multiplier"],
        entry_observed_at=(
            datetime.fromisoformat(row["entry_observed_at"])
            if row["entry_observed_at"]
            else None
        ),
        entry_bid=row["entry_bid"],
        entry_ask=row["entry_ask"],
        entry_underlying=row["entry_underlying"],
        entry_delta=row["entry_delta"],
        entry_vega=row["entry_vega"],
        entry_iv=row["entry_iv"],
        entry_volume=row["entry_volume"],
        entry_open_interest=row["entry_open_interest"],
        entry_quote_time_ms=row["entry_quote_time_ms"],
        entry_quote_quality=row["entry_quote_quality"],
        last_error=row["last_error"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()
