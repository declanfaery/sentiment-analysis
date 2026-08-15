from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from sentiment_analysis.storage import init_db
from sentiment_analysis.trading.models import ACTIVE_TRADE_STATUSES, TradePlan


TRADING_SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_plans (
    id TEXT PRIMARY KEY,
    scan_run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    option_type TEXT NOT NULL,
    earnings_date TEXT NOT NULL,
    earnings_timing TEXT NOT NULL,
    entry_at TEXT NOT NULL,
    exit_at TEXT NOT NULL,
    allocation_pct REAL NOT NULL,
    recommendation TEXT NOT NULL,
    recommendation_score INTEGER NOT NULL,
    status TEXT NOT NULL,
    skip_reason TEXT,
    option_symbol TEXT,
    option_expiration TEXT,
    strike_price REAL,
    contract_multiplier INTEGER NOT NULL DEFAULT 100,
    quantity INTEGER NOT NULL DEFAULT 0,
    entry_order_id TEXT,
    exit_order_id TEXT,
    entry_limit_price REAL,
    entry_fill_price REAL,
    exit_limit_price REAL,
    exit_fill_price REAL,
    filled_quantity INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (scan_run_id, ticker),
    FOREIGN KEY (scan_run_id) REFERENCES scan_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS trade_plans_status_entry_idx
ON trade_plans(status, entry_at);

CREATE INDEX IF NOT EXISTS trade_plans_status_exit_idx
ON trade_plans(status, exit_at);

CREATE TABLE IF NOT EXISTS trade_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    FOREIGN KEY (plan_id) REFERENCES trade_plans(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS automation_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def init_trading_db(path: Path) -> None:
    init_db(path)
    with _connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(TRADING_SCHEMA)


def save_trade_plans(path: Path, plans: Iterable[TradePlan]) -> int:
    init_trading_db(path)
    rows = [_plan_row(plan) for plan in plans]
    if not rows:
        return 0
    with _connect(path) as connection:
        cursor = connection.executemany(
            """
            INSERT OR IGNORE INTO trade_plans (
                id, scan_run_id, ticker, direction, option_type, earnings_date,
                earnings_timing, entry_at, exit_at, allocation_pct, recommendation,
                recommendation_score, status, skip_reason, option_symbol,
                option_expiration, strike_price, contract_multiplier, quantity,
                entry_order_id, exit_order_id, entry_limit_price, entry_fill_price,
                exit_limit_price, exit_fill_price, filled_quantity, last_error,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cursor.rowcount


def update_trade_plan(path: Path, plan: TradePlan) -> None:
    init_trading_db(path)
    updated = plan.with_updates(updated_at=datetime.now(timezone.utc))
    row = _plan_row(updated)
    with _connect(path) as connection:
        connection.execute(
            """
            UPDATE trade_plans SET
                scan_run_id=?, ticker=?, direction=?, option_type=?, earnings_date=?,
                earnings_timing=?, entry_at=?, exit_at=?, allocation_pct=?, recommendation=?,
                recommendation_score=?, status=?, skip_reason=?, option_symbol=?,
                option_expiration=?, strike_price=?, contract_multiplier=?, quantity=?,
                entry_order_id=?, exit_order_id=?, entry_limit_price=?, entry_fill_price=?,
                exit_limit_price=?, exit_fill_price=?, filled_quantity=?, last_error=?,
                created_at=?, updated_at=?
            WHERE id=?
            """,
            row[1:] + (row[0],),
        )


def get_trade_plan(path: Path, plan_id: str) -> TradePlan:
    init_trading_db(path)
    with _connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM trade_plans WHERE id = ?", (plan_id,)).fetchone()
    if row is None:
        raise KeyError(plan_id)
    return _plan_from_row(row)


def list_trade_plans(
    path: Path,
    *,
    limit: int = 200,
    active_only: bool = False,
    scan_run_id: str | None = None,
) -> list[TradePlan]:
    init_trading_db(path)
    where: list[str] = []
    params: list[object] = []
    if active_only:
        placeholders = ",".join("?" for _ in ACTIVE_TRADE_STATUSES)
        where.append(f"status IN ({placeholders})")
        params.extend(sorted(ACTIVE_TRADE_STATUSES))
    if scan_run_id:
        where.append("scan_run_id = ?")
        params.append(scan_run_id)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(max(1, limit))
    with _connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"SELECT * FROM trade_plans {clause} ORDER BY entry_at ASC, ticker ASC LIMIT ?",
            params,
        ).fetchall()
    return [_plan_from_row(row) for row in rows]


def open_exposure(path: Path) -> float:
    init_trading_db(path)
    with _connect(path) as connection:
        row = connection.execute(
            """
            SELECT COALESCE(SUM(
                COALESCE(entry_fill_price, entry_limit_price, 0) *
                CASE
                    WHEN status IN ('entry_submitting', 'entry_submitted')
                        THEN quantity + filled_quantity
                    WHEN filled_quantity > 0 THEN filled_quantity
                    ELSE quantity
                END *
                contract_multiplier
            ), 0)
            FROM trade_plans
            WHERE status IN ('entry_submitting', 'entry_submitted', 'open', 'exit_submitting', 'exit_submitted')
            """
        ).fetchone()
    return float(row[0] or 0.0)


def realized_pnl(path: Path) -> float:
    init_trading_db(path)
    with _connect(path) as connection:
        row = connection.execute(
            """
            SELECT COALESCE(SUM(
                (exit_fill_price - entry_fill_price) * filled_quantity * contract_multiplier
            ), 0)
            FROM trade_plans
            WHERE status = 'closed'
              AND entry_fill_price IS NOT NULL
              AND exit_fill_price IS NOT NULL
              AND filled_quantity > 0
            """
        ).fetchone()
    return float(row[0] or 0.0)


def strategy_equity(path: Path, *, starting_capital: float) -> float:
    return max(0.0, float(starting_capital) + realized_pnl(path))


def record_trade_event(path: Path, plan_id: str, event_type: str, message: str) -> None:
    init_trading_db(path)
    with _connect(path) as connection:
        connection.execute(
            "INSERT INTO trade_events (plan_id, created_at, event_type, message) VALUES (?, ?, ?, ?)",
            (plan_id, datetime.now(timezone.utc).isoformat(), event_type, message[:2000]),
        )


def get_automation_state(path: Path, key: str) -> str | None:
    init_trading_db(path)
    with _connect(path) as connection:
        row = connection.execute("SELECT value FROM automation_state WHERE key = ?", (key,)).fetchone()
    return str(row[0]) if row else None


def set_automation_state(path: Path, key: str, value: str) -> None:
    init_trading_db(path)
    with _connect(path) as connection:
        connection.execute(
            """
            INSERT INTO automation_state (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (key, value, datetime.now(timezone.utc).isoformat()),
        )


def latest_scan_run_id(path: Path) -> str | None:
    init_db(path)
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT id FROM scan_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    return str(row[0]) if row else None


def _plan_row(plan: TradePlan) -> tuple[object, ...]:
    now = datetime.now(timezone.utc)
    created_at = plan.created_at or now
    updated_at = plan.updated_at or now
    return (
        plan.id,
        plan.scan_run_id,
        plan.ticker,
        plan.direction,
        plan.option_type,
        plan.earnings_date.isoformat(),
        plan.earnings_timing,
        plan.entry_at.isoformat(),
        plan.exit_at.isoformat(),
        plan.allocation_pct,
        plan.recommendation,
        plan.recommendation_score,
        plan.status,
        plan.skip_reason,
        plan.option_symbol,
        plan.option_expiration.isoformat() if plan.option_expiration else None,
        plan.strike_price,
        plan.contract_multiplier,
        plan.quantity,
        plan.entry_order_id,
        plan.exit_order_id,
        plan.entry_limit_price,
        plan.entry_fill_price,
        plan.exit_limit_price,
        plan.exit_fill_price,
        plan.filled_quantity,
        plan.last_error,
        created_at.isoformat(),
        updated_at.isoformat(),
    )


def _plan_from_row(row: sqlite3.Row) -> TradePlan:
    return TradePlan(
        id=row["id"],
        scan_run_id=row["scan_run_id"],
        ticker=row["ticker"],
        direction=row["direction"],
        option_type=row["option_type"],
        earnings_date=date.fromisoformat(row["earnings_date"]),
        earnings_timing=row["earnings_timing"],
        entry_at=datetime.fromisoformat(row["entry_at"]),
        exit_at=datetime.fromisoformat(row["exit_at"]),
        allocation_pct=row["allocation_pct"],
        recommendation=row["recommendation"],
        recommendation_score=row["recommendation_score"],
        status=row["status"],
        skip_reason=row["skip_reason"],
        option_symbol=row["option_symbol"],
        option_expiration=_optional_date(row["option_expiration"]),
        strike_price=row["strike_price"],
        contract_multiplier=row["contract_multiplier"],
        quantity=row["quantity"],
        entry_order_id=row["entry_order_id"],
        exit_order_id=row["exit_order_id"],
        entry_limit_price=row["entry_limit_price"],
        entry_fill_price=row["entry_fill_price"],
        exit_limit_price=row["exit_limit_price"],
        exit_fill_price=row["exit_fill_price"],
        filled_quantity=row["filled_quantity"],
        last_error=row["last_error"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _optional_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()
