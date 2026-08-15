from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo


NEW_YORK = ZoneInfo("America/New_York")


def earnings_trade_times(
    earnings_date: date,
    timing: str,
    *,
    exit_delay_minutes: int = 15,
    entry_before_close_minutes: int = 15,
) -> tuple[datetime, datetime]:
    normalized = normalize_earnings_timing(timing)
    schedule = _schedule(earnings_date - timedelta(days=10), earnings_date + timedelta(days=10))
    sessions = [timestamp.date() for timestamp in schedule.index]

    if normalized == "AMC":
        session = _on_or_before(sessions, earnings_date)
        if session != earnings_date:
            raise ValueError(f"{earnings_date.isoformat()} is not a trading session")
        entry_at = _session_value(schedule, session, "market_close") - timedelta(
            minutes=entry_before_close_minutes
        )
        exit_session = _next_session(sessions, session)
        exit_at = _session_value(schedule, exit_session, "market_open") + timedelta(
            minutes=exit_delay_minutes
        )
        return entry_at, exit_at

    if normalized == "BMO":
        exit_session = _on_or_after(sessions, earnings_date)
        if exit_session != earnings_date:
            raise ValueError(f"{earnings_date.isoformat()} is not a trading session")
        entry_session = _previous_session(sessions, exit_session)
        entry_at = _session_value(schedule, entry_session, "market_close") - timedelta(
            minutes=entry_before_close_minutes
        )
        exit_at = _session_value(schedule, exit_session, "market_open") + timedelta(
            minutes=exit_delay_minutes
        )
        return entry_at, exit_at

    raise ValueError(f"unsupported earnings timing: {timing or 'unknown'}")


def week_key(moment: datetime) -> str:
    local = _aware_utc(moment).astimezone(NEW_YORK)
    monday = local.date() - timedelta(days=local.weekday())
    return monday.isoformat()


def weekly_scan_due(
    moment: datetime,
    *,
    completed_week_key: str | None,
    hour: int = 8,
    minute: int = 0,
) -> bool:
    local = _aware_utc(moment).astimezone(NEW_YORK)
    if local.weekday() >= 5:
        return False
    if local.weekday() == 4 and (local.hour, local.minute) >= (16, 0):
        return False
    monday = local.date() - timedelta(days=local.weekday())
    scheduled = datetime.combine(monday, datetime.min.time(), NEW_YORK).replace(
        hour=hour,
        minute=minute,
    )
    return local >= scheduled and completed_week_key != monday.isoformat()


def normalize_earnings_timing(value: str | None) -> str:
    clean = str(value or "").strip().upper().replace(".", "")
    if clean in {"AMC", "AFTER MARKET CLOSE", "AFTER HOURS", "PM"}:
        return "AMC"
    if clean in {"BMO", "BEFORE MARKET OPEN", "PRE-MARKET", "PREMARKET", "AM"}:
        return "BMO"
    return "UNKNOWN"


def _schedule(start_date: date, end_date: date):
    import pandas_market_calendars as mcal

    return mcal.get_calendar("NYSE").schedule(start_date=start_date, end_date=end_date)


def _session_value(schedule, session: date, column: str) -> datetime:
    value = schedule.loc[session.isoformat(), column]
    converted = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
    return _aware_utc(converted)


def _on_or_before(sessions: list[date], wanted: date) -> date:
    matches = [session for session in sessions if session <= wanted]
    if not matches:
        raise ValueError(f"no trading session on or before {wanted.isoformat()}")
    return matches[-1]


def _on_or_after(sessions: list[date], wanted: date) -> date:
    matches = [session for session in sessions if session >= wanted]
    if not matches:
        raise ValueError(f"no trading session on or after {wanted.isoformat()}")
    return matches[0]


def _previous_session(sessions: list[date], session: date) -> date:
    index = sessions.index(session)
    if index == 0:
        raise ValueError(f"no previous trading session for {session.isoformat()}")
    return sessions[index - 1]


def _next_session(sessions: list[date], session: date) -> date:
    index = sessions.index(session)
    if index + 1 >= len(sessions):
        raise ValueError(f"no next trading session for {session.isoformat()}")
    return sessions[index + 1]


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
