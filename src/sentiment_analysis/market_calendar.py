from __future__ import annotations

from datetime import date, timedelta


def earnings_week_window(today: date | None = None) -> tuple[date, date]:
    """Return the upcoming weekday window ending on Friday.

    Sunday runs cover Monday-Friday. Thursday runs cover Thursday-Friday.
    Weekend runs roll forward to Monday-Friday.
    """
    current = today or date.today()
    if current.weekday() >= 5:
        days_until_monday = 7 - current.weekday()
        start = current + timedelta(days=days_until_monday)
    else:
        start = current

    end = start + timedelta(days=4 - start.weekday())
    return start, end


def days_window(days: int, today: date | None = None) -> tuple[date, date]:
    current = today or date.today()
    return current, current + timedelta(days=max(0, days))

