from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from sentiment_analysis.models import EarningsEvent
from sentiment_analysis.providers.base import EarningsProvider


class NullEarningsProvider(EarningsProvider):
    def upcoming(
        self,
        tickers: Sequence[str],
        *,
        horizon_days: int,
    ) -> dict[str, EarningsEvent]:
        del horizon_days
        return {ticker: EarningsEvent(ticker=ticker, earnings_date=None, source="none") for ticker in tickers}


class CsvEarningsProvider(EarningsProvider):
    def __init__(self, path: Path, *, today: date | None = None) -> None:
        self.path = path
        self.today = today

    def upcoming(
        self,
        tickers: Sequence[str],
        *,
        horizon_days: int,
    ) -> dict[str, EarningsEvent]:
        today = self.today or date.today()
        horizon = today + timedelta(days=horizon_days)
        wanted = {ticker.upper() for ticker in tickers}
        events = {ticker: EarningsEvent(ticker=ticker, earnings_date=None, source="csv") for ticker in wanted}

        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"symbol", "earnings_date"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{self.path} is missing required columns: {sorted(missing)}")

            for row in reader:
                symbol = row["symbol"].strip().upper()
                if symbol not in wanted:
                    continue
                earnings_date = _parse_date(row["earnings_date"])
                if earnings_date and today <= earnings_date <= horizon:
                    events[symbol] = EarningsEvent(
                        ticker=symbol,
                        earnings_date=earnings_date,
                        source="csv",
                        company=row.get("name") or row.get("company"),
                        market_cap=_parse_float(row.get("market_cap")),
                        timing=row.get("timing") or None,
                    )

        return events

    def between(
        self,
        start_date: date,
        end_date: date,
        *,
        ticker_universe: set[str] | None = None,
    ) -> dict[str, EarningsEvent]:
        wanted = {symbol.upper() for symbol in ticker_universe} if ticker_universe else None
        events: dict[str, EarningsEvent] = {}

        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"symbol", "earnings_date"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{self.path} is missing required columns: {sorted(missing)}")

            for row in reader:
                symbol = row["symbol"].strip().upper()
                if wanted is not None and symbol not in wanted:
                    continue
                earnings_date = _parse_date(row["earnings_date"])
                if earnings_date and start_date <= earnings_date <= end_date:
                    events[symbol] = EarningsEvent(
                        ticker=symbol,
                        earnings_date=earnings_date,
                        source="csv",
                        company=row.get("name") or row.get("company"),
                        market_cap=_parse_float(row.get("market_cap")),
                        timing=row.get("timing") or None,
                    )

        return events


class YFinanceEarningsProvider(EarningsProvider):
    def __init__(self, *, max_workers: int = 8) -> None:
        self.max_workers = max(1, max_workers)

    def upcoming(
        self,
        tickers: Sequence[str],
        *,
        horizon_days: int,
    ) -> dict[str, EarningsEvent]:
        clean_tickers = sorted({ticker.upper() for ticker in tickers})
        if not clean_tickers:
            return {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_one, ticker, horizon_days): ticker
                for ticker in clean_tickers
            }
            events: dict[str, EarningsEvent] = {}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    events[ticker] = future.result()
                except Exception:
                    events[ticker] = EarningsEvent(ticker=ticker, earnings_date=None, source="yfinance")
            return events

    def _fetch_one(self, ticker: str, horizon_days: int) -> EarningsEvent:
        import yfinance as yf

        today = date.today()
        horizon = today + timedelta(days=horizon_days)
        ticker_obj = yf.Ticker(ticker)

        earnings_date = self._from_earnings_dates(ticker_obj, today, horizon)
        if earnings_date is None:
            earnings_date = self._from_calendar(ticker_obj, today, horizon)

        return EarningsEvent(ticker=ticker, earnings_date=earnings_date, source="yfinance")

    def _from_earnings_dates(self, ticker_obj: Any, today: date, horizon: date) -> date | None:
        try:
            frame = ticker_obj.get_earnings_dates(limit=12)
        except Exception:
            return None
        if frame is None or getattr(frame, "empty", True):
            return None

        dates: list[date] = []
        for value in frame.index:
            parsed = _coerce_date(value)
            if parsed and today <= parsed <= horizon:
                dates.append(parsed)
        return min(dates) if dates else None

    def _from_calendar(self, ticker_obj: Any, today: date, horizon: date) -> date | None:
        try:
            calendar = ticker_obj.calendar
        except Exception:
            return None

        values: list[Any] = []
        if isinstance(calendar, dict):
            values.extend(_flatten(calendar.get("Earnings Date")))
            values.extend(_flatten(calendar.get("EarningsDate")))
        else:
            try:
                if "Earnings Date" in calendar.index:
                    values.extend(_flatten(calendar.loc["Earnings Date"].tolist()))
            except Exception:
                return None

        dates = [
            parsed
            for parsed in (_coerce_date(value) for value in values)
            if parsed and today <= parsed <= horizon
        ]
        return min(dates) if dates else None


class YFinanceCalendarEarningsProvider(EarningsProvider):
    def __init__(
        self,
        *,
        market_cap_min: float | None = None,
        filter_most_active: bool = False,
        max_pages: int = 10,
    ) -> None:
        self.market_cap_min = market_cap_min
        self.filter_most_active = filter_most_active
        self.max_pages = max(1, max_pages)

    def upcoming(
        self,
        tickers: Sequence[str],
        *,
        horizon_days: int,
    ) -> dict[str, EarningsEvent]:
        today = date.today()
        end_date = today + timedelta(days=horizon_days)
        return self.between(today, end_date, ticker_universe=set(tickers))

    def between(
        self,
        start_date: date,
        end_date: date,
        *,
        ticker_universe: set[str] | None = None,
    ) -> dict[str, EarningsEvent]:
        import yfinance as yf

        events: dict[str, EarningsEvent] = {}
        wanted = {symbol.upper() for symbol in ticker_universe} if ticker_universe else None
        # yfinance's calendar query treats end as exclusive, so request one
        # extra day and keep the explicit inclusive filter below.
        query_end_date = end_date + timedelta(days=1)
        calendar = yf.Calendars(start=start_date, end=query_end_date)

        for page_num in range(self.max_pages):
            try:
                frame = calendar.get_earnings_calendar(
                    market_cap=self.market_cap_min,
                    filter_most_active=self.filter_most_active,
                    start=start_date,
                    end=query_end_date,
                    limit=100,
                    offset=page_num * 100,
                    force=True,
                )
            except Exception:
                break
            if frame is None or getattr(frame, "empty", True):
                break

            rows = frame.reset_index().to_dict("records")
            for row in rows:
                symbol = str(row.get("Symbol", "")).strip().upper()
                if not symbol or (wanted is not None and symbol not in wanted):
                    continue
                earnings_date = _coerce_date(row.get("Event Start Date"))
                if earnings_date and start_date <= earnings_date <= end_date:
                    existing = events.get(symbol)
                    if existing is None or existing.earnings_date is None or earnings_date < existing.earnings_date:
                        events[symbol] = EarningsEvent(
                            ticker=symbol,
                            earnings_date=earnings_date,
                            source="yfinance_calendar",
                            company=_none_if_nan(row.get("Company")),
                            market_cap=_parse_float(row.get("Marketcap")),
                            timing=_none_if_nan(row.get("Timing")),
                        )

            if len(rows) < 100:
                break

        return events


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _parse_date(raw: str) -> date | None:
    value = raw.strip()
    if not value:
        return None
    return _coerce_date(value)


def _parse_float(value: Any) -> float | None:
    value = _none_if_nan(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _none_if_nan(value: Any) -> Any | None:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except TypeError:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        if value.tzinfo:
            return value.astimezone(timezone.utc).date()
        return value.date()
    if hasattr(value, "to_pydatetime"):
        return _coerce_date(value.to_pydatetime())
    if isinstance(value, str):
        clean = value.strip()
        if not clean:
            return None
        if clean.endswith("Z"):
            clean = f"{clean[:-1]}+00:00"
        try:
            return datetime.fromisoformat(clean).date()
        except ValueError:
            try:
                return date.fromisoformat(clean[:10])
            except ValueError:
                return None
    return None
