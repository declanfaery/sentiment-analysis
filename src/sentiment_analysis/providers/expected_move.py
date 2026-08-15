from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

from sentiment_analysis.models import EarningsEvent


@dataclass(frozen=True)
class ExpectedMove:
    ticker: str
    expected_move_pct: float
    expected_move_abs: float
    spot_price: float
    strike: float | None
    options_expiration: date
    source: str


class YFinanceExpectedMoveProvider:
    def __init__(self, *, max_workers: int = 8) -> None:
        self.max_workers = max(1, max_workers)

    def enrich(self, events: Iterable[EarningsEvent]) -> dict[str, EarningsEvent]:
        clean_events = [event for event in events if event.earnings_date is not None]
        if not clean_events:
            return {}

        enriched: dict[str, EarningsEvent] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(clean_events))) as executor:
            futures = {
                executor.submit(self.expected_move_for, event.ticker, event.earnings_date): event
                for event in clean_events
            }
            for future in as_completed(futures):
                event = futures[future]
                try:
                    expected_move = future.result()
                except Exception:
                    expected_move = None
                enriched[event.ticker] = with_expected_move(event, expected_move)
        return enriched

    def expected_move_for(self, ticker: str, earnings_date: date) -> ExpectedMove | None:
        import yfinance as yf

        ticker_obj = yf.Ticker(ticker)
        spot_price = _spot_price(ticker_obj)
        if spot_price is None or spot_price <= 0:
            return None

        expiration_text = _select_expiration(ticker_obj.options, earnings_date)
        if expiration_text is None:
            return None

        expiration = date.fromisoformat(expiration_text)
        chain = ticker_obj.option_chain(expiration_text)
        straddle_move = expected_move_from_chain(
            calls=chain.calls,
            puts=chain.puts,
            spot_price=spot_price,
            expiration=expiration,
        )
        if straddle_move:
            return straddle_move

        return implied_volatility_move_from_chain(
            calls=chain.calls,
            puts=chain.puts,
            spot_price=spot_price,
            expiration=expiration,
        )


def with_expected_move(event: EarningsEvent, expected_move: ExpectedMove | None) -> EarningsEvent:
    if expected_move is None:
        return event
    return EarningsEvent(
        ticker=event.ticker,
        earnings_date=event.earnings_date,
        source=event.source,
        company=event.company,
        market_cap=event.market_cap,
        timing=event.timing,
        expected_move_pct=expected_move.expected_move_pct,
        expected_move_abs=expected_move.expected_move_abs,
        expected_move_source=expected_move.source,
        options_expiration=expected_move.options_expiration,
    )


def expected_move_from_chain(
    *,
    calls: Any,
    puts: Any,
    spot_price: float,
    expiration: date,
) -> ExpectedMove | None:
    common_strikes = sorted(set(_strikes(calls)).intersection(_strikes(puts)))
    if not common_strikes:
        return None

    strike = min(common_strikes, key=lambda value: abs(value - spot_price))
    call_row = _row_for_strike(calls, strike)
    put_row = _row_for_strike(puts, strike)
    call_mid = _option_mid(call_row)
    put_mid = _option_mid(put_row)
    if call_mid is None or put_mid is None:
        return None

    expected_move_abs = call_mid + put_mid
    if expected_move_abs <= 0:
        return None
    return ExpectedMove(
        ticker="",
        expected_move_pct=expected_move_abs / spot_price * 100,
        expected_move_abs=expected_move_abs,
        spot_price=spot_price,
        strike=strike,
        options_expiration=expiration,
        source="atm_straddle",
    )


def implied_volatility_move_from_chain(
    *,
    calls: Any,
    puts: Any,
    spot_price: float,
    expiration: date,
    today: date | None = None,
) -> ExpectedMove | None:
    current = today or date.today()
    dte = max((expiration - current).days, 1)
    rows = []
    for frame in (calls, puts):
        try:
            ordered = frame.assign(_distance=(frame["strike"] - spot_price).abs()).sort_values("_distance")
            rows.extend(ordered.head(3).to_dict("records"))
        except Exception:
            continue
    ivs = [_to_float(row.get("impliedVolatility")) for row in rows]
    ivs = [iv for iv in ivs if iv is not None and iv > 0]
    if not ivs:
        return None

    avg_iv = sum(ivs) / len(ivs)
    expected_move_pct = avg_iv * math.sqrt(dte / 365.0) * 100
    return ExpectedMove(
        ticker="",
        expected_move_pct=expected_move_pct,
        expected_move_abs=spot_price * expected_move_pct / 100,
        spot_price=spot_price,
        strike=None,
        options_expiration=expiration,
        source="atm_iv",
    )


def _spot_price(ticker_obj: Any) -> float | None:
    try:
        fast_info = ticker_obj.fast_info
        for key in ("last_price", "regular_market_price", "previous_close"):
            value = fast_info.get(key) if hasattr(fast_info, "get") else getattr(fast_info, key, None)
            parsed = _to_float(value)
            if parsed and parsed > 0:
                return parsed
    except Exception:
        pass

    try:
        history = ticker_obj.history(period="5d")
        if history is not None and not getattr(history, "empty", True):
            return _to_float(history["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None


def _select_expiration(expiration_texts: Iterable[str], earnings_date: date) -> str | None:
    expirations = []
    for raw in expiration_texts:
        try:
            expirations.append((date.fromisoformat(str(raw)), str(raw)))
        except ValueError:
            continue
    if not expirations:
        return None
    future = [item for item in expirations if item[0] >= earnings_date]
    if future:
        return min(future, key=lambda item: item[0])[1]
    return min(expirations, key=lambda item: abs((item[0] - earnings_date).days))[1]


def _strikes(frame: Any) -> list[float]:
    try:
        return [_to_float(value) for value in frame["strike"].tolist() if _to_float(value) is not None]
    except Exception:
        return []


def _row_for_strike(frame: Any, strike: float) -> dict[str, Any]:
    rows = frame[frame["strike"] == strike].to_dict("records")
    return rows[0] if rows else {}


def _option_mid(row: dict[str, Any]) -> float | None:
    bid = _to_float(row.get("bid"))
    ask = _to_float(row.get("ask"))
    if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2
    last_price = _to_float(row.get("lastPrice"))
    if last_price is not None and last_price > 0:
        return last_price
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed

