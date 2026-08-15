from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx

from sentiment_analysis.providers.political_activity import (
    PoliticalActivityProvider,
    empty_political_row,
)


SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{document}"
INSIDER_FORMS = {"3", "3/A", "4", "4/A", "5", "5/A"}


def build_market_signal_rows(
    tickers: Iterable[str],
    *,
    option_expirations: int = 2,
    option_min_volume: int = 250,
    option_min_vol_oi_ratio: float = 2.0,
    option_min_premium: float = 500_000,
    option_max_moneyness_distance_pct: float = 35.0,
    insider_lookback_days: int = 30,
    political_activity_enabled: bool = True,
    political_lookback_days: int = 365,
    political_cache_hours: int = 24,
    political_cache_dir: Path = Path("data/political_cache"),
    sec_user_agent: str,
    max_workers: int = 4,
) -> list[dict[str, object]]:
    clean_tickers = list(dict.fromkeys(ticker.upper() for ticker in tickers if ticker.strip()))
    if not clean_tickers:
        return []

    options_provider = YFinanceOptionsActivityProvider(
        expirations=option_expirations,
        min_volume=option_min_volume,
        min_vol_oi_ratio=option_min_vol_oi_ratio,
        min_premium=option_min_premium,
        max_moneyness_distance_pct=option_max_moneyness_distance_pct,
        max_workers=max_workers,
    )
    insider_provider = SecInsiderActivityProvider(
        user_agent=sec_user_agent,
        lookback_days=insider_lookback_days,
    )
    political_provider = PoliticalActivityProvider(
        lookback_days=political_lookback_days,
        cache_dir=political_cache_dir,
        cache_hours=political_cache_hours,
    )

    try:
        option_rows = options_provider.activity_for_many(clean_tickers)
    except Exception as exc:
        option_rows = {
            ticker: {**_empty_option_row(ticker), "option_error": str(exc)}
            for ticker in clean_tickers
        }
    try:
        insider_rows = insider_provider.activity_for_many(clean_tickers)
    except Exception as exc:
        insider_rows = {
            ticker: {**_empty_insider_row(ticker), "insider_error": str(exc)}
            for ticker in clean_tickers
        }
    if political_activity_enabled:
        try:
            political_rows = political_provider.activity_for_many(clean_tickers)
        except Exception as exc:
            political_rows = {
                ticker: {**empty_political_row(ticker), "political_error": str(exc)}
                for ticker in clean_tickers
            }
    else:
        political_rows = {
            ticker: {**empty_political_row(ticker), "political_alert": "disabled"}
            for ticker in clean_tickers
        }
    rows = []
    for ticker in clean_tickers:
        rows.append(
            {
                "ticker": ticker,
                **option_rows.get(ticker, _empty_option_row(ticker)),
                **insider_rows.get(ticker, _empty_insider_row(ticker)),
                **political_rows.get(ticker, empty_political_row(ticker)),
            }
        )
    return rows


class YFinanceOptionsActivityProvider:
    def __init__(
        self,
        *,
        expirations: int = 2,
        min_volume: int = 250,
        min_vol_oi_ratio: float = 2.0,
        min_premium: float = 500_000,
        max_moneyness_distance_pct: float = 35.0,
        max_workers: int = 4,
    ) -> None:
        self.expirations = max(1, expirations)
        self.min_volume = max(1, min_volume)
        self.min_vol_oi_ratio = max(0.0, min_vol_oi_ratio)
        self.min_premium = max(0.0, min_premium)
        self.max_moneyness_distance_pct = max(0.0, max_moneyness_distance_pct)
        self.max_workers = max(1, max_workers)

    def activity_for_many(self, tickers: Iterable[str]) -> dict[str, dict[str, object]]:
        clean_tickers = list(dict.fromkeys(ticker.upper() for ticker in tickers if ticker.strip()))
        if not clean_tickers:
            return {}

        rows: dict[str, dict[str, object]] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(clean_tickers))) as executor:
            futures = {
                executor.submit(self.activity_for, ticker): ticker
                for ticker in clean_tickers
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    rows[ticker] = future.result()
                except Exception as exc:
                    rows[ticker] = {**_empty_option_row(ticker), "option_error": str(exc)}
        return rows

    def activity_for(self, ticker: str) -> dict[str, object]:
        import yfinance as yf

        ticker_obj = yf.Ticker(ticker)
        spot_price = _spot_price(ticker_obj)
        rows = []
        for expiration_text in list(ticker_obj.options or ())[: self.expirations]:
            chain = ticker_obj.option_chain(expiration_text)
            rows.extend(_option_rows_from_frame(chain.calls, expiration_text, "C", spot_price))
            rows.extend(_option_rows_from_frame(chain.puts, expiration_text, "P", spot_price))
        return summarize_option_activity(
            ticker,
            rows,
            min_volume=self.min_volume,
            min_vol_oi_ratio=self.min_vol_oi_ratio,
            min_premium=self.min_premium,
            max_moneyness_distance_pct=self.max_moneyness_distance_pct,
        )


class SecInsiderActivityProvider:
    def __init__(
        self,
        *,
        user_agent: str,
        lookback_days: int = 30,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.user_agent = user_agent
        self.lookback_days = max(1, lookback_days)
        self.timeout_seconds = timeout_seconds
        self._ticker_to_cik: dict[str, str] | None = None

    def activity_for_many(self, tickers: Iterable[str]) -> dict[str, dict[str, object]]:
        clean_tickers = list(dict.fromkeys(ticker.upper() for ticker in tickers if ticker.strip()))
        if not clean_tickers:
            return {}

        rows: dict[str, dict[str, object]] = {}
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        with httpx.Client(headers=headers, timeout=self.timeout_seconds) as client:
            ticker_to_cik = self._load_ticker_to_cik(client)
            for ticker in clean_tickers:
                cik = ticker_to_cik.get(ticker)
                if not cik:
                    rows[ticker] = _empty_insider_row(ticker)
                    continue
                try:
                    response = client.get(SEC_SUBMISSIONS_URL.format(cik=cik))
                    response.raise_for_status()
                    rows[ticker] = summarize_insider_activity(
                        ticker,
                        response.json(),
                        cik=cik,
                        lookback_days=self.lookback_days,
                    )
                except Exception as exc:
                    rows[ticker] = {**_empty_insider_row(ticker), "insider_error": str(exc)}
        return rows

    def _load_ticker_to_cik(self, client: httpx.Client) -> dict[str, str]:
        if self._ticker_to_cik is not None:
            return self._ticker_to_cik
        response = client.get(SEC_TICKER_URL)
        response.raise_for_status()
        mapping = {}
        for row in response.json().values():
            ticker = str(row.get("ticker", "")).upper()
            cik = str(row.get("cik_str", "")).zfill(10)
            if ticker and cik:
                mapping[ticker] = cik
        self._ticker_to_cik = mapping
        return mapping


def summarize_option_activity(
    ticker: str,
    rows: Iterable[dict[str, Any]],
    *,
    min_volume: int,
    min_vol_oi_ratio: float,
    min_premium: float,
    max_moneyness_distance_pct: float,
) -> dict[str, object]:
    clean_rows = list(rows)
    checked_rows = [
        row
        for row in clean_rows
        if _moneyness_distance_pct(row) <= max_moneyness_distance_pct
    ]
    call_volume = sum(_to_int(row.get("volume")) for row in checked_rows if row.get("type") == "C")
    put_volume = sum(_to_int(row.get("volume")) for row in checked_rows if row.get("type") == "P")
    total_volume = call_volume + put_volume
    unusual_rows = [
        row
        for row in checked_rows
        if _to_int(row.get("volume")) >= min_volume
        and _volume_oi_ratio(row) >= min_vol_oi_ratio
        and _premium_volume(row) >= min_premium
    ]
    top_row = max(unusual_rows, key=_option_priority, default=None)
    option_alert = "unusual" if unusual_rows else "active" if total_volume >= min_volume * 4 else "quiet"
    bias_rows = unusual_rows if unusual_rows else checked_rows
    option_bias = _option_bias(
        sum(_to_int(row.get("volume")) for row in bias_rows if row.get("type") == "C"),
        sum(_to_int(row.get("volume")) for row in bias_rows if row.get("type") == "P"),
    )

    return {
        "option_alert": option_alert,
        "option_bias": option_bias,
        "option_call_volume": call_volume,
        "option_put_volume": put_volume,
        "option_total_volume": total_volume,
        "option_unusual_contracts": len(unusual_rows),
        "option_top_contract": _contract_label(top_row) if top_row else None,
        "option_top_volume": _to_int(top_row.get("volume")) if top_row else None,
        "option_top_open_interest": _to_int(top_row.get("open_interest")) if top_row else None,
        "option_top_vol_oi": round(_volume_oi_ratio(top_row), 2) if top_row else None,
        "option_top_premium": round(_premium_volume(top_row), 2) if top_row else None,
        "option_source": "yfinance_options_chain",
    }


def summarize_insider_activity(
    ticker: str,
    payload: dict[str, Any],
    *,
    cik: str,
    lookback_days: int,
    today: date | None = None,
) -> dict[str, object]:
    current = today or datetime.now(timezone.utc).date()
    cutoff = current - timedelta(days=lookback_days)
    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    documents = recent.get("primaryDocument", [])

    matches = []
    for idx, form in enumerate(forms):
        if form not in INSIDER_FORMS:
            continue
        filing_date = _parse_date(_get_index(filing_dates, idx))
        if filing_date is None or filing_date < cutoff:
            continue
        accession = str(_get_index(accessions, idx) or "")
        document = str(_get_index(documents, idx) or "")
        matches.append(
            {
                "form": form,
                "filing_date": filing_date,
                "url": _sec_archive_url(cik, accession, document),
            }
        )

    matches.sort(key=lambda row: row["filing_date"], reverse=True)
    latest = matches[0] if matches else None
    return {
        "insider_alert": "recent" if matches else "none",
        "insider_recent_count": len(matches),
        "insider_latest_date": latest["filing_date"].isoformat() if latest else None,
        "insider_latest_form": latest["form"] if latest else None,
        "insider_latest_url": latest["url"] if latest else None,
        "insider_source": "sec_edgar_submissions",
    }


def _option_rows_from_frame(
    frame: Any,
    expiration: str,
    option_type: str,
    spot_price: float | None,
) -> list[dict[str, object]]:
    try:
        records = frame.to_dict("records")
    except Exception:
        return []

    rows = []
    for record in records:
        rows.append(
            {
                "expiration": expiration,
                "type": option_type,
                "strike": _to_float(record.get("strike")),
                "volume": _to_int(record.get("volume")),
                "open_interest": _to_int(record.get("openInterest")),
                "bid": _to_float(record.get("bid")),
                "ask": _to_float(record.get("ask")),
                "last_price": _to_float(record.get("lastPrice")),
                "spot_price": spot_price,
            }
        )
    return rows


def _empty_option_row(ticker: str) -> dict[str, object]:
    del ticker
    return {
        "option_alert": "unknown",
        "option_bias": "unknown",
        "option_call_volume": 0,
        "option_put_volume": 0,
        "option_total_volume": 0,
        "option_unusual_contracts": 0,
        "option_top_contract": None,
        "option_top_volume": None,
        "option_top_open_interest": None,
        "option_top_vol_oi": None,
        "option_top_premium": None,
        "option_source": "none",
    }


def _empty_insider_row(ticker: str) -> dict[str, object]:
    del ticker
    return {
        "insider_alert": "unknown",
        "insider_recent_count": 0,
        "insider_latest_date": None,
        "insider_latest_form": None,
        "insider_latest_url": None,
        "insider_source": "none",
    }


def _option_bias(call_volume: int, put_volume: int) -> str:
    if call_volume <= 0 and put_volume <= 0:
        return "quiet"
    if call_volume >= put_volume * 1.25:
        return "calls"
    if put_volume >= call_volume * 1.25:
        return "puts"
    return "mixed"


def _option_priority(row: dict[str, Any]) -> tuple[float, float]:
    return (_premium_volume(row), _volume_oi_ratio(row))


def _premium_volume(row: dict[str, Any] | None) -> float:
    if not row:
        return 0.0
    mid = _option_mid(row)
    return _to_int(row.get("volume")) * mid * 100


def _option_mid(row: dict[str, Any]) -> float:
    bid = _to_float(row.get("bid"))
    ask = _to_float(row.get("ask"))
    if bid is not None and ask is not None and ask >= bid and ask > 0:
        return (bid + ask) / 2
    last_price = _to_float(row.get("last_price"))
    return last_price or 0.0


def _volume_oi_ratio(row: dict[str, Any] | None) -> float:
    if not row:
        return 0.0
    volume = _to_int(row.get("volume"))
    open_interest = _to_int(row.get("open_interest"))
    return volume / max(open_interest, 1)


def _moneyness_distance_pct(row: dict[str, Any]) -> float:
    spot_price = _to_float(row.get("spot_price"))
    strike = _to_float(row.get("strike"))
    if spot_price is None or strike is None or spot_price <= 0:
        return 0.0
    return abs(strike - spot_price) / spot_price * 100


def _contract_label(row: dict[str, Any] | None) -> str | None:
    if not row:
        return None
    strike = row.get("strike")
    strike_text = f"{strike:g}" if isinstance(strike, (int, float)) else "?"
    return f"{row.get('expiration')} {strike_text}{row.get('type')}"


def _sec_archive_url(cik: str, accession: str, document: str) -> str | None:
    if not accession or not document:
        return None
    clean_accession = accession.replace("-", "")
    return SEC_ARCHIVE_URL.format(
        cik_int=int(cik),
        accession=clean_accession,
        document=document,
    )


def _get_index(values: list[Any], idx: int) -> Any:
    try:
        return values[idx]
    except IndexError:
        return None


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


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


def _to_int(value: Any) -> int:
    parsed = _to_float(value)
    return int(parsed) if parsed is not None else 0


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
