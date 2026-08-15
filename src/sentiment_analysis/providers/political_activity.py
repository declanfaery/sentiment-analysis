from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

import httpx


CAPITOL_MARKETS_TICKER_URL = "https://capitolmarkets.org/tickers/{ticker}"
HOUSE_DISCLOSURE_URL = "https://disclosures-clerk.house.gov/FinancialDisclosure/ViewSearch"
SENATE_DISCLOSURE_URL = "https://efdsearch.senate.gov/search/home/"
POLITICAL_DATA_DISCLAIMER = (
    "Disclosure activity is delayed, reported in value ranges, and may include a spouse or child. "
    "'No later sale found' is not proof that a position is still held."
)

# Current 119th Congress leaders plus the two high-profile names explicitly used by the strategy.
# POTUS rows are also detected from the chamber field, so presidential weighting is not name-only.
HIGH_PROFILE_ROLES = {
    "donald j trump": (4, "President"),
    "nancy pelosi": (3, "Former Speaker"),
    "mike johnson": (3, "Speaker"),
    "steve scalise": (3, "House Majority Leader"),
    "hakeem jeffries": (3, "House Minority Leader"),
    "john thune": (3, "Senate Majority Leader"),
    "charles e schumer": (3, "Senate Minority Leader"),
    "chuck schumer": (3, "Senate Minority Leader"),
}


@dataclass(frozen=True)
class PoliticalTrade:
    ticker: str
    person: str
    side: str
    transaction_date: date
    filed_date: date | None
    amount_text: str
    amount_min: int
    amount_max: int
    chamber: str
    party: str | None = None
    conflict: str | None = None
    profile_url: str | None = None

    def to_json(self) -> dict[str, object]:
        row = asdict(self)
        row["transaction_date"] = self.transaction_date.isoformat()
        row["filed_date"] = self.filed_date.isoformat() if self.filed_date else None
        return row

    @classmethod
    def from_json(cls, row: dict[str, object]) -> "PoliticalTrade":
        return cls(
            ticker=str(row["ticker"]),
            person=str(row["person"]),
            side=str(row["side"]),
            transaction_date=date.fromisoformat(str(row["transaction_date"])),
            filed_date=date.fromisoformat(str(row["filed_date"])) if row.get("filed_date") else None,
            amount_text=str(row.get("amount_text") or ""),
            amount_min=int(row.get("amount_min") or 0),
            amount_max=int(row.get("amount_max") or 0),
            chamber=str(row.get("chamber") or "unknown"),
            party=str(row["party"]) if row.get("party") else None,
            conflict=str(row["conflict"]) if row.get("conflict") else None,
            profile_url=str(row["profile_url"]) if row.get("profile_url") else None,
        )


class PoliticalActivityProvider:
    def __init__(
        self,
        *,
        lookback_days: int = 365,
        cache_dir: Path = Path("data/political_cache"),
        cache_hours: int = 24,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.lookback_days = max(1, lookback_days)
        self.cache_dir = cache_dir
        self.cache_hours = max(1, cache_hours)
        self.timeout_seconds = timeout_seconds

    def activity_for_many(self, tickers: Iterable[str]) -> dict[str, dict[str, object]]:
        clean_tickers = list(dict.fromkeys(ticker.upper() for ticker in tickers if ticker.strip()))
        if not clean_tickers:
            return {}

        rows: dict[str, dict[str, object]] = {}
        headers = {
            "User-Agent": "sentiment-analysis/0.1 political-disclosure research",
            "Accept": "text/html,application/xhtml+xml",
        }
        with httpx.Client(headers=headers, timeout=self.timeout_seconds, follow_redirects=True) as client:
            for ticker in clean_tickers:
                try:
                    trades, stale = self._trades_for(ticker, client)
                    rows[ticker] = summarize_political_activity(
                        ticker,
                        trades,
                        lookback_days=self.lookback_days,
                    )
                    rows[ticker]["political_stale"] = stale
                except Exception as exc:
                    rows[ticker] = {**empty_political_row(ticker), "political_error": str(exc)}
        return rows

    def _trades_for(
        self,
        ticker: str,
        client: httpx.Client,
    ) -> tuple[list[PoliticalTrade], bool]:
        cached = self._read_cache(ticker)
        if cached and self._cache_is_fresh(cached[0]):
            return cached[1], False

        try:
            response = client.get(CAPITOL_MARKETS_TICKER_URL.format(ticker=ticker))
            response.raise_for_status()
            trades = parse_capitol_markets_page(ticker, response.text)
            self._write_cache(ticker, trades)
            return trades, False
        except Exception:
            if cached:
                return cached[1], True
            raise

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker.upper()}.json"

    def _read_cache(self, ticker: str) -> tuple[datetime, list[PoliticalTrade]] | None:
        path = self._cache_path(ticker)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            fetched_at = datetime.fromisoformat(str(payload["fetched_at"]))
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
            trades = [PoliticalTrade.from_json(row) for row in payload.get("trades", [])]
            return fetched_at.astimezone(timezone.utc), trades
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    def _write_cache(self, ticker: str, trades: list[PoliticalTrade]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source_url": CAPITOL_MARKETS_TICKER_URL.format(ticker=ticker),
            "trades": [trade.to_json() for trade in trades],
        }
        self._cache_path(ticker).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _cache_is_fresh(self, fetched_at: datetime) -> bool:
        return datetime.now(timezone.utc) - fetched_at <= timedelta(hours=self.cache_hours)


def parse_capitol_markets_page(ticker: str, html: str) -> list[PoliticalTrade]:
    parser = _TradeTableParser(ticker.upper())
    parser.feed(html)
    return parser.trades


def summarize_political_activity(
    ticker: str,
    trades: Iterable[PoliticalTrade],
    *,
    lookback_days: int,
    today: date | None = None,
) -> dict[str, object]:
    current = today or datetime.now(timezone.utc).date()
    cutoff = current - timedelta(days=max(1, lookback_days))
    relevant = [
        trade
        for trade in trades
        if cutoff <= trade.transaction_date <= current
    ]
    relevant.sort(key=_trade_sort_key)

    by_person: dict[str, list[PoliticalTrade]] = {}
    for trade in relevant:
        by_person.setdefault(_normalize_name(trade.person), []).append(trade)

    latest_buys: list[PoliticalTrade] = []
    latest_sells: list[PoliticalTrade] = []
    for person_trades in by_person.values():
        latest = max(person_trades, key=_trade_sort_key)
        if latest.side == "BUY":
            latest_buys.append(latest)
        elif latest.side == "SELL":
            latest_sells.append(latest)

    latest_buys.sort(key=_signal_sort_key, reverse=True)
    latest_sells.sort(key=_signal_sort_key, reverse=True)
    buy_strength = sum(_trade_weight(trade) for trade in latest_buys)
    sell_strength = sum(_trade_weight(trade) for trade in latest_sells)
    net_strength = buy_strength - sell_strength
    high_profile_buyers = [trade for trade in latest_buys if _profile_weight(trade)[0] >= 3]
    high_profile_sellers = [trade for trade in latest_sells if _profile_weight(trade)[0] >= 3]

    # The strategy's political thesis is specifically accumulation followed by no
    # later disclosed sale. A cluster of three such filers, or one high-profile
    # filer, is therefore stronger than broad sell noise from unrelated filers.
    strong_bullish_hold = len(latest_buys) >= 3 or bool(high_profile_buyers)
    strong_bearish_sale = not strong_bullish_hold and (
        len(latest_sells) >= 3 or bool(high_profile_sellers)
    )
    if strong_bullish_hold:
        bias = "bullish"
        alert = "strong"
        directional_strength = buy_strength
        directional_high_profile = high_profile_buyers
        signal_basis = "no_later_sale_cluster"
    elif strong_bearish_sale:
        bias = "bearish"
        alert = "strong"
        directional_strength = -sell_strength
        directional_high_profile = high_profile_sellers
        signal_basis = "latest_sale_cluster"
    elif net_strength >= 2:
        bias = "bullish"
        alert = "active"
        directional_strength = net_strength
        directional_high_profile = high_profile_buyers
        signal_basis = "latest_side_net"
    elif net_strength <= -2:
        bias = "bearish"
        alert = "active"
        directional_strength = net_strength
        directional_high_profile = high_profile_sellers
        signal_basis = "latest_side_net"
    elif relevant:
        bias = "mixed"
        alert = "mixed"
        directional_strength = net_strength
        directional_high_profile = []
        signal_basis = "latest_side_mixed"
    else:
        bias = "none"
        alert = "none"
        directional_strength = 0
        directional_high_profile = []
        signal_basis = "none"

    latest = max(relevant, key=_trade_sort_key, default=None)
    high_profile = [
        _politician_summary(trade)
        for trade in latest_buys + latest_sells
        if _profile_weight(trade)[0] >= 3
    ]
    source_url = CAPITOL_MARKETS_TICKER_URL.format(ticker=ticker.upper())
    return {
        "political_alert": alert,
        "political_bias": bias,
        "political_strength": directional_strength,
        "political_net_strength": net_strength,
        "political_buy_strength": buy_strength,
        "political_sell_strength": sell_strength,
        "political_buyers": len(latest_buys),
        "political_sellers": len(latest_sells),
        "political_buy_transactions": sum(trade.side == "BUY" for trade in relevant),
        "political_sell_transactions": sum(trade.side == "SELL" for trade in relevant),
        "political_no_later_sale_count": len(latest_buys),
        "political_buyer_names": [_politician_summary(trade) for trade in latest_buys],
        "political_seller_names": [_politician_summary(trade) for trade in latest_sells],
        "political_high_profile_names": high_profile,
        "political_directional_high_profile_names": [
            _politician_summary(trade) for trade in directional_high_profile
        ],
        "political_signal_basis": signal_basis,
        "political_latest_date": latest.transaction_date.isoformat() if latest else None,
        "political_source": "capitol_markets_public_disclosures",
        "political_source_url": source_url,
        "political_official_house_url": HOUSE_DISCLOSURE_URL,
        "political_official_senate_url": SENATE_DISCLOSURE_URL,
        "political_disclaimer": POLITICAL_DATA_DISCLAIMER,
        "political_stale": False,
    }


def empty_political_row(ticker: str) -> dict[str, object]:
    return {
        "political_alert": "unknown",
        "political_bias": "unknown",
        "political_strength": 0,
        "political_net_strength": 0,
        "political_buy_strength": 0,
        "political_sell_strength": 0,
        "political_buyers": 0,
        "political_sellers": 0,
        "political_buy_transactions": 0,
        "political_sell_transactions": 0,
        "political_no_later_sale_count": 0,
        "political_buyer_names": [],
        "political_seller_names": [],
        "political_high_profile_names": [],
        "political_directional_high_profile_names": [],
        "political_signal_basis": "none",
        "political_latest_date": None,
        "political_source": "none",
        "political_source_url": CAPITOL_MARKETS_TICKER_URL.format(ticker=ticker.upper()),
        "political_official_house_url": HOUSE_DISCLOSURE_URL,
        "political_official_senate_url": SENATE_DISCLOSURE_URL,
        "political_disclaimer": POLITICAL_DATA_DISCLAIMER,
        "political_stale": False,
    }


class _TradeTableParser(HTMLParser):
    def __init__(self, ticker: str) -> None:
        super().__init__(convert_charrefs=True)
        self.ticker = ticker
        self.trades: list[PoliticalTrade] = []
        self._in_row = False
        self._in_cell = False
        self._cells: list[str] = []
        self._cell_parts: list[str] = []
        self._person: str | None = None
        self._profile_url: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "tr":
            self._in_row = True
            self._cells = []
            self._person = None
            self._profile_url = None
        elif tag == "td" and self._in_row:
            self._in_cell = True
            self._cell_parts = []
        elif tag == "img" and self._in_cell and not self._person:
            alt = _clean_text(attrs_dict.get("alt") or "")
            if alt:
                self._person = alt
        elif tag == "a" and self._in_cell:
            href = attrs_dict.get("href") or ""
            if href.startswith("/politicians/"):
                self._profile_url = f"https://capitolmarkets.org{href}"

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._in_cell:
            self._cells.append(_clean_text(" ".join(self._cell_parts)))
            self._in_cell = False
            self._cell_parts = []
        elif tag == "tr" and self._in_row:
            trade = self._build_trade()
            if trade:
                self.trades.append(trade)
            self._in_row = False

    def _build_trade(self) -> PoliticalTrade | None:
        if len(self._cells) < 5 or self._cells[1] not in {"BUY", "SELL"}:
            return None
        transaction_date = _parse_display_date(self._cells[3])
        if transaction_date is None:
            return None
        first_cell = self._cells[0]
        chamber_match = re.search(r"(House|Senate|POTUS)$", first_cell, re.IGNORECASE)
        chamber = chamber_match.group(1).upper() if chamber_match else "UNKNOWN"
        party_match = re.search(r"([DRI])\s*(?:House|Senate|POTUS)$", first_cell, re.IGNORECASE)
        person = self._person or re.sub(r"[DRI]?\s*(?:House|Senate|POTUS)$", "", first_cell).strip()
        amount_min, amount_max = _amount_range(self._cells[2])
        conflict = self._cells[5].replace("\u2691", "").strip() if len(self._cells) > 5 else ""
        return PoliticalTrade(
            ticker=self.ticker,
            person=person,
            side=self._cells[1],
            transaction_date=transaction_date,
            filed_date=_parse_display_date(self._cells[4]),
            amount_text=self._cells[2],
            amount_min=amount_min,
            amount_max=amount_max,
            chamber=chamber,
            party=party_match.group(1).upper() if party_match else None,
            conflict=conflict if conflict and conflict != "-" else None,
            profile_url=self._profile_url,
        )


def _profile_weight(trade: PoliticalTrade) -> tuple[int, str]:
    if trade.chamber == "POTUS":
        return 4, "President"
    return HIGH_PROFILE_ROLES.get(_normalize_name(trade.person), (1, trade.chamber.title()))


def _trade_weight(trade: PoliticalTrade) -> int:
    profile_weight, _ = _profile_weight(trade)
    size_bonus = 2 if trade.amount_max >= 500_000 else 1 if trade.amount_max >= 100_000 else 0
    return profile_weight + size_bonus


def _politician_summary(trade: PoliticalTrade) -> str:
    weight, role = _profile_weight(trade)
    del weight
    return f"{trade.person} ({role}, {trade.amount_text})"


def _signal_sort_key(trade: PoliticalTrade) -> tuple[int, int, date]:
    return (_trade_weight(trade), trade.amount_max, trade.transaction_date)


def _trade_sort_key(trade: PoliticalTrade) -> tuple[date, date]:
    return (trade.transaction_date, trade.filed_date or trade.transaction_date)


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value)).strip()


def _parse_display_date(value: str) -> date | None:
    try:
        return datetime.strptime(value.strip(), "%b %d, %Y").date()
    except ValueError:
        return None


def _amount_range(value: str) -> tuple[int, int]:
    numbers = re.findall(r"\$?([0-9.]+)\s*([KMB]?)", value.upper())
    parsed = [_amount_value(number, suffix) for number, suffix in numbers]
    if not parsed:
        return 0, 0
    if len(parsed) == 1:
        return parsed[0], parsed[0]
    return parsed[0], parsed[1]


def _amount_value(number: str, suffix: str) -> int:
    multiplier = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]
    return int(float(number) * multiplier)
