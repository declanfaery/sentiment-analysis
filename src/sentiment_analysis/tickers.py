from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

CASHTAG_RE = re.compile(r"(?<![A-Z0-9])\$([A-Z][A-Z0-9]{0,5})(?:\.[A-Z])?(?![A-Z0-9])")
PLAIN_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")
SIMPLE_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9]{0,5}$")
SPECIAL_SECURITY_RE = re.compile(
    r"\b(right|rights|unit|units|warrant|warrants|preferred|preference|note|notes|debenture|debentures)\b",
    re.IGNORECASE,
)

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

DEFAULT_DISCOVERY_QUERIES = (
    '("earnings" OR "guidance" OR "EPS" OR "revenue" OR "beat" OR "miss") has:cashtags lang:en -is:retweet',
    '("calls" OR "puts" OR "options" OR "stock" OR "stocks" OR "market") has:cashtags lang:en -is:retweet',
    '("breakout" OR "squeeze" OR "short interest" OR "upgrade" OR "downgrade") has:cashtags lang:en -is:retweet',
    '("AI" OR "chips" OR "semis" OR "datacenter" OR "GPU") has:cashtags lang:en -is:retweet',
    '("crypto" OR "bitcoin" OR "ethereum" OR "coinbase") has:cashtags lang:en -is:retweet',
    '("yolo" OR "fomo" OR "retail" OR "squeeze" OR "bagholder") has:cashtags lang:en -is:retweet',
)

TREND_FINANCE_KEYWORDS = {
    "ai",
    "bank",
    "banks",
    "bitcoin",
    "call",
    "calls",
    "chip",
    "chips",
    "crypto",
    "earnings",
    "economy",
    "eps",
    "fed",
    "fomc",
    "guidance",
    "inflation",
    "market",
    "markets",
    "nasdaq",
    "options",
    "puts",
    "recession",
    "revenue",
    "semis",
    "short",
    "stock",
    "stocks",
    "tech",
    "tesla",
}

COMMON_FALSE_POSITIVES = {
    "A",
    "AI",
    "ALL",
    "ATH",
    "BE",
    "CEO",
    "CFO",
    "DD",
    "EPS",
    "ETF",
    "FOMO",
    "GDP",
    "IPO",
    "IT",
    "IV",
    "LOL",
    "MOM",
    "NAV",
    "PM",
    "PR",
    "RH",
    "SEC",
    "USA",
    "YOLO",
}


@dataclass(frozen=True)
class TickerListing:
    symbol: str
    name: str


def load_ticker_universe(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "symbol" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a 'symbol' column")
        return {
            row["symbol"].strip().upper()
            for row in reader
            if row.get("symbol") and row["symbol"].strip()
        }


def refresh_us_ticker_universe(
    path: Path,
    *,
    include_etfs: bool = False,
    include_special_securities: bool = False,
) -> int:
    import httpx

    listings: dict[str, TickerListing] = {}
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        sources = [
            (NASDAQ_LISTED_URL, "Symbol", "Security Name"),
            (OTHER_LISTED_URL, "ACT Symbol", "Security Name"),
        ]
        for url, symbol_column, name_column in sources:
            response = client.get(url)
            response.raise_for_status()
            for listing in parse_nasdaqtrader_symbol_directory(
                response.text,
                symbol_column=symbol_column,
                name_column=name_column,
                include_etfs=include_etfs,
                include_special_securities=include_special_securities,
            ):
                listings[listing.symbol] = listing

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "name"])
        writer.writeheader()
        for listing in sorted(listings.values(), key=lambda item: item.symbol):
            writer.writerow({"symbol": listing.symbol, "name": listing.name})

    return len(listings)


def parse_nasdaqtrader_symbol_directory(
    text: str,
    *,
    symbol_column: str,
    name_column: str,
    include_etfs: bool = False,
    include_special_securities: bool = False,
) -> list[TickerListing]:
    reader = csv.DictReader(io.StringIO(text), delimiter="|")
    listings: list[TickerListing] = []

    for row in reader:
        raw_symbol = (row.get(symbol_column) or "").strip().upper()
        symbol = normalize_symbol(raw_symbol)
        if symbol is None:
            continue
        if row.get("Test Issue", "").strip().upper() == "Y":
            continue
        if not include_etfs and row.get("ETF", "").strip().upper() == "Y":
            continue
        name = (row.get(name_column) or "").strip()
        if not include_special_securities and SPECIAL_SECURITY_RE.search(name):
            continue
        listings.append(TickerListing(symbol=symbol, name=name))

    return listings


def normalize_symbol(symbol: str) -> str | None:
    clean = symbol.strip().upper()
    if not SIMPLE_SYMBOL_RE.fullmatch(clean):
        return None
    if clean in COMMON_FALSE_POSITIVES:
        return None
    return clean


def extract_tickers(
    text: str,
    universe: set[str],
    *,
    include_plain_tickers: bool = False,
) -> tuple[str, ...]:
    cashtags = {match.group(1).upper() for match in CASHTAG_RE.finditer(text.upper())}
    tickers = {ticker for ticker in cashtags if ticker in universe}

    if include_plain_tickers:
        for match in PLAIN_TICKER_RE.finditer(text.upper()):
            symbol = match.group(1).upper()
            if symbol in universe and symbol not in COMMON_FALSE_POSITIVES:
                tickers.add(symbol)

    return tuple(sorted(tickers))


def build_discovery_queries() -> list[str]:
    return list(DEFAULT_DISCOVERY_QUERIES)


def build_trend_queries(
    trend_names: Sequence[str] | Iterable[str],
    universe: set[str],
    *,
    max_queries: int = 25,
) -> list[str]:
    queries: list[str] = []
    seen_queries: set[str] = set()

    for trend_name in trend_names:
        tickers = extract_tickers(trend_name, universe, include_plain_tickers=True)
        for ticker in tickers:
            query = f"${ticker} lang:en -is:retweet"
            if query not in seen_queries:
                seen_queries.add(query)
                queries.append(query)
        if len(queries) >= max_queries:
            return queries[:max_queries]

    for trend_name in trend_names:
        if not _looks_finance_relevant_trend(trend_name):
            continue
        query_term = _trend_query_term(trend_name)
        if not query_term:
            continue
        query = f"{query_term} has:cashtags lang:en -is:retweet"
        if query not in seen_queries:
            seen_queries.add(query)
            queries.append(query)
        if len(queries) >= max_queries:
            return queries[:max_queries]

    return queries[:max_queries]


def _looks_finance_relevant_trend(trend_name: str) -> bool:
    lowered = trend_name.lower()
    words = set(re.findall(r"[a-z0-9]+", lowered))
    return bool(words.intersection(TREND_FINANCE_KEYWORDS))


def _trend_query_term(trend_name: str) -> str | None:
    clean = trend_name.strip()
    if not clean:
        return None
    if clean.startswith("#") and re.fullmatch(r"#[A-Za-z0-9_]+", clean):
        return clean
    if clean.startswith("$") and re.fullmatch(r"\$[A-Za-z][A-Za-z0-9]{0,5}", clean):
        return clean.upper()
    if len(clean) > 60:
        return None
    escaped = clean.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_cashtag_queries(
    symbols: Sequence[str] | Iterable[str],
    *,
    max_chars: int = 512,
    suffix: str = "lang:en -is:retweet",
) -> list[str]:
    clean_symbols = sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()})
    if not clean_symbols:
        return []

    queries: list[str] = []
    current_terms: list[str] = []

    for symbol in clean_symbols:
        term = f"${symbol}"
        candidate_terms = [*current_terms, term]
        candidate = f"({' OR '.join(candidate_terms)}) {suffix}".strip()
        if current_terms and len(candidate) > max_chars:
            queries.append(f"({' OR '.join(current_terms)}) {suffix}".strip())
            current_terms = [term]
        else:
            current_terms = candidate_terms

    if current_terms:
        queries.append(f"({' OR '.join(current_terms)}) {suffix}".strip())

    return queries
