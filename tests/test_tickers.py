from __future__ import annotations

import unittest

from sentiment_analysis.tickers import (
    build_cashtag_queries,
    build_trend_queries,
    extract_tickers,
    parse_nasdaqtrader_symbol_directory,
)


class TickerTests(unittest.TestCase):
    def test_extracts_cashtags_from_universe(self) -> None:
        universe = {"TSLA", "NVDA"}
        self.assertEqual(
            extract_tickers("$TSLA and $NVDA are loud, $FAKE ignored", universe),
            ("NVDA", "TSLA"),
        )

    def test_plain_tickers_are_optional(self) -> None:
        universe = {"TSLA"}
        self.assertEqual(extract_tickers("TSLA is loud", universe), ())
        self.assertEqual(
            extract_tickers("TSLA is loud", universe, include_plain_tickers=True),
            ("TSLA",),
        )

    def test_query_builder_respects_max_length(self) -> None:
        queries = build_cashtag_queries(["AAPL", "MSFT", "NVDA", "TSLA"], max_chars=40, suffix="lang:en")
        self.assertGreater(len(queries), 1)
        self.assertTrue(all(len(query) <= 40 for query in queries))

    def test_parses_nasdaqtrader_symbols(self) -> None:
        text = (
            "Symbol|Security Name|ETF|Test Issue\n"
            "AAPL|Apple Inc.|N|N\n"
            "SPY|SPDR S&P 500 ETF|Y|N\n"
            "AACBR|Artius II Acquisition Inc. - Rights|N|N\n"
            "AACBU|Artius II Acquisition Inc. - Units|N|N\n"
            "TEST|Test Corp|N|Y\n"
            "BRK/A|Berkshire Hathaway|N|N\n"
        )

        listings = parse_nasdaqtrader_symbol_directory(
            text,
            symbol_column="Symbol",
            name_column="Security Name",
        )

        self.assertEqual([listing.symbol for listing in listings], ["AAPL"])

    def test_builds_trend_queries_from_cashtags_and_finance_terms(self) -> None:
        queries = build_trend_queries(
            ["$TSLA", "#AI", "Totally Random"],
            {"TSLA", "NVDA"},
            max_queries=4,
        )

        self.assertIn("$TSLA lang:en -is:retweet", queries)
        self.assertIn("#AI has:cashtags lang:en -is:retweet", queries)
        self.assertNotIn('"Totally Random" has:cashtags lang:en -is:retweet', queries)


if __name__ == "__main__":
    unittest.main()
