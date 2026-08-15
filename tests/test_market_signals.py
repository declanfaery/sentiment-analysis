from __future__ import annotations

from datetime import date
import unittest

from sentiment_analysis.providers.market_signals import (
    summarize_insider_activity,
    summarize_option_activity,
)


class MarketSignalsTests(unittest.TestCase):
    def test_summarizes_unusual_options_activity(self) -> None:
        row = summarize_option_activity(
            "PLTR",
            [
                {
                    "expiration": "2026-08-07",
                    "type": "C",
                    "strike": 125.0,
                    "volume": 900,
                    "open_interest": 100,
                    "bid": 2.0,
                    "ask": 2.2,
                    "spot_price": 123.0,
                },
                {
                    "expiration": "2026-08-07",
                    "type": "P",
                    "strike": 120.0,
                    "volume": 100,
                    "open_interest": 500,
                    "bid": 1.0,
                    "ask": 1.2,
                    "spot_price": 123.0,
                },
            ],
            min_volume=250,
            min_vol_oi_ratio=2.0,
            min_premium=100_000,
            max_moneyness_distance_pct=35.0,
        )

        self.assertEqual(row["option_alert"], "unusual")
        self.assertEqual(row["option_bias"], "calls")
        self.assertEqual(row["option_unusual_contracts"], 1)
        self.assertEqual(row["option_top_contract"], "2026-08-07 125C")
        self.assertEqual(row["option_top_vol_oi"], 9.0)

    def test_low_premium_options_are_active_not_unusual(self) -> None:
        row = summarize_option_activity(
            "TINY",
            [
                {
                    "expiration": "2026-08-07",
                    "type": "C",
                    "strike": 10.0,
                    "volume": 900,
                    "open_interest": 100,
                    "bid": 0.05,
                    "ask": 0.07,
                    "spot_price": 10.0,
                }
            ],
            min_volume=250,
            min_vol_oi_ratio=2.0,
            min_premium=100_000,
            max_moneyness_distance_pct=35.0,
        )

        self.assertEqual(row["option_alert"], "quiet")
        self.assertEqual(row["option_unusual_contracts"], 0)

    def test_far_from_spot_contracts_are_ignored_by_default(self) -> None:
        row = summarize_option_activity(
            "SNDK",
            [
                {
                    "expiration": "2026-08-07",
                    "type": "C",
                    "strike": 1300.0,
                    "volume": 5000,
                    "open_interest": 100,
                    "bid": 5.0,
                    "ask": 5.2,
                    "spot_price": 80.0,
                }
            ],
            min_volume=250,
            min_vol_oi_ratio=2.0,
            min_premium=100_000,
            max_moneyness_distance_pct=35.0,
        )

        self.assertEqual(row["option_alert"], "quiet")
        self.assertEqual(row["option_total_volume"], 0)

    def test_summarizes_recent_insider_filings(self) -> None:
        row = summarize_insider_activity(
            "PLTR",
            {
                "filings": {
                    "recent": {
                        "form": ["4", "8-K", "4/A"],
                        "filingDate": ["2026-08-01", "2026-08-02", "2026-07-01"],
                        "accessionNumber": [
                            "0001321655-26-000001",
                            "0001321655-26-000002",
                            "0001321655-26-000003",
                        ],
                        "primaryDocument": ["form4.xml", "form8k.htm", "form4a.xml"],
                    }
                }
            },
            cik="0001321655",
            lookback_days=14,
            today=date(2026, 8, 3),
        )

        self.assertEqual(row["insider_alert"], "recent")
        self.assertEqual(row["insider_recent_count"], 1)
        self.assertEqual(row["insider_latest_form"], "4")
        self.assertEqual(row["insider_latest_date"], "2026-08-01")
        self.assertIn("000132165526000001", row["insider_latest_url"])


if __name__ == "__main__":
    unittest.main()
