from __future__ import annotations

from datetime import date
import unittest

from sentiment_analysis.providers.political_activity import (
    PoliticalTrade,
    parse_capitol_markets_page,
    summarize_political_activity,
)


class PoliticalActivityTests(unittest.TestCase):
    def test_parses_public_ticker_trade_table(self) -> None:
        html = """
        <table><tbody><tr>
          <td><a href="/politicians/nancy-pelosi"><img alt="Nancy Pelosi">Nancy Pelosi<span>D<span>House</span></span></a></td>
          <td>BUY</td><td>$500K-$1M</td><td>May 29, 2026</td><td>Jun 23, 2026</td><td>-</td>
        </tr></tbody></table>
        """

        rows = parse_capitol_markets_page("UBER", html)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].person, "Nancy Pelosi")
        self.assertEqual(rows[0].side, "BUY")
        self.assertEqual(rows[0].chamber, "HOUSE")
        self.assertEqual(rows[0].party, "D")
        self.assertEqual(rows[0].amount_max, 1_000_000)
        self.assertEqual(rows[0].transaction_date, date(2026, 5, 29))

    def test_three_latest_buyers_create_strong_bullish_signal(self) -> None:
        rows = [
            _trade("Nancy Pelosi", "BUY", "2026-05-29", 500_000, 1_000_000),
            _trade("Julie Johnson", "BUY", "2026-06-10", 1_000, 15_000),
            _trade("Ro Khanna", "BUY", "2026-07-15", 15_000, 50_000),
        ]

        result = summarize_political_activity(
            "PLTR",
            rows,
            lookback_days=365,
            today=date(2026, 8, 8),
        )

        self.assertEqual(result["political_alert"], "strong")
        self.assertEqual(result["political_bias"], "bullish")
        self.assertEqual(result["political_buyers"], 3)
        self.assertEqual(result["political_no_later_sale_count"], 3)
        self.assertIn("Nancy Pelosi", result["political_high_profile_names"][0])

    def test_later_sale_removes_person_from_latest_buyers(self) -> None:
        rows = [
            _trade("Nancy Pelosi", "BUY", "2026-05-29", 500_000, 1_000_000),
            _trade("Nancy Pelosi", "SELL", "2026-07-01", 500_000, 1_000_000),
        ]

        result = summarize_political_activity(
            "UBER",
            rows,
            lookback_days=365,
            today=date(2026, 8, 8),
        )

        self.assertEqual(result["political_buyers"], 0)
        self.assertEqual(result["political_sellers"], 1)
        self.assertEqual(result["political_bias"], "bearish")

    def test_high_profile_hold_cluster_overrides_unrelated_sale_noise(self) -> None:
        rows = [
            _trade("Nancy Pelosi", "BUY", "2026-05-29", 500_000, 1_000_000),
            _trade("Donald J. Trump", "SELL", "2026-06-01", 100_000, 250_000),
            _trade("Member One", "SELL", "2026-06-02", 1_000, 15_000),
            _trade("Member Two", "SELL", "2026-06-03", 1_000, 15_000),
            _trade("Member Three", "SELL", "2026-06-04", 1_000, 15_000),
            _trade("Member Four", "SELL", "2026-06-05", 1_000, 15_000),
        ]

        result = summarize_political_activity(
            "UBER",
            rows,
            lookback_days=365,
            today=date(2026, 8, 8),
        )

        self.assertEqual(result["political_bias"], "bullish")
        self.assertEqual(result["political_alert"], "strong")
        self.assertEqual(result["political_signal_basis"], "no_later_sale_cluster")
        self.assertEqual(len(result["political_high_profile_names"]), 2)
        self.assertEqual(len(result["political_directional_high_profile_names"]), 1)


def _trade(
    person: str,
    side: str,
    transaction_date: str,
    amount_min: int,
    amount_max: int,
) -> PoliticalTrade:
    return PoliticalTrade(
        ticker="PLTR",
        person=person,
        side=side,
        transaction_date=date.fromisoformat(transaction_date),
        filed_date=date.fromisoformat(transaction_date),
        amount_text=f"${amount_min}-${amount_max}",
        amount_min=amount_min,
        amount_max=amount_max,
        chamber="HOUSE",
    )


if __name__ == "__main__":
    unittest.main()
