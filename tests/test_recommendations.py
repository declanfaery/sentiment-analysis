from __future__ import annotations

import unittest

from sentiment_analysis.recommendations import build_research_recommendation


class RecommendationTests(unittest.TestCase):
    def test_options_can_confirm_contrarian_direction(self) -> None:
        rec = build_research_recommendation(
            crowd_bias="bullish",
            days_until_earnings=2,
            option_alert="unusual",
            option_bias="puts",
            insider_recent_count=1,
        )

        self.assertEqual(rec.recommendation, "bearish_watch_options_confirm")
        self.assertEqual(rec.recommendation_direction, "bearish")
        self.assertEqual(rec.recommendation_score, 85)

    def test_options_can_conflict_with_contrarian_direction(self) -> None:
        rec = build_research_recommendation(
            crowd_bias="bullish",
            days_until_earnings=2,
            option_alert="unusual",
            option_bias="calls",
            insider_recent_count=0,
        )

        self.assertEqual(rec.recommendation, "bearish_watch_options_conflict")
        self.assertEqual(rec.recommendation_direction, "bearish")
        self.assertEqual(rec.recommendation_score, 30)

    def test_strong_high_profile_political_signal_can_override_x(self) -> None:
        rec = build_research_recommendation(
            crowd_bias="bullish",
            days_until_earnings=2,
            option_alert="unusual",
            option_bias="calls",
            insider_recent_count=0,
            political_alert="strong",
            political_bias="bullish",
            political_strength=5,
            political_buyers=1,
            political_sellers=0,
            political_high_profile_count=1,
        )

        self.assertEqual(
            rec.recommendation,
            "bullish_watch_politics_override_options_confirm",
        )
        self.assertEqual(rec.recommendation_direction, "bullish")
        self.assertEqual(rec.recommendation_score, 95)
        self.assertTrue(any("overrides" in note for note in rec.recommendation_notes))


if __name__ == "__main__":
    unittest.main()
