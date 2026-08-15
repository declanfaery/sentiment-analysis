from __future__ import annotations

from datetime import date
import unittest

from sentiment_analysis.models import EarningsEvent
from sentiment_analysis.screening import CandidateScreen, screen_earnings_candidates


class ScreeningTests(unittest.TestCase):
    def test_filters_low_expected_move_unknown_and_low_quality_candidates(self) -> None:
        events = [
            EarningsEvent("NVDA", date(2026, 8, 3), "test", company="NVIDIA", expected_move_pct=9.0),
            EarningsEvent("SLEEPY", date(2026, 8, 3), "test", company="Sleepy Corp", expected_move_pct=2.0),
            EarningsEvent("UNK", date(2026, 8, 3), "test", company="Unknown Corp"),
            EarningsEvent("SPACU", date(2026, 8, 3), "test", company="Blank Check Acquisition Units", expected_move_pct=12.0),
        ]

        screened = screen_earnings_candidates(
            events,
            screen=CandidateScreen(
                min_expected_move_pct=5.0,
                require_expected_move=True,
                max_candidates=10,
                skip_low_quality_names=True,
            ),
            today=date(2026, 8, 1),
        )

        self.assertEqual([event.ticker for event in screened], ["NVDA"])

    def test_caps_candidates_by_quality_score(self) -> None:
        events = [
            EarningsEvent("MID", date(2026, 8, 3), "test", expected_move_pct=6.0),
            EarningsEvent("BIGMOVE", date(2026, 8, 3), "test", expected_move_pct=12.0),
        ]

        screened = screen_earnings_candidates(
            events,
            screen=CandidateScreen(
                min_expected_move_pct=5.0,
                require_expected_move=True,
                max_candidates=1,
                skip_low_quality_names=True,
            ),
            today=date(2026, 8, 1),
        )

        self.assertEqual([event.ticker for event in screened], ["BIGMOVE"])


if __name__ == "__main__":
    unittest.main()
