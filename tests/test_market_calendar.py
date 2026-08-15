from __future__ import annotations

from datetime import date
import unittest

from sentiment_analysis.market_calendar import earnings_week_window


class MarketCalendarTests(unittest.TestCase):
    def test_sunday_rolls_to_monday_through_friday(self) -> None:
        start, end = earnings_week_window(date(2026, 8, 2))

        self.assertEqual(start, date(2026, 8, 3))
        self.assertEqual(end, date(2026, 8, 7))

    def test_thursday_keeps_current_week_friday(self) -> None:
        start, end = earnings_week_window(date(2026, 8, 6))

        self.assertEqual(start, date(2026, 8, 6))
        self.assertEqual(end, date(2026, 8, 7))


if __name__ == "__main__":
    unittest.main()

