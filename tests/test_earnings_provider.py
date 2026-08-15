from __future__ import annotations

from datetime import date, datetime, timezone
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from sentiment_analysis.providers.earnings import YFinanceCalendarEarningsProvider


class EarningsProviderTests(unittest.TestCase):
    def test_calendar_queries_one_extra_day_and_filters_inclusively(self) -> None:
        calendar = Mock()
        calendar.get_earnings_calendar.return_value = pd.DataFrame(
            [
                {
                    "Symbol": "FRI",
                    "Company": "Friday Corp",
                    "Marketcap": 10,
                    "Event Start Date": datetime(2026, 8, 7, tzinfo=timezone.utc),
                    "Timing": "BMO",
                },
                {
                    "Symbol": "SAT",
                    "Company": "Saturday Corp",
                    "Marketcap": 10,
                    "Event Start Date": datetime(2026, 8, 8, tzinfo=timezone.utc),
                    "Timing": "BMO",
                },
            ]
        ).set_index("Symbol")

        with patch("yfinance.Calendars", return_value=calendar) as calendars:
            provider = YFinanceCalendarEarningsProvider(max_pages=1)
            events = provider.between(date(2026, 8, 3), date(2026, 8, 7))

        calendars.assert_called_once_with(start=date(2026, 8, 3), end=date(2026, 8, 8))
        self.assertIn("FRI", events)
        self.assertNotIn("SAT", events)


if __name__ == "__main__":
    unittest.main()

