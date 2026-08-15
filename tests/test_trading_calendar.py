from __future__ import annotations

from datetime import date, datetime, timezone
import unittest

from sentiment_analysis.trading.calendar import earnings_trade_times, weekly_scan_due


class TradingCalendarTests(unittest.TestCase):
    def test_after_close_exits_fifteen_minutes_after_next_open(self) -> None:
        entry, exit_at = earnings_trade_times(date(2026, 8, 4), "AMC")

        self.assertEqual(entry, datetime(2026, 8, 4, 19, 45, tzinfo=timezone.utc))
        self.assertEqual(exit_at, datetime(2026, 8, 5, 13, 45, tzinfo=timezone.utc))

    def test_premarket_exits_fifteen_minutes_after_earnings_open(self) -> None:
        entry, exit_at = earnings_trade_times(date(2026, 8, 4), "BMO")

        self.assertEqual(entry, datetime(2026, 8, 3, 19, 45, tzinfo=timezone.utc))
        self.assertEqual(exit_at, datetime(2026, 8, 4, 13, 45, tzinfo=timezone.utc))

    def test_exit_delay_can_be_overridden(self) -> None:
        _, exit_at = earnings_trade_times(
            date(2026, 8, 4),
            "AMC",
            exit_delay_minutes=30,
        )

        self.assertEqual(exit_at, datetime(2026, 8, 5, 14, 0, tzinfo=timezone.utc))

    def test_entry_lead_can_be_overridden(self) -> None:
        entry, _ = earnings_trade_times(
            date(2026, 8, 4),
            "AMC",
            entry_before_close_minutes=30,
        )

        self.assertEqual(entry, datetime(2026, 8, 4, 19, 30, tzinfo=timezone.utc))

    def test_weekly_scan_runs_once_after_monday_morning(self) -> None:
        monday_noon_utc = datetime(2026, 8, 3, 12, 30, tzinfo=timezone.utc)

        self.assertTrue(weekly_scan_due(monday_noon_utc, completed_week_key=None))
        self.assertFalse(
            weekly_scan_due(monday_noon_utc, completed_week_key="2026-08-03")
        )

    def test_weekly_scan_does_not_catch_up_on_weekend(self) -> None:
        saturday = datetime(2026, 8, 8, 14, 0, tzinfo=timezone.utc)

        self.assertFalse(weekly_scan_due(saturday, completed_week_key=None))

    def test_weekly_scan_does_not_catch_up_after_friday_close(self) -> None:
        friday_night = datetime(2026, 8, 8, 2, 0, tzinfo=timezone.utc)

        self.assertFalse(weekly_scan_due(friday_night, completed_week_key=None))


if __name__ == "__main__":
    unittest.main()
