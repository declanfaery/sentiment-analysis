from __future__ import annotations

from datetime import date, datetime, timezone
import unittest

from sentiment_analysis.dashboard import (
    _enforce_cheap_signal_gate,
    _prioritize_for_x_counts,
    _select_tickers_for_posts,
)
from sentiment_analysis.models import EarningsEvent, TickerSignal
from sentiment_analysis.pipeline import PipelineResult


class DashboardSelectionTests(unittest.TestCase):
    def test_x_count_priority_keeps_large_active_names_in_budget(self) -> None:
        events = [
            EarningsEvent(
                "SMALL",
                date(2026, 8, 5),
                "test",
                expected_move_pct=12.0,
                market_cap=50_000_000,
            ),
            EarningsEvent(
                "UBER",
                date(2026, 8, 5),
                "test",
                expected_move_pct=7.0,
                market_cap=143_000_000_000,
                timing="BMO",
            ),
            EarningsEvent(
                "PLTR",
                date(2026, 8, 3),
                "test",
                expected_move_pct=11.0,
                market_cap=295_000_000_000,
                timing="AMC",
            ),
        ]

        prioritized = _prioritize_for_x_counts(events, 2)

        self.assertEqual([event.ticker for event in prioritized], ["PLTR", "UBER"])

    def test_x_count_floor_has_no_below_threshold_fallback(self) -> None:
        selected = _select_tickers_for_posts(
            {"PLTR": 4, "UBER": 1},
            min_mentions=5,
            max_tickers=8,
        )

        self.assertEqual(selected, [])

    def test_final_gate_runs_before_market_and_political_enrichment(self) -> None:
        result = PipelineResult(
            signals=[
                _signal("PASS", move=6.0, x_count=10, mentions=6, earnings=date(2026, 8, 5)),
                _signal("LOWMOVE", move=4.9, x_count=10, mentions=6, earnings=date(2026, 8, 5)),
                _signal("LOWCOUNT", move=6.0, x_count=4, mentions=6, earnings=date(2026, 8, 5)),
                _signal("LOWSAMPLE", move=6.0, x_count=10, mentions=4, earnings=date(2026, 8, 5)),
                _signal("OUTSIDE", move=6.0, x_count=10, mentions=6, earnings=date(2026, 8, 10)),
            ],
            queries=(),
            started_at=datetime(2026, 8, 3, tzinfo=timezone.utc),
            finished_at=datetime(2026, 8, 3, tzinfo=timezone.utc),
            raw_posts=32,
            duplicate_posts=0,
            matched_posts=32,
            tickers_seen=5,
            high_traction_tickers=5,
        )
        logs: list[str] = []

        gated = _enforce_cheap_signal_gate(
            result,
            {"min_mentions": 5, "expected_move_min_pct": 5},
            date(2026, 8, 3),
            date(2026, 8, 7),
            logs.append,
        )

        self.assertEqual([signal.ticker for signal in gated.signals], ["PASS"])
        self.assertEqual(gated.high_traction_tickers, 1)
        self.assertTrue(any("rejected 4" in line for line in logs))


def _signal(
    ticker: str,
    *,
    move: float,
    x_count: int,
    mentions: int,
    earnings: date,
) -> TickerSignal:
    return TickerSignal(
        ticker=ticker,
        mention_count=mentions,
        unique_authors=mentions,
        total_engagement=mentions,
        avg_sentiment=0.2,
        sentiment_stddev=0.0,
        positive_posts=mentions,
        neutral_posts=0,
        negative_posts=0,
        traction_score=float(mentions),
        crowd_bias="bullish",
        earnings_date=earnings,
        days_until_earnings=2,
        contrarian_watch="bearish_watch",
        preflight_count=x_count,
        expected_move_pct=move,
    )


if __name__ == "__main__":
    unittest.main()
