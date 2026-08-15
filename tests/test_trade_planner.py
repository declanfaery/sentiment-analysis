from __future__ import annotations

from datetime import date, datetime, timezone
import unittest

from sentiment_analysis.models import TickerSignal
from sentiment_analysis.pipeline import PipelineResult
from sentiment_analysis.trading.planner import PlannerOptions, build_trade_plans


class TradePlannerTests(unittest.TestCase):
    def test_confirmed_bearish_signal_gets_seven_percent_put_plan(self) -> None:
        result = _result(timing="AMC", recommendation="bearish_watch_options_confirm", score=80)

        plans = build_trade_plans(
            result,
            "run-1",
            generated_at=datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].status, "planned")
        self.assertEqual(plans[0].option_type, "PUT")
        self.assertEqual(plans[0].allocation_pct, 0.07)

    def test_monday_premarket_signal_is_marked_missed(self) -> None:
        result = _result(
            earnings_date=date(2026, 8, 3),
            timing="BMO",
            recommendation="bearish_watch",
            score=55,
        )

        plan = build_trade_plans(
            result,
            "run-2",
            generated_at=datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc),
        )[0]

        self.assertEqual(plan.status, "skipped")
        self.assertIn("entry time passed", plan.skip_reason or "")

    def test_directional_option_conflict_still_gets_seven_percent_plan(self) -> None:
        result = _result(
            timing="AMC",
            recommendation="bearish_watch_options_conflict",
            score=30,
        )

        plan = build_trade_plans(
            result,
            "run-3",
            generated_at=datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc),
        )[0]

        self.assertEqual(plan.status, "planned")
        self.assertEqual(plan.allocation_pct, 0.05)

    def test_corroboration_gate_rejects_x_only_watch(self) -> None:
        result = _result(
            timing="AMC",
            recommendation="bearish_watch",
            score=80,
        )

        plan = build_trade_plans(
            result,
            "run-4",
            options=PlannerOptions(require_corroborating_signal=True),
            generated_at=datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc),
        )[0]

        self.assertEqual(plan.status, "skipped")
        self.assertIn("corroborating", plan.skip_reason or "")

    def test_corroboration_gate_accepts_political_override(self) -> None:
        result = _result(
            timing="AMC",
            recommendation="bullish_watch_politics_override_options_conflict",
            score=49,
        )

        plan = build_trade_plans(
            result,
            "run-5",
            options=PlannerOptions(require_corroborating_signal=True),
            generated_at=datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc),
        )[0]

        self.assertEqual(plan.status, "planned")


def _result(
    *,
    earnings_date: date = date(2026, 8, 4),
    timing: str,
    recommendation: str,
    score: int,
) -> PipelineResult:
    signal = TickerSignal(
        ticker="AMD",
        mention_count=100,
        unique_authors=80,
        total_engagement=1000,
        avg_sentiment=0.4,
        sentiment_stddev=0.1,
        positive_posts=80,
        neutral_posts=10,
        negative_posts=10,
        traction_score=100,
        crowd_bias="bullish",
        earnings_date=earnings_date,
        days_until_earnings=1,
        contrarian_watch="bearish_watch",
        options_expiration=date(2026, 8, 7),
    )
    now = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
    return PipelineResult(
        signals=[signal],
        queries=(),
        started_at=now,
        finished_at=now,
        raw_posts=100,
        duplicate_posts=0,
        matched_posts=100,
        tickers_seen=1,
        high_traction_tickers=1,
        candidate_rows=(
            {
                "ticker": "AMD",
                "earnings_date": earnings_date.isoformat(),
                "timing": timing,
                "options_expiration": "2026-08-07",
            },
        ),
        market_signal_rows=(
            {
                "ticker": "AMD",
                "recommendation": recommendation,
                "recommendation_direction": "bearish",
                "recommendation_score": score,
            },
        ),
    )


if __name__ == "__main__":
    unittest.main()
