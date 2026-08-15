from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
import tempfile
import unittest

from sentiment_analysis.models import TickerSignal
from sentiment_analysis.pipeline import PipelineResult
from sentiment_analysis.storage import save_scan_result
from sentiment_analysis.trading.config import TradingConfig
from sentiment_analysis.trading.models import TradePlan
from sentiment_analysis.trading.service import AutomationService
from sentiment_analysis.trading.storage import (
    latest_scan_run_id,
    list_trade_plans,
    save_trade_plans,
    strategy_equity,
)


class TradingCapitalTests(unittest.TestCase):
    def test_strategy_equity_compounds_only_its_realized_results(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "small.sqlite3"
            now = datetime(2026, 8, 11, 14, 0, tzinfo=timezone.utc)
            save_trade_plans(
                path,
                [
                    TradePlan(
                        id="j" * 32,
                        scan_run_id="scan-1",
                        ticker="AMD",
                        direction="bearish",
                        option_type="PUT",
                        earnings_date=date(2026, 8, 10),
                        earnings_timing="AMC",
                        entry_at=now,
                        exit_at=now,
                        allocation_pct=0.07,
                        recommendation="bearish_watch",
                        recommendation_score=70,
                        status="closed",
                        entry_fill_price=5.0,
                        exit_fill_price=6.0,
                        quantity=2,
                        filled_quantity=2,
                        created_at=now,
                        updated_at=now,
                    )
                ],
            )

            self.assertEqual(strategy_equity(path, starting_capital=10_000), 10_200)

    def test_small_service_mirrors_primary_scan_without_rescanning(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            source = Path(tempdir) / "primary.sqlite3"
            small = Path(tempdir) / "small.sqlite3"
            now = datetime(2026, 8, 10, 12, 30, tzinfo=timezone.utc)
            source_run_id = save_scan_result(_result(now), {}, source)
            config = replace(
                TradingConfig.from_env(mode="paper"),
                db_path=small,
                scan_source_db_path=source,
                strategy_name="small-10k",
            )
            service = AutomationService(config, log=lambda _: None)

            mirrored_run_id = service.ensure_weekly_scan(now)

            self.assertIsNotNone(mirrored_run_id)
            self.assertNotEqual(mirrored_run_id, source_run_id)
            self.assertEqual(latest_scan_run_id(small), mirrored_run_id)
            self.assertEqual(len(list_trade_plans(small)), 1)
            self.assertIsNone(service.ensure_weekly_scan(now))


def _result(now: datetime) -> PipelineResult:
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
        earnings_date=date(2026, 8, 11),
        days_until_earnings=1,
        contrarian_watch="bearish_watch",
        options_expiration=date(2026, 8, 21),
    )
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
                "earnings_date": "2026-08-11",
                "timing": "AMC",
                "options_expiration": "2026-08-21",
            },
        ),
        market_signal_rows=(
            {
                "ticker": "AMD",
                "recommendation": "bearish_watch",
                "recommendation_direction": "bearish",
                "recommendation_score": 70,
            },
        ),
    )


if __name__ == "__main__":
    unittest.main()
