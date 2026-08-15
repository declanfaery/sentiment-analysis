from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from sentiment_analysis.models import TickerSignal
from sentiment_analysis.pipeline import PipelineResult
from sentiment_analysis.storage import save_scan_result
from sentiment_analysis.trading.models import OptionContract, TradePlan
from sentiment_analysis.trading.research import ResearchOptions, ShadowResearchCollector
from sentiment_analysis.trading.research_storage import (
    list_shadow_positions,
    research_summary,
)
from sentiment_analysis.trading.storage import save_trade_plans


class ResearchBroker:
    def __init__(self) -> None:
        self.universe_calls = 0
        self.quote_calls = 0

    def option_universe(self, ticker, *, expirations, **kwargs):
        self.universe_calls += 1
        expiration = expirations[0]
        return [
            _contract("CALL", expiration, 100, 0.50, 5.0),
            _contract("CALL", expiration, 90, 0.80, 11.0),
            _contract("PUT", expiration, 100, -0.50, 5.0),
            _contract("PUT", expiration, 110, -0.80, 11.0),
        ]

    def option_quotes(self, contracts):
        self.quote_calls += 1
        return [
            OptionContract(
                **{
                    **contract.__dict__,
                    "bid": contract.ask * 1.20,
                    "ask": contract.ask * 1.22,
                    "underlying_price": 105.0,
                }
            )
            for contract in contracts
        ]


class ShadowResearchTests(unittest.TestCase):
    def test_collects_shadow_entries_and_exit_checkpoints_without_orders(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "research.sqlite3"
            scan_at = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
            entry_at = datetime(2026, 8, 18, 19, 45, tzinfo=timezone.utc)
            exit_at = datetime(2026, 8, 19, 13, 45, tzinfo=timezone.utc)
            result = _result(scan_at)
            scan_run_id = save_scan_result(result, {}, path)
            plan = _plan(scan_run_id, entry_at, exit_at)
            save_trade_plans(path, [plan])
            broker = ResearchBroker()
            collector = ShadowResearchCollector(
                db_path=path,
                broker=broker,
                options=ResearchOptions(
                    enabled=True,
                    entry_minutes_before_close=(15,),
                    target_deltas=(0.50, 0.80),
                    expiration_days_after_exit=(7,),
                    observation_minutes_after_open=(0, 15),
                ),
                planned_entry_minutes_before_close=15,
                planned_exit_minutes_after_open=15,
                log=lambda _: None,
            )

            first = collector.prepare_run(
                result,
                [plan],
                scan_run_id=scan_run_id,
                captured_at=scan_at,
            )
            duplicate = collector.prepare_run(
                result,
                [plan],
                scan_run_id=scan_run_id,
                captured_at=scan_at,
            )

            self.assertEqual(first, (1, 4))
            self.assertEqual(duplicate, (0, 0))

            collector.run_once(entry_at)
            entered = list_shadow_positions(path, statuses=("entered",))
            self.assertEqual(len(entered), 4)
            self.assertEqual(broker.universe_calls, 1)

            market_open = exit_at - timedelta(minutes=15)
            collector.run_once(market_open)
            collector.run_once(market_open + timedelta(minutes=15))

            summary = research_summary(path)
            self.assertEqual(summary["feature_snapshots"], 1)
            self.assertEqual(summary["shadow_positions"], 4)
            self.assertEqual(summary["observations"], 8)
            self.assertEqual(summary["statuses"], {"completed": 4})
            self.assertEqual(broker.quote_calls, 2)
            self.assertAlmostEqual(summary["recent"][0]["best_bid_return_pct"], 0.20)

    def test_stale_entry_quote_is_labeled_and_not_simulated(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "stale.sqlite3"
            scan_at = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
            entry_at = datetime(2026, 8, 18, 19, 45, tzinfo=timezone.utc)
            exit_at = datetime(2026, 8, 19, 13, 45, tzinfo=timezone.utc)
            result = _result(scan_at)
            scan_run_id = save_scan_result(result, {}, path)
            plan = _plan(scan_run_id, entry_at, exit_at)
            save_trade_plans(path, [plan])
            broker = ResearchBroker()
            old_time = int((entry_at - timedelta(days=1)).timestamp() * 1000)
            original = broker.option_universe

            def stale_universe(*args, **kwargs):
                return [
                    OptionContract(
                        **{**contract.__dict__, "quote_time_ms": old_time}
                    )
                    for contract in original(*args, **kwargs)
                ]

            broker.option_universe = stale_universe
            collector = ShadowResearchCollector(
                db_path=path,
                broker=broker,
                options=ResearchOptions(
                    enabled=True,
                    entry_minutes_before_close=(15,),
                    target_deltas=(0.50,),
                    expiration_days_after_exit=(7,),
                ),
                planned_entry_minutes_before_close=15,
                planned_exit_minutes_after_open=15,
                log=lambda _: None,
            )
            collector.prepare_run(
                result,
                [plan],
                scan_run_id=scan_run_id,
                captured_at=scan_at,
            )

            collector.run_once(entry_at)

            invalid = list_shadow_positions(
                path,
                statuses=("invalid_entry_quote",),
            )
            self.assertEqual(len(invalid), 2)
            self.assertTrue(all(row.entry_quote_quality == "stale" for row in invalid))


def _contract(
    option_type: str,
    expiration: date,
    strike: float,
    delta: float,
    ask: float,
) -> OptionContract:
    code = "C" if option_type == "CALL" else "P"
    return OptionContract(
        symbol=f"AMD{expiration:%y%m%d}{code}{int(strike * 1000):08d}",
        ticker="AMD",
        option_type=option_type,
        expiration=expiration,
        strike=strike,
        bid=ask - 0.10,
        ask=ask,
        underlying_price=100.0,
        volume=500,
        open_interest=1000,
        delta=delta,
        implied_volatility=0.80,
    )


def _plan(scan_run_id: str, entry_at: datetime, exit_at: datetime) -> TradePlan:
    return TradePlan(
        id="r" * 32,
        scan_run_id=scan_run_id,
        ticker="AMD",
        direction="bearish",
        option_type="PUT",
        earnings_date=date(2026, 8, 18),
        earnings_timing="AMC",
        entry_at=entry_at,
        exit_at=exit_at,
        allocation_pct=0.05,
        recommendation="bearish_watch_options_confirm",
        recommendation_score=80,
        created_at=entry_at - timedelta(days=1),
        updated_at=entry_at - timedelta(days=1),
    )


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
        earnings_date=date(2026, 8, 18),
        days_until_earnings=1,
        contrarian_watch="bearish_watch",
        expected_move_pct=9.0,
        options_expiration=date(2026, 8, 21),
    )
    return PipelineResult(
        signals=[signal],
        queries=("$AMD",),
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
                "earnings_date": "2026-08-18",
                "timing": "AMC",
                "expected_move_pct": 9.0,
            },
        ),
        market_signal_rows=(
            {
                "ticker": "AMD",
                "recommendation": "bearish_watch_options_confirm",
                "recommendation_direction": "bearish",
                "recommendation_score": 80,
            },
        ),
    )


if __name__ == "__main__":
    unittest.main()
