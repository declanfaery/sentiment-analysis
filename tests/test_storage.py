from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from sentiment_analysis.models import TickerSignal
from sentiment_analysis.pipeline import PipelineResult
from sentiment_analysis.storage import load_scan_result, recent_runs, save_scan_result


class StorageTests(unittest.TestCase):
    def test_saves_scan_run(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "scans.sqlite3"
            result = PipelineResult(
                signals=[
                    TickerSignal(
                        ticker="MSFT",
                        mention_count=3,
                        unique_authors=3,
                        total_engagement=12,
                        avg_sentiment=0.2,
                        sentiment_stddev=0.1,
                        positive_posts=2,
                        neutral_posts=1,
                        negative_posts=0,
                        traction_score=10.0,
                        crowd_bias="bullish",
                        earnings_date=None,
                        days_until_earnings=None,
                        contrarian_watch="no_earnings_date",
                        preflight_count=42,
                    )
                ],
                queries=("$MSFT lang:en -is:retweet",),
                started_at=datetime.now(timezone.utc),
                finished_at=datetime.now(timezone.utc),
                raw_posts=3,
                duplicate_posts=0,
                matched_posts=3,
                tickers_seen=1,
                high_traction_tickers=1,
                preflight_counts={"MSFT": 42},
                selected_tickers=("MSFT",),
                candidate_rows=(
                    {
                        "ticker": "MSFT",
                        "company": "Microsoft",
                        "earnings_date": None,
                        "expected_move_pct": 6.5,
                        "x_count": 42,
                    },
                ),
                market_signal_rows=(
                    {
                        "ticker": "MSFT",
                        "option_alert": "unusual",
                        "insider_recent_count": 1,
                    },
                ),
            )

            run_id = save_scan_result(result, {"X_BEARER_TOKEN": "secret"}, path)
            runs = recent_runs(path)

            self.assertEqual(len(run_id), 32)
            self.assertEqual(runs[0]["high_traction_tickers"], 1)
            self.assertEqual(runs[0]["selected_tickers"], ["MSFT"])

            loaded, options = load_scan_result(path, run_id)
            self.assertEqual(options, {})
            self.assertEqual(loaded.signals[0].ticker, "MSFT")
            self.assertEqual(loaded.preflight_counts, {"MSFT": 42})
            self.assertEqual(loaded.selected_tickers, ("MSFT",))
            self.assertEqual(loaded.candidate_rows[0]["ticker"], "MSFT")
            self.assertEqual(loaded.market_signal_rows[0]["option_alert"], "unusual")


if __name__ == "__main__":
    unittest.main()
