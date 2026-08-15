from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path

from sentiment_analysis.pipeline import PipelineOptions, SentimentPipeline
from sentiment_analysis.providers.csv_posts import CsvPostProvider
from sentiment_analysis.providers.earnings import CsvEarningsProvider
from sentiment_analysis.sentiment.vader import VaderSentimentAnalyzer


class PipelineTests(unittest.TestCase):
    def test_pipeline_filters_by_mentions_and_adds_earnings(self) -> None:
        pipeline = SentimentPipeline(
            post_provider=CsvPostProvider(Path("examples/sample_posts.csv")),
            earnings_provider=CsvEarningsProvider(
                Path("examples/sample_earnings.csv"),
                today=date(2026, 7, 30),
            ),
            sentiment_analyzer=VaderSentimentAnalyzer(),
            options=PipelineOptions(
                ticker_universe={"TSLA", "NVDA", "GME", "SOFI"},
                min_mentions=2,
                earnings_horizon_days=60,
                hours_back=24 * 365,
            ),
        )

        signals = pipeline.run()
        tickers = {signal.ticker for signal in signals}

        self.assertIn("TSLA", tickers)
        self.assertIn("NVDA", tickers)
        self.assertIn("GME", tickers)
        self.assertNotIn("SOFI", tickers)
        self.assertTrue(all(signal.earnings_date is not None for signal in signals))


if __name__ == "__main__":
    unittest.main()
