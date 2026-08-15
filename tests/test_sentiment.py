from __future__ import annotations

import unittest

from sentiment_analysis.sentiment.vader import VaderSentimentAnalyzer


class SentimentTests(unittest.TestCase):
    def test_scores_finance_language(self) -> None:
        analyzer = VaderSentimentAnalyzer()

        bullish = analyzer.score("$NVDA blowout beat incoming")
        bearish = analyzer.score("$TSLA miss setup looks weak")

        self.assertGreater(bullish.compound, 0)
        self.assertLess(bearish.compound, 0)


if __name__ == "__main__":
    unittest.main()

