from __future__ import annotations

import re

from sentiment_analysis.models import SentimentScore


FINANCE_LEXICON = {
    "beat": 1.8,
    "beats": 1.8,
    "blowout": 2.4,
    "breakout": 1.8,
    "bullish": 1.9,
    "calls": 0.7,
    "crush": 2.0,
    "dump": -2.0,
    "euphoric": 1.5,
    "guidance": 0.1,
    "miss": -1.9,
    "misses": -1.9,
    "moon": 1.7,
    "printing": 1.3,
    "puts": -0.7,
    "rally": 1.5,
    "rug": -2.1,
    "scary": -1.4,
    "squeeze": 1.3,
    "weak": -1.4,
}


class VaderSentimentAnalyzer:
    def __init__(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError:
            self._analyzer = None
        else:
            self._analyzer = SentimentIntensityAnalyzer()
            self._analyzer.lexicon.update(FINANCE_LEXICON)

    def score(self, text: str) -> SentimentScore:
        if self._analyzer is None:
            return _fallback_score(text)

        raw = self._analyzer.polarity_scores(text)
        compound = float(raw["compound"])
        return SentimentScore(
            compound=compound,
            positive=float(raw["pos"]),
            neutral=float(raw["neu"]),
            negative=float(raw["neg"]),
            label=_label(compound),
        )


def _fallback_score(text: str) -> SentimentScore:
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return SentimentScore(compound=0.0, positive=0.0, neutral=1.0, negative=0.0, label="neutral")

    raw_score = sum(FINANCE_LEXICON.get(token, 0.0) for token in tokens)
    compound = max(-1.0, min(1.0, raw_score / 6.0))
    positive = max(0.0, compound)
    negative = max(0.0, -compound)
    neutral = max(0.0, 1.0 - positive - negative)
    return SentimentScore(
        compound=compound,
        positive=positive,
        neutral=neutral,
        negative=negative,
        label=_label(compound),
    )


def _label(compound: float) -> str:
    if compound >= 0.15:
        return "positive"
    if compound <= -0.15:
        return "negative"
    return "neutral"

