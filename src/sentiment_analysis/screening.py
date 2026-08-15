from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable

from sentiment_analysis.models import EarningsEvent


LOW_QUALITY_NAME_RE = re.compile(
    r"\b(acquisition|blank check|right|rights|unit|units|warrant|warrants|preferred|note|notes)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CandidateScreen:
    min_expected_move_pct: float = 5.0
    require_expected_move: bool = True
    max_candidates: int = 30
    skip_low_quality_names: bool = True


def screen_earnings_candidates(
    events: Iterable[EarningsEvent],
    *,
    screen: CandidateScreen,
    today: date | None = None,
) -> list[EarningsEvent]:
    current = today or date.today()
    candidates = [
        event
        for event in events
        if is_worth_counting(event, screen=screen)
    ]
    candidates.sort(
        key=lambda event: candidate_quality_score(event, today=current),
        reverse=True,
    )
    return candidates[: max(0, screen.max_candidates)]


def is_worth_counting(event: EarningsEvent, *, screen: CandidateScreen) -> bool:
    if event.earnings_date is None:
        return False
    if screen.skip_low_quality_names and event.company and LOW_QUALITY_NAME_RE.search(event.company):
        return False
    if event.expected_move_pct is None:
        return not screen.require_expected_move
    if event.expected_move_pct < screen.min_expected_move_pct:
        return False
    return True


def candidate_quality_score(event: EarningsEvent, *, today: date | None = None) -> float:
    current = today or date.today()
    days_until = (event.earnings_date - current).days if event.earnings_date else 99
    expected_move_score = event.expected_move_pct or 0.0
    timing_bonus = 0.2 if event.timing in {"BMO", "AMC"} else 0.0
    proximity_bonus = max(0.0, 6.0 - max(days_until, 0)) * 0.25
    return expected_move_score + timing_bonus + proximity_bonus
