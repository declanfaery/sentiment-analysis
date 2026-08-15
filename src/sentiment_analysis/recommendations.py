from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResearchRecommendation:
    recommendation: str
    recommendation_score: int
    recommendation_direction: str
    recommendation_notes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "recommendation": self.recommendation,
            "recommendation_score": self.recommendation_score,
            "recommendation_direction": self.recommendation_direction,
            "recommendation_notes": list(self.recommendation_notes),
        }


def build_research_recommendation(
    *,
    crowd_bias: str,
    days_until_earnings: int | None,
    option_alert: str | None,
    option_bias: str | None,
    insider_recent_count: int | None,
    political_alert: str | None = None,
    political_bias: str | None = None,
    political_strength: int | None = None,
    political_buyers: int | None = None,
    political_sellers: int | None = None,
    political_high_profile_count: int | None = None,
) -> ResearchRecommendation:
    contrarian_direction = _contrarian_direction(crowd_bias)
    if days_until_earnings is None:
        return _recommendation("skip_no_earnings_date", 0, "none", ["no earnings date"])
    if days_until_earnings < 0:
        return _recommendation("skip_earnings_passed", 0, "none", ["earnings already passed"])
    if days_until_earnings > 14:
        return _recommendation("skip_earnings_not_close", 10, "none", ["earnings not close"])
    political_direction = _political_direction(political_alert, political_bias)
    political_is_strong = political_alert == "strong" and political_direction in {"bullish", "bearish"}
    if contrarian_direction == "neutral" and not political_is_strong:
        return _recommendation("neutral_skip", 20, "none", ["X sentiment is neutral"])

    direction = contrarian_direction
    option_direction = _option_direction(option_alert, option_bias)
    insider_count = int(insider_recent_count or 0)
    political_net = int(political_strength or 0)
    political_buyer_count = int(political_buyers or 0)
    political_seller_count = int(political_sellers or 0)
    high_profile_count = int(political_high_profile_count or 0)
    notes = [f"X crowd is {crowd_bias}; contrarian direction is {contrarian_direction}"]
    score = 55
    tags: list[str] = []

    if political_is_strong:
        political_note = (
            f"strong disclosed political activity leans {political_direction} "
            f"({political_buyer_count} latest-buy filer(s), {political_seller_count} latest-sell filer(s), "
            f"signal strength {political_net:+d}, {high_profile_count} high-profile)"
        )
        if contrarian_direction == "neutral":
            direction = political_direction
            score = 65 + min(abs(political_net), 10)
            tags.append("politics_lead")
            notes.append(political_note)
        elif political_direction != contrarian_direction:
            direction = political_direction
            score = 65 + min(abs(political_net), 10)
            tags.append("politics_override")
            notes.append(f"{political_note}; overrides the X-only contrarian direction")
        else:
            score += 20
            tags.append("politics_confirm")
            notes.append(f"{political_note}; confirms the contrarian direction")
    elif political_direction in {"bullish", "bearish"}:
        if political_direction == direction:
            score += 10
            tags.append("politics_support")
            notes.append(f"disclosed political activity modestly supports {direction} (net {political_net:+d})")
        else:
            score -= 15
            tags.append("politics_conflict")
            notes.append(f"disclosed political activity leans {political_direction}, against {direction}")
    elif political_alert == "mixed":
        notes.append("disclosed political activity is mixed")
    else:
        notes.append("no directional political-disclosure signal")

    if option_direction == direction:
        score += 25
        tags.append("options_confirm")
        notes.append(f"unusual options lean {option_direction}")
    elif option_direction in {"bullish", "bearish"} and option_direction != direction:
        score -= 25
        tags.append("options_conflict")
        notes.append(f"unusual options lean {option_direction}, opposite the final setup")
    elif option_alert == "unusual":
        score -= 5
        tags.append("mixed_options")
        notes.append("unusual options are mixed")
    else:
        notes.append("no unusual directional options confirmation")

    if insider_count > 0:
        score += 5
        notes.append(f"{insider_count} recent insider filing(s); review Form 4 details")

    label = f"{direction}_watch"
    if tags:
        label = f"{label}_{'_'.join(tags)}"
    return _recommendation(label, _clamp(score, 0, 100), direction, notes)


def _recommendation(
    label: str,
    score: int,
    direction: str,
    notes: list[str],
) -> ResearchRecommendation:
    return ResearchRecommendation(
        recommendation=label,
        recommendation_score=score,
        recommendation_direction=direction,
        recommendation_notes=tuple(notes),
    )


def _contrarian_direction(crowd_bias: str) -> str:
    if crowd_bias == "bullish":
        return "bearish"
    if crowd_bias == "bearish":
        return "bullish"
    return "neutral"


def _option_direction(option_alert: str | None, option_bias: str | None) -> str:
    if option_alert != "unusual":
        return "none"
    if option_bias == "calls":
        return "bullish"
    if option_bias == "puts":
        return "bearish"
    return "mixed"


def _political_direction(political_alert: str | None, political_bias: str | None) -> str:
    if political_alert not in {"strong", "active"}:
        return "none"
    if political_bias in {"bullish", "bearish"}:
        return political_bias
    return "none"


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))
