from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timezone

from sentiment_analysis.pipeline import PipelineResult
from sentiment_analysis.trading.calendar import earnings_trade_times, normalize_earnings_timing
from sentiment_analysis.trading.models import TradePlan


@dataclass(frozen=True)
class PlannerOptions:
    base_allocation_pct: float = 0.05
    confirmed_allocation_pct: float = 0.07
    min_recommendation_score: int = 25
    require_corroborating_signal: bool = False
    exit_delay_minutes: int = 15
    entry_before_close_minutes: int = 15


def build_trade_plans(
    result: PipelineResult,
    scan_run_id: str,
    *,
    options: PlannerOptions | None = None,
    generated_at: datetime | None = None,
) -> list[TradePlan]:
    settings = options or PlannerOptions()
    now = _aware_utc(generated_at or datetime.now(timezone.utc))
    candidates = {str(row.get("ticker")): row for row in result.candidate_rows}
    market_rows = {str(row.get("ticker")): row for row in result.market_signal_rows}
    plans: list[TradePlan] = []

    for signal in result.signals:
        candidate = candidates.get(signal.ticker, {})
        market = market_rows.get(signal.ticker, {})
        earnings_date = _parse_date(candidate.get("earnings_date")) or signal.earnings_date
        timing = normalize_earnings_timing(_optional_text(candidate.get("timing")))
        direction = str(market.get("recommendation_direction") or "none").lower()
        recommendation = str(market.get("recommendation") or signal.contrarian_watch)
        score = _as_int(market.get("recommendation_score"), 0)
        allocation = (
            settings.confirmed_allocation_pct
            if "options_confirm" in recommendation
            else settings.base_allocation_pct
        )
        plan_id = hashlib.sha256(f"{scan_run_id}:{signal.ticker}".encode("ascii")).hexdigest()[:32]
        status = "planned"
        reason = None

        if earnings_date is None:
            status, reason = "skipped", "missing earnings date"
        elif timing == "UNKNOWN":
            status, reason = "skipped", "earnings timing is unknown"
        elif direction not in {"bullish", "bearish"}:
            status, reason = "skipped", "recommendation has no trade direction"
        elif score < settings.min_recommendation_score:
            status, reason = "skipped", f"recommendation score {score} is below {settings.min_recommendation_score}"
        elif (
            settings.require_corroborating_signal
            and not _has_corroborating_signal(recommendation)
        ):
            status, reason = (
                "skipped",
                "recommendation lacks a corroborating options or political signal",
            )

        entry_at = now
        exit_at = now
        if earnings_date is not None and timing != "UNKNOWN":
            try:
                entry_at, exit_at = earnings_trade_times(
                    earnings_date,
                    timing,
                    exit_delay_minutes=settings.exit_delay_minutes,
                    entry_before_close_minutes=settings.entry_before_close_minutes,
                )
            except ValueError as exc:
                status, reason = "skipped", str(exc)
            else:
                if status == "planned" and entry_at <= now:
                    status, reason = "skipped", "entry time passed before the trade plan was created"

        plans.append(
            TradePlan(
                id=plan_id,
                scan_run_id=scan_run_id,
                ticker=signal.ticker,
                direction=direction,
                option_type="CALL" if direction == "bullish" else "PUT",
                earnings_date=earnings_date or now.date(),
                earnings_timing=timing,
                entry_at=entry_at,
                exit_at=exit_at,
                allocation_pct=allocation,
                recommendation=recommendation,
                recommendation_score=score,
                status=status,
                skip_reason=reason,
                option_expiration=_parse_date(candidate.get("options_expiration"))
                or signal.options_expiration,
                created_at=now,
                updated_at=now,
            )
        )
    return plans


def _parse_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if value:
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None
    return None


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    clean = str(value).strip()
    return clean or None


def _has_corroborating_signal(recommendation: str) -> bool:
    return any(
        tag in recommendation
        for tag in (
            "options_confirm",
            "politics_confirm",
            "politics_override",
            "politics_lead",
            "politics_support",
        )
    )


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
