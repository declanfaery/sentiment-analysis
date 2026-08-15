from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from sentiment_analysis.pipeline import PipelineResult
from sentiment_analysis.trading.broker import OptionsBroker, choose_option_contract
from sentiment_analysis.trading.models import OptionContract, ShadowPosition, TradePlan
from sentiment_analysis.trading.research_storage import (
    list_due_entry_group,
    list_shadow_positions,
    mark_shadow_completed,
    observation_qualities,
    record_shadow_entry,
    record_shadow_failure,
    save_feature_snapshot,
    save_missing_observation,
    save_shadow_observation,
    save_shadow_positions,
)


@dataclass(frozen=True)
class ResearchOptions:
    enabled: bool = False
    max_tickers: int = 8
    entry_minutes_before_close: tuple[int, ...] = (60, 30, 15)
    target_deltas: tuple[float, ...] = (0.50, 0.65, 0.80)
    expiration_days_after_exit: tuple[int, ...] = (7, 14, 21, 28)
    observation_minutes_after_open: tuple[int, ...] = (0, 5, 15, 30)
    entry_grace_minutes: int = 12
    observation_grace_minutes: int = 4
    max_quote_age_seconds: int = 180
    max_spread_pct: float = 0.30
    max_otm_pct: float = 0.05
    min_open_interest: int = 0


class ShadowResearchCollector:
    def __init__(
        self,
        *,
        db_path: Path,
        broker: OptionsBroker,
        options: ResearchOptions,
        planned_entry_minutes_before_close: int,
        planned_exit_minutes_after_open: int,
        log=print,
    ) -> None:
        self.db_path = db_path
        self.broker = broker
        self.options = options
        self.planned_entry_minutes_before_close = planned_entry_minutes_before_close
        self.planned_exit_minutes_after_open = planned_exit_minutes_after_open
        self.log = log

    def prepare_run(
        self,
        result: PipelineResult,
        plans: list[TradePlan],
        *,
        scan_run_id: str,
        captured_at: datetime,
    ) -> tuple[int, int]:
        if not self.options.enabled:
            return 0, 0
        plan_by_ticker = {plan.ticker: plan for plan in plans}
        candidate_by_ticker = {
            str(row.get("ticker")): row for row in result.candidate_rows
        }
        market_by_ticker = {
            str(row.get("ticker")): row for row in result.market_signal_rows
        }
        feature_count = 0
        positions: list[ShadowPosition] = []
        for signal in result.signals[: self.options.max_tickers]:
            plan = plan_by_ticker.get(signal.ticker)
            if plan is None:
                continue
            features = {
                "scan_started_at": result.started_at.isoformat(),
                "captured_at": captured_at.isoformat(),
                "signal": signal.as_dict(),
                "candidate": dict(candidate_by_ticker.get(signal.ticker, {})),
                "market": dict(market_by_ticker.get(signal.ticker, {})),
                "trade_plan": plan.as_dict(),
            }
            feature_count += int(
                save_feature_snapshot(
                    self.db_path,
                    scan_run_id=scan_run_id,
                    plan_id=plan.id,
                    ticker=signal.ticker,
                    captured_at=captured_at,
                    features=features,
                )
            )
            positions.extend(self._positions_for_plan(plan, captured_at))
        inserted = save_shadow_positions(self.db_path, positions)
        self.log(
            f"Shadow research froze {feature_count} feature snapshot(s) and "
            f"scheduled {inserted} virtual position(s)."
        )
        return feature_count, inserted

    def run_once(self, now: datetime | None = None) -> None:
        if not self.options.enabled:
            return
        moment = _aware_utc(now or datetime.now(timezone.utc))
        try:
            if self._collect_observation(moment):
                return
            self._collect_entry(moment)
        except Exception as exc:
            self.log(f"Shadow research collection will retry: {exc}")

    def _positions_for_plan(
        self,
        plan: TradePlan,
        captured_at: datetime,
    ) -> list[ShadowPosition]:
        if plan.earnings_timing == "UNKNOWN":
            return []
        close_at = plan.entry_at + timedelta(
            minutes=self.planned_entry_minutes_before_close
        )
        market_open_at = plan.exit_at - timedelta(
            minutes=self.planned_exit_minutes_after_open
        )
        positions: list[ShadowPosition] = []
        for entry_minutes in self.options.entry_minutes_before_close:
            scheduled_entry = close_at - timedelta(minutes=entry_minutes)
            initial_status = (
                "missed_entry"
                if scheduled_entry
                < captured_at - timedelta(minutes=self.options.entry_grace_minutes)
                else "scheduled"
            )
            for option_type in ("CALL", "PUT"):
                for expiration_days in self.options.expiration_days_after_exit:
                    expiration_target = _next_friday(
                        market_open_at.date() + timedelta(days=expiration_days)
                    )
                    for target_delta in self.options.target_deltas:
                        identity = (
                            f"{plan.id}:{option_type}:{target_delta:.4f}:"
                            f"{expiration_days}:{entry_minutes}"
                        )
                        position_id = hashlib.sha256(
                            identity.encode("ascii")
                        ).hexdigest()[:32]
                        positions.append(
                            ShadowPosition(
                                id=position_id,
                                scan_run_id=plan.scan_run_id,
                                plan_id=plan.id,
                                ticker=plan.ticker,
                                earnings_date=plan.earnings_date,
                                earnings_timing=plan.earnings_timing,
                                option_type=option_type,
                                target_delta=target_delta,
                                target_expiration_days=expiration_days,
                                entry_minutes_before_close=entry_minutes,
                                scheduled_entry_at=scheduled_entry,
                                market_open_at=market_open_at,
                                expiration_target=expiration_target,
                                status=initial_status,
                                last_error=(
                                    "shadow entry time passed before research was scheduled"
                                    if initial_status == "missed_entry"
                                    else None
                                ),
                                created_at=captured_at,
                                updated_at=captured_at,
                            )
                        )
        return positions

    def _collect_entry(self, now: datetime) -> bool:
        positions = list_due_entry_group(
            self.db_path,
            now=now,
            grace_minutes=self.options.entry_grace_minutes,
        )
        if not positions:
            return False
        expirations = sorted({position.expiration_target for position in positions})
        universe = self.broker.option_universe(
            positions[0].ticker,
            expirations=expirations,
            max_otm_pct=self.options.max_otm_pct,
        )
        entered = 0
        rejected = 0
        for position in positions:
            candidates = [
                contract
                for contract in universe
                if contract.option_type == position.option_type
                and contract.expiration == position.expiration_target
            ]
            contract = choose_option_contract(
                candidates,
                max_contract_cost=10_000_000,
                max_spread_pct=self.options.max_spread_pct,
                target_itm_pct=_target_itm(position.target_delta),
                target_delta=position.target_delta,
                max_otm_pct=self.options.max_otm_pct,
                max_breakeven_move_pct=1.0,
                min_volume=0,
                min_open_interest=self.options.min_open_interest,
            )
            if contract is None:
                record_shadow_failure(
                    self.db_path,
                    position.id,
                    status="no_contract",
                    message="no executable contract matched the shadow policy",
                    observed_at=now,
                )
                rejected += 1
                continue
            quality = quote_quality(
                contract,
                now,
                max_age_seconds=self.options.max_quote_age_seconds,
            )
            record_shadow_entry(
                self.db_path,
                position.id,
                contract=contract,
                observed_at=now,
                quote_quality=quality,
            )
            if quality == "stale":
                rejected += 1
            else:
                entered += 1
        self.log(
            f"Shadow entry captured {positions[0].ticker}: {entered} virtual "
            f"position(s), {rejected} rejected or invalid."
        )
        return True

    def _collect_observation(self, now: datetime) -> bool:
        positions = list_shadow_positions(
            self.db_path,
            statuses=("entered",),
        )
        if not positions:
            return False
        existing = observation_qualities(self.db_path)
        due: list[tuple[datetime, int, ShadowPosition]] = []
        checkpoints = self.options.observation_minutes_after_open
        final_checkpoint = max(checkpoints)
        grace = timedelta(minutes=self.options.observation_grace_minutes)
        for position in positions:
            for checkpoint in checkpoints:
                scheduled_at = position.market_open_at + timedelta(minutes=checkpoint)
                current_quality = existing.get((position.id, checkpoint))
                if current_quality in {"fresh", "unknown"} or now < scheduled_at:
                    continue
                if now > scheduled_at + grace:
                    if current_quality is None:
                        save_missing_observation(
                            self.db_path,
                            position_id=position.id,
                            checkpoint_minutes=checkpoint,
                            scheduled_at=scheduled_at,
                            observed_at=now,
                        )
                    if checkpoint == final_checkpoint:
                        mark_shadow_completed(self.db_path, position.id, now=now)
                    continue
                due.append((scheduled_at, checkpoint, position))
        if not due:
            return False
        due.sort(key=lambda item: (item[0], item[2].ticker, item[1]))
        first_schedule, first_checkpoint, first_position = due[0]
        group = [
            position
            for scheduled, checkpoint, position in due
            if scheduled == first_schedule
            and checkpoint == first_checkpoint
            and position.ticker == first_position.ticker
        ]
        contracts_by_symbol = {
            position.contract_symbol: position.contract()
            for position in group
            if position.contract_symbol
        }
        quotes = self.broker.option_quotes(list(contracts_by_symbol.values()))
        quote_by_symbol = {quote.symbol: quote for quote in quotes}
        saved = 0
        for position in group:
            quote = quote_by_symbol.get(position.contract_symbol or "")
            if quote is None:
                continue
            quality = quote_quality(
                quote,
                now,
                max_age_seconds=self.options.max_quote_age_seconds,
            )
            save_shadow_observation(
                self.db_path,
                position=position,
                checkpoint_minutes=first_checkpoint,
                scheduled_at=first_schedule,
                observed_at=now,
                quote=quote,
                quote_quality=quality,
            )
            if first_checkpoint == final_checkpoint and quality in {"fresh", "unknown"}:
                mark_shadow_completed(self.db_path, position.id, now=now)
            saved += 1
        self.log(
            f"Shadow checkpoint +{first_checkpoint}m captured "
            f"{first_position.ticker}: {saved} quote(s)."
        )
        return True


def quote_quality(
    contract: OptionContract,
    observed_at: datetime,
    *,
    max_age_seconds: int,
) -> str:
    if contract.quote_time_ms is None:
        return "unknown"
    quote_time = datetime.fromtimestamp(
        contract.quote_time_ms / 1000,
        timezone.utc,
    )
    age = _aware_utc(observed_at) - quote_time
    if timedelta(seconds=-30) <= age <= timedelta(seconds=max_age_seconds):
        return "fresh"
    return "stale"


def _target_itm(target_delta: float) -> float:
    if target_delta >= 0.75:
        return 0.10
    if target_delta >= 0.60:
        return 0.05
    return 0.0


def _next_friday(value: date) -> date:
    return value + timedelta(days=(4 - value.weekday()) % 7)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
