from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sentiment_analysis.dashboard import DEFAULT_OPTIONS, run_dashboard_scan
from sentiment_analysis.storage import load_scan_result, save_scan_result
from sentiment_analysis.trading.broker import OptionsBroker
from sentiment_analysis.trading.calendar import week_key, weekly_scan_due
from sentiment_analysis.trading.config import TradingConfig
from sentiment_analysis.trading.executor import TradeExecutor
from sentiment_analysis.trading.paper import PaperOptionsBroker
from sentiment_analysis.trading.planner import build_trade_plans
from sentiment_analysis.trading.research import ShadowResearchCollector
from sentiment_analysis.trading.storage import (
    get_automation_state,
    latest_scan_run_id,
    save_trade_plans,
    set_automation_state,
)
from sentiment_analysis.trading.webull import WebullOptionsBroker


WEEKLY_SCAN_STATE_KEY = "weekly_scan_week"
SANDBOX_SMOKE_STATE_KEY = "webull_sandbox_smoke_passed"


class AutomationService:
    def __init__(self, config: TradingConfig, *, broker: OptionsBroker | None = None, log=print) -> None:
        self.config = config
        self.log = log
        self._scan_retry_after: datetime | None = None
        self._sandbox_gate_logged = False
        self.broker = broker or build_broker(config)
        self.executor = TradeExecutor(
            db_path=config.db_path,
            broker=self.broker,
            options=config.execution,
            log=log,
        )
        self.research = ShadowResearchCollector(
            db_path=config.db_path,
            broker=self.broker,
            options=config.research,
            planned_entry_minutes_before_close=(
                config.planner.entry_before_close_minutes
            ),
            planned_exit_minutes_after_open=config.planner.exit_delay_minutes,
            log=log,
        )

    def run_once(self, now: datetime | None = None, *, skip_scan: bool = False) -> None:
        moment = _aware_utc(now or datetime.now(timezone.utc))
        scan_error: Exception | None = None
        if not skip_scan and (self._scan_retry_after is None or moment >= self._scan_retry_after):
            try:
                self.ensure_weekly_scan(moment)
                self._scan_retry_after = None
            except Exception as exc:
                scan_error = exc
                self._scan_retry_after = moment + timedelta(seconds=self.config.scan_retry_seconds)
                self.log(
                    f"Weekly scan failed; next attempt after "
                    f"{self._scan_retry_after.isoformat()}: {exc}"
                )
        sandbox_ready = (
            self.config.mode != "sandbox"
            or bool(get_automation_state(self.config.db_path, SANDBOX_SMOKE_STATE_KEY))
        )
        if sandbox_ready:
            self._sandbox_gate_logged = False
            self.executor.run_once(moment)
        elif not self._sandbox_gate_logged:
            self.log("Sandbox strategy execution is waiting for the one-time Webull smoke test.")
            self._sandbox_gate_logged = True
        self.research.run_once(moment)
        if scan_error is not None:
            raise scan_error

    def ensure_weekly_scan(self, now: datetime) -> str | None:
        if self.config.scan_source_db_path is not None:
            return self._ensure_mirrored_weekly_scan(now)

        completed_week = get_automation_state(self.config.db_path, WEEKLY_SCAN_STATE_KEY)
        if not weekly_scan_due(
            now,
            completed_week_key=completed_week,
            hour=self.config.scan_hour_et,
            minute=self.config.scan_minute_et,
        ):
            return None

        current_week = week_key(now)
        run_id = latest_scan_run_id(self.config.db_path)
        if run_id:
            saved_result, _ = load_scan_result(self.config.db_path, run_id)
            if week_key(saved_result.started_at) == current_week:
                self._plan_saved_run(run_id, saved_result, now)
                set_automation_state(self.config.db_path, WEEKLY_SCAN_STATE_KEY, current_week)
                self.log(f"Using existing scan {run_id} for week {current_week}.")
                return run_id

        scan_options = {**DEFAULT_OPTIONS, **self.config.scan_options()}
        scan_options["sqlite_path"] = str(self.config.db_path)
        self.log(f"Starting weekly sentiment scan for week {current_week}.")
        result = run_dashboard_scan(scan_options, self.log)
        run_id = latest_scan_run_id(self.config.db_path)
        if not run_id:
            raise RuntimeError("weekly scan completed without a saved run ID")
        self._plan_saved_run(run_id, result, now)
        set_automation_state(self.config.db_path, WEEKLY_SCAN_STATE_KEY, current_week)
        return run_id

    def _ensure_mirrored_weekly_scan(self, now: datetime) -> str | None:
        completed_week = get_automation_state(self.config.db_path, WEEKLY_SCAN_STATE_KEY)
        if not weekly_scan_due(
            now,
            completed_week_key=completed_week,
            hour=self.config.scan_hour_et,
            minute=self.config.scan_minute_et,
        ):
            return None

        source_path = self.config.scan_source_db_path
        if source_path is None or source_path.resolve() == self.config.db_path.resolve():
            raise ValueError("AUTOMATION_SCAN_SOURCE_DB must differ from TRADING_DB_PATH")
        source_run_id = latest_scan_run_id(source_path)
        if not source_run_id:
            self.log(f"{self.config.strategy_name}: waiting for the primary weekly scan.")
            return None
        result, saved_options = load_scan_result(source_path, source_run_id)
        current_week = week_key(now)
        if week_key(result.started_at) != current_week:
            self.log(f"{self.config.strategy_name}: primary scan for {current_week} is not ready yet.")
            return None

        mirrored_run_id = save_scan_result(result, saved_options, self.config.db_path)
        self._plan_saved_run(mirrored_run_id, result, now)
        set_automation_state(self.config.db_path, WEEKLY_SCAN_STATE_KEY, current_week)
        self.log(
            f"{self.config.strategy_name}: mirrored primary scan {source_run_id} "
            f"as {mirrored_run_id}; no additional X requests were made."
        )
        return mirrored_run_id

    def plan_run(self, run_id: str, *, generated_at: datetime | None = None) -> int:
        result, _ = load_scan_result(self.config.db_path, run_id)
        return self._plan_saved_run(run_id, result, generated_at or datetime.now(timezone.utc))

    def _plan_saved_run(self, run_id: str, result, generated_at: datetime) -> int:
        plans = build_trade_plans(
            result,
            run_id,
            options=self.config.planner,
            generated_at=generated_at,
        )
        inserted = save_trade_plans(self.config.db_path, plans)
        self.research.prepare_run(
            result,
            plans,
            scan_run_id=run_id,
            captured_at=generated_at,
        )
        eligible = sum(plan.status == "planned" for plan in plans)
        self.log(
            f"Built {len(plans)} trade plan(s), {eligible} eligible; "
            f"inserted {inserted} new row(s)."
        )
        return inserted

    def daemon(self) -> None:
        self.log(
            f"Automation '{self.config.strategy_name}' running in {self.config.mode} mode; "
            f"polling every {self.config.poll_seconds}s."
        )
        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self.log(f"Automation iteration failed: {exc}")
            time.sleep(self.config.poll_seconds)


def build_broker(config: TradingConfig) -> OptionsBroker:
    if config.mode == "paper":
        return PaperOptionsBroker(starting_cash=config.paper_cash_balance)
    return WebullOptionsBroker(
        app_key=config.webull_app_key or "",
        app_secret=config.webull_app_secret or "",
        account_id=config.webull_account_id or "",
        environment=config.mode,
        allow_live=config.live_enabled,
    )


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
