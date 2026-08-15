from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentiment_analysis.config import load_env_file
from sentiment_analysis.trading.executor import ExecutionOptions
from sentiment_analysis.trading.planner import PlannerOptions
from sentiment_analysis.trading.research import ResearchOptions


@dataclass(frozen=True)
class TradingConfig:
    mode: str
    db_path: Path
    scan_source_db_path: Path | None
    scan_options_path: Path
    strategy_name: str
    scan_hour_et: int
    scan_minute_et: int
    scan_retry_seconds: int
    poll_seconds: int
    paper_cash_balance: float
    webull_app_key: str | None
    webull_app_secret: str | None
    webull_account_id: str | None
    live_confirmation: str | None
    planner: PlannerOptions
    execution: ExecutionOptions
    research: ResearchOptions

    @classmethod
    def from_env(cls, *, mode: str | None = None) -> "TradingConfig":
        load_env_file()
        selected_mode = (mode or os.getenv("TRADING_MODE", "paper")).strip().lower()
        if selected_mode not in {"paper", "sandbox", "live"}:
            raise ValueError("TRADING_MODE must be paper, sandbox, or live")
        credential_prefix = f"WEBULL_{selected_mode.upper()}" if selected_mode != "paper" else "WEBULL"

        def credential(name: str) -> str | None:
            return os.getenv(f"{credential_prefix}_{name}") or os.getenv(f"WEBULL_{name}") or None

        source_path = os.getenv("AUTOMATION_SCAN_SOURCE_DB")

        return cls(
            mode=selected_mode,
            db_path=Path(os.getenv("TRADING_DB_PATH", "data/scans.sqlite3")),
            scan_source_db_path=Path(source_path) if source_path else None,
            scan_options_path=Path(
                os.getenv("AUTOMATION_SCAN_OPTIONS", "data/automation_scan_options.json")
            ),
            strategy_name=os.getenv("STRATEGY_NAME", "main").strip() or "main",
            scan_hour_et=_env_int("WEEKLY_SCAN_HOUR_ET", 8),
            scan_minute_et=_env_int("WEEKLY_SCAN_MINUTE_ET", 0),
            scan_retry_seconds=max(300, _env_int("WEEKLY_SCAN_RETRY_SECONDS", 3600)),
            poll_seconds=max(15, _env_int("TRADING_POLL_SECONDS", 30)),
            paper_cash_balance=_env_float("PAPER_CASH_BALANCE", 100_000.0),
            webull_app_key=credential("APP_KEY"),
            webull_app_secret=credential("APP_SECRET"),
            webull_account_id=credential("ACCOUNT_ID"),
            live_confirmation=os.getenv("ENABLE_LIVE_TRADING") or None,
            planner=PlannerOptions(
                base_allocation_pct=_bounded_pct("BASE_ALLOCATION_PCT", 0.05),
                confirmed_allocation_pct=_bounded_pct("CONFIRMED_ALLOCATION_PCT", 0.07),
                min_recommendation_score=_env_int("MIN_TRADE_SCORE", 25),
                require_corroborating_signal=_env_bool(
                    "REQUIRE_TRADE_CORROBORATION", False
                ),
                exit_delay_minutes=max(0, _env_int("EXIT_DELAY_MINUTES", 15)),
                entry_before_close_minutes=max(
                    1, _env_int("ENTRY_BEFORE_CLOSE_MINUTES", 15)
                ),
            ),
            execution=ExecutionOptions(
                max_contracts_per_trade=max(1, _env_int("MAX_CONTRACTS_PER_TRADE", 1)),
                max_total_exposure_pct=_bounded_pct("MAX_TOTAL_EXPOSURE_PCT", 0.20),
                max_option_spread_pct=_bounded_pct("MAX_OPTION_SPREAD_PCT", 0.12),
                entry_window_minutes=max(1, _env_int("ENTRY_WINDOW_MINUTES", 14)),
                reprice_after_seconds=max(15, _env_int("REPRICE_AFTER_SECONDS", 60)),
                max_entry_chase_pct=_bounded_pct("MAX_ENTRY_CHASE_PCT", 0.03),
                entry_chase_step_pct=_bounded_pct("ENTRY_CHASE_STEP_PCT", 0.01),
                option_target_itm_pct=_bounded_pct("OPTION_TARGET_ITM_PCT", 0.10),
                option_target_delta=_bounded_pct("OPTION_TARGET_DELTA", 0.80),
                option_max_otm_pct=_nonnegative_pct("OPTION_MAX_OTM_PCT", 0.0),
                max_option_breakeven_pct=_bounded_pct(
                    "MAX_OPTION_BREAKEVEN_PCT", 0.025
                ),
                affordable_option_fallback=_env_bool(
                    "AFFORDABLE_OPTION_FALLBACK", False
                ),
                affordable_fallback_max_otm_pct=_nonnegative_pct(
                    "AFFORDABLE_FALLBACK_MAX_OTM_PCT", 0.03
                ),
                min_option_volume=max(0, _env_int("MIN_OPTION_VOLUME", 0)),
                min_option_open_interest=max(
                    0, _env_int("MIN_OPTION_OPEN_INTEREST", 100)
                ),
                min_expiration_days_after_exit=max(
                    0, _env_int("MIN_EXPIRATION_DAYS_AFTER_EXIT", 14)
                ),
                max_expiration_days_after_exit=max(
                    7, _env_int("MAX_EXPIRATION_DAYS_AFTER_EXIT", 28)
                ),
                early_exit_monitor_minutes=max(
                    0, _env_int("EARLY_EXIT_MONITOR_MINUTES", 15)
                ),
                early_exit_profit_pct=_bounded_pct(
                    "EARLY_EXIT_PROFIT_PCT", 0.05
                ),
                exit_chase_step_pct=_bounded_pct(
                    "EXIT_CHASE_STEP_PCT", 0.10
                ),
                strategy_starting_capital=_optional_positive_float(
                    "STRATEGY_STARTING_CAPITAL"
                ),
            ),
            research=ResearchOptions(
                enabled=_env_bool("SHADOW_RESEARCH_ENABLED", False),
                max_tickers=max(1, _env_int("SHADOW_MAX_TICKERS", 8)),
                entry_minutes_before_close=_env_int_tuple(
                    "SHADOW_ENTRY_MINUTES_BEFORE_CLOSE",
                    (60, 30, 15),
                ),
                target_deltas=_env_float_tuple(
                    "SHADOW_TARGET_DELTAS",
                    (0.50, 0.65, 0.80),
                ),
                expiration_days_after_exit=_env_int_tuple(
                    "SHADOW_EXPIRATION_DAYS_AFTER_EXIT",
                    (7, 14, 21, 28),
                ),
                observation_minutes_after_open=_env_nonnegative_int_tuple(
                    "SHADOW_OBSERVATION_MINUTES_AFTER_OPEN",
                    (0, 5, 15, 30),
                ),
                entry_grace_minutes=max(
                    1,
                    _env_int("SHADOW_ENTRY_GRACE_MINUTES", 12),
                ),
                observation_grace_minutes=max(
                    1,
                    _env_int("SHADOW_OBSERVATION_GRACE_MINUTES", 4),
                ),
                max_quote_age_seconds=max(
                    30,
                    _env_int("SHADOW_MAX_QUOTE_AGE_SECONDS", 180),
                ),
                max_spread_pct=_bounded_pct("SHADOW_MAX_SPREAD_PCT", 0.30),
                max_otm_pct=_nonnegative_pct("SHADOW_MAX_OTM_PCT", 0.05),
                min_open_interest=max(
                    0,
                    _env_int("SHADOW_MIN_OPEN_INTEREST", 0),
                ),
            ),
        )

    @property
    def live_enabled(self) -> bool:
        return self.live_confirmation == "I_UNDERSTAND_THE_RISK"

    def scan_options(self) -> dict[str, Any]:
        if not self.scan_options_path.exists():
            return {}
        payload = json.loads(self.scan_options_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{self.scan_options_path} must contain a JSON object")
        return payload


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _bounded_pct(name: str, default: float) -> float:
    value = _env_float(name, default)
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be greater than 0 and at most 1")
    return value


def _nonnegative_pct(name: str, default: float) -> float:
    value = _env_float(name, default)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be at least 0 and at most 1")
    return value


def _optional_positive_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    value = _env_float(name, 0.0)
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def _env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    values = _parse_tuple(name, default, int)
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive integers")
    return tuple(dict.fromkeys(values))


def _env_nonnegative_int_tuple(
    name: str,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    values = _parse_tuple(name, default, int)
    if any(value < 0 for value in values):
        raise ValueError(f"{name} values must be nonnegative integers")
    return tuple(sorted(set(values)))


def _env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    values = _parse_tuple(name, default, float)
    if any(not 0 < value <= 1 for value in values):
        raise ValueError(f"{name} values must be greater than 0 and at most 1")
    return tuple(sorted(set(values)))


def _parse_tuple(name: str, default: tuple, parser):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        values = tuple(parser(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated numeric list") from exc
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values
