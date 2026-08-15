from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from sentiment_analysis.trading.broker import BrokerError, OptionsBroker, option_limit_price
from sentiment_analysis.trading.calendar import NEW_YORK
from sentiment_analysis.trading.models import OptionContract, OrderUpdate, TradePlan
from sentiment_analysis.trading.storage import (
    list_trade_plans,
    open_exposure,
    record_trade_event,
    strategy_equity,
    update_trade_plan,
)


@dataclass(frozen=True)
class ExecutionOptions:
    max_contracts_per_trade: int = 1
    max_total_exposure_pct: float = 0.20
    max_option_spread_pct: float = 0.12
    entry_window_minutes: int = 14
    reprice_after_seconds: int = 60
    max_entry_chase_pct: float = 0.03
    entry_chase_step_pct: float = 0.01
    option_target_itm_pct: float = 0.10
    option_target_delta: float = 0.80
    option_max_otm_pct: float = 0.0
    max_option_breakeven_pct: float = 0.025
    affordable_option_fallback: bool = False
    affordable_fallback_max_otm_pct: float = 0.03
    min_option_volume: int = 0
    min_option_open_interest: int = 100
    min_expiration_days_after_exit: int = 14
    max_expiration_days_after_exit: int = 28
    early_exit_monitor_minutes: int = 15
    early_exit_profit_pct: float = 0.05
    exit_chase_step_pct: float = 0.10
    strategy_starting_capital: float | None = None


class TradeExecutor:
    def __init__(
        self,
        *,
        db_path: Path,
        broker: OptionsBroker,
        options: ExecutionOptions | None = None,
        log=print,
    ) -> None:
        self.db_path = db_path
        self.broker = broker
        self.options = options or ExecutionOptions()
        self.log = log

    def run_once(self, now: datetime | None = None) -> None:
        moment = _aware_utc(now or datetime.now(timezone.utc))
        plans = list_trade_plans(self.db_path, active_only=True)

        for plan in plans:
            if plan.status in {"entry_submitting", "entry_submitted"}:
                self._reconcile_entry(plan, moment)
            elif plan.status in {"exit_submitting", "exit_submitted"}:
                self._reconcile_exit(plan, moment)

        plans = list_trade_plans(self.db_path, active_only=True)
        for plan in plans:
            if plan.status != "open":
                continue
            monitor_at = plan.exit_at - timedelta(
                minutes=self.options.early_exit_monitor_minutes
            )
            if moment >= monitor_at and self._reconcile_external_close(plan):
                continue
            if moment >= plan.exit_at:
                self._submit_exit(plan, event_type="hard_exit_submitted")
                # Each option order uses quote, preview, and placement calls.
                # Stagger exits to stay within Webull's burst limits.
                break
            if moment >= monitor_at and plan.entry_fill_price:
                target = plan.entry_fill_price * (
                    1 + self.options.early_exit_profit_pct
                )
                self._submit_exit(
                    plan,
                    exact_limit_price=target,
                    event_type="take_profit_submitted",
                )
                break

        plans = list_trade_plans(self.db_path, active_only=True)
        for plan in plans:
            if plan.status == "planned" and moment >= plan.entry_at:
                if self._submit_entry(plan, moment):
                    # Stagger accepted entries across poll cycles to avoid broker
                    # request bursts and make each submission independently observable.
                    break

    def _submit_entry(self, plan: TradePlan, now: datetime) -> bool:
        deadline = plan.entry_at + timedelta(minutes=self.options.entry_window_minutes)
        if now > deadline:
            self._save(
                plan.with_updates(status="skipped", skip_reason="entry window expired", last_error=None),
                "entry_skipped",
                "Entry window expired before an order was submitted.",
            )
            return False

        current = plan
        try:
            cash = self.broker.cash_balance()
            exposure = open_exposure(self.db_path)
            broker_equity = cash + exposure
            account_basis = broker_equity
            if self.options.strategy_starting_capital is not None:
                account_basis = min(
                    broker_equity,
                    strategy_equity(
                        self.db_path,
                        starting_capital=self.options.strategy_starting_capital,
                    ),
                )
            per_trade_budget = max(0.0, account_basis * plan.allocation_pct)
            total_remaining = max(0.0, account_basis * self.options.max_total_exposure_pct - exposure)
            budget = min(per_trade_budget, total_remaining)
            if budget < 1:
                raise BrokerError("available option budget is below $1 after exposure limits")
            selection_budget = budget / (1 + self.options.max_entry_chase_pct)

            minimum_expiration = (
                plan.exit_at.astimezone(NEW_YORK).date()
                + timedelta(days=self.options.min_expiration_days_after_exit)
            )
            maximum_expiration = (
                plan.exit_at.astimezone(NEW_YORK).date()
                + timedelta(days=self.options.max_expiration_days_after_exit)
            )
            contract = self.broker.select_option(
                plan.ticker,
                plan.option_type,
                minimum_expiration=minimum_expiration,
                maximum_expiration=maximum_expiration,
                preferred_expiration=plan.option_expiration,
                max_contract_cost=selection_budget,
                max_spread_pct=self.options.max_option_spread_pct,
                target_itm_pct=self.options.option_target_itm_pct,
                target_delta=self.options.option_target_delta,
                max_otm_pct=self.options.option_max_otm_pct,
                max_breakeven_move_pct=self.options.max_option_breakeven_pct,
                min_volume=self.options.min_option_volume,
                min_open_interest=self.options.min_option_open_interest,
            )
            used_fallback = False
            if contract is None and self.options.affordable_option_fallback:
                contract = self.broker.select_option(
                    plan.ticker,
                    plan.option_type,
                    minimum_expiration=(
                        plan.exit_at.astimezone(NEW_YORK).date() + timedelta(days=1)
                    ),
                    maximum_expiration=(
                        plan.exit_at.astimezone(NEW_YORK).date() + timedelta(days=10)
                    ),
                    preferred_expiration=plan.option_expiration,
                    max_contract_cost=selection_budget,
                    max_spread_pct=max(self.options.max_option_spread_pct, 0.15),
                    target_itm_pct=0.01,
                    target_delta=0.50,
                    max_otm_pct=self.options.affordable_fallback_max_otm_pct,
                    max_breakeven_move_pct=max(
                        self.options.max_option_breakeven_pct,
                        0.10,
                    ),
                    min_volume=self.options.min_option_volume,
                    min_open_interest=min(self.options.min_option_open_interest, 50),
                )
                used_fallback = contract is not None
            if contract is None:
                raise BrokerError(
                    f"no qualifying liquid {plan.option_type.lower()} fit the "
                    f"${budget:,.2f} budget and event-risk limits"
                )
            quantity = min(
                max(1, self.options.max_contracts_per_trade),
                int(budget // (contract.ask_cost * (1 + self.options.max_entry_chase_pct))),
            )
            if quantity < 1:
                raise BrokerError("selected contract exceeded the current trade budget")
            limit_price = option_limit_price(contract.ask, penny_program=contract.penny_program)
            order_id = _client_order_id("en", plan.id)
            submitting = plan.with_updates(
                status="entry_submitting",
                option_symbol=contract.symbol,
                option_expiration=contract.expiration,
                strike_price=contract.strike,
                contract_multiplier=contract.multiplier,
                quantity=quantity,
                entry_order_id=order_id,
                entry_limit_price=limit_price,
                last_error=None,
            )
            update_trade_plan(self.db_path, submitting)
            if used_fallback:
                record_trade_event(
                    self.db_path,
                    plan.id,
                    "contract_fallback",
                    "Preferred contract was unavailable; selected the bounded near-ATM fallback.",
                )
            current = submitting
            update = self.broker.place_option_order(
                contract,
                side="BUY",
                quantity=quantity,
                limit_price=limit_price,
                client_order_id=order_id,
            )
        except Exception as exc:
            self._store_error(current, f"entry failed: {exc}")
            return False

        self._apply_entry_update(submitting, update)
        return True

    def _reconcile_entry(self, plan: TradePlan, now: datetime) -> None:
        if not plan.entry_order_id:
            self._save(plan.with_updates(status="error", last_error="entry order ID is missing"), "error", "Entry order ID is missing.")
            return
        try:
            update = self.broker.get_order(plan.entry_order_id)
        except Exception as exc:
            if (
                plan.status == "entry_submitting"
                and _is_order_not_found(exc)
            ):
                self._save(
                    plan.with_updates(
                        status="planned",
                        entry_order_id=None,
                        last_error=None,
                    ),
                    "entry_retry",
                    "Broker confirmed the prior entry was not accepted; retrying submission.",
                )
                return
            self._store_error(plan, f"could not reconcile entry order: {exc}")
            return

        deadline = plan.entry_at + timedelta(minutes=self.options.entry_window_minutes)
        if update.status in {"SUBMITTED", "PENDING", "PARTIAL_FILLED"} and now > deadline:
            try:
                self.broker.cancel_order(plan.entry_order_id)
            except Exception as exc:
                self._store_error(plan, f"entry cancellation failed: {exc}")
                return
            if update.filled_quantity > 0:
                update = OrderUpdate(
                    client_order_id=update.client_order_id,
                    status="FILLED",
                    filled_quantity=update.filled_quantity,
                    filled_price=update.filled_price,
                    raw_status="PARTIAL_FILL_AFTER_CANCEL",
                )
            else:
                self._save(
                    plan.with_updates(
                        status="open" if plan.filled_quantity > 0 else "skipped",
                        quantity=plan.filled_quantity or plan.quantity,
                        skip_reason=None if plan.filled_quantity > 0 else "entry limit did not fill",
                        last_error=None,
                    ),
                    "entry_cancelled",
                    "Additional entry quantity did not fill before the cutoff."
                    if plan.filled_quantity > 0
                    else "Entry limit did not fill before the cutoff.",
                )
                return
        elif update.status in {"SUBMITTED", "PENDING"} and _reprice_due(plan, now, self.options.reprice_after_seconds):
            self._reprice(plan, side="BUY")
            return
        self._apply_entry_update(plan, update)

    def _submit_exit(
        self,
        plan: TradePlan,
        *,
        exact_limit_price: float | None = None,
        event_type: str,
    ) -> None:
        if plan.filled_quantity <= 0:
            self._save(plan.with_updates(status="error", last_error="no filled contracts to close"), "error", "No filled contracts were recorded for exit.")
            return
        current = plan
        try:
            contract = self._contract(plan)
            if exact_limit_price is not None:
                order_contract = contract
                limit_price = _ceil_option_price(
                    exact_limit_price,
                    penny_program=contract.penny_program,
                )
            else:
                order_contract = self.broker.option_quote(contract)
                reference_price, _ = self._exit_reference(contract, order_contract)
                if reference_price <= 0:
                    raise BrokerError(f"no executable bid for {order_contract.symbol}")
                limit_price = option_limit_price(
                    reference_price,
                    penny_program=order_contract.penny_program,
                )
            order_id = _client_order_id("ex", plan.id)
            submitting = plan.with_updates(
                status="exit_submitting",
                exit_order_id=order_id,
                exit_limit_price=limit_price,
                last_error=None,
            )
            update_trade_plan(self.db_path, submitting)
            current = submitting
            update = self.broker.place_option_order(
                order_contract,
                side="SELL",
                quantity=plan.filled_quantity,
                limit_price=limit_price,
                client_order_id=order_id,
            )
        except Exception as exc:
            self._store_error(current, f"exit failed and will retry: {exc}")
            return
        record_trade_event(
            self.db_path,
            plan.id,
            event_type,
            f"Submitted sell-to-close limit for {plan.filled_quantity} contract(s) at {limit_price:.2f}.",
        )
        self._apply_exit_update(submitting, update)

    def _reconcile_exit(self, plan: TradePlan, now: datetime) -> None:
        if not plan.exit_order_id:
            self._save(plan.with_updates(status="error", last_error="exit order ID is missing"), "error", "Exit order ID is missing.")
            return
        try:
            update = self.broker.get_order(plan.exit_order_id)
        except Exception as exc:
            if plan.status == "exit_submitting" and _is_order_not_found(exc):
                try:
                    remaining = self.broker.option_position_quantity(self._contract(plan))
                except Exception as position_exc:
                    self._store_error(
                        plan,
                        f"could not verify position after missing exit order: {position_exc}",
                    )
                    return
                if remaining <= 0:
                    if self._reconcile_external_close(
                        plan,
                        position_known_empty=True,
                    ):
                        return
                    self._save(
                        plan.with_updates(
                            status="closed",
                            exit_fill_price=plan.exit_limit_price,
                            last_error=None,
                        ),
                        "exit_reconciled",
                        "Broker reports no remaining position; reconciled the exit at the submitted limit.",
                    )
                else:
                    self._save(
                        plan.with_updates(
                            status="open",
                            exit_order_id=None,
                            last_error=None,
                        ),
                        "exit_retry",
                        f"Broker reports {remaining} contract(s) still open; retrying the exit.",
                    )
                return
            self._store_error(plan, f"could not reconcile exit order: {exc}")
            return
        if update.status in {"SUBMITTED", "PENDING", "PARTIAL_FILLED"}:
            if now < plan.exit_at:
                self._apply_exit_update(plan, update)
                return
            if _reprice_due(plan, now, self.options.reprice_after_seconds):
                if self._reconcile_external_close(plan):
                    return
                self._reprice(plan, side="SELL")
                return
        elif update.status in {"FAILED", "CANCELLED"}:
            if self._reconcile_external_close(plan):
                return
        self._apply_exit_update(plan, update)

    def _reconcile_external_close(
        self,
        plan: TradePlan,
        *,
        position_known_empty: bool = False,
    ) -> bool:
        contract = self._contract(plan)
        if not position_known_empty:
            try:
                if self.broker.option_position_quantity(contract) > 0:
                    return False
            except Exception as exc:
                self._store_error(plan, f"could not check for a manual exit: {exc}")
                return False
        finder = getattr(self.broker, "latest_option_close", None)
        if not callable(finder):
            return False
        try:
            update = finder(contract, since=plan.entry_at)
        except Exception as exc:
            self._store_error(plan, f"could not reconcile manual exit history: {exc}")
            return False
        if update is None or update.status != "FILLED":
            return False
        if plan.exit_order_id and plan.exit_order_id != update.client_order_id:
            try:
                self.broker.cancel_order(plan.exit_order_id)
            except Exception:
                pass
        closed = plan.with_updates(
            status="closed",
            exit_order_id=update.client_order_id,
            exit_fill_price=update.filled_price or plan.exit_limit_price,
            last_error=None,
        )
        self._save(
            closed,
            "external_exit_reconciled",
            f"Broker history shows {update.filled_quantity or plan.filled_quantity} "
            f"contract(s) sold at {closed.exit_fill_price}.",
        )
        return True

    def _reprice(self, plan: TradePlan, *, side: str) -> None:
        try:
            contract = self.broker.option_quote(self._contract(plan))
            fresh_exit_quote = True
            if side == "BUY":
                price = contract.ask
            else:
                price, fresh_exit_quote = self._exit_reference(
                    self._contract(plan), contract
                )
            if price <= 0:
                raise BrokerError(f"no executable {'ask' if side == 'BUY' else 'bid'} for {contract.symbol}")
            if side == "BUY":
                cash = self.broker.cash_balance()
                exposure = open_exposure(self.db_path)
                broker_equity = cash + exposure
                account_basis = broker_equity
                if self.options.strategy_starting_capital is not None:
                    account_basis = min(
                        broker_equity,
                        strategy_equity(
                            self.db_path,
                            starting_capital=self.options.strategy_starting_capital,
                        ),
                    )
                target_budget = account_basis * plan.allocation_pct
                existing_cost = (
                    plan.filled_quantity
                    * (plan.entry_fill_price or 0)
                    * plan.contract_multiplier
                )
                remaining_budget = max(0.0, target_budget - existing_cost)
                max_limit_price = remaining_budget / max(
                    1,
                    plan.quantity * plan.contract_multiplier,
                )
                current_limit = plan.entry_limit_price or price
                chased_price = max(
                    price,
                    current_limit * (1 + self.options.entry_chase_step_pct),
                )
                limit_price = min(
                    option_limit_price(chased_price, penny_program=contract.penny_program),
                    _floor_option_price(max_limit_price, penny_program=contract.penny_program),
                )
                if limit_price <= 0:
                    raise BrokerError("entry allocation has no remaining repricing capacity")
            else:
                if not fresh_exit_quote and plan.exit_limit_price:
                    price = min(
                        price,
                        plan.exit_limit_price
                        * (1 - self.options.exit_chase_step_pct),
                    )
                limit_price = option_limit_price(price, penny_program=contract.penny_program)
            order_id = plan.entry_order_id if side == "BUY" else plan.exit_order_id
            if not order_id:
                raise BrokerError("order ID is missing")
            update = self.broker.replace_order(
                order_id,
                quantity=plan.quantity if side == "BUY" else plan.filled_quantity,
                limit_price=limit_price,
            )
            changed = (
                plan.with_updates(entry_limit_price=limit_price)
                if side == "BUY"
                else plan.with_updates(exit_limit_price=limit_price)
            )
            update_trade_plan(self.db_path, changed)
        except Exception as exc:
            self._store_error(plan, f"{side.lower()} repricing failed: {exc}")
            return
        if side == "BUY":
            self._apply_entry_update(changed, update)
        else:
            self._apply_exit_update(changed, update)

    def _exit_reference(
        self,
        contract: OptionContract,
        quote: OptionContract,
    ) -> tuple[float, bool]:
        now = datetime.now(timezone.utc)
        if quote.quote_time_ms is not None:
            quote_time = datetime.fromtimestamp(quote.quote_time_ms / 1000, timezone.utc)
            age = now - quote_time
            if timedelta(seconds=-30) <= age <= timedelta(minutes=2):
                return quote.bid, True
        try:
            mark = self.broker.option_position_mark(contract)
        except Exception:
            mark = None
        return (mark if mark is not None and mark > 0 else quote.bid), False

    def _apply_entry_update(self, plan: TradePlan, update: OrderUpdate) -> None:
        if update.status == "FILLED":
            added_quantity = update.filled_quantity or plan.quantity
            existing_quantity = plan.filled_quantity
            quantity = existing_quantity + added_quantity
            added_price = update.filled_price or plan.entry_limit_price
            fill_price = _weighted_fill_price(
                existing_quantity=existing_quantity,
                existing_price=plan.entry_fill_price,
                added_quantity=added_quantity,
                added_price=added_price,
            )
            filled = plan.with_updates(
                status="open",
                quantity=quantity,
                filled_quantity=quantity,
                entry_fill_price=fill_price,
                last_error=None,
            )
            self._save(
                filled,
                "entry_filled",
                f"Bought {quantity} {plan.option_symbol} at {filled.entry_fill_price}.",
            )
        elif update.status in {"FAILED", "CANCELLED"}:
            if plan.filled_quantity > 0:
                self._save(
                    plan.with_updates(
                        status="open",
                        quantity=plan.filled_quantity,
                        last_error=None,
                    ),
                    "entry_addon_failed",
                    f"Additional entry order ended with status {update.status}; existing position remains open.",
                )
            else:
                self._save(
                    plan.with_updates(status="error", last_error=f"entry order {update.status.lower()}"),
                    "entry_failed",
                    f"Entry order ended with status {update.status}.",
                )
        else:
            if plan.status != "entry_submitted":
                update_trade_plan(self.db_path, plan.with_updates(status="entry_submitted"))

    def _apply_exit_update(self, plan: TradePlan, update: OrderUpdate) -> None:
        if update.status == "FILLED":
            quantity = update.filled_quantity or plan.filled_quantity
            closed = plan.with_updates(
                status="closed",
                exit_fill_price=update.filled_price or plan.exit_limit_price,
                last_error=None,
            )
            self._save(
                closed,
                "exit_filled",
                f"Sold {quantity} {plan.option_symbol} at {closed.exit_fill_price}.",
            )
        elif update.status in {"FAILED", "CANCELLED"}:
            self._save(
                plan.with_updates(status="error", last_error=f"exit order {update.status.lower()}; position may still be open"),
                "exit_failed",
                f"Exit order ended with status {update.status}; verify the position manually.",
            )
        else:
            if plan.status != "exit_submitted":
                update_trade_plan(self.db_path, plan.with_updates(status="exit_submitted"))

    def _contract(self, plan: TradePlan) -> OptionContract:
        if not plan.option_symbol or not plan.option_expiration or plan.strike_price is None:
            raise BrokerError("trade plan does not contain a selected option contract")
        return OptionContract(
            symbol=plan.option_symbol,
            ticker=plan.ticker,
            option_type=plan.option_type,
            expiration=plan.option_expiration,
            strike=plan.strike_price,
            bid=0.0,
            ask=0.0,
            underlying_price=0.0,
            multiplier=plan.contract_multiplier,
        )

    def _store_error(self, plan: TradePlan, message: str) -> None:
        update_trade_plan(self.db_path, plan.with_updates(last_error=message))
        record_trade_event(self.db_path, plan.id, "warning", message)
        self.log(f"{plan.ticker}: {message}")

    def _save(self, plan: TradePlan, event_type: str, message: str) -> None:
        update_trade_plan(self.db_path, plan)
        record_trade_event(self.db_path, plan.id, event_type, message)
        self.log(f"{plan.ticker}: {message}")


def _reprice_due(plan: TradePlan, now: datetime, seconds: int) -> bool:
    if plan.updated_at is None:
        return True
    return now - _aware_utc(plan.updated_at) >= timedelta(seconds=max(15, seconds))


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _is_order_not_found(exc: Exception) -> bool:
    message = str(exc).upper()
    return "ORDER_NOT_FOUND" in message or "ORDER WAS NOT FOUND" in message


def _weighted_fill_price(
    *,
    existing_quantity: int,
    existing_price: float | None,
    added_quantity: int,
    added_price: float | None,
) -> float | None:
    total_quantity = existing_quantity + added_quantity
    if total_quantity <= 0:
        return None
    if existing_quantity <= 0:
        return added_price
    if added_quantity <= 0:
        return existing_price
    if existing_price is None or added_price is None:
        return added_price or existing_price
    return (
        existing_quantity * existing_price + added_quantity * added_price
    ) / total_quantity


def _client_order_id(prefix: str, plan_id: str) -> str:
    return f"{prefix[:2]}{plan_id[:20]}{uuid4().hex[:10]}"


def _floor_option_price(price: float, *, penny_program: bool) -> float:
    increment = 0.01 if price < 3 else 0.05
    return math.floor(max(0.0, price) / increment) * increment


def _ceil_option_price(price: float, *, penny_program: bool) -> float:
    increment = 0.01 if price < 3 else 0.05
    return round(math.ceil(max(0.0, price) / increment) * increment, 2)
