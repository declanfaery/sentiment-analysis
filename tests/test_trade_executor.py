from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from sentiment_analysis.trading.executor import (
    ExecutionOptions,
    TradeExecutor,
    _ceil_option_price,
    _weighted_fill_price,
)
from sentiment_analysis.trading.models import OptionContract, OrderUpdate, TradePlan
from sentiment_analysis.trading.storage import get_trade_plan, save_trade_plans


class ImmediateFillBroker:
    def __init__(self) -> None:
        self.orders: dict[str, OrderUpdate] = {}
        self.contract = OptionContract(
            symbol="AMD260807P00150000",
            ticker="AMD",
            option_type="PUT",
            expiration=date(2026, 8, 7),
            strike=150,
            bid=4.90,
            ask=5.00,
            underlying_price=150,
        )

    def cash_balance(self) -> float:
        return 10_000

    def underlying_price(self, ticker: str) -> float:
        return self.contract.underlying_price

    def select_option(self, *args, **kwargs):
        return self.contract

    def option_quote(self, contract):
        return self.contract

    def place_option_order(self, contract, *, side, quantity, limit_price, client_order_id):
        update = OrderUpdate(client_order_id, "FILLED", quantity, limit_price)
        self.orders[client_order_id] = update
        return update

    def get_order(self, client_order_id):
        return self.orders[client_order_id]

    def replace_order(self, client_order_id, *, quantity, limit_price):
        return self.orders[client_order_id]

    def cancel_order(self, client_order_id):
        return None

    def option_position_quantity(self, contract):
        return 0

    def option_position_mark(self, contract):
        return contract.bid

    def latest_option_close(self, contract, *, since=None):
        return None


class MissingOrderBroker(ImmediateFillBroker):
    def get_order(self, client_order_id):
        raise RuntimeError("OAUTH_OPENAPI_ORDER_NOT_FOUND: The requested order was not found")


class MillionDollarBroker(ImmediateFillBroker):
    def cash_balance(self) -> float:
        return 1_000_000


class FirstTickerTooExpensiveBroker(ImmediateFillBroker):
    def select_option(self, ticker, *args, **kwargs):
        if ticker == "AAA":
            return None
        return self.contract


class AffordableFallbackBroker(ImmediateFillBroker):
    def __init__(self) -> None:
        super().__init__()
        self.selection_calls: list[dict[str, object]] = []

    def select_option(self, ticker, *args, **kwargs):
        self.selection_calls.append(kwargs)
        return self.contract if float(kwargs.get("max_otm_pct", 0)) > 0 else None


class MissingExitOrderBroker(ImmediateFillBroker):
    def get_order(self, client_order_id):
        raise RuntimeError("OAUTH_OPENAPI_ORDER_NOT_FOUND: The requested order was not found")


class ProfitableOptionBroker(ImmediateFillBroker):
    def __init__(self) -> None:
        super().__init__()
        self.contract = OptionContract(
            **{
                **self.contract.__dict__,
                "bid": 5.30,
                "ask": 5.40,
                "quote_time_ms": 1786455000000,
            }
        )


class StaleOptionLivePositionMarkBroker(ImmediateFillBroker):
    def option_quote(self, contract):
        return OptionContract(
            **{
                **self.contract.__dict__,
                "bid": 4.0,
                "ask": 4.2,
                "quote_time_ms": 1786368600000,
            }
        )

    def option_position_mark(self, contract):
        return 5.30


class WorkingExitBroker(ImmediateFillBroker):
    def __init__(self) -> None:
        super().__init__()
        self.replace_prices: list[float] = []

    def place_option_order(self, contract, *, side, quantity, limit_price, client_order_id):
        update = OrderUpdate(client_order_id, "PENDING", 0, None)
        self.orders[client_order_id] = update
        return update

    def replace_order(self, client_order_id, *, quantity, limit_price):
        self.replace_prices.append(limit_price)
        update = OrderUpdate(client_order_id, "PENDING", 0, None)
        self.orders[client_order_id] = update
        return update

    def option_position_quantity(self, contract):
        return 1


class ManuallyClosedBroker(ImmediateFillBroker):
    def __init__(self) -> None:
        super().__init__()
        self.placement_count = 0

    def place_option_order(self, contract, *, side, quantity, limit_price, client_order_id):
        self.placement_count += 1
        return super().place_option_order(
            contract,
            side=side,
            quantity=quantity,
            limit_price=limit_price,
            client_order_id=client_order_id,
        )

    def latest_option_close(self, contract, *, since=None):
        return OrderUpdate("manual-close", "FILLED", 1, 6.25, "EXTERNAL_FILLED")


class TradeExecutorTests(unittest.TestCase):
    def test_profit_floor_rounds_up_to_valid_option_tick(self) -> None:
        self.assertEqual(_ceil_option_price(9.555, penny_program=True), 9.60)

    def test_opens_and_closes_due_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "trades.sqlite3"
            now = datetime(2026, 8, 4, 19, 0, tzinfo=timezone.utc)
            plan = TradePlan(
                id="a" * 32,
                scan_run_id="scan-1",
                ticker="AMD",
                direction="bearish",
                option_type="PUT",
                earnings_date=date(2026, 8, 4),
                earnings_timing="AMC",
                entry_at=now,
                exit_at=now + timedelta(days=1),
                allocation_pct=0.07,
                recommendation="bearish_watch",
                recommendation_score=55,
                created_at=now,
                updated_at=now,
            )
            save_trade_plans(path, [plan])
            executor = TradeExecutor(db_path=path, broker=ImmediateFillBroker(), log=lambda _: None)

            executor.run_once(now)
            opened = get_trade_plan(path, plan.id)
            self.assertEqual(opened.status, "open")
            self.assertEqual(opened.quantity, 1)
            self.assertEqual(opened.entry_fill_price, 5.0)

            executor.run_once(plan.exit_at)
            closed = get_trade_plan(path, plan.id)
            self.assertEqual(closed.status, "closed")
            self.assertEqual(closed.exit_fill_price, 4.9)

    def test_retries_entry_when_broker_confirms_submission_was_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "trades.sqlite3"
            now = datetime(2026, 8, 10, 19, 2, tzinfo=timezone.utc)
            plan = TradePlan(
                id="b" * 32,
                scan_run_id="scan-1",
                ticker="RKLB",
                direction="bearish",
                option_type="PUT",
                earnings_date=date(2026, 8, 10),
                earnings_timing="AMC",
                entry_at=now - timedelta(minutes=2),
                exit_at=now + timedelta(days=1),
                allocation_pct=0.07,
                recommendation="bearish_watch",
                recommendation_score=75,
                status="entry_submitting",
                entry_order_id="missing-order",
                created_at=now - timedelta(minutes=2),
                updated_at=now - timedelta(minutes=2),
            )
            save_trade_plans(path, [plan])
            executor = TradeExecutor(
                db_path=path,
                broker=MissingOrderBroker(),
                log=lambda _: None,
            )

            executor.run_once(now)

            retried = get_trade_plan(path, plan.id)
            self.assertEqual(retried.status, "open")
            self.assertIsNotNone(retried.entry_order_id)
            self.assertNotEqual(retried.entry_order_id, "missing-order")
            self.assertLessEqual(len(retried.entry_order_id), 32)

    def test_staggers_multiple_due_entries_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "trades.sqlite3"
            now = datetime(2026, 8, 10, 19, 0, tzinfo=timezone.utc)
            plans = [
                TradePlan(
                    id=letter * 32,
                    scan_run_id="scan-1",
                    ticker=ticker,
                    direction="bearish",
                    option_type="PUT",
                    earnings_date=date(2026, 8, 10),
                    earnings_timing="AMC",
                    entry_at=now,
                    exit_at=now + timedelta(days=1),
                    allocation_pct=0.07,
                    recommendation="bearish_watch",
                    recommendation_score=75,
                    created_at=now,
                    updated_at=now,
                )
                for letter, ticker in (("c", "ASTS"), ("d", "RKLB"))
            ]
            save_trade_plans(path, plans)
            executor = TradeExecutor(
                db_path=path,
                broker=ImmediateFillBroker(),
                log=lambda _: None,
            )

            executor.run_once(now)

            statuses = {get_trade_plan(path, plan.id).status for plan in plans}
            self.assertEqual(statuses, {"open", "planned"})

    def test_unaffordable_candidate_does_not_block_later_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "trades.sqlite3"
            now = datetime(2026, 8, 11, 19, 45, tzinfo=timezone.utc)
            plans = [
                TradePlan(
                    id=letter * 32,
                    scan_run_id="scan-1",
                    ticker=ticker,
                    direction="bullish",
                    option_type="CALL",
                    earnings_date=date(2026, 8, 11),
                    earnings_timing="AMC",
                    entry_at=now,
                    exit_at=now + timedelta(days=1),
                    allocation_pct=0.07,
                    recommendation="bullish_watch",
                    recommendation_score=75,
                    created_at=now,
                    updated_at=now,
                )
                for letter, ticker in (("i", "AAA"), ("j", "BBB"))
            ]
            save_trade_plans(path, plans)
            executor = TradeExecutor(
                db_path=path,
                broker=FirstTickerTooExpensiveBroker(),
                log=lambda _: None,
            )

            executor.run_once(now)

            blocked = get_trade_plan(path, plans[0].id)
            opened = get_trade_plan(path, plans[1].id)
            self.assertEqual(blocked.status, "planned")
            self.assertIn("no qualifying liquid", blocked.last_error or "")
            self.assertEqual(opened.status, "open")

    def test_bounded_affordable_fallback_runs_after_preferred_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "trades.sqlite3"
            now = datetime(2026, 8, 11, 19, 45, tzinfo=timezone.utc)
            plan = TradePlan(
                id="k" * 32,
                scan_run_id="scan-1",
                ticker="QNT",
                direction="bearish",
                option_type="PUT",
                earnings_date=date(2026, 8, 11),
                earnings_timing="AMC",
                entry_at=now,
                exit_at=now + timedelta(days=1),
                allocation_pct=0.07,
                recommendation="bearish_watch",
                recommendation_score=60,
                created_at=now,
                updated_at=now,
            )
            save_trade_plans(path, [plan])
            broker = AffordableFallbackBroker()
            executor = TradeExecutor(
                db_path=path,
                broker=broker,
                options=ExecutionOptions(affordable_option_fallback=True),
                log=lambda _: None,
            )

            executor.run_once(now)

            self.assertEqual(get_trade_plan(path, plan.id).status, "open")
            self.assertEqual(len(broker.selection_calls), 2)
            self.assertEqual(broker.selection_calls[0]["max_otm_pct"], 0.0)
            self.assertEqual(broker.selection_calls[1]["max_otm_pct"], 0.03)

    def test_staggers_multiple_due_exits_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "trades.sqlite3"
            now = datetime(2026, 8, 11, 13, 45, tzinfo=timezone.utc)
            plans = [
                TradePlan(
                    id=letter * 32,
                    scan_run_id="scan-1",
                    ticker=ticker,
                    direction="bearish",
                    option_type="PUT",
                    earnings_date=date(2026, 8, 10),
                    earnings_timing="AMC",
                    entry_at=now - timedelta(days=1),
                    exit_at=now,
                    allocation_pct=0.07,
                    recommendation="bearish_watch",
                    recommendation_score=75,
                    status="open",
                    option_symbol=f"{ticker}260814P00150000",
                    option_expiration=date(2026, 8, 14),
                    strike_price=150,
                    quantity=1,
                    filled_quantity=1,
                    entry_fill_price=5.0,
                    created_at=now - timedelta(days=1),
                    updated_at=now - timedelta(days=1),
                )
                for letter, ticker in (("f", "ASTS"), ("g", "RKLB"))
            ]
            save_trade_plans(path, plans)
            executor = TradeExecutor(
                db_path=path,
                broker=ImmediateFillBroker(),
                log=lambda _: None,
            )

            executor.run_once(now)

            statuses = {get_trade_plan(path, plan.id).status for plan in plans}
            self.assertEqual(statuses, {"closed", "open"})

    def test_missing_exit_order_reconciles_when_position_is_gone(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "trades.sqlite3"
            now = datetime(2026, 8, 11, 14, 0, tzinfo=timezone.utc)
            plan = TradePlan(
                id="h" * 32,
                scan_run_id="scan-1",
                ticker="RKLB",
                direction="bearish",
                option_type="PUT",
                earnings_date=date(2026, 8, 10),
                earnings_timing="AMC",
                entry_at=now - timedelta(days=1),
                exit_at=now - timedelta(minutes=15),
                allocation_pct=0.07,
                recommendation="bearish_watch",
                recommendation_score=75,
                status="exit_submitting",
                option_symbol="RKLB260814P00081000",
                option_expiration=date(2026, 8, 14),
                strike_price=81,
                quantity=123,
                filled_quantity=123,
                entry_fill_price=5.55,
                exit_order_id="missing-exit",
                exit_limit_price=4.25,
                created_at=now - timedelta(days=1),
                updated_at=now - timedelta(minutes=15),
            )
            save_trade_plans(path, [plan])
            executor = TradeExecutor(
                db_path=path,
                broker=MissingExitOrderBroker(),
                log=lambda _: None,
            )

            executor.run_once(now)

            reconciled = get_trade_plan(path, plan.id)
            self.assertEqual(reconciled.status, "closed")
            self.assertEqual(reconciled.exit_fill_price, 4.25)
            self.assertIsNone(reconciled.last_error)

    def test_places_resting_profit_order_during_opening_monitor_window(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "trades.sqlite3"
            exit_at = datetime(2026, 8, 11, 13, 45, tzinfo=timezone.utc)
            now = exit_at - timedelta(minutes=15)
            plan = TradePlan(
                id="i" * 32,
                scan_run_id="scan-1",
                ticker="AMD",
                direction="bearish",
                option_type="PUT",
                earnings_date=date(2026, 8, 10),
                earnings_timing="AMC",
                entry_at=now - timedelta(days=1),
                exit_at=exit_at,
                allocation_pct=0.07,
                recommendation="bearish_watch",
                recommendation_score=75,
                status="open",
                option_symbol="AMD260821P00150000",
                option_expiration=date(2026, 8, 21),
                strike_price=150,
                quantity=1,
                filled_quantity=1,
                entry_fill_price=5.0,
                created_at=now - timedelta(days=1),
                updated_at=now - timedelta(days=1),
            )
            save_trade_plans(path, [plan])
            executor = TradeExecutor(
                db_path=path,
                broker=ProfitableOptionBroker(),
                log=lambda _: None,
            )

            executor.run_once(now)

            closed = get_trade_plan(path, plan.id)
            self.assertEqual(closed.status, "closed")
            self.assertEqual(closed.exit_fill_price, 5.25)

    def test_resting_profit_order_does_not_depend_on_stale_option_quote(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "trades.sqlite3"
            exit_at = datetime(2026, 8, 12, 13, 45, tzinfo=timezone.utc)
            now = exit_at - timedelta(minutes=15)
            plan = TradePlan(
                id="m" * 32,
                scan_run_id="scan-1",
                ticker="AMD",
                direction="bearish",
                option_type="PUT",
                earnings_date=date(2026, 8, 11),
                earnings_timing="AMC",
                entry_at=now - timedelta(days=1),
                exit_at=exit_at,
                allocation_pct=0.07,
                recommendation="bearish_watch",
                recommendation_score=75,
                status="open",
                option_symbol="AMD260821P00150000",
                option_expiration=date(2026, 8, 21),
                strike_price=150,
                quantity=1,
                filled_quantity=1,
                entry_fill_price=5.0,
                created_at=now - timedelta(days=1),
                updated_at=now - timedelta(days=1),
            )
            save_trade_plans(path, [plan])
            executor = TradeExecutor(
                db_path=path,
                broker=StaleOptionLivePositionMarkBroker(),
                log=lambda _: None,
            )

            executor.run_once(now)

            closed = get_trade_plan(path, plan.id)
            self.assertEqual(closed.status, "closed")
            self.assertEqual(closed.exit_fill_price, 5.25)

    def test_profit_order_is_not_repriced_until_hard_exit_time(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "trades.sqlite3"
            exit_at = datetime.now(timezone.utc) + timedelta(minutes=20)
            monitor_at = exit_at - timedelta(minutes=15)
            plan = TradePlan(
                id="n" * 32,
                scan_run_id="scan-1",
                ticker="AMD",
                direction="bearish",
                option_type="PUT",
                earnings_date=date(2026, 8, 11),
                earnings_timing="AMC",
                entry_at=monitor_at - timedelta(days=1),
                exit_at=exit_at,
                allocation_pct=0.05,
                recommendation="bearish_watch_options_confirm",
                recommendation_score=75,
                status="open",
                option_symbol="AMD260821P00150000",
                option_expiration=date(2026, 8, 21),
                strike_price=150,
                quantity=1,
                filled_quantity=1,
                entry_fill_price=5.0,
                created_at=monitor_at - timedelta(days=1),
                updated_at=monitor_at - timedelta(days=1),
            )
            save_trade_plans(path, [plan])
            broker = WorkingExitBroker()
            executor = TradeExecutor(db_path=path, broker=broker, log=lambda _: None)

            executor.run_once(monitor_at)
            submitted = get_trade_plan(path, plan.id)
            self.assertEqual(submitted.status, "exit_submitted")
            self.assertEqual(submitted.exit_limit_price, 5.25)

            executor.run_once(monitor_at + timedelta(minutes=5))
            self.assertEqual(broker.replace_prices, [])

            executor.run_once(exit_at)
            self.assertEqual(broker.replace_prices, [4.7])

    def test_manual_close_is_reconciled_at_actual_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "trades.sqlite3"
            exit_at = datetime(2026, 8, 13, 13, 45, tzinfo=timezone.utc)
            now = exit_at - timedelta(minutes=15)
            plan = TradePlan(
                id="o" * 32,
                scan_run_id="scan-1",
                ticker="AMD",
                direction="bullish",
                option_type="CALL",
                earnings_date=date(2026, 8, 12),
                earnings_timing="AMC",
                entry_at=now - timedelta(days=1),
                exit_at=exit_at,
                allocation_pct=0.05,
                recommendation="bullish_watch_politics_override",
                recommendation_score=80,
                status="open",
                option_symbol="AMD260821C00150000",
                option_expiration=date(2026, 8, 21),
                strike_price=150,
                quantity=1,
                filled_quantity=1,
                entry_fill_price=5.0,
                created_at=now - timedelta(days=1),
                updated_at=now - timedelta(days=1),
            )
            save_trade_plans(path, [plan])
            broker = ManuallyClosedBroker()
            executor = TradeExecutor(db_path=path, broker=broker, log=lambda _: None)

            executor.run_once(now)

            closed = get_trade_plan(path, plan.id)
            self.assertEqual(closed.status, "closed")
            self.assertEqual(closed.exit_order_id, "manual-close")
            self.assertEqual(closed.exit_fill_price, 6.25)
            self.assertEqual(broker.placement_count, 0)

    def test_sizes_contracts_to_allocation_when_safety_cap_allows_it(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "trades.sqlite3"
            now = datetime(2026, 8, 10, 19, 0, tzinfo=timezone.utc)
            plan = TradePlan(
                id="e" * 32,
                scan_run_id="scan-1",
                ticker="ASTS",
                direction="bearish",
                option_type="PUT",
                earnings_date=date(2026, 8, 10),
                earnings_timing="AMC",
                entry_at=now,
                exit_at=now + timedelta(days=1),
                allocation_pct=0.07,
                recommendation="bearish_watch",
                recommendation_score=80,
                created_at=now,
                updated_at=now,
            )
            save_trade_plans(path, [plan])
            executor = TradeExecutor(
                db_path=path,
                broker=MillionDollarBroker(),
                options=ExecutionOptions(max_contracts_per_trade=1_000),
                log=lambda _: None,
            )

            executor.run_once(now)

            opened = get_trade_plan(path, plan.id)
            self.assertEqual(opened.quantity, 135)
            self.assertEqual(opened.filled_quantity, 135)
            self.assertLessEqual(opened.quantity * opened.entry_fill_price * 100, 70_000)

    def test_virtual_capital_limits_a_million_dollar_broker_account(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "small.sqlite3"
            now = datetime(2026, 8, 10, 19, 45, tzinfo=timezone.utc)
            plan = TradePlan(
                id="k" * 32,
                scan_run_id="scan-1",
                ticker="AMD",
                direction="bearish",
                option_type="PUT",
                earnings_date=date(2026, 8, 10),
                earnings_timing="AMC",
                entry_at=now,
                exit_at=now + timedelta(days=1),
                allocation_pct=0.07,
                recommendation="bearish_watch",
                recommendation_score=80,
                created_at=now,
                updated_at=now,
            )
            save_trade_plans(path, [plan])
            executor = TradeExecutor(
                db_path=path,
                broker=MillionDollarBroker(),
                options=ExecutionOptions(
                    max_contracts_per_trade=1_000,
                    strategy_starting_capital=10_000,
                ),
                log=lambda _: None,
            )

            executor.run_once(now)

            opened = get_trade_plan(path, plan.id)
            self.assertEqual(opened.quantity, 1)
            self.assertLessEqual(opened.quantity * opened.entry_fill_price * 100, 700)

    def test_combines_existing_and_supplemental_fill_price(self) -> None:
        price = _weighted_fill_price(
            existing_quantity=1,
            existing_price=4.2,
            added_quantity=159,
            added_price=4.4,
        )

        self.assertAlmostEqual(price, 4.39875)


if __name__ == "__main__":
    unittest.main()
