from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

from sentiment_analysis.trading.broker import option_limit_price
from sentiment_analysis.trading.calendar import NEW_YORK
from sentiment_analysis.trading.config import TradingConfig
from sentiment_analysis.trading.service import AutomationService, SANDBOX_SMOKE_STATE_KEY
from sentiment_analysis.trading.storage import (
    latest_scan_run_id,
    list_trade_plans,
    open_exposure,
    set_automation_state,
    strategy_equity,
)
from sentiment_analysis.trading.research_storage import research_summary
from sentiment_analysis.trading.webull import WebullOptionsBroker, list_webull_accounts


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            return plan_command(args)
        if args.command == "run-once":
            return run_once_command(args)
        if args.command == "daemon":
            return daemon_command(args)
        if args.command == "status":
            return status_command(args)
        if args.command == "research-status":
            return research_status_command(args)
        if args.command == "verify-webull":
            return verify_webull_command(args)
        if args.command == "smoke-webull":
            return smoke_webull_command(args)
    except KeyboardInterrupt:
        print("Automation stopped.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentiment-trader")
    subparsers = parser.add_subparsers(dest="command")

    plan = subparsers.add_parser("plan", help="Build trade plans from a saved dashboard scan.")
    plan.add_argument("--run-id", default="latest")

    once = subparsers.add_parser("run-once", help="Run one scheduler/executor iteration.")
    once.add_argument("--mode", choices=["paper", "sandbox", "live"])
    once.add_argument("--skip-scan", action="store_true")

    daemon = subparsers.add_parser("daemon", help="Run the always-on scheduler and executor.")
    daemon.add_argument("--mode", choices=["paper", "sandbox", "live"])

    status = subparsers.add_parser("status", help="Show persisted trade plans.")
    status.add_argument("--limit", type=int, default=30)
    status.add_argument("--json", action="store_true")

    research_status = subparsers.add_parser(
        "research-status",
        help="Show shadow-research collection status and recent simulations.",
    )
    research_status.add_argument("--limit", type=int, default=20)
    research_status.add_argument("--json", action="store_true")

    verify = subparsers.add_parser("verify-webull", help="Verify official Webull credentials read-only.")
    verify.add_argument("--environment", choices=["sandbox", "live"], default="sandbox")
    verify.add_argument("--ticker", default="SPY", help="Underlying used for the read-only option quote check.")

    smoke = subparsers.add_parser(
        "smoke-webull",
        help="Preview or round-trip one Webull sandbox option contract.",
    )
    smoke.add_argument("--ticker", default="SPY")
    smoke.add_argument("--max-cost", type=float, default=1000.0)
    smoke.add_argument("--timeout", type=int, default=180)
    smoke.add_argument("--execute", action="store_true", help="Place and close the sandbox order.")
    return parser


def plan_command(args: argparse.Namespace) -> int:
    config = TradingConfig.from_env()
    run_id = latest_scan_run_id(config.db_path) if args.run_id == "latest" else args.run_id
    if not run_id:
        raise ValueError("there are no saved dashboard scans to plan")
    service = AutomationService(config)
    inserted = service.plan_run(run_id)
    print(f"Created {inserted} new trade plan(s) from scan {run_id}.")
    return 0


def run_once_command(args: argparse.Namespace) -> int:
    config = TradingConfig.from_env(mode=args.mode)
    AutomationService(config).run_once(skip_scan=args.skip_scan)
    return 0


def daemon_command(args: argparse.Namespace) -> int:
    config = TradingConfig.from_env(mode=args.mode)
    AutomationService(config).daemon()
    return 0


def status_command(args: argparse.Namespace) -> int:
    config = TradingConfig.from_env()
    plans = list_trade_plans(config.db_path, limit=max(1, args.limit))
    if args.json:
        print(json.dumps([plan.as_dict() for plan in plans], indent=2))
        return 0
    if not plans:
        print("No trade plans saved.")
        return 0
    if config.execution.strategy_starting_capital is not None:
        equity = strategy_equity(
            config.db_path,
            starting_capital=config.execution.strategy_starting_capital,
        )
        print(
            f"Strategy {config.strategy_name}: virtual equity ${equity:,.2f}; "
            f"open premium ${open_exposure(config.db_path):,.2f}."
        )
    for plan in plans:
        print(
            f"{plan.ticker:6} {plan.option_type:4} {plan.status:17} "
            f"entry={plan.entry_at.isoformat()} allocation={plan.allocation_pct:.0%} "
            f"score={plan.recommendation_score}"
        )
        if plan.skip_reason or plan.last_error:
            print(f"       {plan.skip_reason or plan.last_error}")
    return 0


def research_status_command(args: argparse.Namespace) -> int:
    config = TradingConfig.from_env()
    summary = research_summary(config.db_path, limit=max(1, args.limit))
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
        return 0
    print(
        f"Shadow research: {summary['feature_snapshots']} feature snapshot(s), "
        f"{summary['shadow_positions']} virtual position(s), "
        f"{summary['observations']} outcome observation(s)."
    )
    statuses = summary.get("statuses", {})
    if statuses:
        print(
            "Statuses: "
            + ", ".join(
                f"{status}={count}"
                for status, count in sorted(statuses.items())
            )
        )
    for row in summary.get("recent", []):
        best = row.get("best_bid_return_pct")
        result = "na" if best is None else f"{float(best):+.1%} best"
        print(
            f"{row['ticker']:6} {row['option_type']:4} "
            f"d={float(row['target_delta']):.2f} "
            f"DTE+={row['target_expiration_days']:2} "
            f"entry=-{row['entry_minutes_before_close']:2}m "
            f"{row['status']:20} {result}"
        )
    return 0


def verify_webull_command(args: argparse.Namespace) -> int:
    config = TradingConfig.from_env(mode=args.environment)
    if not config.webull_account_id:
        accounts = list_webull_accounts(
            app_key=config.webull_app_key or "",
            app_secret=config.webull_app_secret or "",
            environment=args.environment,
        )
        _print_accounts(args.environment, accounts)
        variable = f"WEBULL_{args.environment.upper()}_ACCOUNT_ID"
        print(f"Add the option-enabled Account ID above to {variable} in .env, then run this command again.")
        return 0

    broker = WebullOptionsBroker(
        app_key=config.webull_app_key or "",
        app_secret=config.webull_app_secret or "",
        account_id=config.webull_account_id or "",
        environment=args.environment,
        allow_live=True,
    )
    summary = broker.account_summary()
    accounts = summary.get("accounts", [])
    _print_accounts(args.environment, accounts if isinstance(accounts, list) else [])
    balance = summary.get("balance", {})
    cash = balance.get("total_cash_balance", "unknown") if isinstance(balance, dict) else "unknown"
    print(f"Configured account balance read succeeded. Cash balance: {cash}")
    ticker = args.ticker.strip().upper()
    contract = broker.select_option(
        ticker,
        "CALL",
        minimum_expiration=date.today() + timedelta(days=1),
        preferred_expiration=None,
        max_contract_cost=100_000,
        max_spread_pct=1.0,
    )
    if contract is None:
        raise RuntimeError(f"Webull returned no usable {ticker} option quote; verify OpenAPI OPRA access")
    print(
        f"OpenAPI option data succeeded: {contract.symbol} "
        f"bid={contract.bid:.2f} ask={contract.ask:.2f}. No order was placed."
    )
    return 0


def smoke_webull_command(args: argparse.Namespace) -> int:
    config = TradingConfig.from_env(mode="sandbox")
    broker = WebullOptionsBroker(
        app_key=config.webull_app_key or "",
        app_secret=config.webull_app_secret or "",
        account_id=config.webull_account_id or "",
        environment="sandbox",
    )
    ticker = args.ticker.strip().upper()
    contract = broker.select_option(
        ticker,
        "CALL",
        minimum_expiration=date.today() + timedelta(days=1),
        preferred_expiration=None,
        max_contract_cost=max(1.0, args.max_cost),
        max_spread_pct=0.50,
    )
    if contract is None:
        raise RuntimeError(f"no liquid {ticker} sandbox call fit the ${args.max_cost:,.2f} limit")

    entry_price = option_limit_price(contract.ask, penny_program=contract.penny_program)
    entry_id = f"smk-en-{uuid4().hex[:24]}"
    if not args.execute:
        broker.preview_option_order(
            contract,
            side="BUY",
            quantity=1,
            limit_price=entry_price,
            client_order_id=entry_id,
        )
        print(
            f"Sandbox preview passed for 1 {contract.symbol} at {entry_price:.2f}. "
            "No order was placed."
        )
        return 0

    _require_regular_market_session()
    print(f"Submitting sandbox buy: 1 {contract.symbol} at {entry_price:.2f}.")
    broker.place_option_order(
        contract,
        side="BUY",
        quantity=1,
        limit_price=entry_price,
        client_order_id=entry_id,
    )
    entry = _wait_for_smoke_fill(
        broker,
        contract,
        entry_id,
        side="BUY",
        timeout=max(30, args.timeout),
    )
    print(f"Sandbox buy filled at {entry.filled_price or entry_price}.")

    quote = broker.option_quote(contract)
    exit_price = option_limit_price(quote.bid, penny_program=quote.penny_program)
    if exit_price <= 0:
        raise RuntimeError(f"sandbox position opened, but {contract.symbol} has no exit bid")
    exit_id = f"smk-ex-{uuid4().hex[:24]}"
    print(f"Submitting sandbox close: 1 {contract.symbol} at {exit_price:.2f}.")
    broker.place_option_order(
        quote,
        side="SELL",
        quantity=1,
        limit_price=exit_price,
        client_order_id=exit_id,
    )
    exit_update = _wait_for_smoke_fill(
        broker,
        contract,
        exit_id,
        side="SELL",
        timeout=max(30, args.timeout),
    )
    set_automation_state(
        config.db_path,
        SANDBOX_SMOKE_STATE_KEY,
        datetime.now(timezone.utc).isoformat(),
    )
    print(f"Sandbox close filled at {exit_update.filled_price or exit_price}. Smoke test passed.")
    return 0


def _wait_for_smoke_fill(broker, contract, order_id: str, *, side: str, timeout: int):
    deadline = time.monotonic() + timeout
    next_reprice = time.monotonic() + 15
    while time.monotonic() < deadline:
        time.sleep(2)
        update = broker.get_order(order_id)
        if update.status == "FILLED":
            return update
        if update.status in {"FAILED", "CANCELLED"}:
            raise RuntimeError(f"sandbox {side.lower()} ended with status {update.status}")
        if time.monotonic() >= next_reprice:
            quote = broker.option_quote(contract)
            raw_price = quote.ask if side == "BUY" else quote.bid
            limit_price = option_limit_price(raw_price, penny_program=quote.penny_program)
            if limit_price > 0:
                try:
                    broker.replace_order(order_id, quantity=1, limit_price=limit_price)
                except Exception as exc:
                    print(
                        f"Sandbox {side.lower()} reprice was rejected; "
                        f"continuing to monitor the original order: {exc}"
                    )
            next_reprice = time.monotonic() + 15
    broker.cancel_order(order_id)
    raise RuntimeError(f"sandbox {side.lower()} did not fill within {timeout}s and was cancelled")


def _require_regular_market_session() -> None:
    import pandas_market_calendars as mcal

    now = datetime.now(timezone.utc)
    local_date = now.astimezone(NEW_YORK).date()
    schedule = mcal.get_calendar("NYSE").schedule(start_date=local_date, end_date=local_date)
    if schedule.empty:
        raise RuntimeError("sandbox execution smoke test requires an NYSE trading session")
    market_open = schedule.iloc[0]["market_open"].to_pydatetime()
    market_close = schedule.iloc[0]["market_close"].to_pydatetime()
    if not market_open <= now <= market_close:
        raise RuntimeError("sandbox execution smoke test requires regular market hours")


def _print_accounts(environment: str, accounts: list[dict[str, object]]) -> None:
    print(f"Webull {environment} authentication succeeded. Available account(s):")
    for account in accounts:
        account_id = str(account.get("account_id") or account.get("id") or "unknown")
        account_type = str(account.get("account_type") or account.get("type") or "unknown")
        account_number = str(account.get("account_number") or account.get("account_no") or "")
        masked_number = f"...{account_number[-4:]}" if account_number else "not returned"
        print(f"  Account ID: {account_id} | type: {account_type} | number: {masked_number}")


if __name__ == "__main__":
    raise SystemExit(main())
