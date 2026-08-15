from __future__ import annotations

from datetime import date, datetime
from typing import Protocol

from sentiment_analysis.trading.models import OptionContract, OrderUpdate


class BrokerError(RuntimeError):
    pass


class OptionsBroker(Protocol):
    def cash_balance(self) -> float:
        ...

    def underlying_price(self, ticker: str) -> float:
        ...

    def select_option(
        self,
        ticker: str,
        option_type: str,
        *,
        minimum_expiration: date,
        maximum_expiration: date,
        preferred_expiration: date | None,
        max_contract_cost: float,
        max_spread_pct: float,
        target_itm_pct: float,
        target_delta: float,
        max_otm_pct: float,
        max_breakeven_move_pct: float,
        min_volume: int,
        min_open_interest: int,
    ) -> OptionContract | None:
        ...

    def option_quote(self, contract: OptionContract) -> OptionContract:
        ...

    def option_universe(
        self,
        ticker: str,
        *,
        expirations: list[date],
        max_otm_pct: float = 0.05,
        max_itm_pct: float = 0.25,
        limit_per_type: int = 24,
    ) -> list[OptionContract]:
        ...

    def option_quotes(
        self,
        contracts: list[OptionContract],
    ) -> list[OptionContract]:
        ...

    def place_option_order(
        self,
        contract: OptionContract,
        *,
        side: str,
        quantity: int,
        limit_price: float,
        client_order_id: str,
    ) -> OrderUpdate:
        ...

    def get_order(self, client_order_id: str) -> OrderUpdate:
        ...

    def replace_order(self, client_order_id: str, *, quantity: int, limit_price: float) -> OrderUpdate:
        ...

    def cancel_order(self, client_order_id: str) -> None:
        ...

    def option_position_quantity(self, contract: OptionContract) -> int:
        ...

    def option_position_mark(self, contract: OptionContract) -> float | None:
        ...

    def latest_option_close(
        self,
        contract: OptionContract,
        *,
        since: datetime | None = None,
    ) -> OrderUpdate | None:
        ...


def choose_option_contract(
    contracts: list[OptionContract],
    *,
    max_contract_cost: float,
    max_spread_pct: float,
    target_itm_pct: float = 0.0,
    target_delta: float = 0.70,
    max_otm_pct: float = 0.0,
    max_breakeven_move_pct: float = 1.0,
    min_volume: int = 0,
    min_open_interest: int = 0,
) -> OptionContract | None:
    eligible = [
        contract
        for contract in contracts
        if contract.ask > 0
        and contract.bid > 0
        and contract.ask_cost <= max_contract_cost
        and contract.spread_pct <= max_spread_pct
        and contract.itm_pct >= -max(0.0, max_otm_pct)
        and contract.breakeven_move_pct <= max_breakeven_move_pct
        and contract.volume >= min_volume
        and contract.open_interest >= min_open_interest
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda contract: (
            abs(abs(contract.delta) - target_delta)
            if contract.delta is not None
            else abs(contract.itm_pct - target_itm_pct),
            abs(contract.itm_pct - target_itm_pct),
            contract.breakeven_move_pct,
            contract.spread_pct,
            -contract.open_interest,
            -contract.volume,
        ),
    )


def option_limit_price(price: float, *, penny_program: bool = True) -> float:
    if price <= 0:
        return 0.0
    # Webull OpenAPI enforces nickel ticks for every option premium at $3+,
    # including contracts the reference data labels as penny-program eligible.
    increment = 0.01 if price < 3 else 0.05
    ticks = round(price / increment)
    return round(max(increment, ticks * increment), 2)
