from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sentiment_analysis.trading.broker import BrokerError, choose_option_contract
from sentiment_analysis.trading.models import OptionContract, OrderUpdate


class PaperOptionsBroker:
    """Local fill simulator backed by delayed yfinance option quotes."""

    def __init__(self, *, starting_cash: float = 100_000.0) -> None:
        self.starting_cash = max(0.0, starting_cash)
        self._orders: dict[str, OrderUpdate] = {}
        self._contracts: dict[str, OptionContract] = {}
        self._positions: dict[str, int] = {}

    def cash_balance(self) -> float:
        return self.starting_cash

    def underlying_price(self, ticker: str) -> float:
        import yfinance as yf

        return _spot_price(yf.Ticker(ticker))

    def select_option(
        self,
        ticker: str,
        option_type: str,
        *,
        minimum_expiration: date,
        maximum_expiration: date | None = None,
        preferred_expiration: date | None,
        max_contract_cost: float,
        max_spread_pct: float,
        target_itm_pct: float = 0.0,
        target_delta: float = 0.70,
        max_otm_pct: float = 0.0,
        max_breakeven_move_pct: float = 1.0,
        min_volume: int = 0,
        min_open_interest: int = 0,
    ) -> OptionContract | None:
        import yfinance as yf

        ticker_obj = yf.Ticker(ticker)
        expirations = sorted(
            date.fromisoformat(value)
            for value in ticker_obj.options
            if date.fromisoformat(value) >= minimum_expiration
            and (maximum_expiration is None or date.fromisoformat(value) <= maximum_expiration)
        )
        if not expirations:
            return None
        expiration = (
            preferred_expiration
            if preferred_expiration in expirations
            else expirations[0]
        )
        spot = _spot_price(ticker_obj)
        chain = ticker_obj.option_chain(expiration.isoformat())
        frame = chain.calls if option_type.upper() == "CALL" else chain.puts
        contracts = [
            _contract_from_yfinance(ticker, option_type, expiration, spot, row)
            for row in frame.to_dict("records")
        ]
        selected = choose_option_contract(
            contracts,
            max_contract_cost=max_contract_cost,
            max_spread_pct=max_spread_pct,
            target_itm_pct=target_itm_pct,
            target_delta=target_delta,
            max_otm_pct=max_otm_pct,
            max_breakeven_move_pct=max_breakeven_move_pct,
            min_volume=min_volume,
            min_open_interest=min_open_interest,
        )
        if selected:
            self._contracts[selected.symbol] = selected
        return selected

    def option_quote(self, contract: OptionContract) -> OptionContract:
        import yfinance as yf

        ticker_obj = yf.Ticker(contract.ticker)
        chain = ticker_obj.option_chain(contract.expiration.isoformat())
        frame = chain.calls if contract.option_type == "CALL" else chain.puts
        matches = frame[frame["contractSymbol"] == contract.symbol]
        if matches.empty:
            raise BrokerError(f"no paper quote for {contract.symbol}")
        refreshed = _contract_from_yfinance(
            contract.ticker,
            contract.option_type,
            contract.expiration,
            _spot_price(ticker_obj),
            matches.iloc[0].to_dict(),
        )
        self._contracts[refreshed.symbol] = refreshed
        return refreshed

    def option_universe(
        self,
        ticker: str,
        *,
        expirations: list[date],
        max_otm_pct: float = 0.05,
        max_itm_pct: float = 0.25,
        limit_per_type: int = 24,
    ) -> list[OptionContract]:
        import yfinance as yf

        ticker_obj = yf.Ticker(ticker)
        available = {date.fromisoformat(value) for value in ticker_obj.options}
        spot = _spot_price(ticker_obj)
        contracts: list[OptionContract] = []
        for expiration in sorted(set(expirations) & available):
            chain = ticker_obj.option_chain(expiration.isoformat())
            for option_type, frame in (("CALL", chain.calls), ("PUT", chain.puts)):
                candidates = [
                    _contract_from_yfinance(
                        ticker,
                        option_type,
                        expiration,
                        spot,
                        row,
                    )
                    for row in frame.to_dict("records")
                ]
                candidates = [
                    contract
                    for contract in candidates
                    if -max_otm_pct <= contract.itm_pct <= max_itm_pct
                ]
                candidates.sort(
                    key=lambda contract: min(
                        abs(contract.itm_pct - target)
                        for target in (0.0, 0.05, 0.10)
                    )
                )
                contracts.extend(candidates[: max(1, limit_per_type)])
        self._contracts.update({contract.symbol: contract for contract in contracts})
        return contracts

    def option_quotes(
        self,
        contracts: list[OptionContract],
    ) -> list[OptionContract]:
        refreshed: list[OptionContract] = []
        for contract in contracts:
            try:
                refreshed.append(self.option_quote(contract))
            except BrokerError:
                continue
        return refreshed

    def place_option_order(
        self,
        contract: OptionContract,
        *,
        side: str,
        quantity: int,
        limit_price: float,
        client_order_id: str,
    ) -> OrderUpdate:
        position_change = quantity if side.upper() == "BUY" else -quantity
        self._positions[contract.symbol] = max(
            0,
            self._positions.get(contract.symbol, 0) + position_change,
        )
        update = OrderUpdate(
            client_order_id=client_order_id,
            status="FILLED",
            filled_quantity=quantity,
            filled_price=limit_price,
            raw_status="PAPER_FILLED",
        )
        self._orders[client_order_id] = update
        return update

    def get_order(self, client_order_id: str) -> OrderUpdate:
        try:
            return self._orders[client_order_id]
        except KeyError as exc:
            raise BrokerError(f"paper order {client_order_id} is not in this process") from exc

    def replace_order(self, client_order_id: str, *, quantity: int, limit_price: float) -> OrderUpdate:
        update = OrderUpdate(
            client_order_id=client_order_id,
            status="FILLED",
            filled_quantity=quantity,
            filled_price=limit_price,
            raw_status="PAPER_REPLACED_AND_FILLED",
        )
        self._orders[client_order_id] = update
        return update

    def cancel_order(self, client_order_id: str) -> None:
        current = self._orders.get(client_order_id)
        self._orders[client_order_id] = OrderUpdate(
            client_order_id=client_order_id,
            status="CANCELLED",
            filled_quantity=current.filled_quantity if current else 0,
            filled_price=current.filled_price if current else None,
            raw_status="PAPER_CANCELLED",
        )

    def option_position_quantity(self, contract: OptionContract) -> int:
        return self._positions.get(contract.symbol, 0)

    def option_position_mark(self, contract: OptionContract) -> float | None:
        quote = self.option_quote(contract)
        return quote.bid if quote.bid > 0 else None

    def latest_option_close(
        self,
        contract: OptionContract,
        *,
        since: datetime | None = None,
    ) -> OrderUpdate | None:
        return None


def _spot_price(ticker_obj: Any) -> float:
    try:
        price = float(ticker_obj.fast_info["last_price"])
    except Exception as exc:
        raise BrokerError("could not load paper underlying quote") from exc
    if price <= 0:
        raise BrokerError("paper underlying quote is not positive")
    return price


def _contract_from_yfinance(
    ticker: str,
    option_type: str,
    expiration: date,
    spot: float,
    row: dict[str, Any],
) -> OptionContract:
    return OptionContract(
        symbol=str(row.get("contractSymbol") or ""),
        ticker=ticker,
        option_type=option_type.upper(),
        expiration=expiration,
        strike=_float(row.get("strike")),
        bid=_float(row.get("bid")),
        ask=_float(row.get("ask")),
        underlying_price=spot,
        volume=_int(row.get("volume")),
        open_interest=_int(row.get("openInterest")),
        implied_volatility=_optional_float(row.get("impliedVolatility")),
    )


def _float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number == number else 0.0


def _int(value: object) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None
