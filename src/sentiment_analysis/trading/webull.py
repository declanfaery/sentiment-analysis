from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sentiment_analysis.trading.broker import BrokerError, choose_option_contract
from sentiment_analysis.trading.models import OptionContract, OrderUpdate


def list_webull_accounts(
    *,
    app_key: str,
    app_secret: str,
    environment: str = "sandbox",
) -> list[dict[str, Any]]:
    """Authenticate read-only and return the accounts available to an app key."""
    if not app_key or not app_secret:
        raise BrokerError("WEBULL_APP_KEY and WEBULL_APP_SECRET are required")
    trade_client, _ = _build_clients(app_key, app_secret, environment)
    payload = _response_json(trade_client.account_v2.get_account_list(), "account list")
    rows = _rows(payload)
    if not rows:
        raise BrokerError("Webull returned no accounts for these credentials")
    return rows


class WebullOptionsBroker:
    """Official Webull OpenAPI adapter for single-leg long options."""

    def __init__(
        self,
        *,
        app_key: str,
        app_secret: str,
        account_id: str,
        environment: str = "sandbox",
        allow_live: bool = False,
    ) -> None:
        if not app_key or not app_secret or not account_id:
            raise BrokerError("WEBULL_APP_KEY, WEBULL_APP_SECRET, and WEBULL_ACCOUNT_ID are required")
        if environment == "live" and not allow_live:
            raise BrokerError("live Webull trading is locked; set ENABLE_LIVE_TRADING=I_UNDERSTAND_THE_RISK")

        self.trade_client, self.data_client = _build_clients(app_key, app_secret, environment)
        self.account_id = account_id
        self.environment = environment
        self._cash_cache: tuple[float, float] | None = None

    def account_summary(self) -> dict[str, object]:
        accounts = self._json(self.trade_client.account_v2.get_account_list(), "account list")
        balance = self._json(
            self.trade_client.account_v2.get_account_balance(self.account_id),
            "account balance",
        )
        return {"environment": self.environment, "accounts": accounts, "balance": balance}

    def cash_balance(self) -> float:
        now = time.monotonic()
        if self._cash_cache and now - self._cash_cache[0] < 2:
            return self._cash_cache[1]
        payload = self._json(
            self.trade_client.account_v2.get_account_balance(self.account_id),
            "account balance",
        )
        if not isinstance(payload, dict):
            raise BrokerError("Webull account balance response was not an object")
        cash = _float(payload.get("total_cash_balance"))
        usd = next(
            (
                row
                for row in payload.get("account_currency_assets", [])
                if isinstance(row, dict) and row.get("currency") == "USD"
            ),
            {},
        )
        option_buying_power = _optional_float(usd.get("option_buying_power"))
        if option_buying_power is not None:
            available = max(0.0, min(cash, option_buying_power))
        else:
            available = max(0.0, cash)
        self._cash_cache = (now, available)
        return available

    def underlying_price(self, ticker: str) -> float:
        rows = self._json(
            self.data_client.market_data.get_snapshot(ticker, "US_STOCK"),
            f"{ticker} stock snapshot",
        )
        price = _float(_first_row(rows).get("price"))
        if price <= 0:
            raise BrokerError(f"Webull returned no live underlying price for {ticker}")
        return price

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
        spot_rows = self._json(
            self.data_client.market_data.get_snapshot(ticker, "US_STOCK"),
            f"{ticker} stock snapshot",
        )
        spot_row = _first_row(spot_rows)
        spot = _float(spot_row.get("price"))
        if spot <= 0:
            raise BrokerError(f"Webull returned no live underlying price for {ticker}")

        for expiration in _candidate_expirations(
            minimum_expiration,
            preferred_expiration,
            maximum_expiration,
        ):
            response = self.data_client.instrument.get_option_contracts(
                category="US_OPTION",
                underlying_symbols=ticker,
                status="LISTING",
                start_date=expiration.isoformat(),
                end_date=expiration.isoformat(),
                option_type=option_type.upper(),
                show_deliverables=False,
                page_size=1000,
            )
            rows = self._json(response, f"{ticker} option contracts")
            static_contracts = [
                row
                for row in _rows(rows)
                if str(row.get("option_type", "")).upper() == option_type.upper()
                and str(row.get("tradable_status", "OC")).upper() == "OC"
                and str(row.get("def_type", "STANDARD")).upper() == "STANDARD"
                and _int(row.get("multiplier"), 100) == 100
            ]
            if not static_contracts:
                continue
            static_contracts = _prioritize_static_contracts(
                static_contracts,
                option_type=option_type,
                spot=spot,
                target_itm_pct=target_itm_pct,
                max_otm_pct=max_otm_pct,
            )

            snapshots: dict[str, dict[str, Any]] = {}
            symbols = [str(row.get("symbol") or "") for row in static_contracts if row.get("symbol")]
            for chunk in _chunks(symbols, 20):
                quote_rows = self._json(
                    self.data_client.option_market_data.get_option_snapshot(chunk, "US_OPTION"),
                    f"{ticker} option snapshots",
                )
                snapshots.update(
                    {
                        str(row.get("symbol")): row
                        for row in _rows(quote_rows)
                        if row.get("symbol")
                    }
                )

            contracts = [
                _merge_contract(ticker, option_type, expiration, spot, row, snapshots.get(str(row.get("symbol"))))
                for row in static_contracts
                if snapshots.get(str(row.get("symbol")))
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
                return selected
        return None

    def option_quote(self, contract: OptionContract) -> OptionContract:
        rows = self._json(
            self.data_client.option_market_data.get_option_snapshot(contract.symbol, "US_OPTION"),
            f"{contract.symbol} option snapshot",
        )
        row = _first_row(rows)
        return OptionContract(
            **{
                **contract.__dict__,
                "bid": _float(row.get("bid")),
                "ask": _float(row.get("ask")),
                "volume": _int(row.get("volume")),
                "open_interest": _int(row.get("open_interest")),
                "quote_time_ms": _int(row.get("quote_time")) or None,
            }
        )

    def option_universe(
        self,
        ticker: str,
        *,
        expirations: list[date],
        max_otm_pct: float = 0.05,
        max_itm_pct: float = 0.25,
        limit_per_type: int = 24,
    ) -> list[OptionContract]:
        spot = self.underlying_price(ticker)
        contracts: list[OptionContract] = []
        for expiration in sorted(set(expirations)):
            rows = self._json(
                self.data_client.instrument.get_option_contracts(
                    category="US_OPTION",
                    underlying_symbols=ticker,
                    status="LISTING",
                    start_date=expiration.isoformat(),
                    end_date=expiration.isoformat(),
                    strike_price_gte=f"{spot * (1 - max_itm_pct):.4f}",
                    strike_price_lte=f"{spot * (1 + max_itm_pct):.4f}",
                    show_deliverables=False,
                    page_size=1000,
                ),
                f"{ticker} research option contracts",
            )
            static_contracts = _research_static_contracts(
                _rows(rows),
                spot=spot,
                max_otm_pct=max_otm_pct,
                max_itm_pct=max_itm_pct,
                limit_per_type=limit_per_type,
            )
            snapshots: dict[str, dict[str, Any]] = {}
            symbols = [
                str(row.get("symbol") or "")
                for row in static_contracts
                if row.get("symbol")
            ]
            for chunk in _chunks(symbols, 20):
                quote_rows = self._json(
                    self.data_client.option_market_data.get_option_snapshot(
                        chunk,
                        "US_OPTION",
                    ),
                    f"{ticker} research option snapshots",
                )
                snapshots.update(
                    {
                        str(row.get("symbol")): row
                        for row in _rows(quote_rows)
                        if row.get("symbol")
                    }
                )
            contracts.extend(
                _merge_contract(
                    ticker,
                    str(row.get("option_type") or "").upper(),
                    expiration,
                    spot,
                    row,
                    snapshots.get(str(row.get("symbol"))),
                )
                for row in static_contracts
                if snapshots.get(str(row.get("symbol")))
            )
        return contracts

    def option_quotes(
        self,
        contracts: list[OptionContract],
    ) -> list[OptionContract]:
        refreshed: list[OptionContract] = []
        for ticker in sorted({contract.ticker for contract in contracts}):
            ticker_contracts = [
                contract for contract in contracts if contract.ticker == ticker
            ]
            spot = self.underlying_price(ticker)
            snapshots: dict[str, dict[str, Any]] = {}
            for chunk in _chunks(
                [contract.symbol for contract in ticker_contracts],
                20,
            ):
                rows = self._json(
                    self.data_client.option_market_data.get_option_snapshot(
                        chunk,
                        "US_OPTION",
                    ),
                    f"{ticker} research option snapshots",
                )
                snapshots.update(
                    {
                        str(row.get("symbol")): row
                        for row in _rows(rows)
                        if row.get("symbol")
                    }
                )
            for contract in ticker_contracts:
                row = snapshots.get(contract.symbol)
                if row is None:
                    continue
                refreshed.append(_refresh_contract(contract, spot, row))
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
        order = _option_order(
            contract,
            side=side,
            quantity=quantity,
            limit_price=limit_price,
            client_order_id=client_order_id,
        )
        self._preview_order(order, side)
        self._json(
            self.trade_client.order_v3.place_order(self.account_id, [order]),
            f"place {side.lower()} order",
        )
        return OrderUpdate(client_order_id=client_order_id, status="SUBMITTED", raw_status="SUBMITTED")

    def preview_option_order(
        self,
        contract: OptionContract,
        *,
        side: str,
        quantity: int,
        limit_price: float,
        client_order_id: str,
    ) -> None:
        order = _option_order(
            contract,
            side=side,
            quantity=quantity,
            limit_price=limit_price,
            client_order_id=client_order_id,
        )
        self._preview_order(order, side)

    def _preview_order(self, order: dict[str, object], side: str) -> None:
        self._json(
            self.trade_client.order_v3.preview_order(self.account_id, [order]),
            f"preview {side.lower()} order",
        )

    def get_order(self, client_order_id: str) -> OrderUpdate:
        row = self._order_detail(client_order_id)
        raw_status = str(row.get("status") or "UNKNOWN").upper()
        return OrderUpdate(
            client_order_id=client_order_id,
            status=raw_status,
            filled_quantity=_int(row.get("filled_quantity")),
            filled_price=_optional_float(row.get("filled_price")),
            raw_status=raw_status,
        )

    def replace_order(self, client_order_id: str, *, quantity: int, limit_price: float) -> OrderUpdate:
        row = self._order_detail(client_order_id)
        modify_order = _replacement_order(
            client_order_id,
            quantity=quantity,
            limit_price=limit_price,
            order_row=row,
        )
        self._json(
            self.trade_client.order_v3.replace_order(
                self.account_id,
                [modify_order],
            ),
            "replace order",
        )
        return self.get_order(client_order_id)

    def cancel_order(self, client_order_id: str) -> None:
        self._json(
            self.trade_client.order_v3.cancel_order(self.account_id, client_order_id),
            "cancel order",
        )

    def option_position_quantity(self, contract: OptionContract) -> int:
        payload = self._json(
            self.trade_client.account_v2.get_account_position(self.account_id),
            "account positions",
        )
        for row in _rows(payload):
            if str(row.get("instrument_type") or "").upper() != "OPTION":
                continue
            for leg in row.get("legs", []):
                if not isinstance(leg, dict):
                    continue
                if (
                    str(leg.get("symbol") or "").upper() == contract.ticker.upper()
                    and str(leg.get("option_type") or "").upper()
                    == contract.option_type.upper()
                    and str(leg.get("option_expire_date") or "")
                    == contract.expiration.isoformat()
                    and abs(
                        _float(
                            leg.get("option_exercise_price")
                            or leg.get("strike_price")
                        )
                        - contract.strike
                    )
                    < 0.001
                ):
                    return max(0, _int(row.get("quantity") or leg.get("quantity")))
        return 0

    def option_position_mark(self, contract: OptionContract) -> float | None:
        payload = self._json(
            self.trade_client.account_v2.get_account_position(self.account_id),
            "account positions",
        )
        for row in _rows(payload):
            if str(row.get("instrument_type") or "").upper() != "OPTION":
                continue
            for leg in row.get("legs", []):
                if not isinstance(leg, dict):
                    continue
                if (
                    str(leg.get("symbol") or "").upper() == contract.ticker.upper()
                    and str(leg.get("option_type") or "").upper()
                    == contract.option_type.upper()
                    and str(leg.get("option_expire_date") or "")
                    == contract.expiration.isoformat()
                    and abs(
                        _float(leg.get("option_exercise_price")) - contract.strike
                    )
                    < 0.001
                ):
                    mark = _optional_float(leg.get("last_price") or row.get("last_price"))
                    return mark if mark is not None and mark > 0 else None
        return None

    def latest_option_close(
        self,
        contract: OptionContract,
        *,
        since: datetime | None = None,
    ) -> OrderUpdate | None:
        start = (since or datetime.now(timezone.utc) - timedelta(days=7)).date()
        # Webull treats end_date as an exclusive boundary.
        end = datetime.now(timezone.utc).date() + timedelta(days=1)
        payload = self._json(
            self.trade_client.order_v3.get_order_history(
                self.account_id,
                page_size=100,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
            ),
            "order history",
        )
        return _latest_option_close_from_history(payload, contract, since=since)

    def _order_detail(self, client_order_id: str) -> dict[str, Any]:
        payload = self._json(
            self.trade_client.order_v3.get_order_detail(self.account_id, client_order_id),
            "order detail",
        )
        rows = payload.get("orders", []) if isinstance(payload, dict) else payload
        return _first_row(rows)

    @staticmethod
    def _json(response: Any, operation: str) -> Any:
        return _response_json(response, operation)


def _build_clients(app_key: str, app_secret: str, environment: str) -> tuple[Any, Any]:
    if environment not in {"sandbox", "live"}:
        raise BrokerError("Webull environment must be sandbox or live")
    try:
        from webull.core.client import ApiClient
        from webull.data.data_client import DataClient
        from webull.trade.trade_client import TradeClient
    except ImportError as exc:
        raise BrokerError("install webull-openapi-python-sdk before using Webull") from exc

    # The SDK logs signed request headers, including the App Key, on HTTP errors.
    endpoint = "api.sandbox.webull.com" if environment == "sandbox" else "api.webull.com"
    api_client = ApiClient(app_key, app_secret, "us")
    api_client.add_endpoint("us", endpoint)
    trade_client = TradeClient(api_client)
    data_client = DataClient(api_client)
    logging.getLogger("webull").setLevel(logging.CRITICAL)
    logging.getLogger("webull.core.client").setLevel(logging.CRITICAL)
    return trade_client, data_client


def _response_json(response: Any, operation: str) -> Any:
    status_code = int(getattr(response, "status_code", 0))
    if status_code != 200:
        detail = getattr(response, "text", "")
        raise BrokerError(f"Webull {operation} failed ({status_code}): {detail[:500]}")
    try:
        return response.json()
    except Exception as exc:
        raise BrokerError(f"Webull {operation} returned invalid JSON") from exc


def _option_order(
    contract: OptionContract,
    *,
    side: str,
    quantity: int,
    limit_price: float,
    client_order_id: str,
) -> dict[str, object]:
    clean_side = side.upper()
    position_intent = "BUY_TO_OPEN" if clean_side == "BUY" else "SELL_TO_CLOSE"
    leg = {
        "side": clean_side,
        "quantity": str(quantity),
        "symbol": contract.ticker,
        "strike_price": f"{contract.strike:.3f}".rstrip("0").rstrip("."),
        "option_expire_date": contract.expiration.isoformat(),
        "instrument_type": "OPTION",
        "option_type": contract.option_type,
        "market": "US",
    }
    return {
        "client_order_id": client_order_id[:32],
        "combo_type": "NORMAL",
        "option_strategy": "SINGLE",
        "order_type": "LIMIT",
        "limit_price": f"{limit_price:.2f}",
        "quantity": str(quantity),
        "side": clean_side,
        "time_in_force": "DAY",
        "entrust_type": "QTY",
        "instrument_type": "OPTION",
        "market": "US",
        "symbol": contract.ticker,
        "position_intent": position_intent,
        "legs": [leg],
    }


def _replacement_order(
    client_order_id: str,
    *,
    quantity: int,
    limit_price: float,
    order_row: dict[str, Any],
) -> dict[str, object]:
    replacement: dict[str, object] = {
        "client_order_id": client_order_id,
        "quantity": str(quantity),
        "limit_price": f"{limit_price:.2f}",
    }
    if str(order_row.get("instrument_type") or "").upper() == "OPTION":
        legs = [
            {"id": str(leg["id"]), "quantity": str(quantity)}
            for leg in order_row.get("legs", [])
            if isinstance(leg, dict) and leg.get("id")
        ]
        if not legs:
            raise BrokerError("Webull option order detail did not include leg IDs")
        replacement["legs"] = legs
    return replacement


def _latest_option_close_from_history(
    payload: Any,
    contract: OptionContract,
    *,
    since: datetime | None = None,
) -> OrderUpdate | None:
    earliest_ms = int(since.timestamp() * 1000) if since is not None else 0
    matches: list[tuple[int, str, dict[str, Any]]] = []
    for group in _rows(payload):
        parent_id = str(group.get("client_order_id") or "")
        orders = group.get("orders")
        if not isinstance(orders, list):
            orders = [group]
        for order in orders:
            if not isinstance(order, dict):
                continue
            if str(order.get("instrument_type") or "").upper() != "OPTION":
                continue
            if str(order.get("side") or "").upper() != "SELL":
                continue
            if str(order.get("status") or "").upper() != "FILLED":
                continue
            if not any(
                _option_leg_matches(leg, contract)
                for leg in order.get("legs", [])
                if isinstance(leg, dict)
            ):
                continue
            filled_time = _int(order.get("filled_time"))
            if filled_time < earliest_ms:
                continue
            client_order_id = str(order.get("client_order_id") or parent_id)
            matches.append((filled_time, client_order_id, order))
    if not matches:
        return None
    _, client_order_id, order = max(matches, key=lambda row: row[0])
    return OrderUpdate(
        client_order_id=client_order_id,
        status="FILLED",
        filled_quantity=_int(order.get("filled_quantity")),
        filled_price=_optional_float(order.get("filled_price")),
        raw_status="EXTERNAL_FILLED",
    )


def _option_leg_matches(leg: dict[str, Any], contract: OptionContract) -> bool:
    return (
        str(leg.get("symbol") or "").upper() == contract.ticker.upper()
        and str(leg.get("option_type") or "").upper() == contract.option_type.upper()
        and str(leg.get("option_expire_date") or "") == contract.expiration.isoformat()
        and abs(_float(leg.get("strike_price")) - contract.strike) < 0.001
    )


def _merge_contract(
    ticker: str,
    option_type: str,
    expiration: date,
    spot: float,
    static: dict[str, Any],
    quote: dict[str, Any] | None,
) -> OptionContract:
    quote = quote or {}
    return OptionContract(
        symbol=str(static.get("symbol") or quote.get("symbol") or ""),
        ticker=ticker,
        option_type=option_type.upper(),
        expiration=expiration,
        strike=_float(static.get("strike_price") or quote.get("strike_price")),
        bid=_float(quote.get("bid")),
        ask=_float(quote.get("ask")),
        underlying_price=spot,
        multiplier=_int(static.get("multiplier"), 100),
        volume=_int(quote.get("volume")),
        open_interest=_int(quote.get("open_interest")),
        penny_program=bool(static.get("ppind", True)),
        delta=_optional_float(quote.get("delta")),
        vega=_optional_float(quote.get("vega")),
        implied_volatility=_optional_float(quote.get("imp_vol")),
        quote_time_ms=_int(quote.get("quote_time")) or None,
    )


def _prioritize_static_contracts(
    contracts: list[dict[str, Any]],
    *,
    option_type: str,
    spot: float,
    target_itm_pct: float,
    max_otm_pct: float,
    limit: int = 40,
) -> list[dict[str, Any]]:
    def itm_pct(row: dict[str, Any]) -> float:
        strike = _float(row.get("strike_price"))
        if spot <= 0:
            return float("-inf")
        if option_type.upper() == "PUT":
            return (strike - spot) / spot
        return (spot - strike) / spot

    in_the_money = [
        row for row in contracts if -max(0.0, max_otm_pct) <= itm_pct(row) <= 0.40
    ]
    return sorted(
        in_the_money,
        key=lambda row: abs(itm_pct(row) - target_itm_pct),
    )[: max(1, limit)]


def _research_static_contracts(
    contracts: list[dict[str, Any]],
    *,
    spot: float,
    max_otm_pct: float,
    max_itm_pct: float,
    limit_per_type: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for option_type in ("CALL", "PUT"):
        rows: list[tuple[float, dict[str, Any]]] = []
        for row in contracts:
            if str(row.get("option_type") or "").upper() != option_type:
                continue
            if str(row.get("tradable_status", "OC")).upper() != "OC":
                continue
            if str(row.get("def_type", "STANDARD")).upper() != "STANDARD":
                continue
            if _int(row.get("multiplier"), 100) != 100 or spot <= 0:
                continue
            strike = _float(row.get("strike_price"))
            itm_pct = (
                (strike - spot) / spot
                if option_type == "PUT"
                else (spot - strike) / spot
            )
            if not -max_otm_pct <= itm_pct <= max_itm_pct:
                continue
            distance = min(abs(itm_pct - target) for target in (0.0, 0.05, 0.10))
            rows.append((distance, row))
        selected.extend(
            row
            for _, row in sorted(rows, key=lambda item: item[0])[
                : max(1, limit_per_type)
            ]
        )
    return selected


def _refresh_contract(
    contract: OptionContract,
    spot: float,
    row: dict[str, Any],
) -> OptionContract:
    delta = _optional_float(row.get("delta"))
    vega = _optional_float(row.get("vega"))
    implied_volatility = _optional_float(row.get("imp_vol"))
    return OptionContract(
        **{
            **contract.__dict__,
            "bid": _float(row.get("bid")),
            "ask": _float(row.get("ask")),
            "underlying_price": spot,
            "volume": _int(row.get("volume")),
            "open_interest": _int(row.get("open_interest")),
            "delta": contract.delta if delta is None else delta,
            "vega": contract.vega if vega is None else vega,
            "implied_volatility": (
                contract.implied_volatility
                if implied_volatility is None
                else implied_volatility
            ),
            "quote_time_ms": _int(row.get("quote_time")) or None,
        }
    )


def _candidate_expirations(
    minimum: date,
    preferred: date | None,
    maximum: date | None = None,
) -> list[date]:
    values: list[date] = []
    if preferred and preferred >= minimum and (maximum is None or preferred <= maximum):
        values.append(preferred)
    cursor = minimum
    while cursor.weekday() != 4:
        cursor += timedelta(days=1)
    for offset in range(0, 36, 7):
        expiration = cursor + timedelta(days=offset)
        if maximum is not None and expiration > maximum:
            break
        if expiration not in values:
            values.append(expiration)
    return values


def _rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "items", "contracts", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def _first_row(payload: Any) -> dict[str, Any]:
    rows = _rows(payload)
    if rows:
        return rows[0]
    if isinstance(payload, dict) and not payload.get("error_code"):
        return payload
    raise BrokerError("Webull response contained no rows")


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _float(value: object) -> float:
    parsed = _optional_float(value)
    return parsed or 0.0


def _optional_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default
