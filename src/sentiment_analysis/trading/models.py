from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime


ACTIVE_TRADE_STATUSES = frozenset(
    {
        "planned",
        "entry_submitting",
        "entry_submitted",
        "open",
        "exit_submitting",
        "exit_submitted",
    }
)


@dataclass(frozen=True)
class OptionContract:
    symbol: str
    ticker: str
    option_type: str
    expiration: date
    strike: float
    bid: float
    ask: float
    underlying_price: float
    multiplier: int = 100
    volume: int = 0
    open_interest: int = 0
    penny_program: bool = True
    delta: float | None = None
    vega: float | None = None
    implied_volatility: float | None = None
    quote_time_ms: int | None = None

    @property
    def ask_cost(self) -> float:
        return self.ask * self.multiplier

    @property
    def spread_pct(self) -> float:
        midpoint = (self.bid + self.ask) / 2
        if midpoint <= 0:
            return float("inf")
        return max(0.0, self.ask - self.bid) / midpoint

    @property
    def intrinsic_value(self) -> float:
        if self.option_type.upper() == "PUT":
            return max(0.0, self.strike - self.underlying_price)
        return max(0.0, self.underlying_price - self.strike)

    @property
    def extrinsic_value(self) -> float:
        return max(0.0, self.ask - self.intrinsic_value)

    @property
    def breakeven_move_pct(self) -> float:
        if self.underlying_price <= 0:
            return float("inf")
        return self.extrinsic_value / self.underlying_price

    @property
    def itm_pct(self) -> float:
        if self.underlying_price <= 0:
            return float("-inf")
        if self.option_type.upper() == "PUT":
            return (self.strike - self.underlying_price) / self.underlying_price
        return (self.underlying_price - self.strike) / self.underlying_price


@dataclass(frozen=True)
class OrderUpdate:
    client_order_id: str
    status: str
    filled_quantity: int = 0
    filled_price: float | None = None
    raw_status: str | None = None


@dataclass(frozen=True)
class TradePlan:
    id: str
    scan_run_id: str
    ticker: str
    direction: str
    option_type: str
    earnings_date: date
    earnings_timing: str
    entry_at: datetime
    exit_at: datetime
    allocation_pct: float
    recommendation: str
    recommendation_score: int
    status: str = "planned"
    skip_reason: str | None = None
    option_symbol: str | None = None
    option_expiration: date | None = None
    strike_price: float | None = None
    contract_multiplier: int = 100
    quantity: int = 0
    entry_order_id: str | None = None
    exit_order_id: str | None = None
    entry_limit_price: float | None = None
    entry_fill_price: float | None = None
    exit_limit_price: float | None = None
    exit_fill_price: float | None = None
    filled_quantity: int = 0
    last_error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def with_updates(self, **changes: object) -> "TradePlan":
        return replace(self, **changes)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "scan_run_id": self.scan_run_id,
            "ticker": self.ticker,
            "direction": self.direction,
            "option_type": self.option_type,
            "earnings_date": self.earnings_date.isoformat(),
            "earnings_timing": self.earnings_timing,
            "entry_at": self.entry_at.isoformat(),
            "exit_at": self.exit_at.isoformat(),
            "allocation_pct": self.allocation_pct,
            "recommendation": self.recommendation,
            "recommendation_score": self.recommendation_score,
            "status": self.status,
            "skip_reason": self.skip_reason,
            "option_symbol": self.option_symbol,
            "option_expiration": self.option_expiration.isoformat()
            if self.option_expiration
            else None,
            "strike_price": self.strike_price,
            "quantity": self.quantity,
            "entry_order_id": self.entry_order_id,
            "exit_order_id": self.exit_order_id,
            "entry_limit_price": self.entry_limit_price,
            "entry_fill_price": self.entry_fill_price,
            "exit_limit_price": self.exit_limit_price,
            "exit_fill_price": self.exit_fill_price,
            "filled_quantity": self.filled_quantity,
            "last_error": self.last_error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass(frozen=True)
class ShadowPosition:
    id: str
    scan_run_id: str
    plan_id: str
    ticker: str
    earnings_date: date
    earnings_timing: str
    option_type: str
    target_delta: float
    target_expiration_days: int
    entry_minutes_before_close: int
    scheduled_entry_at: datetime
    market_open_at: datetime
    expiration_target: date
    status: str = "scheduled"
    contract_symbol: str | None = None
    contract_expiration: date | None = None
    strike_price: float | None = None
    multiplier: int = 100
    entry_observed_at: datetime | None = None
    entry_bid: float | None = None
    entry_ask: float | None = None
    entry_underlying: float | None = None
    entry_delta: float | None = None
    entry_vega: float | None = None
    entry_iv: float | None = None
    entry_volume: int = 0
    entry_open_interest: int = 0
    entry_quote_time_ms: int | None = None
    entry_quote_quality: str | None = None
    last_error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def contract(self) -> OptionContract:
        if (
            not self.contract_symbol
            or self.contract_expiration is None
            or self.strike_price is None
        ):
            raise ValueError("shadow position does not have a selected contract")
        return OptionContract(
            symbol=self.contract_symbol,
            ticker=self.ticker,
            option_type=self.option_type,
            expiration=self.contract_expiration,
            strike=self.strike_price,
            bid=self.entry_bid or 0.0,
            ask=self.entry_ask or 0.0,
            underlying_price=self.entry_underlying or 0.0,
            multiplier=self.multiplier,
            volume=self.entry_volume,
            open_interest=self.entry_open_interest,
            delta=self.entry_delta,
            vega=self.entry_vega,
            implied_volatility=self.entry_iv,
            quote_time_ms=self.entry_quote_time_ms,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "scan_run_id": self.scan_run_id,
            "plan_id": self.plan_id,
            "ticker": self.ticker,
            "earnings_date": self.earnings_date.isoformat(),
            "earnings_timing": self.earnings_timing,
            "option_type": self.option_type,
            "target_delta": self.target_delta,
            "target_expiration_days": self.target_expiration_days,
            "entry_minutes_before_close": self.entry_minutes_before_close,
            "scheduled_entry_at": self.scheduled_entry_at.isoformat(),
            "market_open_at": self.market_open_at.isoformat(),
            "expiration_target": self.expiration_target.isoformat(),
            "status": self.status,
            "contract_symbol": self.contract_symbol,
            "contract_expiration": self.contract_expiration.isoformat()
            if self.contract_expiration
            else None,
            "strike_price": self.strike_price,
            "entry_observed_at": self.entry_observed_at.isoformat()
            if self.entry_observed_at
            else None,
            "entry_bid": self.entry_bid,
            "entry_ask": self.entry_ask,
            "entry_underlying": self.entry_underlying,
            "entry_delta": self.entry_delta,
            "entry_vega": self.entry_vega,
            "entry_iv": self.entry_iv,
            "entry_volume": self.entry_volume,
            "entry_open_interest": self.entry_open_interest,
            "entry_quote_time_ms": self.entry_quote_time_ms,
            "entry_quote_quality": self.entry_quote_quality,
            "last_error": self.last_error,
        }
