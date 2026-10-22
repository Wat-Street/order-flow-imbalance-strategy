from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal


def _require_utc(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC")


@dataclass(frozen=True, slots=True)
class Quote:
    source: str
    symbol: str
    timestamp: datetime
    available_at: datetime
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    book_stale: bool = False
    book_dead: bool = False
    valid: bool = True

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "timestamp")
        _require_utc(self.available_at, "available_at")
        if not self.source or not self.symbol:
            raise ValueError("source and symbol must be nonempty")
        if self.available_at < self.timestamp:
            raise ValueError("quote cannot be available before its event timestamp")
        if self.valid and (
            not all(
                math.isfinite(value)
                for value in (self.bid_price, self.ask_price, self.bid_size, self.ask_size)
            )
            or self.bid_price <= 0
            or self.ask_price <= self.bid_price
            or self.bid_size < 0
            or self.ask_size < 0
        ):
            raise ValueError(
                "valid quotes require positive, uncrossed prices and nonnegative sizes"
            )

    @property
    def midpoint(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0


@dataclass(frozen=True, slots=True)
class Prediction:
    source: str
    symbol: str
    timestamp: datetime
    available_at: datetime
    score: float | None
    feature_ready: bool
    model_id: str

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "timestamp")
        _require_utc(self.available_at, "available_at")
        if self.available_at < self.timestamp:
            raise ValueError("prediction cannot be available before its row timestamp")
        if not self.source or not self.symbol or not self.model_id:
            raise ValueError("source, symbol, and model_id must be nonempty")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("prediction score must be finite or null")


@dataclass(frozen=True, slots=True)
class Opportunity:
    quote: Quote
    prediction: Prediction | None = None
    raw_ofi: float | None = None
    momentum_score: float | None = None
    feature_ready: bool = True

    def __post_init__(self) -> None:
        if self.prediction is not None:
            if (self.prediction.source, self.prediction.symbol) != (
                self.quote.source,
                self.quote.symbol,
            ):
                raise ValueError("prediction and quote venue/symbol identities must match")
            if self.prediction.timestamp != self.quote.timestamp:
                raise ValueError("prediction and quote timestamps must match the declared clock")

    @property
    def available_at(self) -> datetime:
        if self.prediction is None:
            return self.quote.available_at
        return max(self.quote.available_at, self.prediction.available_at)


@dataclass(frozen=True, slots=True)
class FundingPayment:
    source: str
    symbol: str
    settled_at: datetime
    rate: float
    mark_price: float
    mark_available_at: datetime
    mark_kind: Literal["observed", "causal_proxy"] = "observed"

    def __post_init__(self) -> None:
        _require_utc(self.settled_at, "settled_at")
        _require_utc(self.mark_available_at, "mark_available_at")
        if self.mark_available_at > self.settled_at:
            raise ValueError("funding mark must be available no later than settlement")
        if self.mark_price <= 0:
            raise ValueError("funding mark_price must be positive")
        if not math.isfinite(self.rate) or not math.isfinite(self.mark_price):
            raise ValueError("funding rate and mark_price must be finite")
        if not self.source or not self.symbol:
            raise ValueError("source and symbol must be nonempty")


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    initial_cash: float = 10_000.0
    latency: timedelta = timedelta(seconds=1)
    max_quote_age: timedelta = timedelta(seconds=1)
    fee_bps: float = 4.0
    extra_slippage_bps: float = 0.5
    participation_rate: float = 0.10
    tick_size: float = 0.01
    lot_size: float = 0.001
    require_funding: bool = False
    funding_interval: timedelta | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.latency < timedelta(0) or self.max_quote_age < timedelta(0):
            raise ValueError("latency and max_quote_age cannot be negative")
        if (
            not math.isfinite(self.fee_bps)
            or not math.isfinite(self.extra_slippage_bps)
            or self.fee_bps < 0
            or self.extra_slippage_bps < 0
        ):
            raise ValueError("fees and slippage cannot be negative")
        if not 0 < self.participation_rate <= 1:
            raise ValueError("participation_rate must be in (0, 1]")
        if (
            not math.isfinite(self.tick_size)
            or not math.isfinite(self.lot_size)
            or self.tick_size <= 0
            or self.lot_size <= 0
        ):
            raise ValueError("tick_size and lot_size must be positive")
        if self.funding_interval is not None and self.funding_interval <= timedelta(0):
            raise ValueError("funding_interval must be positive when configured")


@dataclass(frozen=True, slots=True)
class TradingConfig:
    order_quantity: float = 0.001
    max_position: float = 0.001
    holding_time: timedelta = timedelta(seconds=30)
    long_threshold: float = 0.2
    short_threshold: float = -0.2

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.order_quantity)
            or not math.isfinite(self.max_position)
            or self.order_quantity <= 0
            or self.max_position <= 0
        ):
            raise ValueError("order_quantity and max_position must be positive")
        if self.holding_time <= timedelta(0):
            raise ValueError("holding_time must be positive")
        if self.short_threshold >= self.long_threshold:
            raise ValueError("short_threshold must be below long_threshold")


@dataclass(frozen=True, slots=True)
class Order:
    order_id: str
    strategy: str
    source: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    reduce_only: bool
    decision_at: datetime
    eligible_at: datetime
    reason: str


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: str
    strategy: str
    source: str
    symbol: str
    side: Literal["buy", "sell"]
    timestamp: datetime
    quantity: float
    reference_price: float
    price: float
    fee: float
    slippage: float
    reduce_only: bool


@dataclass(frozen=True, slots=True)
class OrderEvent:
    order_id: str
    timestamp: datetime
    status: Literal["submitted", "waiting", "partial", "filled", "rejected", "cancelled"]
    remaining_quantity: float
    reason: str


@dataclass(frozen=True, slots=True)
class Decision:
    strategy: str
    source: str
    symbol: str
    timestamp: datetime
    available_at: datetime
    action: Literal["enter_long", "enter_short", "hold", "skip", "exit"]
    reason: str
    score: float | None = None


@dataclass(frozen=True, slots=True)
class FundingRecord:
    payment: FundingPayment
    position_before: float
    cash_flow: float


@dataclass(frozen=True, slots=True)
class EquityPoint:
    timestamp: datetime
    clock_timestamp: datetime | None
    cash: float
    position: float
    mark_price: float | None
    equity: float | None
    valuation_complete: bool


@dataclass(frozen=True, slots=True)
class AccountState:
    cash: float
    position: float
    fees_paid: float
    slippage_paid: float
    funding_paid: float
    terminal_inventory: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    strategy: str
    source: str
    symbol: str
    model_id: str | None
    config: BacktestConfig
    trading_config: TradingConfig
    decisions: tuple[Decision, ...] = field(default_factory=tuple)
    orders: tuple[Order, ...] = field(default_factory=tuple)
    order_events: tuple[OrderEvent, ...] = field(default_factory=tuple)
    fills: tuple[Fill, ...] = field(default_factory=tuple)
    funding: tuple[FundingRecord, ...] = field(default_factory=tuple)
    equity: tuple[EquityPoint, ...] = field(default_factory=tuple)
    account: AccountState | None = None
    coverage: CoverageDiagnostics | None = None


@dataclass(frozen=True, slots=True)
class CoverageDiagnostics:
    """Per-row coverage summary saved alongside each account's results."""

    total_rows: int
    tradable_rows: int
    stale_rows: int
    dead_rows: int
    invalid_quote_rows: int
    feature_ready_rows: int
    prediction_ready_rows: int
    tradable_fraction: float
