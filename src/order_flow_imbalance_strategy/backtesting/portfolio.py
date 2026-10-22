from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import AccountState, Fill, FundingPayment, FundingRecord


@dataclass(slots=True)
class PortfolioLedger:
    initial_cash: float
    cash: float = field(init=False)
    position: float = 0.0
    fees_paid: float = 0.0
    slippage_paid: float = 0.0
    funding_paid: float = 0.0
    funding_records: list[FundingRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.cash = self.initial_cash

    def apply_fill(self, fill: Fill) -> None:
        direction = 1 if fill.side == "buy" else -1
        if fill.quantity <= 0 or fill.price <= 0:
            raise ValueError("fill quantity and price must be positive")
        if fill.reduce_only:
            position_direction = 1 if self.position > 0 else -1
            if self.position == 0 or direction == position_direction:
                raise ValueError("reduce-only fill does not reduce the current position")
            if fill.quantity > abs(self.position) + 1e-10:
                raise ValueError("reduce-only fill exceeds the open position")
        self.cash -= direction * fill.quantity * fill.price
        self.cash -= fill.fee
        self.position += direction * fill.quantity
        self.fees_paid += fill.fee
        self.slippage_paid += fill.slippage
        if abs(self.position) < 1e-12:
            self.position = 0.0

    def apply_funding(self, payment: FundingPayment) -> FundingRecord:
        position_before = self.position
        cash_flow = -position_before * payment.mark_price * payment.rate
        self.cash += cash_flow
        self.funding_paid -= cash_flow
        record = FundingRecord(payment, position_before, cash_flow)
        self.funding_records.append(record)
        return record

    def snapshot(self) -> AccountState:
        return AccountState(
            cash=self.cash,
            position=self.position,
            fees_paid=self.fees_paid,
            slippage_paid=self.slippage_paid,
            funding_paid=self.funding_paid,
            terminal_inventory=self.position,
        )


def reconcile_accounting(
    initial_cash: float,
    fills: tuple[Fill, ...] | list[Fill],
    funding: tuple[FundingRecord, ...] | list[FundingRecord],
) -> tuple[float, float]:
    """Independently derive cash and inventory from the immutable execution log."""
    cash = initial_cash
    position = 0.0
    events = [(fill.timestamp, 1, fill) for fill in fills]
    events.extend((record.payment.settled_at, 0, record) for record in funding)
    for _, event_type, event in sorted(events, key=lambda item: (item[0], item[1])):
        if event_type == 0:
            cash += event.cash_flow
        else:
            direction = 1 if event.side == "buy" else -1
            cash -= direction * event.quantity * event.price
            cash -= event.fee
            position += direction * event.quantity
    return cash, position
