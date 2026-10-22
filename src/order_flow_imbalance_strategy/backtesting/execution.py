from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta

from .contracts import BacktestConfig, Fill, Opportunity, Order


@dataclass(frozen=True, slots=True)
class FillAttempt:
    fill: Fill | None
    reason: str


def _round_quantity_down(quantity: float, lot_size: float) -> float:
    lots = math.floor((quantity + lot_size * 1e-10) / lot_size)
    return round(lots * lot_size, 12)


def attempt_fill(
    order: Order,
    opportunity: Opportunity,
    remaining_quantity: float,
    config: BacktestConfig,
) -> FillAttempt:
    """Try one market-order slice against this row's displayed top-of-book size."""
    quote = opportunity.quote
    if not quote.valid or quote.book_stale or quote.book_dead:
        return FillAttempt(None, "invalid_or_stale_quote")
    if opportunity.available_at < order.eligible_at:
        return FillAttempt(None, "waiting_for_latency")
    quote_age = opportunity.available_at - quote.timestamp
    if quote_age < timedelta(0) or quote_age > config.max_quote_age:
        return FillAttempt(None, "quote_too_old")

    displayed_size = quote.ask_size if order.side == "buy" else quote.bid_size
    available_quantity = displayed_size * config.participation_rate
    quantity = _round_quantity_down(min(remaining_quantity, available_quantity), config.lot_size)
    if quantity <= 0:
        return FillAttempt(None, "insufficient_displayed_size")

    reference_price = quote.ask_price if order.side == "buy" else quote.bid_price
    direction = 1 if order.side == "buy" else -1
    # The executed price is the crossed quote plus adverse slippage. The spec's
    # independent accounting fixture (buy ask 101 -> 101.00505, sell bid 99 ->
    # 98.99505) is defined on this slipped price directly, so no tick-grid snap
    # is applied to the fill price; tick_size remains a quote-validity concern.
    price = reference_price * (1 + direction * config.extra_slippage_bps / 10_000)
    notional = quantity * price
    fee = notional * config.fee_bps / 10_000
    slippage = quantity * (price - reference_price) * direction
    return FillAttempt(
        Fill(
            order_id=order.order_id,
            strategy=order.strategy,
            source=order.source,
            symbol=order.symbol,
            side=order.side,
            timestamp=opportunity.available_at,
            quantity=quantity,
            reference_price=reference_price,
            price=price,
            fee=fee,
            slippage=slippage,
            reduce_only=order.reduce_only,
        ),
        "filled",
    )
