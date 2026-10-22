from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime

from .contracts import (
    BacktestConfig,
    BacktestResult,
    Decision,
    EquityPoint,
    FundingPayment,
    Opportunity,
    Order,
    OrderEvent,
    TradingConfig,
)
from .entry_schedule import (
    ENTRY_ALLOWED,
    CoverageTracker,
    entry_block_reason,
)
from .execution import attempt_fill
from .portfolio import PortfolioLedger
from .strategies import Strategy


def _make_order(
    order_id: str,
    strategy: str,
    opportunity: Opportunity,
    side: str,
    quantity: float,
    reduce_only: bool,
    decision_at: datetime,
    eligible_at: datetime,
    reason: str,
) -> Order:
    return Order(
        order_id=order_id,
        strategy=strategy,
        source=opportunity.quote.source,
        symbol=opportunity.quote.symbol,
        side=side,
        quantity=quantity,
        reduce_only=reduce_only,
        decision_at=decision_at,
        eligible_at=eligible_at,
        reason=reason,
    )


def run_backtest(
    strategy: Strategy,
    opportunities: Sequence[Opportunity],
    funding: Sequence[FundingPayment] = (),
    config: BacktestConfig | None = None,
    trading: TradingConfig | None = None,
) -> BacktestResult:
    """Replay one isolated account over the complete supplied decision clock.

    Prediction scores are directional scores (``2 * probability - 1`` for the
    selected binary target). Funding must include every settlement in the run;
    marks are required to be available by their settlement timestamps.
    """
    if not opportunities:
        raise ValueError("at least one full-clock opportunity is required")
    config = config or BacktestConfig()
    trading = trading or TradingConfig()

    rows = tuple(opportunities)
    first_quote = rows[0].quote
    source, symbol = first_quote.source, first_quote.symbol
    previous_timestamp: datetime | None = None
    previous_available: datetime | None = None
    model_ids: set[str] = set()
    for row in rows:
        quote = row.quote
        if (quote.source, quote.symbol) != (source, symbol):
            raise ValueError("one backtest account cannot mix venue or native-symbol identities")
        if previous_timestamp is not None and quote.timestamp <= previous_timestamp:
            raise ValueError("quote timestamps must be strictly increasing and unique")
        if previous_available is not None and row.available_at < previous_available:
            raise ValueError("opportunity availability times must be nondecreasing")
        previous_timestamp = quote.timestamp
        previous_available = row.available_at
        if row.prediction is not None:
            model_ids.add(row.prediction.model_id)
    if len(model_ids) > 1:
        raise ValueError("a backtest run cannot mix model bundle identities")

    funding_rows = tuple(sorted(funding, key=lambda item: item.settled_at))
    if config.require_funding and not funding_rows:
        raise ValueError("strict funding mode requires a complete funding settlement series")
    if config.require_funding and config.funding_interval is None:
        raise ValueError("strict funding mode requires a configured funding_interval")
    if any((item.source, item.symbol) != (source, symbol) for item in funding_rows):
        raise ValueError("funding venue/symbol identity must match the quote stream")
    if config.require_funding:
        interval_seconds = config.funding_interval.total_seconds()
        first_epoch = rows[0].quote.timestamp.timestamp()
        last_epoch = rows[-1].quote.timestamp.timestamp()
        first_index = math.ceil(first_epoch / interval_seconds)
        expected = set()
        index = first_index
        while index * interval_seconds <= last_epoch:
            expected.add(datetime.fromtimestamp(index * interval_seconds, UTC))
            index += 1
        observed = {payment.settled_at for payment in funding_rows}
        missing_settlements = sorted(expected - observed)
        if missing_settlements:
            raise ValueError(f"strict funding mode is missing settlements: {missing_settlements}")

    ledger = PortfolioLedger(config.initial_cash)
    coverage = CoverageTracker()
    decisions: list[Decision] = []
    orders: list[Order] = []
    order_events: list[OrderEvent] = []
    fills = []
    equity: list[EquityPoint] = []
    pending_entry: tuple[Order, float] | None = None
    pending_exit: tuple[Order, float] | None = None
    opened_at: datetime | None = None
    funding_index = 0
    order_number = 0

    def submit(order: Order, now: datetime) -> None:
        orders.append(order)
        order_events.append(
            OrderEvent(order.order_id, now, "submitted", order.quantity, order.reason)
        )

    def process_pending(
        pending: tuple[Order, float] | None,
        row: Opportunity,
        is_exit: bool,
    ) -> tuple[Order, float] | None:
        nonlocal opened_at
        if pending is None:
            return None
        order, remaining = pending
        attempt = attempt_fill(order, row, remaining, config)
        if attempt.fill is None:
            if attempt.reason == "insufficient_displayed_size" and not is_exit:
                order_events.append(
                    OrderEvent(
                        order.order_id, row.available_at, "rejected", remaining, attempt.reason
                    )
                )
                return None
            order_events.append(
                OrderEvent(order.order_id, row.available_at, "waiting", remaining, attempt.reason)
            )
            return pending

        fill = attempt.fill
        if is_exit:
            quantity = min(fill.quantity, abs(ledger.position))
            fill = replace(
                fill,
                quantity=quantity,
                fee=fill.fee * quantity / fill.quantity,
                slippage=fill.slippage * quantity / fill.quantity,
                reduce_only=True,
            )
        position_before = ledger.position
        ledger.apply_fill(fill)
        if not is_exit and position_before == 0 and ledger.position != 0:
            opened_at = row.available_at
        fills.append(fill)
        remaining = max(0.0, remaining - fill.quantity)
        status = "filled" if remaining < config.lot_size / 2 else "partial"
        order_events.append(
            OrderEvent(order.order_id, row.available_at, status, remaining, attempt.reason)
        )
        return None if status == "filled" else (order, remaining)

    for row in rows:
        now = row.available_at
        coverage.observe(row)
        while funding_index < len(funding_rows) and funding_rows[funding_index].settled_at <= now:
            payment = funding_rows[funding_index]
            ledger.apply_funding(payment)
            funding_index += 1

        if ledger.position and opened_at is not None and pending_exit is None:
            exit_due = opened_at + trading.holding_time
            if now >= exit_due:
                if pending_entry is not None:
                    entry_order, entry_remaining = pending_entry
                    order_events.append(
                        OrderEvent(
                            entry_order.order_id,
                            now,
                            "cancelled",
                            entry_remaining,
                            "holding_time_expired",
                        )
                    )
                    pending_entry = None
                order_number += 1
                exit_order = _make_order(
                    f"{strategy.name}-{order_number:08d}",
                    strategy.name,
                    row,
                    "sell" if ledger.position > 0 else "buy",
                    abs(ledger.position),
                    True,
                    exit_due,
                    exit_due + config.latency,
                    "holding_time_expired",
                )
                pending_exit = (exit_order, abs(ledger.position))
                submit(exit_order, now)
                decisions.append(
                    Decision(
                        strategy.name,
                        source,
                        symbol,
                        row.quote.timestamp,
                        now,
                        "exit",
                        "holding_time_expired",
                    )
                )

        pending_exit = process_pending(pending_exit, row, True)
        pending_entry = process_pending(pending_entry, row, False)
        if ledger.position == 0:
            opened_at = None
            pending_exit = None

        block_reason = entry_block_reason(
            row,
            position=ledger.position,
            has_pending_entry=pending_entry is not None,
            has_pending_exit=pending_exit is not None,
        )
        # Only emit an entry decision when the shared schedule says this row is
        # eligible and the account is flat with no pending orders; otherwise the
        # row was already handled above (exit/fill) or is simply holding.
        if block_reason == ENTRY_ALLOWED:
            side = strategy.decide(row, trading)
            if side:
                quantity = min(trading.order_quantity, trading.max_position)
                order_number += 1
                entry_order = _make_order(
                    f"{strategy.name}-{order_number:08d}",
                    strategy.name,
                    row,
                    "buy" if side > 0 else "sell",
                    quantity,
                    False,
                    now,
                    now + config.latency,
                    "strategy_entry",
                )
                pending_entry = (entry_order, quantity)
                submit(entry_order, now)
                action = "enter_long" if side > 0 else "enter_short"
                reason = "strategy_signal"
            else:
                action = "hold"
                reason = "no_entry_signal"
            score = (
                row.prediction.score
                if row.prediction is not None
                else row.raw_ofi
                if strategy.name == "raw_ofi"
                else row.momentum_score
            )
            decisions.append(
                Decision(
                    strategy.name,
                    source,
                    symbol,
                    row.quote.timestamp,
                    now,
                    action,
                    reason,
                    score,
                )
            )
        elif ledger.position == 0 and pending_entry is None and pending_exit is None:
            # Flat with no pending orders but the shared schedule blocked entry
            # (stale/dead quote or features not ready). Record the reason so the
            # skipped opportunity stays inspectable in the report.
            decisions.append(
                Decision(
                    strategy.name,
                    source,
                    symbol,
                    row.quote.timestamp,
                    now,
                    "skip",
                    block_reason,
                )
            )

        quote = row.quote
        mark_is_valid = quote.valid and not quote.book_stale and not quote.book_dead
        marked = ledger.position == 0 or mark_is_valid
        equity.append(
            EquityPoint(
                timestamp=now,
                clock_timestamp=quote.timestamp,
                cash=ledger.cash,
                position=ledger.position,
                mark_price=quote.midpoint if mark_is_valid else None,
                equity=ledger.cash + ledger.position * quote.midpoint if marked else None,
                valuation_complete=marked,
            )
        )

    account = ledger.snapshot()
    for pending in (pending_entry, pending_exit):
        if pending is not None:
            order, remaining = pending
            order_events.append(
                OrderEvent(
                    order.order_id,
                    rows[-1].available_at,
                    "cancelled",
                    remaining,
                    "end_of_input_window",
                )
            )
    return BacktestResult(
        strategy=strategy.name,
        source=source,
        symbol=symbol,
        model_id=next(iter(model_ids), None),
        config=config,
        trading_config=trading,
        decisions=tuple(decisions),
        orders=tuple(orders),
        order_events=tuple(order_events),
        fills=tuple(fills),
        funding=tuple(ledger.funding_records),
        equity=tuple(equity),
        account=account,
        coverage=coverage.diagnostics(),
    )
