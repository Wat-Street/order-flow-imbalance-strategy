"""Deterministic, hand-checkable tests for the Phase 3 backtester.

The arithmetic fixtures here are taken directly from the project specification
(the independent fill/accounting check: a one-unit buy at ask 101 and sell at
bid 99 with 0.5 bps adverse slippage and 4 bps fees per fill produces prices
101.00505 / 98.99505, total fees 0.08000004 and net -2.09000004).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from order_flow_imbalance_strategy.backtesting import (
    BacktestConfig,
    Cash,
    FundingPayment,
    MLFilteredOFI,
    Momentum,
    Opportunity,
    Order,
    PrecomputedMomentum,
    Prediction,
    Quote,
    RandomEntry,
    RawOFI,
    TradingConfig,
    compute_risk_metrics,
    entry_block_reason,
    holm_adjust,
    paired_stationary_bootstrap,
    reconcile_accounting,
    run_backtest,
    run_required_baselines,
)
from order_flow_imbalance_strategy.backtesting.contracts import EquityPoint
from order_flow_imbalance_strategy.backtesting.entry_schedule import (
    ENTRY_ALLOWED,
    ENTRY_BLOCKED_FEATURE_NOT_READY,
    ENTRY_BLOCKED_POSITION_OPEN,
    ENTRY_BLOCKED_STALE,
)
from order_flow_imbalance_strategy.backtesting.execution import attempt_fill

T0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
SOURCE = "binance_usdm"
SYMBOL = "BTCUSDT"


def quote(
    index: int,
    bid_price: float = 100.0,
    ask_price: float = 101.0,
    bid_size: float = 10.0,
    ask_size: float = 10.0,
    *,
    book_stale: bool = False,
    book_dead: bool = False,
    valid: bool = True,
) -> Quote:
    ts = T0 + timedelta(seconds=index)
    return Quote(
        source=SOURCE,
        symbol=SYMBOL,
        timestamp=ts,
        available_at=ts,
        bid_price=bid_price,
        ask_price=ask_price,
        bid_size=bid_size,
        ask_size=ask_size,
        book_stale=book_stale,
        book_dead=book_dead,
        valid=valid,
    )


def opportunity(
    index: int,
    *,
    score: float | None = None,
    raw_ofi: float | None = None,
    momentum_score: float | None = None,
    feature_ready: bool = True,
    **quote_kwargs: object,
) -> Opportunity:
    q = quote(index, **quote_kwargs)  # type: ignore[arg-type]
    prediction = None
    if score is not None:
        prediction = Prediction(
            source=SOURCE,
            symbol=SYMBOL,
            timestamp=q.timestamp,
            available_at=q.available_at,
            score=score,
            feature_ready=feature_ready,
            model_id="model-abc",
        )
    return Opportunity(
        quote=q,
        prediction=prediction,
        raw_ofi=raw_ofi,
        momentum_score=momentum_score,
        feature_ready=feature_ready,
    )


def _order(side: str, index: int, latency_s: float = 1.0) -> Order:
    decision = T0 + timedelta(seconds=index)
    return Order(
        order_id="o1",
        strategy="test",
        source=SOURCE,
        symbol=SYMBOL,
        side=side,
        quantity=1.0,
        reduce_only=False,
        decision_at=decision,
        eligible_at=decision + timedelta(seconds=latency_s),
        reason="strategy_entry",
    )


# --------------------------------------------------------------------------- #
# Mandatory spec fixture: exact fill prices, fees, and net PnL.
# --------------------------------------------------------------------------- #


def test_spec_fill_prices_fees_and_net_pnl() -> None:
    config = BacktestConfig(fee_bps=4.0, extra_slippage_bps=0.5, lot_size=1.0)
    # Buy one unit at ask 101, filled on the quote at t+1 (after 1s latency).
    buy_order = _order("buy", 0)
    buy_quote = opportunity(1, bid_price=99.0, ask_price=101.0, bid_size=10.0, ask_size=10.0)
    buy = attempt_fill(buy_order, buy_quote, 1.0, config).fill
    assert buy is not None
    assert buy.price == pytest.approx(101.00505, abs=1e-9)

    sell_order = _order("sell", 2)
    sell_quote = opportunity(3, bid_price=99.0, ask_price=101.0, bid_size=10.0, ask_size=10.0)
    sell = attempt_fill(sell_order, sell_quote, 1.0, config).fill
    assert sell is not None
    assert sell.price == pytest.approx(98.99505, abs=1e-9)

    total_fees = buy.fee + sell.fee
    assert total_fees == pytest.approx(0.08000004, abs=1e-9)

    net = (sell.price - buy.price) - total_fees
    assert net == pytest.approx(-2.09000004, abs=1e-9)


# --------------------------------------------------------------------------- #
# Fill mechanics: causality guards.
# --------------------------------------------------------------------------- #


def test_fill_waits_for_latency() -> None:
    config = BacktestConfig()
    order = _order("buy", 0, latency_s=1.0)
    # Quote available at t+0 is earlier than eligible_at = t+1 -> must wait.
    attempt = attempt_fill(order, opportunity(0), 1.0, config)
    assert attempt.fill is None
    assert attempt.reason == "waiting_for_latency"


def test_fill_rejects_stale_dead_and_old_quotes() -> None:
    config = BacktestConfig(max_quote_age=timedelta(seconds=1))
    order = _order("buy", 0)
    assert attempt_fill(order, opportunity(1, book_stale=True), 1.0, config).reason == (
        "invalid_or_stale_quote"
    )
    assert attempt_fill(order, opportunity(1, book_dead=True), 1.0, config).reason == (
        "invalid_or_stale_quote"
    )
    # A quote whose event time is 3s before availability exceeds max_quote_age.
    old = Quote(
        source=SOURCE,
        symbol=SYMBOL,
        timestamp=T0,
        available_at=T0 + timedelta(seconds=3),
        bid_price=99.0,
        ask_price=101.0,
        bid_size=10.0,
        ask_size=10.0,
    )
    old_opp = Opportunity(quote=old)
    assert attempt_fill(order, old_opp, 1.0, config).reason == "quote_too_old"


def test_participation_cap_limits_fill_and_supports_partials() -> None:
    config = BacktestConfig(participation_rate=0.10, lot_size=0.001)
    order = _order("buy", 0)
    # Displayed ask size 10 * 10% = 1.0 available this row.
    attempt = attempt_fill(order, opportunity(1, ask_size=10.0), 5.0, config)
    assert attempt.fill is not None
    assert attempt.fill.quantity == pytest.approx(1.0)


def test_zero_displayed_size_is_insufficient() -> None:
    config = BacktestConfig()
    order = _order("buy", 0)
    attempt = attempt_fill(order, opportunity(1, ask_size=0.0), 1.0, config)
    assert attempt.fill is None
    assert attempt.reason == "insufficient_displayed_size"


def test_buy_crosses_ask_and_sell_crosses_bid() -> None:
    config = BacktestConfig(extra_slippage_bps=0.0, lot_size=1.0)
    buy = attempt_fill(_order("buy", 0), opportunity(1), 1.0, config).fill
    sell = attempt_fill(_order("sell", 0), opportunity(1), 1.0, config).fill
    assert buy is not None and sell is not None
    assert buy.reference_price == 101.0  # ask
    assert sell.reference_price == 100.0  # bid


# --------------------------------------------------------------------------- #
# Portfolio accounting reconciles independently from the fill log.
# --------------------------------------------------------------------------- #


def test_independent_reconciliation_matches_engine() -> None:
    config = BacktestConfig(fee_bps=4.0, extra_slippage_bps=0.5, lot_size=0.001)
    trading = TradingConfig(
        order_quantity=0.001, max_position=0.001, holding_time=timedelta(seconds=2)
    )
    # Strong long signal at t0, held 2s, then forced exit.
    rows = [opportunity(i, score=0.9, feature_ready=True) for i in range(8)]
    result = run_backtest(MLFilteredOFI(), rows, config=config, trading=trading)
    assert result.fills, "expected at least one entry fill"
    cash, position = reconcile_accounting(config.initial_cash, result.fills, result.funding)
    assert cash == pytest.approx(result.account.cash, abs=1e-9)
    assert position == pytest.approx(result.account.position, abs=1e-12)


def test_funding_cash_flow_sign() -> None:
    # Long position pays funding when the rate is positive.
    config = BacktestConfig(fee_bps=0.0, extra_slippage_bps=0.0, lot_size=1.0)
    trading = TradingConfig(
        order_quantity=1.0, max_position=1.0, holding_time=timedelta(seconds=100)
    )
    rows = [opportunity(i, score=0.9) for i in range(10)]
    settle = T0 + timedelta(seconds=5)
    funding = [
        FundingPayment(
            source=SOURCE,
            symbol=SYMBOL,
            settled_at=settle,
            rate=0.0001,
            mark_price=100.5,
            mark_available_at=settle,
        )
    ]
    result = run_backtest(MLFilteredOFI(), rows, funding=funding, config=config, trading=trading)
    assert result.funding, "funding should have settled while a position was open"
    record = result.funding[0]
    assert record.position_before > 0
    assert record.cash_flow < 0  # long pays positive funding


# --------------------------------------------------------------------------- #
# Strategies.
# --------------------------------------------------------------------------- #


def test_ml_strategy_respects_thresholds_and_readiness() -> None:
    trading = TradingConfig(long_threshold=0.2, short_threshold=-0.2)
    strat = MLFilteredOFI()
    assert strat.decide(opportunity(0, score=0.5), trading) == 1
    assert strat.decide(opportunity(0, score=-0.5), trading) == -1
    assert strat.decide(opportunity(0, score=0.1), trading) == 0
    assert strat.decide(opportunity(0, score=0.5, feature_ready=False), trading) == 0


def test_raw_ofi_is_symmetric_and_cash_never_trades() -> None:
    trading = TradingConfig()
    assert RawOFI(threshold=1.0).decide(opportunity(0, raw_ofi=2.0), trading) == 1
    assert RawOFI(threshold=1.0).decide(opportunity(0, raw_ofi=-2.0), trading) == -1
    assert RawOFI(threshold=1.0).decide(opportunity(0, raw_ofi=0.5), trading) == 0
    assert Cash().decide(opportunity(0, score=0.9), trading) == 0


def test_precomputed_momentum_reads_column() -> None:
    trading = TradingConfig()
    assert (
        PrecomputedMomentum(threshold=0.0).decide(opportunity(0, momentum_score=0.1), trading) == 1
    )
    assert (
        PrecomputedMomentum(threshold=0.0).decide(opportunity(0, momentum_score=-0.1), trading)
        == -1
    )
    assert (
        PrecomputedMomentum(threshold=1.0).decide(opportunity(0, momentum_score=0.5), trading) == 0
    )


def test_random_calibration_uses_only_training_rows() -> None:
    training = [opportunity(i, score=0.9 if i % 2 == 0 else 0.0) for i in range(10)]
    calibrated = RandomEntry.calibrate(training, threshold=0.2, seed=7)
    assert 0 <= calibrated.attempt_probability <= 1
    assert 0 <= calibrated.long_probability <= 1
    # Deterministic for a fixed seed.
    other = RandomEntry.calibrate(training, threshold=0.2, seed=7)
    rows = [opportunity(i, score=0.9) for i in range(20)]
    decisions_a = [calibrated.decide(r, TradingConfig()) for r in rows]
    decisions_b = [other.decide(r, TradingConfig()) for r in rows]
    assert decisions_a == decisions_b


# --------------------------------------------------------------------------- #
# Engine invariants.
# --------------------------------------------------------------------------- #


def test_engine_rejects_mixed_venue_identity() -> None:
    a = opportunity(0, score=0.9)
    b_quote = Quote(
        source="hyperliquid",
        symbol="BTC",
        timestamp=T0 + timedelta(seconds=1),
        available_at=T0 + timedelta(seconds=1),
        bid_price=99.0,
        ask_price=101.0,
        bid_size=10.0,
        ask_size=10.0,
    )
    b = Opportunity(quote=b_quote)
    with pytest.raises(ValueError, match="venue or native-symbol"):
        run_backtest(Cash(), [a, b])


def test_strict_funding_requires_complete_series() -> None:
    config = BacktestConfig(require_funding=True, funding_interval=timedelta(hours=8))
    rows = [opportunity(i, score=0.0) for i in range(5)]
    with pytest.raises(ValueError, match="funding"):
        run_backtest(Cash(), rows, funding=(), config=config)


def test_cash_account_equity_is_flat_and_complete() -> None:
    rows = [opportunity(i, score=0.0) for i in range(5)]
    result = run_backtest(Cash(), rows)
    assert all(point.equity == pytest.approx(10_000.0) for point in result.equity)
    assert all(point.valuation_complete for point in result.equity)
    assert result.account.position == 0.0


# --------------------------------------------------------------------------- #
# Risk metrics refuse to manufacture values.
# --------------------------------------------------------------------------- #


def _equity_point(index: int, equity: float | None, *, complete: bool = True) -> EquityPoint:
    ts = T0 + timedelta(seconds=index)
    return EquityPoint(
        timestamp=ts,
        clock_timestamp=ts,
        cash=equity if equity is not None else 0.0,
        position=0.0,
        mark_price=100.0,
        equity=equity,
        valuation_complete=complete,
    )


def test_risk_insufficient_days() -> None:
    metrics = compute_risk_metrics([_equity_point(0, 10_000.0)])
    assert metrics.sharpe.status == "insufficient_days"


def test_risk_incomplete_valuation() -> None:
    points = [_equity_point(0, 10_000.0), _equity_point(1, None, complete=False)]
    metrics = compute_risk_metrics(points)
    assert metrics.sharpe.status == "incomplete_valuation"
    assert metrics.max_drawdown.status == "incomplete_valuation"


def test_risk_rejects_nonincreasing_timestamps() -> None:
    p0 = _equity_point(0, 10_000.0)
    p1 = _equity_point(0, 10_000.0)  # duplicate timestamp
    metrics = compute_risk_metrics([p0, p1])
    assert metrics.sharpe.status == "invalid_input"


# --------------------------------------------------------------------------- #
# Dependence-aware comparison helpers.
# --------------------------------------------------------------------------- #


def test_paired_bootstrap_is_deterministic_and_bracketed() -> None:
    first = [0.01, 0.02, -0.01, 0.03, 0.00, 0.02, -0.02, 0.01]
    second = [0.00, 0.01, -0.02, 0.01, -0.01, 0.00, -0.03, 0.00]
    a = paired_stationary_bootstrap(first, second, block_length=3, resamples=500, seed=1)
    b = paired_stationary_bootstrap(first, second, block_length=3, resamples=500, seed=1)
    assert a == b
    assert a.confidence_low <= a.mean_difference <= a.confidence_high
    assert 0.0 <= a.two_sided_p_value <= 1.0


def test_holm_adjustment_monotonic_and_bounded() -> None:
    raw = {"a": 0.01, "b": 0.04, "c": 0.03}
    adjusted = holm_adjust(raw)
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())
    # Holm never decreases a p-value.
    assert all(adjusted[name] >= raw[name] - 1e-12 for name in raw)


# --------------------------------------------------------------------------- #
# Full comparison harness.
# --------------------------------------------------------------------------- #


def test_run_required_baselines_produces_all_five_families() -> None:
    training = [opportunity(i, score=0.9 if i % 3 == 0 else 0.0) for i in range(30)]
    evaluation = [opportunity(30 + i, score=0.9 if i % 2 == 0 else -0.9) for i in range(30)]
    results = run_required_baselines(
        evaluation,
        training,
        raw_ofi_threshold=0.0,
        momentum_threshold=0.0,
        random_seeds=(11, 29, 47),
    )
    assert {"ml_filtered_ofi", "raw_ofi", "momentum", "cash"} <= set(results)
    assert sum(name.startswith("random_seed_") for name in results) == 3


# --------------------------------------------------------------------------- #
# Trailing-price momentum computes from observed prices, not a column.
# --------------------------------------------------------------------------- #


def _rising_rows(n: int, step: float = 0.1) -> list[Opportunity]:
    rows = []
    for i in range(n):
        mid_shift = step * i
        rows.append(
            opportunity(
                i,
                bid_price=100.0 + mid_shift,
                ask_price=101.0 + mid_shift,
            )
        )
    return rows


def test_trailing_momentum_goes_long_on_rising_prices() -> None:
    strat = Momentum(lookback=timedelta(seconds=3), threshold=0.0)
    trading = TradingConfig()
    rows = _rising_rows(6)
    decisions = [strat.decide(row, trading) for row in rows]
    # Early rows have no full 3s window yet -> 0; once the window fills on a
    # strictly rising series, the signal is long.
    assert decisions[0] == 0
    assert decisions[-1] == 1


def test_trailing_momentum_goes_short_on_falling_prices() -> None:
    strat = Momentum(lookback=timedelta(seconds=3), threshold=0.0)
    trading = TradingConfig()
    rows = [opportunity(i, bid_price=100.0 - 0.1 * i, ask_price=101.0 - 0.1 * i) for i in range(6)]
    decisions = [strat.decide(row, trading) for row in rows]
    assert decisions[-1] == -1


def test_trailing_momentum_resets_across_day_boundary() -> None:
    strat = Momentum(lookback=timedelta(seconds=3), threshold=0.0)
    trading = TradingConfig()
    # Feed a rising series, then a row on the next UTC day: the window must reset
    # so it cannot compare across the boundary.
    for row in _rising_rows(5):
        strat.decide(row, trading)
    next_day_ts = T0 + timedelta(days=1)
    next_day = Opportunity(
        quote=Quote(
            source=SOURCE,
            symbol=SYMBOL,
            timestamp=next_day_ts,
            available_at=next_day_ts,
            bid_price=200.0,
            ask_price=201.0,
            bid_size=10.0,
            ask_size=10.0,
        )
    )
    assert strat.decide(next_day, trading) == 0


def test_momentum_select_uses_training_only_and_returns_configured() -> None:
    training = _rising_rows(40)
    chosen = Momentum.select(
        training,
        lookbacks=(timedelta(seconds=2), timedelta(seconds=5)),
        thresholds=(0.0,),
    )
    assert isinstance(chosen, Momentum)
    assert chosen.lookback in (timedelta(seconds=2), timedelta(seconds=5))


# --------------------------------------------------------------------------- #
# Shared entry schedule.
# --------------------------------------------------------------------------- #


def test_entry_schedule_reasons() -> None:
    row = opportunity(0, score=0.9)
    assert (
        entry_block_reason(row, position=0.0, has_pending_entry=False, has_pending_exit=False)
        == ENTRY_ALLOWED
    )
    assert (
        entry_block_reason(row, position=0.5, has_pending_entry=False, has_pending_exit=False)
        == ENTRY_BLOCKED_POSITION_OPEN
    )
    stale = opportunity(0, score=0.9, book_stale=True)
    assert (
        entry_block_reason(stale, position=0.0, has_pending_entry=False, has_pending_exit=False)
        == ENTRY_BLOCKED_STALE
    )
    not_ready = opportunity(0, score=0.9, feature_ready=False)
    assert (
        entry_block_reason(not_ready, position=0.0, has_pending_entry=False, has_pending_exit=False)
        == ENTRY_BLOCKED_FEATURE_NOT_READY
    )


def test_engine_records_skip_reason_for_stale_rows() -> None:
    rows = [opportunity(i, score=0.9, book_stale=True) for i in range(4)]
    result = run_backtest(MLFilteredOFI(), rows)
    skips = [d for d in result.decisions if d.action == "skip"]
    assert skips, "a stale, flat row should record a skip decision"
    assert all(d.reason == ENTRY_BLOCKED_STALE for d in skips)


# --------------------------------------------------------------------------- #
# Coverage diagnostics.
# --------------------------------------------------------------------------- #


def test_coverage_diagnostics_counts() -> None:
    rows = [
        opportunity(0, score=0.9),
        opportunity(1, score=0.9, book_stale=True),
        opportunity(2, score=0.9, book_dead=True),
    ]
    result = run_backtest(Cash(), rows)
    cov = result.coverage
    assert cov is not None
    assert cov.total_rows == 3
    assert cov.tradable_rows == 1
    assert cov.stale_rows == 1
    assert cov.dead_rows == 1
    assert cov.tradable_fraction == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# Streaming reader equivalence.
# --------------------------------------------------------------------------- #


def test_streaming_matches_eager_load(tmp_path) -> None:
    import polars as pl

    from order_flow_imbalance_strategy.backtesting.io import load_opportunities
    from order_flow_imbalance_strategy.backtesting.streaming import stream_opportunities

    # Two UTC days of a few rows each.
    records = []
    for day in range(2):
        for sec in range(3):
            ts = T0 + timedelta(days=day, seconds=sec)
            records.append(
                {
                    "source": SOURCE,
                    "symbol": SYMBOL,
                    "timestamp": ts,
                    "available_at": ts,
                    "bid_price": 100.0,
                    "ask_price": 101.0,
                    "bid_size": 10.0,
                    "ask_size": 10.0,
                    "book_stale": False,
                    "book_dead": False,
                    "raw_ofi": 0.5,
                    "feature_ready": True,
                }
            )
    path = tmp_path / "features.parquet"
    pl.DataFrame(records).write_parquet(path)

    eager = load_opportunities(path)
    streamed = list(stream_opportunities(path))
    assert len(streamed) == len(eager) == 6
    assert [o.quote.timestamp for o in streamed] == [o.quote.timestamp for o in eager]
