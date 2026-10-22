from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from itertools import product

from .contracts import (
    BacktestConfig,
    BacktestResult,
    Fill,
    FundingPayment,
    FundingRecord,
    Opportunity,
    TradingConfig,
)
from .engine import run_backtest
from .strategies import Strategy


@dataclass(frozen=True, slots=True)
class FixedFillCostScenario:
    fee_bps: float
    extra_slippage_bps: float
    pnl_after_spread_and_funding: float
    fees: float
    extra_slippage: float
    net_pnl: float
    break_even_fee_bps: float | None


def fixed_fill_cost_grid(
    fills: Sequence[Fill],
    funding: Sequence[FundingRecord],
    fee_bps_values: Sequence[float],
    slippage_bps_values: Sequence[float],
) -> tuple[FixedFillCostScenario, ...]:
    """Reprice fixed fills under fee/slippage assumptions without changing execution."""
    if any(value < 0 for value in (*fee_bps_values, *slippage_bps_values)):
        raise ValueError("cost assumptions must be nonnegative")
    signed_reference_pnl = sum(
        fill.quantity * fill.reference_price * (1 if fill.side == "sell" else -1) for fill in fills
    )
    funding_pnl = sum(record.cash_flow for record in funding)
    pnl_before_extra_costs = signed_reference_pnl + funding_pnl
    notional = sum(fill.quantity * fill.price for fill in fills)
    scenarios: list[FixedFillCostScenario] = []
    for fee_bps, slippage_bps in product(fee_bps_values, slippage_bps_values):
        fees = sum(fill.quantity * fill.price * fee_bps / 10_000 for fill in fills)
        extra_slippage = sum(
            fill.quantity * fill.reference_price * slippage_bps / 10_000 for fill in fills
        )
        net = pnl_before_extra_costs - fees - extra_slippage
        break_even = (
            (pnl_before_extra_costs - extra_slippage) / notional * 10_000 if notional > 0 else None
        )
        scenarios.append(
            FixedFillCostScenario(
                fee_bps=fee_bps,
                extra_slippage_bps=slippage_bps,
                pnl_after_spread_and_funding=pnl_before_extra_costs,
                fees=fees,
                extra_slippage=extra_slippage,
                net_pnl=net,
                break_even_fee_bps=break_even,
            )
        )
    return tuple(scenarios)


def replay_latency_grid(
    strategy: Strategy,
    opportunities: Sequence[Opportunity],
    latencies: Sequence[timedelta] = (
        timedelta(0),
        timedelta(seconds=1),
        timedelta(seconds=2),
    ),
    funding: Sequence[FundingPayment] = (),
    config: BacktestConfig | None = None,
    trading: TradingConfig | None = None,
) -> dict[float, BacktestResult]:
    """Replay orders under each latency; eligibility and fills are recomputed."""
    if not latencies:
        raise ValueError("at least one latency scenario is required")
    config = config or BacktestConfig()
    trading = trading or TradingConfig()
    results: dict[float, BacktestResult] = {}
    for latency in latencies:
        if latency < timedelta(0):
            raise ValueError("latency scenarios cannot be negative")
        scenario_config = replace(config, latency=latency)
        scenario_strategy = copy.deepcopy(strategy)
        results[latency.total_seconds()] = run_backtest(
            scenario_strategy,
            opportunities,
            funding=funding,
            config=scenario_config,
            trading=trading,
        )
    return results
