from __future__ import annotations

from collections.abc import Sequence

from .contracts import BacktestConfig, BacktestResult, FundingPayment, Opportunity, TradingConfig
from .engine import run_backtest
from .strategies import Cash, MLFilteredOFI, Momentum, RandomEntry, RawOFI


def run_required_baselines(
    opportunities: Sequence[Opportunity],
    training_opportunities: Sequence[Opportunity],
    config: BacktestConfig | None = None,
    trading: TradingConfig | None = None,
    random_seeds: tuple[int, ...] = (11, 29, 47),
    raw_ofi_threshold: float = 0.0,
    momentum_threshold: float = 0.0,
    funding: Sequence[FundingPayment] = (),
) -> dict[str, BacktestResult]:
    """Run every required account on the same clock, capital, and cost settings."""
    if not training_opportunities:
        raise ValueError("training opportunities are required to calibrate random baseline")
    if opportunities and max(row.available_at for row in training_opportunities) >= min(
        row.available_at for row in opportunities
    ):
        raise ValueError("random-baseline calibration data must precede evaluation data")
    config = config or BacktestConfig()
    trading = trading or TradingConfig()
    calibrated = [
        RandomEntry.calibrate(
            list(training_opportunities),
            threshold=trading.long_threshold,
            seed=seed,
        )
        for seed in random_seeds
    ]
    # Trailing-price momentum selects its lookback/threshold on training data
    # only, mirroring how the random baseline is calibrated on training.
    del momentum_threshold  # superseded by training-only selection
    momentum = Momentum.select(list(training_opportunities))
    strategies = [
        MLFilteredOFI(),
        RawOFI(raw_ofi_threshold),
        momentum,
        Cash(),
        *calibrated,
    ]
    results: dict[str, BacktestResult] = {}
    for strategy in strategies:
        if isinstance(strategy, RandomEntry):
            strategy.name = f"random_seed_{strategy.seed}"
        results[strategy.name] = run_backtest(
            strategy,
            opportunities,
            funding=funding,
            config=config,
            trading=trading,
        )
    return results
