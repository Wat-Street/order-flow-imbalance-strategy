from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

from .comparison import run_required_baselines
from .contracts import BacktestConfig, TradingConfig
from .engine import run_backtest
from .io import load_funding, load_opportunities, sha256_file
from .reporting import write_comparison_json, write_result_json
from .strategies import Cash, MLFilteredOFI, Momentum, RandomEntry, RawOFI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a cost-aware OFI execution study.")
    parser.add_argument("--input", type=Path, required=True, help="joined full-clock Parquet input")
    parser.add_argument("--output", type=Path, required=True, help="atomic JSON report output")
    parser.add_argument(
        "--strategy",
        choices=("all", "ml", "raw_ofi", "random", "momentum", "cash"),
        default="all",
    )
    parser.add_argument(
        "--calibration-input", type=Path, help="earlier rows used only for random calibration"
    )
    parser.add_argument("--funding", type=Path, help="CSV or Parquet funding settlement records")
    parser.add_argument("--model-id", help="saved model bundle identity for prediction rows")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=0.5)
    parser.add_argument("--latency-seconds", type=float, default=1.0)
    parser.add_argument("--max-quote-age-seconds", type=float, default=1.0)
    parser.add_argument("--participation-rate", type=float, default=0.10)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--lot-size", type=float, default=0.001)
    parser.add_argument("--quantity", type=float, default=0.001)
    parser.add_argument("--max-position", type=float, default=0.001)
    parser.add_argument("--holding-seconds", type=float, default=30.0)
    parser.add_argument("--long-threshold", type=float, default=0.2)
    parser.add_argument("--short-threshold", type=float, default=-0.2)
    parser.add_argument("--raw-ofi-threshold", type=float, default=0.0)
    parser.add_argument("--momentum-threshold", type=float, default=0.0)
    parser.add_argument(
        "--momentum-lookback-seconds",
        type=float,
        default=60.0,
        help="trailing-price momentum window when no calibration input is given",
    )
    parser.add_argument("--strict-funding", action="store_true")
    parser.add_argument(
        "--funding-interval-hours",
        type=float,
        help="expected settlement cadence for strict funding coverage checks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BacktestConfig(
        initial_cash=args.initial_cash,
        latency=timedelta(seconds=args.latency_seconds),
        max_quote_age=timedelta(seconds=args.max_quote_age_seconds),
        fee_bps=args.fee_bps,
        extra_slippage_bps=args.slippage_bps,
        participation_rate=args.participation_rate,
        tick_size=args.tick_size,
        lot_size=args.lot_size,
        require_funding=args.strict_funding,
        funding_interval=(
            timedelta(hours=args.funding_interval_hours)
            if args.funding_interval_hours is not None
            else None
        ),
    )
    trading = TradingConfig(
        order_quantity=args.quantity,
        max_position=args.max_position,
        holding_time=timedelta(seconds=args.holding_seconds),
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
    )
    opportunities = load_opportunities(args.input, args.model_id)
    funding = load_funding(args.funding) if args.funding else []
    metadata = {
        "input_path": str(args.input),
        "input_sha256": sha256_file(args.input),
        "funding_path": str(args.funding) if args.funding else None,
        "funding_sha256": sha256_file(args.funding) if args.funding else None,
        "funding_mark_policy": "causal mark supplied per settlement" if funding else "not supplied",
        "calibration_path": str(args.calibration_input) if args.calibration_input else None,
    }

    if args.strategy == "all":
        if args.calibration_input is None:
            raise SystemExit("--calibration-input is required for --strategy all")
        training = load_opportunities(args.calibration_input, args.model_id)
        if max(row.available_at for row in training) >= min(
            row.available_at for row in opportunities
        ):
            raise SystemExit("random-baseline calibration data must precede evaluation data")
        metadata["calibration_sha256"] = sha256_file(args.calibration_input)
        results = run_required_baselines(
            opportunities,
            training,
            config,
            trading,
            tuple(args.seeds),
            args.raw_ofi_threshold,
            args.momentum_threshold,
            funding,
        )
        write_comparison_json(args.output, results, metadata)
        return 0

    if args.strategy == "ml":
        strategy = MLFilteredOFI()
    elif args.strategy == "raw_ofi":
        strategy = RawOFI(args.raw_ofi_threshold)
    elif args.strategy == "momentum":
        if args.calibration_input is not None:
            training = load_opportunities(args.calibration_input, args.model_id)
            if max(row.available_at for row in training) >= min(
                row.available_at for row in opportunities
            ):
                raise SystemExit("momentum selection data must precede evaluation data")
            metadata["calibration_sha256"] = sha256_file(args.calibration_input)
            strategy = Momentum.select(training)
        else:
            strategy = Momentum(
                lookback=timedelta(seconds=args.momentum_lookback_seconds),
                threshold=args.momentum_threshold,
            )
    elif args.strategy == "cash":
        strategy = Cash()
    else:
        if args.calibration_input is None:
            raise SystemExit("--calibration-input is required for --strategy random")
        training = load_opportunities(args.calibration_input, args.model_id)
        if max(row.available_at for row in training) >= min(
            row.available_at for row in opportunities
        ):
            raise SystemExit("random-baseline calibration data must precede evaluation data")
        metadata["calibration_sha256"] = sha256_file(args.calibration_input)
        strategy = RandomEntry.calibrate(
            training,
            threshold=trading.long_threshold,
            seed=args.seeds[0],
        )

    result = run_backtest(strategy, opportunities, funding, config, trading)
    write_result_json(args.output, result, metadata=metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
