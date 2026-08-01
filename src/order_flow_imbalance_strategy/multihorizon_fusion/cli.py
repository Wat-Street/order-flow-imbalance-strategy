from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import polars as pl
from tqdm import tqdm

from order_flow_imbalance_strategy.multihorizon_fusion.compute import (
    compute_fusion,
    validate_fusion,
)
from order_flow_imbalance_strategy.multihorizon_fusion.config import (
    DEFAULT_CONFIG,
    FusionConfig,
    load_config,
)

SYMBOL_PATTERN = re.compile(r"[A-Z0-9]+")


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}, expected YYYY-MM-DD") from exc


def parse_symbol(value: str) -> str:
    """Normalize a Binance symbol and reject values that could escape the data directory."""
    symbol = value.strip().upper()
    if not SYMBOL_PATTERN.fullmatch(symbol):
        raise argparse.ArgumentTypeError(
            f"invalid symbol {value!r}; expected letters and numbers only"
        )
    return symbol


def daterange(start: date, end: date) -> list[date]:
    """Inclusive list of dates from ``start`` to ``end``."""
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def input_path(data_dir: Path, symbol: str, day: date) -> Path:
    return data_dir / "signals" / symbol / f"{symbol}-ofi-B-{day:%Y-%m-%d}.parquet"


def output_path(data_dir: Path, symbol: str, day: date) -> Path:
    return data_dir / "signals" / symbol / f"{symbol}-ofi-C-{day:%Y-%m-%d}.parquet"


def history_path(data_dir: Path, symbol: str, day: date) -> Path:
    """Previous day's stage-B file, used only to warm causal horizon state."""
    return input_path(data_dir, symbol, day - timedelta(days=1))


def build_tasks(
    symbols: list[str],
    start: date,
    end: date,
    data_dir: Path,
) -> list[dict[str, Any]]:
    """Build every requested symbol-day task so missing inputs are reported."""
    tasks: list[dict[str, Any]] = []
    for symbol in dict.fromkeys(symbols):
        for day in daterange(start, end):
            in_path = input_path(data_dir, symbol, day)
            tasks.append(
                {
                    "symbol": symbol,
                    "date": f"{day:%Y-%m-%d}",
                    "in_path": str(in_path),
                    "out_path": str(output_path(data_dir, symbol, day)),
                    "history_path": str(history_path(data_dir, symbol, day)),
                }
            )
    return tasks


def process_file(task: dict[str, Any]) -> dict[str, Any]:
    """Worker: fuse one symbol-day and write the output parquet.

    Never raises; always returns a status dict so a single bad day cannot crash
    the pool. ``status`` is one of ``ok``, ``skipped``, ``no_input``, ``error``.
    """
    symbol = str(task.get("symbol", "<unknown>"))
    day = str(task.get("date", "<unknown>"))
    status: dict[str, Any] = {"symbol": symbol, "date": day, "status": "error"}

    try:
        in_path = Path(task["in_path"])
        out_path = Path(task["out_path"])
        prior_path = Path(task["history_path"]) if task.get("history_path") else None
        overwrite = bool(task.get("overwrite", False))

        if out_path.exists() and not overwrite:
            status["status"] = "skipped"
            status["message"] = "output already exists"
            return status

        if not in_path.exists():
            status["status"] = "no_input"
            status["message"] = f"missing input {in_path}"
            return status

        config = task.get("config", DEFAULT_CONFIG)
        if not isinstance(config, FusionConfig):
            raise TypeError("task config must be a FusionConfig")

        df = pl.read_parquet(in_path)
        history = pl.read_parquet(prior_path) if prior_path and prior_path.is_file() else None
        out = compute_fusion(df, config, history=history)

        problems = validate_fusion(df, out, config)
        if problems:
            status["message"] = "validation failed: " + "; ".join(problems)
            return status

        stats: dict[str, float | None] = {}
        for col in config.output_columns:
            mean = out[col].mean()
            stats[f"mean_{col}"] = None if mean is None else float(mean)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=out_path.parent,
            prefix=f".{out_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
        try:
            out.write_parquet(tmp_path, compression="snappy")
            tmp_path.replace(out_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        status.update(
            status="ok",
            rows=out.height,
            **stats,
        )
        return status
    except Exception as exc:  # noqa: BLE001 - workers must never propagate
        status["message"] = f"{type(exc).__name__}: {exc}"
        return status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 7 - multi-horizon exponential fusion of spoof-adjusted OFI (ofi-B -> ofi-C)"
        )
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        type=parse_symbol,
        help="e.g. BTCUSDT ETHUSDT",
    )
    parser.add_argument("--start", type=parse_date, required=True, help="start date YYYY-MM-DD")
    parser.add_argument("--end", type=parse_date, required=True, help="end date YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=4, help="process pool size (default 4)")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="root data dir (default ./data)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="optional JSON config for horizon half-lives (default: built-in 1m/5m/15m)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing ofi-C outputs atomically",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.end < args.start:
        print(f"error: --end {args.end} is before --start {args.start}", file=sys.stderr)
        return 2

    if args.config is not None and not args.config.is_file():
        print(f"error: config file not found: {args.config}", file=sys.stderr)
        return 2

    if args.workers < 1:
        print("error: --workers must be >= 1", file=sys.stderr)
        return 2

    # Validate config eagerly so misconfiguration fails fast before spawning workers.
    try:
        config = load_config(args.config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    tasks = build_tasks(args.symbols, args.start, args.end, args.data_dir)
    if not tasks:
        print("No ofi-B input files found for the requested symbols/range.")
        return 0

    for task in tasks:
        task["config"] = config
        task["overwrite"] = args.overwrite

    counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    runnable_tasks: list[dict[str, Any]] = []
    for task in tasks:
        input_exists = Path(task["in_path"]).is_file()
        output_can_skip = Path(task["out_path"]).is_file() and not args.overwrite
        if not input_exists and not output_can_skip:
            counts["no_input"] += 1
        else:
            runnable_tasks.append(task)

    if runnable_tasks:
        desc = f"Fusion ({', '.join(config.output_columns)})"

        def record_result(result: dict[str, Any]) -> None:
            counts[result["status"]] += 1
            if result["status"] == "error":
                failures.append(result)

        if args.workers == 1:
            for task in tqdm(runnable_tasks, desc=desc, unit="day"):
                record_result(process_file(task))
        else:
            try:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futures = {pool.submit(process_file, task): task for task in runnable_tasks}
                    for future in tqdm(
                        as_completed(futures), total=len(runnable_tasks), desc=desc, unit="day"
                    ):
                        task = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:  # noqa: BLE001 - failure is isolated by task
                            result = {
                                "symbol": task["symbol"],
                                "date": task["date"],
                                "status": "error",
                                "message": f"worker failed: {type(exc).__name__}: {exc}",
                            }
                        record_result(result)
            except Exception as exc:  # noqa: BLE001 - report process-pool setup failures cleanly
                print(f"error: process pool failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1

    print("\nSummary:")
    for status in ("ok", "skipped", "no_input", "error"):
        if counts[status]:
            print(f"  {status:>9}: {counts[status]}")

    if failures:
        print(f"\n{len(failures)} failure(s):", file=sys.stderr)
        for failure in failures[:20]:
            print(
                f"  - {failure['symbol']} {failure['date']}: {failure.get('message')}",
                file=sys.stderr,
            )
        return 1

    return 0
