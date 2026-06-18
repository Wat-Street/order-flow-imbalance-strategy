# Layout::
#     data/aligned/{SYMBOL}/{SYMBOL}-aligned-{YYYY-MM-DD}.parquet   (input)
#     data/signals/{SYMBOL}/{SYMBOL}-ofi-A-{YYYY-MM-DD}.parquet     (output)
# Run (from the repo root)::
#     python src/order_flow_imbalance_strategy/ofi.py --symbols BTCUSDT ETHUSDT SOLUSDT \
#         --start 2021-01-01 --end 2025-12-31 --workers 4
# or, after ``pip install -e .``::
#     python -m order_flow_imbalance_strategy.ofi --symbols BTCUSDT \
#         --start 2022-01-01 --end 2024-12-31
# Nulling rules (applied *before* trusting any contribution):
#   * Row 0 of each day has no ``t-1`` -> all three outputs null (handled naturally by
#       ``shift(1)`` producing nulls that the null-mask catches).
#   * If any required column or its shifted counterpart is null -> outputs null.
#   * If ``book_stale`` is True for the current OR previous row -> outputs null,
#       because OFI computed from a stale quote is meaningless.
#   * If the gap to the previous snapshot is not exactly 1 second (a missing second in
#       the grid) -> outputs null, because OFI is only defined between adjacent seconds
#   and ``shift(1)`` would otherwise silently bridge the gap.
#   * If the book is crossed or locked (``ask_price <= bid_price``, i.e. non-positive
#       spread) on the current OR previous row -> outputs null, because that is corrupt
#       Level-1 data.


from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from tqdm import tqdm

#: Columns that must be present in an aligned input frame.
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "bid_price",
    "ask_price",
    "bid_qty",
    "ask_qty",
)

#: Columns appended by :func:`compute_ofi`.
OUTPUT_COLUMNS: tuple[str, ...] = ("ofi_1s", "bid_contribution", "ask_contribution")


def compute_ofi(df: pl.DataFrame) -> pl.DataFrame:
    """Compute the raw 1-second OFI signal for a single day's aligned frame.

    Returns a new frame with every original column preserved (same order, same
    row count) plus ``ofi_1s``, ``bid_contribution`` and ``ask_contribution``.

    The input is sorted by ``timestamp`` to guarantee that ``shift(1)`` yields
    the true previous snapshot regardless of how the parquet was written.
    """
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input frame missing required columns: {missing}")

    original_columns = df.columns
    df = df.sort("timestamp")

    # ``book_stale`` is part of the aligned spec; default to "not stale" if a
    # test frame omits it so the rest of the logic stays uniform.
    if "book_stale" in df.columns:
        book_stale = pl.col("book_stale")
    else:
        book_stale = pl.lit(False)  # noqa: FBT003 - boolean literal is intentional

    df = df.with_columns(
        pl.col("bid_price").shift(1).alias("prev_bid_price"),
        pl.col("ask_price").shift(1).alias("prev_ask_price"),
        pl.col("bid_qty").shift(1).alias("prev_bid_qty"),
        pl.col("ask_qty").shift(1).alias("prev_ask_qty"),
        book_stale.alias("_book_stale"),
        book_stale.shift(1).alias("_book_stale_prev"),
    )

    # Bid contribution e_b, piecewise on the best-bid price move.
    e_b = (
        pl.when(pl.col("bid_price") > pl.col("prev_bid_price"))
        .then(pl.col("bid_qty"))
        .when(pl.col("bid_price") == pl.col("prev_bid_price"))
        .then(pl.col("bid_qty") - pl.col("prev_bid_qty"))
        .otherwise(-pl.col("prev_bid_qty"))  # bid_price[t] < bid_price[t-1]
        .cast(pl.Float64)
    )

    # Ask contribution e_a, piecewise on the best-ask price move.
    e_a = (
        pl.when(pl.col("ask_price") > pl.col("prev_ask_price"))
        .then(-pl.col("prev_ask_qty"))
        .when(pl.col("ask_price") == pl.col("prev_ask_price"))
        .then(pl.col("ask_qty") - pl.col("prev_ask_qty"))
        .otherwise(pl.col("ask_qty"))  # ask_price[t] < ask_price[t-1]
        .cast(pl.Float64)
    )

    # Gap to the previous snapshot. Anything other than exactly 1 second means a
    # row for some second is missing, so OFI would be computed across a
    # discontinuity. ``diff()`` is null on row 0 (caught as warmup below).
    seconds_since_prev = pl.col("timestamp").diff() != pl.duration(seconds=1)

    # A row is invalid (-> null outputs) if it is a warmup row, has any missing
    # required value, sits on/after a stale book snapshot, sits across a gap in the
    # 1-second grid, or has a crossed/locked book (non-positive spread) now or at
    # t-1. ``fill_null(True)`` treats an unknown flag/gap as bad (conservative).
    invalid = (
        pl.col("bid_price").is_null()
        | pl.col("ask_price").is_null()
        | pl.col("bid_qty").is_null()
        | pl.col("ask_qty").is_null()
        | pl.col("prev_bid_price").is_null()
        | pl.col("prev_ask_price").is_null()
        | pl.col("prev_bid_qty").is_null()
        | pl.col("prev_ask_qty").is_null()
        | pl.col("_book_stale").fill_null(True)
        | pl.col("_book_stale_prev").fill_null(True)
        | seconds_since_prev.fill_null(True)
        | (pl.col("ask_price") <= pl.col("bid_price"))
        | (pl.col("prev_ask_price") <= pl.col("prev_bid_price"))
    )

    null_f64 = pl.lit(None, dtype=pl.Float64)
    df = df.with_columns(
        pl.when(invalid).then(null_f64).otherwise(e_b).alias("bid_contribution"),
        pl.when(invalid).then(null_f64).otherwise(e_a).alias("ask_contribution"),
    ).with_columns((pl.col("bid_contribution") - pl.col("ask_contribution")).alias("ofi_1s"))

    # Restore the original columns (dropping all helpers) and append outputs.
    return df.select(*original_columns, *OUTPUT_COLUMNS)


def validate_ofi(df: pl.DataFrame, expected_rows: int) -> list[str]:
    """Return a list of hard validation problems; empty means the frame is safe to write.

    Hard checks (block the write):

    * row count exactly matches the input frame,
    * ``ofi_1s`` has no inf / NaN among its non-null rows,
    * wherever ``ofi_1s`` is non-null, both contributions are non-null (and vice versa).
    """
    problems: list[str] = []

    if df.height != expected_rows:
        problems.append(f"row count {df.height} != input {expected_rows}")

    ofi = df["ofi_1s"]
    non_finite = ofi.is_finite().not_() & ofi.is_not_null()
    if non_finite.any():
        problems.append(f"{int(non_finite.sum())} non-finite ofi_1s values (inf/NaN)")

    ofi_present = ofi.is_not_null()
    bid_present = df["bid_contribution"].is_not_null()
    ask_present = df["ask_contribution"].is_not_null()
    inconsistent = (ofi_present != bid_present) | (ofi_present != ask_present)
    if inconsistent.any():
        problems.append(
            f"{int(inconsistent.sum())} rows where ofi_1s/contribution null-state disagrees"
        )

    return problems


def process_file(task: dict[str, Any]) -> dict[str, Any]:
    """Worker: compute OFI for one symbol-day and write the output parquet.

    Never raises; always returns a status dict so a single bad day cannot crash
    the pool. ``status`` is one of ``ok``, ``skipped``, ``no_input``, ``error``.
    """
    symbol = task["symbol"]
    date = task["date"]
    in_path = Path(task["in_path"])
    out_path = Path(task["out_path"])
    status: dict[str, Any] = {"symbol": symbol, "date": date, "status": "error"}

    try:
        if out_path.exists():
            status["status"] = "skipped"
            status["message"] = "output already exists"
            return status

        if not in_path.exists():
            status["status"] = "no_input"
            status["message"] = f"missing input {in_path}"
            return status

        df = pl.read_parquet(in_path)
        expected_rows = df.height

        out = compute_ofi(df)

        problems = validate_ofi(out, expected_rows)
        if problems:
            status["message"] = "validation failed: " + "; ".join(problems)
            return status

        mean_ofi = out["ofi_1s"].mean()
        std_ofi = out["ofi_1s"].std()
        valid_rows = int(out["ofi_1s"].is_not_null().sum())

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        out.write_parquet(tmp_path)
        tmp_path.replace(out_path)

        status.update(
            status="ok",
            rows=out.height,
            valid_rows=valid_rows,
            mean_ofi=None if mean_ofi is None else float(mean_ofi),
            std_ofi=None if std_ofi is None else float(std_ofi),
        )
        return status
    except Exception as exc:  # noqa: BLE001 - workers must never propagate
        status["message"] = f"{type(exc).__name__}: {exc}"
        return status


# --- CLI -------------------------------------------------------------------


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}, expected YYYY-MM-DD") from exc


def daterange(start: date, end: date) -> list[date]:
    """Inclusive list of dates from ``start`` to ``end``."""
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def aligned_path(data_dir: Path, symbol: str, day: date) -> Path:
    return data_dir / "aligned" / symbol / f"{symbol}-aligned-{day:%Y-%m-%d}.parquet"


def signal_path(data_dir: Path, symbol: str, day: date) -> Path:
    return data_dir / "signals" / symbol / f"{symbol}-ofi-A-{day:%Y-%m-%d}.parquet"


def build_tasks(symbols: list[str], start: date, end: date, data_dir: Path) -> list[dict[str, Any]]:
    """Build the full task list upfront (only days whose aligned input exists)."""
    tasks: list[dict[str, Any]] = []
    for symbol in symbols:
        for day in daterange(start, end):
            in_path = aligned_path(data_dir, symbol, day)
            if not in_path.exists():
                continue
            tasks.append(
                {
                    "symbol": symbol,
                    "date": f"{day:%Y-%m-%d}",
                    "in_path": str(in_path),
                    "out_path": str(signal_path(data_dir, symbol, day)),
                }
            )
    return tasks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 5 - compute the raw 1-second Order Flow Imbalance (OFI) signal"
    )
    parser.add_argument("--symbols", nargs="+", required=True, help="e.g. BTCUSDT ETHUSDT")
    parser.add_argument("--start", type=parse_date, required=True, help="start date YYYY-MM-DD")
    parser.add_argument("--end", type=parse_date, required=True, help="end date YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=4, help="process pool size (default 4)")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="root data dir (default ./data)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.end < args.start:
        print(f"error: --end {args.end} is before --start {args.start}", file=sys.stderr)
        return 2

    tasks = build_tasks(args.symbols, args.start, args.end, args.data_dir)
    if not tasks:
        print("No aligned input files found for the requested symbols/range.")
        return 0

    counts: Counter[str] = Counter()
    drift_warnings: list[str] = []
    failures: list[dict[str, Any]] = []

    workers = max(1, args.workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_file, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="OFI", unit="day"):
            result = future.result()  # process_file never raises
            counts[result["status"]] += 1

            if result["status"] == "error":
                failures.append(result)
            elif result["status"] == "ok":
                mean_ofi = result.get("mean_ofi")
                std_ofi = result.get("std_ofi")
                # Flag a suspicious persistent drift: |mean| large vs spread.
                if mean_ofi is not None and std_ofi and std_ofi > 0:
                    if abs(mean_ofi) > 0.1 * std_ofi:
                        drift_warnings.append(
                            f"{result['symbol']} {result['date']}: "
                            f"mean OFI {mean_ofi:.3g} (std {std_ofi:.3g})"
                        )

    print("\nSummary:")
    for status in ("ok", "skipped", "no_input", "error"):
        if counts[status]:
            print(f"  {status:>9}: {counts[status]}")

    if drift_warnings:
        print(f"\n{len(drift_warnings)} day(s) with notable OFI drift (check sign convention):")
        for line in drift_warnings[:20]:
            print(f"  - {line}")
        if len(drift_warnings) > 20:
            print(f"  ... and {len(drift_warnings) - 20} more")

    if failures:
        print(f"\n{len(failures)} failure(s):", file=sys.stderr)
        for f in failures[:20]:
            print(f"  - {f['symbol']} {f['date']}: {f.get('message')}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
