import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

logger = logging.getLogger("EventAlignment")


def build_alignment_engine(
    book_lf: pl.LazyFrame, trades_lf: pl.LazyFrame, klines_lf: pl.LazyFrame
) -> pl.LazyFrame:

    # combines L1 book, filtered trades, and klines into a 1-second grid. Every
    # input carries the normalized ``timestamp`` column as Datetime("ms") (the
    # Stage-2 contract), so we operate on it directly rather than via from_epoch.
    # The output grid is emitted as ``timestamp`` (Datetime), which is exactly the
    # key ofi.py sorts on and diffs to detect gaps.

    # shift klines forward by 1 minute before forward-filling: a bar covering
    # 09:00:00 - 09:00:59 becomes valid exactly at 09:01:00 (no lookahead bias).
    klines = (
        klines_lf.with_columns(
            (pl.col("timestamp") + pl.duration(minutes=1)).dt.truncate("1s").alias("dt_1s")
        )
        .drop("timestamp")
        .sort("dt_1s")
    )

    # truncate book updates to 1s, keep the final state per second. The raw
    # per-update timestamp is kept only as ``exact_time`` (to pick the last update
    # in the second and to measure staleness) and is dropped before the grid join
    # so it never collides with the emitted ``timestamp`` grid key.
    book = (
        book_lf.with_columns(
            [
                pl.col("timestamp").alias("exact_time"),
                pl.col("timestamp").dt.truncate("1s").alias("dt_1s"),
            ]
        )
        .drop("timestamp")
        .group_by("dt_1s")
        .agg(pl.all().sort_by("exact_time").last())
        .sort("dt_1s")
    )

    # aggregate trades per second (volume, signed_volume, VWAP). The trade inputs
    # are the wash-filtered *cleaned-flow* columns from Stage-4: buy_qty / sell_qty
    # / notional carry the runner's renames of buy_qty_clean / sell_qty_clean /
    # notional_clean, so trades flagged as wash never reach the aligned volume,
    # VWAP, or signed volume. Volume and notional are derived from the cleaned
    # split rather than a raw price*quantity so all three agree on the same tape.
    trades = (
        trades_lf.with_columns(pl.col("timestamp").dt.truncate("1s").alias("dt_1s"))
        .drop("timestamp")
        .group_by("dt_1s")
        .agg(
            [
                pl.col("buy_qty").sum().alias("buy_vol"),
                pl.col("sell_qty").sum().alias("sell_vol"),
                pl.col("notional").sum().alias("quote_vol"),
            ]
        )
        .with_columns((pl.col("buy_vol") + pl.col("sell_vol")).alias("volume"))
        .with_columns(
            [
                (pl.col("quote_vol") / pl.col("volume")).alias("vwap"),
                (pl.col("buy_vol") - pl.col("sell_vol")).alias("signed_volume"),
            ]
        )
        .drop(["quote_vol", "buy_vol", "sell_vol"])
        .sort("dt_1s")
    )

    # 1-second base grid spanning the full UTC day of the data (00:00:00 ..
    # 23:59:59), so every per-day aligned file has the same 86,400-row grid
    # regardless of when the first/last book update landed. Seconds before the
    # first native book (nothing to carry yet) are dropped after the join below;
    # trailing seconds are forward-filled and flagged stale. datetime_ranges is
    # inclusive of both bounds.
    day_start = pl.col("dt_1s").min().dt.truncate("1d")
    grid = book.select(
        pl.datetime_ranges(
            day_start,
            day_start + pl.duration(days=1) - pl.duration(seconds=1),
            "1s",
        ).alias("dt_1s")
    ).explode("dt_1s", empty_as_null=True)

    # ASOF join handles the forward-filling
    aligned = (
        grid.join_asof(book, on="dt_1s", strategy="backward")
        .join_asof(klines, on="dt_1s", strategy="backward")
        .join(trades, on="dt_1s", how="left")
    )

    # boundary rules: stale quotes, zeroed volumes
    aligned = aligned.with_columns(
        [
            # book_stale: this grid second had NO native book update (its book was
            # forward-filled from an earlier second). exact_time is the carried
            # update's time; if its truncated second != this second, the quote is
            # stale. OFI computed from a stale quote is meaningless, so any fill is
            # flagged — not just fills older than a fixed tolerance. A null carried
            # book (before the first update of the day) is stale too.
            (pl.col("exact_time").dt.truncate("1s") != pl.col("dt_1s"))
            .fill_null(True)
            .alias("book_stale"),
            # gap_prev_s: whole seconds since the last native book update (0 on a
            # native second, >0 on a forward-filled one). Provenance for OFI/analysis.
            (pl.col("dt_1s") - pl.col("exact_time").dt.truncate("1s"))
            .dt.total_seconds()
            .alias("gap_prev_s"),
            # trade defaults
            pl.col("volume").is_null().alias("no_trades"),
            pl.col("volume").fill_null(0.0),
            pl.col("signed_volume").fill_null(0.0),
        ]
    )
    # Drop the seconds before the first native book update — there is nothing to
    # carry forward there (a null carried book). Trailing seconds after the last
    # update are retained (forward-filled, flagged stale) to keep full-day coverage.
    aligned = aligned.filter(pl.col("exact_time").is_not_null())

    # book_stale_prev: the previous grid second was stale (or absent). ofi.py
    # derives its own t-1 flag internally, but carry it for provenance and parity
    # with the L2 aligned schema. Grid rows are already in ascending time order;
    # computed after the leading-drop so the first retained row's prev is null->True.
    return (
        aligned.with_columns(pl.col("book_stale").shift(1).fill_null(True).alias("book_stale_prev"))
        .drop("exact_time")
        # emit the 1-second grid as ``timestamp`` (Datetime) — the key ofi.py sorts
        # on and diffs; the raw book timestamp was dropped above so there is no clash.
        .rename({"dt_1s": "timestamp"})
    )


# ---------------------------------------------------------------------------
# Stage-3 runner: per-symbol-day I/O, mirroring the other stages (normalize_data,
# klines, filter_wash_trades) — argparse CLI + ProcessPoolExecutor, reading the
# normalized Stage-2/Stage-4 parquet and writing the Stage-5 input that ofi.py
# consumes at data/aligned/{SYMBOL}/{SYMBOL}-aligned-{date}.parquet.
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = "data"
DEFAULT_PROCESSED_DIR = "data/processed"
MIN_OUTPUT_ROWS = 100

# Columns pulled from each normalized input. Trades come from the wash-filtered
# Stage-4 output; its cleaned-flow columns are renamed to the generic names the
# transform expects. Neither trades nor klines carry ``asset`` into the join
# (the book already contributes it) to avoid a duplicate-column clash.
BOOK_COLUMNS = ["timestamp", "asset", "bid_price", "ask_price", "bid_qty", "ask_qty"]
KLINES_COLUMNS = ["timestamp", "close", "realized_vol", "ATR", "vol_regime"]


def book_path(processed_dir, symbol: str, date_str: str) -> Path:
    return Path(processed_dir) / symbol / "bookTicker" / f"{symbol}-bookTicker-{date_str}.parquet"


def trades_path(processed_dir, symbol: str, date_str: str) -> Path:
    return (
        Path(processed_dir)
        / symbol
        / "aggTrades_filtered"
        / f"{symbol}-aggTrades-{date_str}.parquet"
    )


def klines_path(processed_dir, symbol: str, date_str: str) -> Path:
    return Path(processed_dir) / symbol / "klines" / f"{symbol}-klines-{date_str}.parquet"


def aligned_path(data_dir, symbol: str, date_str: str) -> Path:
    """Output path. Matches ofi.py's ``aligned_path`` so Stage 5 reads exactly this."""
    return Path(data_dir) / "aligned" / symbol / f"{symbol}-aligned-{date_str}.parquet"


def align_one(processed_dir, data_dir, symbol: str, date_str: str) -> dict:
    """Align one symbol-day. Never raises; returns a status dict.
    status in {ok, skipped, missing, too_few_rows, error}."""
    try:
        bpath = book_path(processed_dir, symbol, date_str)
        if not bpath.exists():
            # Book defines the grid; without it there is nothing to align.
            return {"status": "missing", "symbol": symbol, "date": date_str}

        dest = aligned_path(data_dir, symbol, date_str)
        if dest.exists() and dest.stat().st_size > 0:
            return {"status": "skipped", "symbol": symbol, "date": date_str}

        book_lf = pl.scan_parquet(bpath).select(BOOK_COLUMNS)

        tpath = trades_path(processed_dir, symbol, date_str)
        if tpath.exists():
            # rename the wash filter's cleaned-flow columns to the generic names
            # the transform expects (see build_alignment_engine trade aggregation).
            trades_lf = pl.scan_parquet(tpath).select(
                pl.col("timestamp"),
                pl.col("buy_qty_clean").alias("buy_qty"),
                pl.col("sell_qty_clean").alias("sell_qty"),
                pl.col("notional_clean").alias("notional"),
            )
        else:
            trades_lf = pl.LazyFrame(
                schema={
                    "timestamp": pl.Datetime("ms"),
                    "buy_qty": pl.Float64,
                    "sell_qty": pl.Float64,
                    "notional": pl.Float64,
                }
            )

        kpath = klines_path(processed_dir, symbol, date_str)
        if kpath.exists():
            klines_lf = pl.scan_parquet(kpath).select(KLINES_COLUMNS)
        else:
            klines_lf = pl.LazyFrame(schema={"timestamp": pl.Datetime("ms"), "close": pl.Float64})

        aligned = build_alignment_engine(book_lf, trades_lf, klines_lf).collect()

        if aligned.height < MIN_OUTPUT_ROWS:
            logger.warning(
                "SKIPPED [%s | %s]: only %s aligned rows.", symbol, date_str, aligned.height
            )
            return {
                "status": "too_few_rows",
                "symbol": symbol,
                "date": date_str,
                "rows": aligned.height,
            }

        dest.parent.mkdir(parents=True, exist_ok=True)
        aligned.write_parquet(dest, compression="snappy")
        logger.info("SUCCESS [%s | %s]: %s rows -> %s", symbol, date_str, aligned.height, dest)
        return {"status": "ok", "symbol": symbol, "date": date_str, "rows": aligned.height}
    except Exception as exc:  # noqa: BLE001 - workers must never propagate
        logger.error("FAILED [%s | %s]: %s", symbol, date_str, exc, exc_info=True)
        return {"status": "error", "symbol": symbol, "date": date_str}


def generate_tasks(symbols, start_date, end_date) -> list[tuple[str, str]]:
    tasks: list[tuple[str, str]] = []
    delta = timedelta(days=1)
    for symbol in symbols:
        current = start_date
        while current <= end_date:
            tasks.append((symbol, current.strftime("%Y-%m-%d")))
            current += delta
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 3 - align L1 book/trades/klines to a 1-second grid"
    )
    parser.add_argument("--symbols", nargs="+", required=True, help="e.g. BTCUSDT ETHUSDT")
    parser.add_argument("--start", required=True, help="start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="end date YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="process pool size")
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="root data dir; aligned files go to <data-dir>/aligned/... (default ./data)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(processName)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if end < start:
        print(f"error: --end {end} is before --start {start}", file=sys.stderr)
        return 2

    tasks = generate_tasks(args.symbols, start, end)
    logger.info("Generated %s alignment tasks across %s workers", len(tasks), args.workers)

    results: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(align_one, args.processed_dir, args.data_dir, s, d) for s, d in tasks]
        for fut in as_completed(futures):
            st = fut.result()["status"]
            results[st] = results.get(st, 0) + 1
    logger.info("Alignment complete. Summary: %s", results)
    return 1 if results.get("error", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
