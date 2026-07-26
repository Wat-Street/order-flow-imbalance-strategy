"""Stage 3 - Event Alignment: resample the three normalized L1 streams onto a
single, regular time grid.

bookTicker (irregular, hundreds/s), aggTrades (irregular), and klines (fixed
1-minute bars) live on three incompatible clocks and cannot be joined directly.
This stage collapses them onto one canonical grid (default 1 second) so each row
is the full market state for that interval.

The grid is the spine: it is built from the UTC calendar day, independently of
the data, and every source is *left-joined onto it* by the rule appropriate to
its nature -- so a second with no data becomes an explicit, flagged row rather
than an absent one.

Levels vs flows (the distinction the whole stage hinges on):

* **Book = level.** A quote persists until the next update, so it is carried
  forward (as-of *backward* join). The carry is *capped* at ``max_book_staleness_s``:
  beyond that horizon the quote is not trusted -- the level columns are nulled and
  ``book_dead`` is set, so a dead feed is never mistaken for a quiet market.
* **Trades = flow.** Volumes accumulate, so they are *summed* per interval and are
  **never** forward-filled -- an empty second is 0, flagged ``no_trades``. Carrying
  a flow would fabricate trades that never happened.
* **Klines = slow context.** A 1-minute bar is upsampled by carrying the most
  recently *completed* bar forward. The bar covering minute M is only known at the
  end of M, so it is shifted one bar (attached to M+1 onward) -- the single most
  important anti-lookahead rule here.

Two staleness signals are emitted so both consumers are served:

* ``book_stale`` -- strict: True on *any* forward-filled interval (no native
  update this second). ofi.py gates on this so OFI is never diffed across a fill.
* ``book_dead`` -- coarse: True when the carried quote is older than
  ``max_book_staleness_s`` (or absent). A feed-health flag for the model/backtester.

Anti-lookahead is guaranteed by construction: every source is joined with an
as-of *backward* strategy (and klines are shifted a full bar first), so no feature
at time t can draw on an input timestamped after t.

Day-boundary policy: each day is processed independently (cold start). The grid
always spans the full UTC day; seconds before the first book update carry a null,
``book_dead`` book, and klines context is null until the first completed bar of
the day is known. Nothing is seeded from the prior day.

Output: data/aligned/{SYMBOL}/{SYMBOL}-aligned-{YYYY-MM-DD}.parquet -- exactly the
artifact ofi.py (Stage 5) consumes.

Run::

    python -m order_flow_imbalance_strategy.event_alignment \\
        --symbols BTCUSDT ETHUSDT SOLUSDT --start 2022-01-01 --end 2024-12-31 \\
        --grid 1s --workers 4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

logger = logging.getLogger("EventAlignment")

# --- column groups ---------------------------------------------------------

# Book level columns from the normalized bookTicker: carried forward within the
# staleness horizon, nulled beyond it. A "level" persists until the next update.
BOOK_LEVEL_COLUMNS = [
    "bid_price",
    "ask_price",
    "bid_qty",
    "ask_qty",
    "mid_price",
    "spread",
    "queue_imbalance",
]

# Slow-context columns from klines: previous completed bar, carried forward.
KLINES_CONTEXT_COLUMNS = ["close", "realized_vol", "ATR", "vol_regime"]

# Flow columns: summed per interval, 0-filled on empty seconds, NEVER carried.
# (vwap is derived and left null on empty seconds -- it is not zero-filled.)
FLOW_PRIMITIVES = ["buy_qty", "sell_qty", "notional", "trade_count"]
FLOW_ZERO_FILL = [*FLOW_PRIMITIVES, "signed_volume", "volume"]

DEFAULT_DATA_DIR = "data"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_TRADES_SUBDIR = "aggTrades_filtered"
DEFAULT_GRID = "1s"
DEFAULT_MAX_BOOK_STALENESS_S = 5.0


# --- pure transform --------------------------------------------------------


def build_day_grid(day: date, grid: str = DEFAULT_GRID) -> pl.Series:
    """Canonical spine for one UTC day, built from the calendar (not the data).

    Half-open ``[00:00:00, next-midnight)``: for a 1s grid that is 86,400 points,
    but the count is *derived* from ``grid`` rather than assumed.
    """
    start = datetime(day.year, day.month, day.day)
    return pl.datetime_range(
        start,
        start + timedelta(days=1),
        interval=grid,
        closed="left",
        time_unit="ms",
        eager=True,
    ).alias("timestamp")


def build_alignment_engine(
    book_lf: pl.LazyFrame,
    trades_lf: pl.LazyFrame,
    klines_lf: pl.LazyFrame,
    *,
    day: date | None = None,
    grid: str = DEFAULT_GRID,
    max_book_staleness_s: float | None = DEFAULT_MAX_BOOK_STALENESS_S,
) -> pl.LazyFrame:
    """Align the three normalized streams onto the grid. Pure (no file I/O).

    Every input carries ``timestamp`` as ``Datetime("ms")`` (the Stage-2 contract).
    ``day`` fixes the calendar spine; when None it is derived from the first book
    update. Returns a LazyFrame keyed on ``timestamp`` -- the column ofi.py sorts
    on and diffs to detect gaps.
    """
    # --- book (level): last update per interval; keep exact time for staleness --
    book = (
        book_lf.with_columns(
            [
                pl.col("timestamp").alias("exact_time"),
                pl.col("timestamp").dt.truncate(grid).alias("ts"),
            ]
        )
        .drop("timestamp")
        .group_by("ts")
        .agg(pl.all().sort_by("exact_time").last())
        .sort("ts")
    )

    # --- trades (flow): sum per interval; NEVER forward-filled ------------------
    # Inputs are the generic cleaned-flow names (buy_qty/sell_qty/notional); the
    # runner maps the wash filter's *_clean columns onto them.
    trades = (
        trades_lf.with_columns(pl.col("timestamp").dt.truncate(grid).alias("ts"))
        .drop("timestamp")
        .group_by("ts")
        .agg(
            [
                pl.col("buy_qty").sum().alias("buy_qty"),
                pl.col("sell_qty").sum().alias("sell_qty"),
                pl.col("notional").sum().alias("notional"),
                pl.len().cast(pl.Int64).alias("trade_count"),
            ]
        )
        .with_columns(
            [
                (pl.col("buy_qty") + pl.col("sell_qty")).alias("volume"),
                (pl.col("buy_qty") - pl.col("sell_qty")).alias("signed_volume"),
            ]
        )
        .with_columns(
            pl.when(pl.col("volume") > 0)
            .then(pl.col("notional") / pl.col("volume"))
            .otherwise(None)
            .alias("vwap")
        )
        .sort("ts")
    )

    # --- klines (slow context): shift one bar => PREVIOUS completed bar only ----
    # timestamp is the bar's open_time; +1 minute makes it known only from the
    # next bar onward, so a bar never attaches to the seconds it summarizes.
    klines = (
        klines_lf.with_columns(
            (pl.col("timestamp") + pl.duration(minutes=1)).dt.truncate(grid).alias("ts")
        )
        .drop("timestamp")
        .sort("ts")
    )

    # --- the spine: full calendar day, independent of the data -----------------
    if day is None:
        first = book.select(pl.col("ts").min()).collect().item()
        if first is None:
            return pl.LazyFrame(schema={"timestamp": pl.Datetime("ms")})
        day = first.date()

    grid_lf = pl.DataFrame(build_day_grid(day, grid)).rename({"timestamp": "ts"}).lazy()

    # as-of BACKWARD forward-fills each level source; left join keeps flows sparse.
    aligned = (
        grid_lf.join_asof(book, on="ts", strategy="backward")
        .join_asof(klines, on="ts", strategy="backward")
        .join(trades, on="ts", how="left")
    )

    # --- staleness flags + capped forward-fill ---------------------------------
    aligned = aligned.with_columns(
        (pl.col("ts") - pl.col("exact_time").dt.truncate(grid))
        .dt.total_seconds()
        .alias("gap_prev_s")
    )
    aligned = aligned.with_columns(
        # strict: any interval with no native update (what ofi.py gates on).
        (pl.col("exact_time").dt.truncate(grid) != pl.col("ts")).fill_null(True).alias("book_stale")
    )
    if max_book_staleness_s and max_book_staleness_s > 0:
        dead = pl.col("gap_prev_s").is_null() | (pl.col("gap_prev_s") > max_book_staleness_s)
    else:
        # horizon disabled: carry the quote indefinitely; only a total absence
        # of any prior book (before the first update) is "dead".
        dead = pl.col("gap_prev_s").is_null()
    aligned = aligned.with_columns(dead.alias("book_dead"))
    # cap the carry: a quote older than the horizon (or absent) is not trusted.
    aligned = aligned.with_columns(
        [
            pl.when(pl.col("book_dead")).then(None).otherwise(pl.col(c)).alias(c)
            for c in BOOK_LEVEL_COLUMNS
        ]
    )

    # --- flow defaults: 0 on empty seconds, flagged; vwap stays null ------------
    aligned = aligned.with_columns(pl.col("trade_count").is_null().alias("no_trades"))
    aligned = aligned.with_columns([pl.col(c).fill_null(0).alias(c) for c in FLOW_ZERO_FILL])

    # book_stale_prev: provenance / parity with the L2 aligned schema.
    aligned = aligned.with_columns(
        pl.col("book_stale").shift(1).fill_null(True).alias("book_stale_prev")
    )

    ordered = [
        "ts",
        *BOOK_LEVEL_COLUMNS,
        "buy_qty",
        "sell_qty",
        "notional",
        "trade_count",
        "signed_volume",
        "volume",
        "vwap",
        *KLINES_CONTEXT_COLUMNS,
        "book_stale",
        "book_dead",
        "book_stale_prev",
        "no_trades",
        "gap_prev_s",
    ]
    present = set(aligned.collect_schema().names())
    return aligned.select([c for c in ordered if c in present]).rename({"ts": "timestamp"})


# --- validation ------------------------------------------------------------


def validate_aligned(df: pl.DataFrame, expected_rows: int) -> None:
    """Pre-write invariants; raises ``AssertionError`` on any violation.

    * exactly ``expected_rows`` rows (the calendar grid count for this day/grid),
    * strictly monotonic, unique ``timestamp`` (the spine is intact),
    * all flow primitives >= 0 (signed_volume is excluded -- it is signed),
    * no null level columns wherever the book is not ``book_dead``.

    Anti-lookahead is guaranteed structurally (backward as-of joins + the klines
    one-bar shift), so it is enforced by construction and covered by tests rather
    than re-derived here.
    """
    if df.height != expected_rows:
        raise AssertionError(f"row count {df.height} != expected grid {expected_rows}")
    if not df["timestamp"].is_sorted():
        raise AssertionError("timestamp is not monotonic")
    if df["timestamp"].n_unique() != df.height:
        raise AssertionError("duplicate timestamps on the grid")
    for c in FLOW_PRIMITIVES + ["volume"]:
        mn = df[c].min()
        if mn is not None and mn < 0:
            raise AssertionError(f"negative flow in {c}: {mn}")
    live = df.filter(~pl.col("book_dead"))
    for c in BOOK_LEVEL_COLUMNS:
        if c in live.columns and live[c].null_count():
            raise AssertionError(f"{live[c].null_count()} null values in level column {c}")


# --- runner: per-symbol-day I/O -------------------------------------------


def book_path(processed_dir, symbol: str, date_str: str) -> Path:
    return Path(processed_dir) / symbol / "bookTicker" / f"{symbol}-bookTicker-{date_str}.parquet"


def trades_path(processed_dir, symbol: str, date_str: str, subdir: str) -> Path:
    return Path(processed_dir) / symbol / subdir / f"{symbol}-aggTrades-{date_str}.parquet"


def klines_path(processed_dir, symbol: str, date_str: str) -> Path:
    return Path(processed_dir) / symbol / "klines" / f"{symbol}-klines-{date_str}.parquet"


def aligned_path(data_dir, symbol: str, date_str: str) -> Path:
    """Output path. Matches ofi.py's ``aligned_path`` so Stage 5 reads exactly this."""
    return Path(data_dir) / "aligned" / symbol / f"{symbol}-aligned-{date_str}.parquet"


def _load_trades(processed_dir, symbol: str, date_str: str, subdir: str) -> pl.LazyFrame:
    """Load trades as (timestamp, buy_qty, sell_qty, notional).

    Reads the configured ``subdir`` (default ``aggTrades_filtered`` -- the wash
    filter's cleaned flow) and maps its ``*_clean`` columns onto the generic
    names. If that dir is missing (wash filter not run, or it skipped a thin day),
    fall back to raw normalized ``aggTrades`` with a loud warning rather than
    silently dropping every trade; only return an empty frame if neither exists.
    """
    empty = pl.LazyFrame(
        schema={
            "timestamp": pl.Datetime("ms"),
            "buy_qty": pl.Float64,
            "sell_qty": pl.Float64,
            "notional": pl.Float64,
        }
    )
    p = trades_path(processed_dir, symbol, date_str, subdir)
    if not p.exists():
        raw = trades_path(processed_dir, symbol, date_str, "aggTrades")
        if subdir != "aggTrades" and raw.exists():
            logger.warning(
                "[%s | %s] trades dir %r missing; falling back to UNWASHED aggTrades "
                "(run the wash filter first for cleaned flow)",
                symbol,
                date_str,
                subdir,
            )
            p = raw
        else:
            return empty
    names = pl.scan_parquet(p).collect_schema().names()
    if "buy_qty_clean" in names:
        return pl.scan_parquet(p).select(
            pl.col("timestamp"),
            pl.col("buy_qty_clean").alias("buy_qty"),
            pl.col("sell_qty_clean").alias("sell_qty"),
            pl.col("notional_clean").alias("notional"),
        )
    return pl.scan_parquet(p).select("timestamp", "buy_qty", "sell_qty", "notional")


def _load_klines(processed_dir, symbol: str, date_str: str) -> pl.LazyFrame:
    p = klines_path(processed_dir, symbol, date_str)
    if not p.exists():
        return pl.LazyFrame(schema={"timestamp": pl.Datetime("ms")})
    names = pl.scan_parquet(p).collect_schema().names()
    keep = ["timestamp", *[c for c in KLINES_CONTEXT_COLUMNS if c in names]]
    return pl.scan_parquet(p).select(keep)


def align_one(
    processed_dir,
    data_dir,
    symbol: str,
    date_str: str,
    grid: str = DEFAULT_GRID,
    max_book_staleness_s: float | None = DEFAULT_MAX_BOOK_STALENESS_S,
    trades_subdir: str = DEFAULT_TRADES_SUBDIR,
) -> dict:
    """Align one symbol-day. Never raises; returns a status dict.
    status in {ok, skipped, missing, empty, failed_validation, error}."""
    try:
        bpath = book_path(processed_dir, symbol, date_str)
        if not bpath.exists():
            # The book defines the grid span; without it there is nothing to align.
            return {"status": "missing", "symbol": symbol, "date": date_str}

        dest = aligned_path(data_dir, symbol, date_str)
        if dest.exists() and dest.stat().st_size > 0:
            return {"status": "skipped", "symbol": symbol, "date": date_str}

        day = date.fromisoformat(date_str)
        book_lf = pl.scan_parquet(bpath).select(["timestamp", *BOOK_LEVEL_COLUMNS])
        trades_lf = _load_trades(processed_dir, symbol, date_str, trades_subdir)
        klines_lf = _load_klines(processed_dir, symbol, date_str)

        aligned = build_alignment_engine(
            book_lf,
            trades_lf,
            klines_lf,
            day=day,
            grid=grid,
            max_book_staleness_s=max_book_staleness_s,
        ).collect()

        if aligned.height == 0:
            return {"status": "empty", "symbol": symbol, "date": date_str}

        aligned = aligned.with_columns(pl.lit(symbol).alias("asset"))
        validate_aligned(aligned, expected_rows=len(build_day_grid(day, grid)))

        dest.parent.mkdir(parents=True, exist_ok=True)
        aligned.write_parquet(dest, compression="snappy")
        logger.info("SUCCESS [%s | %s]: %s rows -> %s", symbol, date_str, aligned.height, dest)
        return {"status": "ok", "symbol": symbol, "date": date_str, "rows": aligned.height}
    except AssertionError as exc:
        logger.error("VALIDATION FAILED [%s | %s]: %s", symbol, date_str, exc)
        return {"status": "failed_validation", "symbol": symbol, "date": date_str}
    except Exception as exc:  # noqa: BLE001 - workers must never propagate
        logger.error("FAILED [%s | %s]: %s", symbol, date_str, exc, exc_info=True)
        return {"status": "error", "symbol": symbol, "date": date_str}


def generate_tasks(symbols, start_date: date, end_date: date) -> list[tuple[str, str]]:
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
        description="Stage 3 - align L1 book/trades/klines to a regular time grid"
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
    parser.add_argument(
        "--grid", default=DEFAULT_GRID, help="grid interval, e.g. 1s, 5s (default 1s)"
    )
    parser.add_argument(
        "--max-book-staleness",
        type=float,
        default=DEFAULT_MAX_BOOK_STALENESS_S,
        help="seconds a quote may be carried before it is nulled and flagged "
        "book_dead; <=0 disables the cap (default 5)",
    )
    parser.add_argument(
        "--trades-subdir",
        default=DEFAULT_TRADES_SUBDIR,
        help="processed subdir for trades; default reads the wash-filtered "
        "cleaned flow, use 'aggTrades' for raw",
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
        futures = [
            ex.submit(
                align_one,
                args.processed_dir,
                args.data_dir,
                s,
                d,
                args.grid,
                args.max_book_staleness,
                args.trades_subdir,
            )
            for s, d in tasks
        ]
        for fut in as_completed(futures):
            st = fut.result()["status"]
            results[st] = results.get(st, 0) + 1
    logger.info("Alignment complete. Summary: %s", results)
    return 1 if (results.get("error", 0) or results.get("failed_validation", 0)) else 0


if __name__ == "__main__":
    sys.exit(main())
