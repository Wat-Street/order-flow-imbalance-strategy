"""Stage 3 (Hyperliquid track) — align normalized L2 snapshots to a 1-second grid.

Parallel to the Event Alignment stage of the design doc. Consumes the per-day
standardized Parquet from
:mod:`order_flow_imbalance_strategy.normalize_hyperliquid` and produces the
committable, AWS-free artifact: one Parquet per coin per day on a regular
1-second grid (86,400 rows/day), which the deep-OFI stage and the M* depth study
read.

**Grid & fill rule** (the doc's bookTicker rule; the L2 book is the direct
analog): for each 1-second interval of the UTC day, take the **last** snapshot
whose event time falls in that second; **forward-fill** seconds that had no native
snapshot.

**Cross-stage contract to OFI** — the aligned frame carries flags so the OFI
stage never computes flow across a discontinuity or fabricates it from a filled
value:

* ``book_stale`` — this second had no native snapshot (its book was
  forward-filled). OFI on a stale quote is meaningless.
* ``book_stale_prev`` — the *previous* second was stale (or the row after a gap
  left by a Stage-2 inverted-book drop). OFI diffs t vs t-1; if either is stale
  the contribution must be null.
* ``levels_valid_bid`` / ``levels_valid_ask`` — usable contiguous depth, carried
  from Stage 2. Deep OFI sums only over ``min(valid_t, valid_{t-1})`` levels; a
  null level truncates usable depth (never zero-filled).
* ``gap_prev_s`` — seconds since the last native snapshot (>1 means the row sits
  after a data gap; the inversion-drop case surfaces here).

Row 0 of each day has no t-1: OFI warmup handles it (documented for the OFI
stage). Seconds before the first snapshot of the day are dropped (no book to
carry yet).

Run::

    python -m order_flow_imbalance_strategy.align_hyperliquid \\
        --symbols BTC --start 2024-01-01 --end 2024-01-07 \\
        --processed-dir data/processed_hl --out data/aligned_hl
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import logging
import logging.config
import os
import sys
from pathlib import Path

import polars as pl
from tqdm import tqdm

from order_flow_imbalance_strategy.normalize_hyperliquid import (
    DEFAULT_COINS,
    FLAGS,
    level_columns,
)
from order_flow_imbalance_strategy.normalize_hyperliquid import out_path as normalized_path

LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"standard": {"format": "%(asctime)s - %(levelname)s - %(message)s"}},
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "stream": "ext://sys.stdout",
            "formatter": "standard",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "logs/hyperliquid_align.log",
            "level": "DEBUG",
            "mode": "a",
            "formatter": "standard",
        },
    },
    "loggers": {
        "hyperliquid_align_logger": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True,
        }
    },
}
Path("logs").mkdir(exist_ok=True)
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger("hyperliquid_align_logger")

SECONDS_PER_DAY = 86_400

# The book state carried by forward-fill (everything except the per-second flags).
CARRY_COLUMNS = [*level_columns(), "mid_price", "spread", "queue_imbalance", *FLAGS]


def out_path(out_dir, coin: str, date_str: str) -> Path:
    return Path(out_dir) / coin / f"{coin}-l2-aligned-{date_str}.parquet"


def build_grid(date_str: str) -> pl.DataFrame:
    """The 86,400 one-second boundaries of the UTC day as a ``ts`` column."""
    day_start = dt.datetime.fromisoformat(date_str).replace(tzinfo=dt.UTC)
    grid = pl.datetime_range(
        day_start,
        day_start + dt.timedelta(days=1),
        interval="1s",
        closed="left",
        time_unit="ms",
        eager=True,
    ).alias("ts")
    return pl.DataFrame(grid)


def align_day_frame(df: pl.DataFrame, date_str: str) -> pl.DataFrame:
    """Resample a normalized per-day frame to the 1-second grid with the
    forward-fill + flag contract. Pure (no I/O) so it is unit-testable."""
    # Last snapshot per second (group by the floor-to-second of the event time).
    # Force the grid key to ms precision + UTC so the join key matches build_grid
    # regardless of the source timestamp's time unit.
    per_sec = (
        df.with_columns(
            pl.col("timestamp")
            .dt.truncate("1s")
            .dt.cast_time_unit("ms")
            .dt.replace_time_zone("UTC")
            .alias("ts")
        )
        .sort("timestamp")
        .group_by("ts", maintain_order=True)
        .last()
    )
    # native_ts marks seconds that actually had a snapshot (pre-fill).
    per_sec = per_sec.with_columns(pl.col("ts").alias("native_ts"))

    grid = build_grid(date_str)
    joined = grid.join(per_sec, on="ts", how="left").sort("ts")

    # book_stale: this second had no native snapshot.
    joined = joined.with_columns((pl.col("native_ts").is_null()).alias("book_stale"))

    # Forward-fill the carried book state + the native_ts (so gap size is known).
    joined = joined.with_columns(
        [pl.col(c).forward_fill() for c in CARRY_COLUMNS] + [pl.col("native_ts").forward_fill()]
    )

    # Drop seconds before the first native snapshot (nothing to carry yet).
    joined = joined.filter(pl.col("native_ts").is_not_null())

    # gap_prev_s: seconds since the last native snapshot for THIS row.
    #  - native rows: 0 handled below; use diff of native_ts in seconds.
    # Compute seconds between this row's ts and the carried native_ts.
    joined = joined.with_columns(
        ((pl.col("ts") - pl.col("native_ts")).dt.total_seconds()).alias("gap_prev_s")
    )

    # book_stale_prev: previous grid second was stale OR we sit after a gap
    # (gap_prev_s>0 means the current second was forward-filled from an earlier
    # native snapshot; the inverted-book drops from Stage 2 manifest as gaps too).
    joined = joined.with_columns(
        (pl.col("book_stale").shift(1).fill_null(True) | (pl.col("gap_prev_s") > 0)).alias(
            "book_stale_prev"
        )
    )

    # event_time_ms of the carried snapshot is retained for provenance.
    ordered = [
        "ts",
        "coin",
        "event_time_ms",
        *level_columns(),
        "mid_price",
        "spread",
        "queue_imbalance",
        *FLAGS,
        "book_stale",
        "book_stale_prev",
        "gap_prev_s",
    ]
    return joined.select([c for c in ordered if c in joined.columns])


def align_day(processed_dir, out_dir, coin: str, date_str: str) -> dict:
    """Align one coin-day. Never raises; returns a status dict.
    status in {ok, skipped, missing, error}."""
    try:
        dest = out_path(out_dir, coin, date_str)
        if dest.exists() and dest.stat().st_size > 0:
            return {"status": "skipped", "coin": coin, "date": date_str}

        src = normalized_path(processed_dir, coin, date_str)
        if not src.exists():
            return {"status": "missing", "coin": coin, "date": date_str}

        df = pl.read_parquet(src)
        aligned = align_day_frame(df, date_str)

        dest.parent.mkdir(parents=True, exist_ok=True)
        aligned.write_parquet(dest, compression="snappy")
        stale_frac = float(aligned["book_stale"].mean()) if aligned.height else 0.0
        mb = dest.stat().st_size / (1024 * 1024)
        logger.info(
            "Aligned: coin=%s date=%s rows=%s stale=%.1f%% size=%.2f MB",
            coin,
            date_str,
            aligned.height,
            stale_frac * 100,
            mb,
        )
        return {"status": "ok", "coin": coin, "date": date_str, "rows": aligned.height}
    except Exception as exc:  # noqa: BLE001 - workers never propagate
        logger.error("Error aligning %s %s: %s", coin, date_str, exc, exc_info=True)
        return {"status": "error", "coin": coin, "date": date_str}


# --- CLI -------------------------------------------------------------------


def generate_tasks(args) -> list[tuple[str, str]]:
    tasks: list[tuple[str, str]] = []
    delta = dt.timedelta(days=1)
    for coin in args.symbols:
        current = args.start
        while current <= args.end:
            tasks.append((coin, current.strftime("%Y-%m-%d")))
            current += delta
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Align normalized Hyperliquid L2 snapshots to a 1-second grid"
    )
    parser.add_argument("--symbols", nargs="+", type=str, default=DEFAULT_COINS)
    parser.add_argument("--start", type=dt.datetime.fromisoformat, required=True)
    parser.add_argument("--end", type=dt.datetime.fromisoformat, required=True)
    parser.add_argument("--processed-dir", default="data/processed_hl")
    parser.add_argument("--out", default="data/aligned_hl")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    if args.start > args.end:
        sys.exit("start date must be before end date")

    tasks = generate_tasks(args)
    logger.info("Generated %s coin-day alignment tasks", len(tasks))
    results: dict[str, int] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(align_day, args.processed_dir, args.out, c, d) for c, d in tasks]
        for fut in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Align L2"
        ):
            st = fut.result()["status"]
            results[st] = results.get(st, 0) + 1
    logger.info("Alignment complete. Summary: %s", results)


if __name__ == "__main__":
    main()
