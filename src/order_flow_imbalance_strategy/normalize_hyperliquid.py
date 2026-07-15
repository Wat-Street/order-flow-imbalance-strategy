"""Stage 2 (Hyperliquid track) — normalize raw L2 snapshots to schema-enforced
per-day Parquet.

Parallel to :mod:`order_flow_imbalance_strategy.normalize_data` (the Binance L1
Stage 2 from the design doc). Consumes the raw newline-JSON produced by
:mod:`order_flow_imbalance_strategy.ingest_hyperliquid` (hourly files) and emits
one standardized Parquet **per coin per day**, with data-quality defects removed
and the L2 ladder validated.

Because the L2 book is a 20-level ladder (not a single L1 quote), it has failure
modes the L1 bookTicker cannot: individual bad deep levels, duplicated price
levels, and ladder inversions. The validation taxonomy (see the design doc /
memory ``l2-pipeline-design``):

* **Category A — top-of-book defect → DROP ROW** (L1-identical): ``bid_px_01<=0``,
  ``ask_px_01<=0``, ``bid_sz_01<=0``, ``ask_sz_01<=0``, crossed touch
  ``ask_px_01<=bid_px_01``, duplicate ``event_time_ms`` (keep last).
* **Category B — deep individual defect** (level k>=2 with ``px<=0`` or ``sz<=0``)
  → **NULL that level** (keep the row; deep OFI tolerates a null level).
* **Category C — monotonicity**: aggregated L2 has strictly monotone prices
  (bids descending, asks ascending).

  * *Equal* adjacent price (``bid_px_k == bid_px_{k+1}``) → degenerate duplicate,
    **null the deeper level**.
  * *Inversion* (bids ascending / asks descending past some level) → the ladder
    sort is corrupt; **DROP THE ROW** and count it (``dropped_inverted``). The
    drop is surfaced to Stage 3/OFI as a discontinuity so OFI is never diffed
    across it.
* **Category D — thin book** (genuinely fewer than 20 levels, trailing NaN) →
  leave null; NOT a defect.
* **Category E** — top-of-book nulled by a cascade → drop row; ``<100`` valid
  rows in the day → don't write (``failed_validation``).

Derived top-of-book fields match L1: ``mid_price``, ``spread``,
``queue_imbalance``. Two extra columns carry the *usable depth* forward to the
OFI stage: ``levels_valid_bid`` / ``levels_valid_ask`` (count of contiguous valid
levels from the touch). Deep OFI must sum only over levels valid in both
consecutive snapshots — a null level truncates usable depth, it is never
zero-filled (that would fabricate order flow).

Run::

    python -m order_flow_imbalance_strategy.normalize_hyperliquid \\
        --symbols BTC --start 2024-01-01 --end 2024-01-07 \\
        --raw-dir data/raw_hl --out data/processed_hl
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
            "filename": "logs/hyperliquid_normalize.log",
            "level": "DEBUG",
            "mode": "a",
            "formatter": "standard",
        },
    },
    "loggers": {
        "hyperliquid_normalize_logger": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True,
        }
    },
}
Path("logs").mkdir(exist_ok=True)
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger("hyperliquid_normalize_logger")

# --- constants -------------------------------------------------------------

DATA_TYPE = "l2Book"
DEFAULT_COINS = ["BTC"]
DEPTH = 20
MIN_ROWS = 100  # <100 valid rows in a day => don't write (doc rule)

# Column-name helpers -------------------------------------------------------


def px(side: str, k: int) -> str:
    return f"{side}_px_{k:02d}"


def sz(side: str, k: int) -> str:
    return f"{side}_sz_{k:02d}"


def cnt(side: str, k: int) -> str:
    return f"{side}_n_{k:02d}"


def level_columns() -> list[str]:
    cols: list[str] = []
    for side in ("bid", "ask"):
        for k in range(1, DEPTH + 1):
            cols += [px(side, k), sz(side, k), cnt(side, k)]
    return cols


DERIVED = ["mid_price", "spread", "queue_imbalance"]
FLAGS = ["levels_valid_bid", "levels_valid_ask"]
OUTPUT_COLUMNS = ["timestamp", "coin", "event_time_ms", *level_columns(), *DERIVED, *FLAGS]

# Top-of-book columns that must never be null after normalization.
CRITICAL_COLUMNS = ["timestamp", px("bid", 1), px("ask", 1), sz("bid", 1), sz("ask", 1), *DERIVED]


# --- path builders ---------------------------------------------------------


def raw_hour_path(raw_dir, coin: str, date_str: str, hour: int) -> Path:
    return Path(raw_dir) / coin / DATA_TYPE / f"{coin}-{DATA_TYPE}-{date_str}-{hour:02d}.jsonl"


def out_path(out_dir, coin: str, date_str: str) -> Path:
    return Path(out_dir) / coin / DATA_TYPE / f"{coin}-{DATA_TYPE}-{date_str}.parquet"


# --- parsing raw JSON -> flat frame ---------------------------------------


def _explode_snapshot(rec: dict) -> dict | None:
    """Flatten one archive line's ``raw.data`` into event_time_ms + 20x2 level
    columns. Returns ``None`` if the line lacks the expected nesting."""
    try:
        data = rec["raw"]["data"]
        bids, asks = data["levels"][0], data["levels"][1]
    except (KeyError, IndexError, TypeError):
        return None
    row: dict[str, object] = {"event_time_ms": int(data["time"])}
    for side, book in (("bid", bids), ("ask", asks)):
        for i in range(DEPTH):
            k = i + 1
            if i < len(book):
                lvl = book[i]
                row[px(side, k)] = float(lvl["px"])
                row[sz(side, k)] = float(lvl["sz"])
                row[cnt(side, k)] = int(lvl["n"])
            else:
                row[px(side, k)] = None
                row[sz(side, k)] = None
                row[cnt(side, k)] = None
    return row


def parse_day(raw_dir, coin: str, date_str: str) -> tuple[pl.DataFrame, int]:
    """Parse all available hourly files for a day into one per-snapshot frame.
    Returns (frame, hours_found). Malformed lines are skipped with a debug log."""
    import json

    rows: list[dict] = []
    hours_found = 0
    for hour in range(24):
        p = raw_hour_path(raw_dir, coin, date_str, hour)
        if not p.exists():
            continue
        hours_found += 1
        with open(p, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    flat = _explode_snapshot(json.loads(line))
                except ValueError:
                    flat = None
                if flat is None:
                    logger.debug("Skipping malformed line %s in %s", lineno, p.name)
                    continue
                rows.append(flat)
    if not rows:
        return pl.DataFrame(), hours_found
    return pl.DataFrame(rows), hours_found


# --- ladder validation (Categories A-E) -----------------------------------


def _detect_inversion(df: pl.DataFrame) -> pl.Series:
    """Boolean Series: True where the ladder is INVERTED (Category C-inv) — bids
    not strictly descending or asks not strictly ascending, considering only
    consecutive non-null price pairs. Equal adjacent prices are handled
    separately (C-eq) and are NOT counted as inversion here."""
    inverted = pl.repeat(False, df.height, eager=True)
    for side in ("bid", "ask"):
        for k in range(1, DEPTH):
            a, b = df[px(side, k)], df[px(side, k + 1)]
            both = a.is_not_null() & b.is_not_null()
            if side == "bid":
                # descending expected: violation if a < b (strictly ascending)
                bad = both & (a < b)
            else:
                # ascending expected: violation if a > b (strictly descending)
                bad = both & (a > b)
            inverted = inverted | bad.fill_null(False)
    return inverted


def validate_ladder(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Apply Categories A-E. Returns (clean_frame, stats). Row order preserved
    except for de-duplication (keep last per event_time_ms)."""
    stats = {"raw_rows": df.height}

    # --- Category A: top-of-book row drops ---------------------------------
    touch_ok = (
        (pl.col(px("bid", 1)) > 0)
        & (pl.col(px("ask", 1)) > 0)
        & (pl.col(sz("bid", 1)) > 0)
        & (pl.col(sz("ask", 1)) > 0)
        & (pl.col(px("ask", 1)) > pl.col(px("bid", 1)))  # not crossed at touch
    )
    df = df.filter(touch_ok.fill_null(False))
    stats["after_touch"] = df.height
    if df.height == 0:
        return df, stats

    # Category A: duplicate event-time, keep last (stable: last occurrence wins).
    before = df.height
    df = df.unique(subset=["event_time_ms"], keep="last", maintain_order=True)
    stats["dup_time_dropped"] = before - df.height

    # --- Category C-inv: drop inverted-ladder rows -------------------------
    inverted = _detect_inversion(df)
    stats["dropped_inverted"] = int(inverted.sum())
    df = df.filter(~inverted)
    if df.height == 0:
        return df, stats

    # --- Category B + C-eq: null bad / duplicate deep levels ---------------
    # Build expressions per side/level. A level k (k>=2) is nulled if its px<=0,
    # sz<=0, or its price equals the shallower level's price (degenerate dup).
    exprs: list[pl.Expr] = []
    for side in ("bid", "ask"):
        for k in range(2, DEPTH + 1):
            bad_level = (
                (pl.col(px(side, k)) <= 0)
                | (pl.col(sz(side, k)) <= 0)
                | (pl.col(px(side, k)) == pl.col(px(side, k - 1)))  # C-eq duplicate
            ).fill_null(False)
            for col_fn in (px, sz, cnt):
                c = col_fn(side, k)
                exprs.append(
                    pl.when(bad_level).then(None).otherwise(pl.col(c)).alias(c)
                )
    df = df.with_columns(exprs)

    return df, stats


def _levels_valid_expr(side: str) -> pl.Expr:
    """Count of contiguous valid (non-null px & sz) levels from the touch. Once a
    level is null, deeper levels don't count even if present — usable depth for
    OFI is the contiguous run from level 1."""
    # Build cumulative "all shallower-or-equal levels are valid" then sum.
    valid_run = pl.lit(True)
    per_level: list[pl.Expr] = []
    for k in range(1, DEPTH + 1):
        this_valid = pl.col(px(side, k)).is_not_null() & pl.col(sz(side, k)).is_not_null()
        valid_run = valid_run & this_valid
        per_level.append(valid_run.cast(pl.Int32))
    return sum(per_level)  # type: ignore[return-value]


def add_derived(df: pl.DataFrame) -> pl.DataFrame:
    """Top-of-book derived fields + usable-depth counts."""
    return df.with_columns(
        [
            ((pl.col(px("bid", 1)) + pl.col(px("ask", 1))) / 2).alias("mid_price"),
            (pl.col(px("ask", 1)) - pl.col(px("bid", 1))).alias("spread"),
            (
                (pl.col(sz("bid", 1)) - pl.col(sz("ask", 1)))
                / (pl.col(sz("bid", 1)) + pl.col(sz("ask", 1)))
            ).alias("queue_imbalance"),
            _levels_valid_expr("bid").alias("levels_valid_bid"),
            _levels_valid_expr("ask").alias("levels_valid_ask"),
        ]
    )


def assert_schema(df: pl.DataFrame) -> None:
    """Pre-write assertions (doc Stage 2 rule). Nulls allowed only in deep
    (level>=2) columns; forbidden in top-of-book + derived. No infs; spread>=0."""
    for c in CRITICAL_COLUMNS:
        n_null = df[c].null_count()
        if n_null:
            raise AssertionError(f"{n_null} nulls in critical column {c}")
    # No infinities in float columns.
    for c in df.columns:
        if df[c].dtype in (pl.Float32, pl.Float64):
            if df[c].is_infinite().any():
                raise AssertionError(f"infinite values in {c}")
    if (df["spread"] < 0).any():
        raise AssertionError("negative spread present")
    if df["mid_price"].min() is not None and df["mid_price"].min() <= 0:
        raise AssertionError("non-positive mid_price")


# --- worker ----------------------------------------------------------------


def normalize_day(raw_dir, out_dir, coin: str, date_str: str) -> dict:
    """Normalize one coin-day. Never raises; returns a status dict.
    status in {ok, skipped, empty, failed_validation, error}."""
    try:
        dest = out_path(out_dir, coin, date_str)
        if dest.exists() and dest.stat().st_size > 0:
            return {"status": "skipped", "coin": coin, "date": date_str}

        df, hours_found = parse_day(raw_dir, coin, date_str)
        if hours_found == 0 or df.height == 0:
            return {"status": "empty", "coin": coin, "date": date_str, "rows": 0}

        df, stats = validate_ladder(df)
        if df.height < MIN_ROWS:
            logger.warning(
                "Too few valid rows, not writing: coin=%s date=%s rows=%s stats=%s",
                coin,
                date_str,
                df.height,
                stats,
            )
            return {
                "status": "failed_validation",
                "coin": coin,
                "date": date_str,
                "rows": df.height,
            }

        df = add_derived(df)
        # Cast timestamp from epoch-ms, order columns, enforce dtypes.
        df = df.with_columns(
            [
                pl.col("event_time_ms").cast(pl.Datetime("ms")).alias("timestamp"),
                pl.lit(coin).cast(pl.Categorical).alias("coin"),
            ]
        ).select(OUTPUT_COLUMNS)
        assert_schema(df)

        dest.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(dest, compression="snappy")
        mb = dest.stat().st_size / (1024 * 1024)
        logger.info(
            "Normalized: coin=%s date=%s rows=%s inverted=%s dup=%s size=%.2f MB",
            coin,
            date_str,
            df.height,
            stats.get("dropped_inverted", 0),
            stats.get("dup_time_dropped", 0),
            mb,
        )
        return {
            "status": "ok",
            "coin": coin,
            "date": date_str,
            "rows": df.height,
            "dropped_inverted": stats.get("dropped_inverted", 0),
        }
    except Exception as exc:  # noqa: BLE001 - workers never propagate
        logger.error("Error normalizing %s %s: %s", coin, date_str, exc, exc_info=True)
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
        description="Normalize raw Hyperliquid L2 snapshots to schema-enforced per-day Parquet"
    )
    parser.add_argument("--symbols", nargs="+", type=str, default=DEFAULT_COINS)
    parser.add_argument("--start", type=dt.datetime.fromisoformat, required=True)
    parser.add_argument("--end", type=dt.datetime.fromisoformat, required=True)
    parser.add_argument("--raw-dir", default="data/raw_hl")
    parser.add_argument("--out", default="data/processed_hl")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    if args.start > args.end:
        sys.exit("start date must be before end date")

    tasks = generate_tasks(args)
    logger.info("Generated %s coin-day normalization tasks", len(tasks))
    results: dict[str, int] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(normalize_day, args.raw_dir, args.out, c, d) for c, d in tasks]
        for fut in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Normalize L2"
        ):
            st = fut.result()["status"]
            results[st] = results.get(st, 0) + 1
    logger.info("Normalization complete. Summary: %s", results)


if __name__ == "__main__":
    main()
