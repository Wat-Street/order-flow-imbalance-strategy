# L1 cancellation-suspicion filter: down-weights OFI by unexplained queue removal.
# note: aligned flow is zero-filled, so a missing trade feed reads as "no trades"
# run: python -m order_flow_imbalance_strategy.spoofing --symbols BTCUSDT
#          --start 2024-01-01 --end 2024-01-31 [--baseline]

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
from tqdm import tqdm

from order_flow_imbalance_strategy.ofi import daterange, parse_date, signal_path

logger = logging.getLogger("SpoofingFilter")

# required and produced columns for each stage
REMOVAL_INPUT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "bid_price",
    "ask_price",
    "bid_qty",
    "ask_qty",
    "buy_qty",
    "sell_qty",
    "book_stale",
    "book_dead",
)

REMOVAL_OUTPUT_COLUMNS: tuple[str, ...] = (
    "bid_removal",
    "ask_removal",
    "bid_identifiable",
    "ask_identifiable",
)

STATS_INPUT_COLUMNS: tuple[str, ...] = ("timestamp", "book_dead", "bid_removal", "ask_removal")

STATS_OUTPUT_COLUMNS: tuple[str, ...] = (
    "warm_up",
    "bid_prior_n",
    "bid_prior_median",
    "bid_prior_std",
    "ask_prior_n",
    "ask_prior_median",
    "ask_prior_std",
)

SCORE_INPUT_COLUMNS: tuple[str, ...] = tuple(
    f"{side}_{c}" for side in ("bid", "ask") for c in ("removal", "prior_median", "prior_std")
)

SCORE_OUTPUT_COLUMNS: tuple[str, ...] = ("bid_z", "bid_score", "ask_z", "ask_score")

CLEAN_INPUT_COLUMNS: tuple[str, ...] = ("ofi_1s", "bid_score", "ask_score")

CLEAN_OUTPUT_COLUMNS: tuple[str, ...] = ("spoof_score", "ofi_clean", "spoof_unknown")

PIPELINE_INPUT_COLUMNS: tuple[str, ...] = (*REMOVAL_INPUT_COLUMNS, "ofi_1s")
BASELINE_INPUT_COLUMNS: tuple[str, ...] = ("timestamp", "ofi_1s")

CONFIG_STAMP_COLUMNS: tuple[str, ...] = ("spoof_config_version", "spoof_config_id")

# warn when more than 10% of scored rows score above 0.5
WARN_SCORE = 0.5
WARN_FRACTION = 0.10


# scoring settings, frozen so they can't change mid-evaluation
@dataclass(frozen=True)
class SpoofConfig:
    version: str = "l1-depletion-v1"
    filter_enabled: bool = True
    window_s: int = 3600
    min_obs: int = 2
    z_floor: float = 2.0
    z_cap: float = 6.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.window_s < 1:
            raise ValueError(f"window_s must be >= 1, got {self.window_s}")
        if self.min_obs < 2:
            raise ValueError(f"min_obs must be >= 2 (a sample std needs two), got {self.min_obs}")
        if self.z_cap <= self.z_floor:
            raise ValueError(f"z_cap ({self.z_cap}) must exceed z_floor ({self.z_floor})")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")

    # short hash of the settings, stamped on every scored row
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


# default settings, and the no-filter baseline for the ablation
DEFAULT_CONFIG = SpoofConfig()
NO_FILTER_CONFIG = SpoofConfig(version="no-filter-v1", filter_enabled=False)


# run every stage on one aligned OFI frame and stamp the config
def score_spoofing(df: pl.DataFrame, cfg: SpoofConfig = DEFAULT_CONFIG) -> pl.DataFrame:
    if cfg.filter_enabled:
        _check_columns(df, PIPELINE_INPUT_COLUMNS, CONFIG_STAMP_COLUMNS)
        df = clean_ofi(side_scores(trailing_stats(unexplained_removal(df), cfg), cfg))
    else:
        _check_columns(df, BASELINE_INPUT_COLUMNS, (*CLEAN_OUTPUT_COLUMNS, *CONFIG_STAMP_COLUMNS))
        # baseline: zero suspicion on valid OFI only
        df = df.with_columns(
            pl.when(pl.col("ofi_1s").is_not_null()).then(0.0).alias("spoof_score"),
            pl.col("ofi_1s").alias("ofi_clean"),
            pl.lit(False).alias("spoof_unknown"),  # noqa: FBT003 - boolean literal is intentional
        )
    return df.with_columns(
        pl.lit(cfg.version).alias("spoof_config_version"),
        pl.lit(cfg.fingerprint()).alias("spoof_config_id"),
    )


# summary counts for a scored day; warns if the score fires too often
def diagnostics(df: pl.DataFrame) -> dict:
    _check_columns(df, ("ofi_1s", "spoof_score", "spoof_unknown", "spoof_config_version"), ())

    # unknown scores are left out of the fraction
    n_scored = int(df["spoof_score"].is_not_null().sum())
    n_high = int((df["spoof_score"] > WARN_SCORE).sum())
    frac_high = n_high / n_scored if n_scored else None
    warned = frac_high is not None and frac_high > WARN_FRACTION
    if warned:
        logger.warning(
            "[%s] %.1f%% of %d scored rows have spoof_score > %s (limit %.0f%%); "
            "inspect the thresholds",
            df["spoof_config_version"][0],
            100 * frac_high,
            n_scored,
            WARN_SCORE,
            100 * WARN_FRACTION,
        )
    return {
        "n_rows": df.height,
        "n_ofi_valid": int(df["ofi_1s"].is_not_null().sum()),
        "n_scored": n_scored,
        "n_unknown": int(df["spoof_unknown"].sum()),
        "n_warm_up": int(df["warm_up"].sum()) if "warm_up" in df.columns else None,
        "n_high": n_high,
        "frac_high": frac_high,
        "warned": warned,
    }


# queue drop at an unchanged price that same-second trades don't explain (null = unknown)
def unexplained_removal(df: pl.DataFrame) -> pl.DataFrame:
    _check_columns(df, REMOVAL_INPUT_COLUMNS, REMOVAL_OUTPUT_COLUMNS)

    original_columns = df.columns
    df = df.sort("timestamp")

    adjacent = (pl.col("timestamp").diff() == pl.duration(seconds=1)).fill_null(False)
    unusable = pl.col("book_stale") | pl.col("book_dead")
    clean_pair = adjacent & ~unusable & ~unusable.shift(1).fill_null(True)

    df = df.with_columns(
        _identifiable("bid_price", "bid_qty", "sell_qty", clean_pair).alias("bid_identifiable"),
        _identifiable("ask_price", "ask_qty", "buy_qty", clean_pair).alias("ask_identifiable"),
    )
    df = df.with_columns(
        pl.when(pl.col("bid_identifiable"))
        .then(_depletion("bid_qty", "sell_qty"))
        .alias("bid_removal"),
        pl.when(pl.col("ask_identifiable"))
        .then(_depletion("ask_qty", "buy_qty"))
        .alias("ask_removal"),
    )
    return df.select(*original_columns, *REMOVAL_OUTPUT_COLUMNS)


# median and std of each side's removals over the previous window_s seconds
def trailing_stats(df: pl.DataFrame, cfg: SpoofConfig = DEFAULT_CONFIG) -> pl.DataFrame:
    _check_columns(df, STATS_INPUT_COLUMNS, STATS_OUTPUT_COLUMNS)

    original_columns = df.columns
    df = df.sort("timestamp")

    # reset on a gap, a new day or a dead book (a stale book is just quiet, so no reset)
    t = pl.col("timestamp")
    dead = pl.col("book_dead")
    starts_segment = (
        (t.diff() != pl.duration(seconds=1)).fill_null(True)
        | (t.dt.date() != t.dt.date().shift(1)).fill_null(True)
        | dead
        | dead.shift(1).fill_null(False)
    )
    df = df.with_columns(starts_segment.cast(pl.Int64).cum_sum().alias("_segment"))
    df = df.with_columns((pl.int_range(pl.len()).over("_segment") < cfg.window_s).alias("warm_up"))

    for side in ("bid", "ask"):
        # shift first so the current second is never in its own window
        prev = pl.col(f"{side}_removal").shift(1)
        df = df.with_columns(
            prev.is_not_null()
            .cast(pl.Int64)
            .rolling_sum(window_size=cfg.window_s, min_samples=1)
            .over("_segment")
            .alias(f"{side}_prior_n"),
            prev.rolling_median(window_size=cfg.window_s, min_samples=1)
            .over("_segment")
            .alias(f"{side}_prior_median"),
            prev.rolling_std(window_size=cfg.window_s, min_samples=1, ddof=1)
            .over("_segment")
            .alias(f"{side}_prior_std"),
        )
        usable = ~pl.col("warm_up") & (pl.col(f"{side}_prior_n") >= cfg.min_obs)
        df = df.with_columns(
            pl.when(usable).then(pl.col(f"{side}_prior_median")).alias(f"{side}_prior_median"),
            pl.when(usable).then(pl.col(f"{side}_prior_std")).alias(f"{side}_prior_std"),
        )

    return df.select(*original_columns, *STATS_OUTPUT_COLUMNS)


# z-score of each side's removal, mapped to a 0-1 score
def side_scores(df: pl.DataFrame, cfg: SpoofConfig = DEFAULT_CONFIG) -> pl.DataFrame:
    _check_columns(df, SCORE_INPUT_COLUMNS, SCORE_OUTPUT_COLUMNS)

    for side in ("bid", "ask"):
        z = (pl.col(f"{side}_removal") - pl.col(f"{side}_prior_median")) / (
            pl.col(f"{side}_prior_std") + cfg.eps
        )
        df = df.with_columns(z.alias(f"{side}_z"))
        ramp = (pl.col(f"{side}_z") - cfg.z_floor) / (cfg.z_cap - cfg.z_floor)
        df = df.with_columns(ramp.clip(0.0, 1.0).alias(f"{side}_score"))
    return df


# combine the side scores and down-weight OFI; unknown scores pass OFI through flagged
def clean_ofi(df: pl.DataFrame) -> pl.DataFrame:
    _check_columns(df, CLEAN_INPUT_COLUMNS, CLEAN_OUTPUT_COLUMNS)

    # max_horizontal skips nulls, so require both sides to be known
    both_known = pl.col("bid_score").is_not_null() & pl.col("ask_score").is_not_null()
    df = df.with_columns(
        pl.when(both_known).then(pl.max_horizontal("bid_score", "ask_score")).alias("spoof_score")
    )
    ofi_valid = pl.col("ofi_1s").is_not_null()
    score_known = pl.col("spoof_score").is_not_null()
    return df.with_columns(
        pl.when(score_known)
        .then(pl.col("ofi_1s") * (1 - pl.col("spoof_score")))
        .otherwise(pl.col("ofi_1s"))
        .alias("ofi_clean"),
        (ofi_valid & ~score_known).alias("spoof_unknown"),
    )


# fail early on missing inputs or outputs that already exist
def _check_columns(df: pl.DataFrame, required: tuple[str, ...], produced: tuple[str, ...]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"input frame missing required columns: {missing}")
    conflicting = [c for c in produced if c in df.columns]
    if conflicting:
        raise ValueError(f"input frame already contains output columns: {conflicting}")


# same price, clean adjacent rows, and known quantities and trades
def _identifiable(price: str, qty: str, opposing_flow: str, clean_pair: pl.Expr) -> pl.Expr:
    same_price = (pl.col(price) == pl.col(price).shift(1)).fill_null(False)
    known = (
        pl.col(qty).is_not_null()
        & pl.col(qty).shift(1).is_not_null()
        & pl.col(opposing_flow).is_not_null()
    )
    return clean_pair & same_price & known


# max(0, prev_qty - qty - opposing_flow)
def _depletion(qty: str, opposing_flow: str) -> pl.Expr:
    return (pl.col(qty).shift(1) - pl.col(qty) - pl.col(opposing_flow)).clip(lower_bound=0.0)


# output file for one symbol-day; the config version keeps filtered and baseline runs apart
def spoof_path(data_dir: Path, symbol: str, day: date, cfg: SpoofConfig) -> Path:
    name = f"{symbol}-ofi-spoof-{cfg.version}-{day:%Y-%m-%d}.parquet"
    return data_dir / "signals" / symbol / name


# one task per day that has an OFI file
def build_tasks(
    symbols: list[str], start: date, end: date, data_dir: Path, cfg: SpoofConfig
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for symbol in symbols:
        for day in daterange(start, end):
            in_path = signal_path(data_dir, symbol, day)
            if not in_path.exists():
                continue
            tasks.append(
                {
                    "symbol": symbol,
                    "date": f"{day:%Y-%m-%d}",
                    "in_path": str(in_path),
                    "out_path": str(spoof_path(data_dir, symbol, day, cfg)),
                    "cfg": cfg,
                }
            )
    return tasks


# score and write one symbol-day; never raises so one bad day can't crash the pool
def process_file(task: dict[str, Any]) -> dict[str, Any]:
    in_path = Path(task["in_path"])
    out_path = Path(task["out_path"])
    status: dict[str, Any] = {"symbol": task["symbol"], "date": task["date"], "status": "error"}

    try:
        if out_path.exists():
            status["status"] = "skipped"
            return status
        if not in_path.exists():
            status["status"] = "no_input"
            status["message"] = f"missing input {in_path}"
            return status

        df = pl.read_parquet(in_path)
        out = score_spoofing(df, task["cfg"])

        # raw OFI has to come back unchanged, row for row
        raw = df.sort("timestamp")["ofi_1s"]
        if out.height != df.height or not out.sort("timestamp")["ofi_1s"].equals(raw):
            status["message"] = "validation failed: ofi_1s changed"
            return status

        report = diagnostics(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        out.write_parquet(tmp_path)
        tmp_path.replace(out_path)

        status.update(status="ok", **report)
        return status
    except Exception as exc:  # noqa: BLE001 - workers must never propagate
        status["message"] = f"{type(exc).__name__}: {exc}"
        return status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score OFI days for L1 cancellation suspicion")
    parser.add_argument("--symbols", nargs="+", required=True, help="e.g. BTCUSDT ETHUSDT")
    parser.add_argument("--start", type=parse_date, required=True, help="start date YYYY-MM-DD")
    parser.add_argument("--end", type=parse_date, required=True, help="end date YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=4, help="process pool size (default 4)")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="root data dir (default ./data)"
    )
    parser.add_argument(
        "--baseline", action="store_true", help="write the no-filter baseline instead"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.end < args.start:
        print(f"error: --end {args.end} is before --start {args.start}", file=sys.stderr)
        return 2

    cfg = NO_FILTER_CONFIG if args.baseline else DEFAULT_CONFIG
    tasks = build_tasks(args.symbols, args.start, args.end, args.data_dir, cfg)
    if not tasks:
        print("No OFI input files found for the requested symbols/range.")
        return 0

    counts: Counter[str] = Counter()
    noisy_days: list[str] = []
    failures: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(process_file, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Spoof", unit="day"):
            result = future.result()
            counts[result["status"]] += 1
            if result["status"] == "error":
                failures.append(result)
            elif result["status"] == "ok" and result["warned"]:
                noisy_days.append(
                    f"{result['symbol']} {result['date']}: {100 * result['frac_high']:.1f}%"
                )

    print(f"\nSummary ({cfg.version}, id {cfg.fingerprint()}):")
    for status in ("ok", "skipped", "no_input", "error"):
        if counts[status]:
            print(f"  {status:>9}: {counts[status]}")

    if noisy_days:
        print(f"\n{len(noisy_days)} day(s) with >10% of scored rows above 0.5:")
        for line in noisy_days[:20]:
            print(f"  - {line}")

    if failures:
        print(f"\n{len(failures)} failure(s):", file=sys.stderr)
        for f in failures[:20]:
            print(f"  - {f['symbol']} {f['date']}: {f.get('message')}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
