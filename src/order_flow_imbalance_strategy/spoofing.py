# L1 cancellation-suspicion filter: down-weights OFI by unexplained queue removal.
# note: aligned flow is zero-filled, so a missing trade feed reads as "no trades"
# note: the ablation's "trade_excluded" variant is our reading of the spec
# run: python -m order_flow_imbalance_strategy.spoofing score --symbols BTCUSDT
#          --start 2024-01-01 --end 2024-01-31
#      python -m order_flow_imbalance_strategy.spoofing ablate --symbol BTCUSDT
#          --select-start 2024-01-01 --select-end 2024-02-29
#          --confirm-start 2024-03-01 --confirm-end 2024-03-31

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from tqdm import tqdm

from order_flow_imbalance_strategy.ofi import daterange, parse_date, signal_path

logger = logging.getLogger("SpoofingFilter")

# columns score_spoofing() reads from an OFI file
INPUT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "bid_price",
    "ask_price",
    "bid_qty",
    "ask_qty",
    "buy_qty",
    "sell_qty",
    "book_stale",
    "book_dead",
    "ofi_1s",
)

# warn when more than 10% of scored rows score above 0.5
WARN_SCORE = 0.5
WARN_FRACTION = 0.10

# thresholds are picked at one predeclared horizon; the rest are reported alongside
SELECTION_HORIZON_S = 30
REPORT_HORIZONS_S = (10, 30, 60)
THRESHOLD_GRID = tuple((f, c) for f in (1.5, 2.0, 3.0) for c in (4.0, 6.0, 8.0))

EXCLUDE_ABOVE = 0.5
VARIANTS = ("raw", "diagnostic", "down_weighted", "trade_excluded")
N_BOOT = 10_000
BOOT_SEED = 7


# scoring settings, frozen so they can't change mid-evaluation
@dataclass(frozen=True)
class SpoofConfig:
    version: str = "l1-depletion-v1"
    window_s: int = 3600
    min_obs: int = 2
    z_floor: float = 2.0
    z_cap: float = 6.0
    eps: float = 1e-8

    # short hash of the settings, stamped on every scored row
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


DEFAULT_CONFIG = SpoofConfig()


# run every stage on one OFI frame and stamp the config
def score_spoofing(df: pl.DataFrame, cfg: SpoofConfig = DEFAULT_CONFIG) -> pl.DataFrame:
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input frame missing required columns: {missing}")
    df = clean_ofi(side_scores(trailing_stats(unexplained_removal(df), cfg), cfg))
    return df.with_columns(
        pl.lit(cfg.version).alias("spoof_config_version"),
        pl.lit(cfg.fingerprint()).alias("spoof_config_id"),
    )


# summary counts for a scored day; warns if the score fires too often
def diagnostics(df: pl.DataFrame) -> dict:
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
        "n_scored": n_scored,
        "n_unknown": int(df["spoof_unknown"].sum()),
        "n_warm_up": int(df["warm_up"].sum()),
        "frac_high": frac_high,
        "warned": warned,
    }


# queue drop at an unchanged price that same-second trades don't explain (null = unknown)
def unexplained_removal(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort("timestamp")
    adjacent = (pl.col("timestamp").diff() == pl.duration(seconds=1)).fill_null(False)
    unusable = pl.col("book_stale") | pl.col("book_dead")
    clean_pair = adjacent & ~unusable & ~unusable.shift(1).fill_null(True)
    bid_ok = _identifiable("bid_price", "bid_qty", "sell_qty", clean_pair)
    ask_ok = _identifiable("ask_price", "ask_qty", "buy_qty", clean_pair)
    return df.with_columns(
        pl.when(bid_ok).then(_depletion("bid_qty", "sell_qty")).alias("bid_removal"),
        pl.when(ask_ok).then(_depletion("ask_qty", "buy_qty")).alias("ask_removal"),
        bid_ok.alias("bid_identifiable"),
        ask_ok.alias("ask_identifiable"),
    )


# median and std of each side's removals over the previous window_s seconds
def trailing_stats(df: pl.DataFrame, cfg: SpoofConfig = DEFAULT_CONFIG) -> pl.DataFrame:
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

    return df.drop("_segment")


# z-score of each side's removal, mapped to a 0-1 score
def side_scores(df: pl.DataFrame, cfg: SpoofConfig = DEFAULT_CONFIG) -> pl.DataFrame:
    for side in ("bid", "ask"):
        z = (pl.col(f"{side}_removal") - pl.col(f"{side}_prior_median")) / (
            pl.col(f"{side}_prior_std") + cfg.eps
        )
        ramp = (z - cfg.z_floor) / (cfg.z_cap - cfg.z_floor)
        df = df.with_columns(z.alias(f"{side}_z"), ramp.clip(0.0, 1.0).alias(f"{side}_score"))
    return df


# combine the side scores and down-weight OFI; unknown scores pass OFI through flagged
def clean_ofi(df: pl.DataFrame) -> pl.DataFrame:
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


# output file for one symbol-day, named by config version
def spoof_path(data_dir: Path, symbol: str, day: date) -> Path:
    name = f"{symbol}-ofi-spoof-{DEFAULT_CONFIG.version}-{day:%Y-%m-%d}.parquet"
    return data_dir / "signals" / symbol / name


# one task per day that has an OFI file
def build_tasks(symbols: list[str], start: date, end: date, data_dir: Path) -> list[dict[str, Any]]:
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
                    "out_path": str(spoof_path(data_dir, symbol, day)),
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

        out = score_spoofing(pl.read_parquet(in_path))
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


# log mid change h seconds ahead, only between two valid quotes on the same UTC day
def forward_return(df: pl.DataFrame, horizon_s: int) -> pl.DataFrame:
    t = pl.col("timestamp")
    mid = (pl.col("bid_price") + pl.col("ask_price")) / 2
    quote_ok = (
        ~pl.col("book_stale") & ~pl.col("book_dead") & (pl.col("ask_price") > pl.col("bid_price"))
    ).fill_null(False)
    ok = (
        quote_ok
        & quote_ok.shift(-horizon_s).fill_null(False)
        & (t.shift(-horizon_s) - t == pl.duration(seconds=horizon_s)).fill_null(False)
        & (t.shift(-horizon_s).dt.date() == t.dt.date()).fill_null(False)
    )
    ret = (mid.shift(-horizon_s) / mid).log()
    return df.sort("timestamp").with_columns(pl.when(ok).then(ret).alias(f"fwd_ret_{horizon_s}s"))


# rerun the production scoring with other thresholds on an already-scored frame
def rescore(df: pl.DataFrame, cfg: SpoofConfig) -> pl.DataFrame:
    scored = ["bid_z", "bid_score", "ask_z", "ask_score", "spoof_score", "ofi_clean"]
    scored += ["spoof_unknown", "spoof_config_version", "spoof_config_id"]
    return clean_ofi(side_scores(df.drop(scored, strict=False), cfg))


def add_variants(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("ofi_1s").alias("feat_raw"),
        pl.col("ofi_clean").alias("feat_down_weighted"),
        pl.when(pl.col("spoof_score") > EXCLUDE_ABOVE)
        .then(0.0)
        .otherwise(pl.col("ofi_1s"))
        .alias("feat_trade_excluded"),
    )


# Spearman IC of each feature per UTC day, keeping days where every IC is defined
def daily_ics(df: pl.DataFrame, target: str, features: list[str]) -> pl.DataFrame:
    return (
        df.group_by(pl.col("timestamp").dt.date().alias("day"))
        .agg(pl.corr(f, target, method="spearman").alias(f) for f in features)
        .sort("day")
        .fill_nan(None)
        .drop_nulls()
    )


# mean and 95% CI by resampling days with replacement (days are the independent blocks)
def bootstrap_mean_ci(
    values: list[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("no days to bootstrap")
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def run_ablation(
    selection: pl.DataFrame,
    confirmation: pl.DataFrame,
    selection_horizon_s: int = SELECTION_HORIZON_S,
    report_horizons_s: tuple[int, ...] = REPORT_HORIZONS_S,
) -> dict[str, Any]:
    sel_days = _days(selection)
    conf_days = _days(confirmation)
    if sel_days[-1] >= conf_days[0]:
        raise ValueError("selection days must all end before confirmation starts")

    # --- selection: everything here sees selection days only ---
    target = f"fwd_ret_{selection_horizon_s}s"
    labeled = forward_return(selection, selection_horizon_s)
    grid = []
    for z_floor, z_cap in THRESHOLD_GRID:
        cfg = replace(DEFAULT_CONFIG, z_floor=z_floor, z_cap=z_cap)
        rows = _cohort(add_variants(rescore(labeled, cfg)), target)
        ics = daily_ics(rows, target, ["feat_raw", "feat_down_weighted"])
        if ics.height == 0:
            raise ValueError("no labeled, scored rows in the selection period")
        diff = float((ics["feat_down_weighted"] - ics["feat_raw"]).mean())
        grid.append({"z_floor": z_floor, "z_cap": z_cap, "mean_diff": diff})
    best = max(grid, key=lambda g: g["mean_diff"])
    locked = replace(
        DEFAULT_CONFIG,
        version=f"{DEFAULT_CONFIG.version}-sel-f{best['z_floor']}-c{best['z_cap']}",
        z_floor=best["z_floor"],
        z_cap=best["z_cap"],
    )
    coefs = _fit_diagnostic(_cohort(add_variants(rescore(labeled, locked)), target), target)

    # --- confirmation: the locked config and coefficients, applied once ---
    scored = add_variants(rescore(confirmation, locked))
    ofi_valid = scored["ofi_1s"].is_not_null()
    by_horizon = {}
    for h in report_horizons_s:
        h_target = f"fwd_ret_{h}s"
        rows = _with_diagnostic(_cohort(forward_return(scored, h), h_target), coefs)
        by_horizon[str(h)] = _evaluate(rows, h_target)

    return {
        "horizon_s": selection_horizon_s,
        "selection": {
            "days": [d.isoformat() for d in sel_days],
            "configs_tried": len(grid),
            "grid": grid,
            "locked_config": asdict(locked),
            "locked_id": locked.fingerprint(),
            "diagnostic_coefs": coefs,
        },
        "confirmation": {
            "days": [d.isoformat() for d in conf_days],
            "scored_share_of_valid_ofi": float(
                scored.filter(ofi_valid)["spoof_score"].is_not_null().mean() or 0.0
            ),
            "by_horizon": by_horizon,
        },
        "notes": [
            "trade_excluded zeroes OFI where spoof_score > 0.5; our reading of the spec",
            "diagnostic = OLS on [1, ofi, score, ofi*score] fit on the selection cohort",
            "cohort: valid OFI, known spoof_score and a valid forward return",
            "CI: bootstrap over days of paired per-day IC differences",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L1 cancellation-suspicion filter for OFI")
    commands = parser.add_subparsers(dest="command", required=True)

    score = commands.add_parser("score", help="score OFI days and write them")
    score.add_argument("--symbols", nargs="+", required=True, help="e.g. BTCUSDT ETHUSDT")
    score.add_argument("--start", type=parse_date, required=True, help="start date YYYY-MM-DD")
    score.add_argument("--end", type=parse_date, required=True, help="end date YYYY-MM-DD")
    score.add_argument("--workers", type=int, default=4, help="process pool size (default 4)")

    ablate = commands.add_parser("ablate", help="compare raw and cleaned OFI on later days")
    ablate.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    ablate.add_argument("--select-start", type=parse_date, required=True)
    ablate.add_argument("--select-end", type=parse_date, required=True)
    ablate.add_argument("--confirm-start", type=parse_date, required=True)
    ablate.add_argument("--confirm-end", type=parse_date, required=True)

    for sub in (score, ablate):
        sub.add_argument(
            "--data-dir", type=Path, default=Path("data"), help="root data dir (default ./data)"
        )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return _ablate(args) if args.command == "ablate" else _score_days(args)


def _score_days(args: argparse.Namespace) -> int:
    if args.end < args.start:
        print(f"error: --end {args.end} is before --start {args.start}", file=sys.stderr)
        return 2

    tasks = build_tasks(args.symbols, args.start, args.end, args.data_dir)
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

    print(f"\nSummary ({DEFAULT_CONFIG.version}, id {DEFAULT_CONFIG.fingerprint()}):")
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


def _ablate(args: argparse.Namespace) -> int:
    if not (args.select_start <= args.select_end < args.confirm_start <= args.confirm_end):
        print(
            "error: need select-start <= select-end < confirm-start <= confirm-end", file=sys.stderr
        )
        return 2

    selection = _load_days(args.data_dir, args.symbol, args.select_start, args.select_end)
    confirmation = _load_days(args.data_dir, args.symbol, args.confirm_start, args.confirm_end)
    if selection is None or confirmation is None:
        print("error: no scored spoofing files for one of the periods", file=sys.stderr)
        return 1

    report = {"symbol": args.symbol, **run_ablation(selection, confirmation)}
    out = args.data_dir / "reports"
    out = out / f"spoofing-ablation-{args.symbol}-{args.confirm_start}-{args.confirm_end}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    sel = report["selection"]
    print(f"locked {sel['locked_config']['version']} (id {sel['locked_id']})")
    for h, res in report["confirmation"]["by_horizon"].items():
        print(f"\n{h}s horizon: {res['n_rows']} rows over {res['n_days']} day(s)")
        for name, v in res["variants"].items():
            lo, hi = v["ci95"]
            diff = v["diff_vs_raw"]
            print(
                f"  {name:>15}: IC {v['mean_ic']:+.4f}  vs raw {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            )
    print(f"\nreport: {out}")
    return 0


def _days(df: pl.DataFrame) -> list[date]:
    return df["timestamp"].dt.date().unique().sort().to_list()


# rows where every variant and the label are defined
def _cohort(df: pl.DataFrame, target: str) -> pl.DataFrame:
    return df.filter(
        pl.col("ofi_1s").is_not_null()
        & pl.col("spoof_score").is_not_null()
        & pl.col(target).is_not_null()
    )


def _fit_diagnostic(df: pl.DataFrame, target: str) -> list[float]:
    ofi = df["ofi_1s"].to_numpy()
    score = df["spoof_score"].to_numpy()
    x = np.column_stack([np.ones_like(ofi), ofi, score, ofi * score])
    coefs, *_ = np.linalg.lstsq(x, df[target].to_numpy(), rcond=None)
    return [float(c) for c in coefs]


def _with_diagnostic(df: pl.DataFrame, coefs: list[float]) -> pl.DataFrame:
    b0, b1, b2, b3 = coefs
    ofi, score = pl.col("ofi_1s"), pl.col("spoof_score")
    return df.with_columns((b0 + b1 * ofi + b2 * score + b3 * ofi * score).alias("feat_diagnostic"))


def _evaluate(df: pl.DataFrame, target: str) -> dict[str, Any]:
    ics = daily_ics(df, target, [f"feat_{v}" for v in VARIANTS])
    if ics.height == 0:
        raise ValueError(f"no confirmation days with a defined IC for {target}")
    variants = {}
    for v in VARIANTS:
        ic = ics[f"feat_{v}"]
        if v == "raw":
            diff, lo, hi = 0.0, 0.0, 0.0
        else:
            diff, lo, hi = bootstrap_mean_ci((ic - ics["feat_raw"]).to_list())
        variants[v] = {"mean_ic": float(ic.mean()), "diff_vs_raw": diff, "ci95": [lo, hi]}
    return {"n_rows": df.height, "n_days": ics.height, "variants": variants}


def _load_days(data_dir: Path, symbol: str, start: date, end: date) -> pl.DataFrame | None:
    paths = [spoof_path(data_dir, symbol, d) for d in daterange(start, end)]
    frames = [pl.read_parquet(p) for p in paths if p.exists()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else None


if __name__ == "__main__":
    raise SystemExit(main())
