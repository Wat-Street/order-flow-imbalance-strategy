"""
used this command for testing (short date range currently)
Run via:

python -m research.ml_gate_validation \
  --symbols BTCUSDT ETHUSDT \
  --start 2024-01-01 \
  --end 2024-01-15

"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

HORIZONS = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 120, 300]
PRIMARY_HORIZON = 30
SIGNALS = ["ofi_1s"]  # Expanded to ["ofi_1s", "ofi_clean"] once spoofing filter is integrated
PRIMARY_SIGNAL = "ofi_1s"
LABELS = ["mid", "micro", "vwap"]
PRIMARY_LABEL = "micro"

# Alpha gate thresholds
GATE = dict(
    min_mean_ic=0.02,
    min_ic_ir=3.0,
    min_hit_rate=0.60,
    require_ci_above_zero=True,
)

N_BOOT = 2000
BOOT_SEED = 7
MIN_ROWS_FOR_IC = 500
REGIME_VOL_WINDOW = 300

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ml_gate")


# Align book and trade data.
def _build_second_grid(df_book: pl.DataFrame, df_trade: pl.DataFrame) -> pl.DataFrame:
    df_book = df_book.sort("timestamp")
    df_trade = df_trade.sort("timestamp")

    book_1s = (
        df_book.with_columns(pl.col("timestamp").dt.truncate("1s"))
        .group_by("timestamp", maintain_order=True)
        .last()
    )

    trade_1s = (
        df_trade.with_columns(pl.col("timestamp").dt.truncate("1s"))
        .group_by("timestamp", maintain_order=True)
        .agg(
            [
                pl.col("quantity").sum().alias("sec_qty"),
                pl.col("notional").sum().alias("sec_notional"),
            ]
        )
    )

    t0 = df_book["timestamp"].min().replace(microsecond=0)
    t1 = df_book["timestamp"].max().replace(microsecond=0)
    grid = pl.DataFrame({"timestamp": pl.datetime_range(t0, t1, interval="1s", eager=True)})

    grid = grid.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))
    book_1s = book_1s.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))
    trade_1s = trade_1s.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))

    g = (
        grid.join(book_1s, on="timestamp", how="left")
        .join(trade_1s, on="timestamp", how="left")
        .sort("timestamp")
    )
    book_cols = ["bid_price", "ask_price", "bid_qty", "ask_qty"]
    g = g.with_columns([pl.col(c).forward_fill() for c in book_cols])
    g = g.with_columns([pl.col("sec_qty").fill_null(0.0), pl.col("sec_notional").fill_null(0.0)])

    return g.drop_nulls(subset=book_cols)


# Calculate OFI and price returns.
def _add_signals_and_labels(g: pl.DataFrame) -> pl.DataFrame:
    g = g.with_columns(
        [
            pl.col("bid_price").shift(1).alias("prev_bid_price"),
            pl.col("ask_price").shift(1).alias("prev_ask_price"),
            pl.col("bid_qty").shift(1).alias("prev_bid_qty"),
            pl.col("ask_qty").shift(1).alias("prev_ask_qty"),
        ]
    )

    g = g.with_columns(
        [
            pl.when(pl.col("bid_price") > pl.col("prev_bid_price"))
            .then(pl.col("bid_qty"))
            .when(pl.col("bid_price") == pl.col("prev_bid_price"))
            .then(pl.col("bid_qty") - pl.col("prev_bid_qty"))
            .otherwise(-pl.col("prev_bid_qty"))
            .alias("e_b"),
            pl.when(pl.col("ask_price") > pl.col("prev_ask_price"))
            .then(-pl.col("prev_ask_qty"))
            .when(pl.col("ask_price") == pl.col("prev_ask_price"))
            .then(pl.col("ask_qty") - pl.col("prev_ask_qty"))
            .otherwise(pl.col("ask_qty"))
            .alias("e_a"),
        ]
    )

    g = g.with_columns(
        [
            (pl.col("e_b") - pl.col("e_a")).alias("ofi_1s"),
            ((pl.col("bid_price") + pl.col("ask_price")) / 2.0).alias("mid_price"),
            (
                (pl.col("bid_price") * pl.col("ask_qty") + pl.col("ask_price") * pl.col("bid_qty"))
                / (pl.col("bid_qty") + pl.col("ask_qty"))
            ).alias("micro_price"),
        ]
    )

    for h in HORIZONS:
        g = g.with_columns(
            [
                ((pl.col("mid_price").shift(-h) - pl.col("mid_price")) / pl.col("mid_price")).alias(
                    f"ret_mid_{h}s"
                ),
                (
                    (pl.col("micro_price").shift(-h) - pl.col("micro_price"))
                    / pl.col("micro_price")
                ).alias(f"ret_micro_{h}s"),
            ]
        )

    g = g.with_columns(
        [
            pl.col("sec_notional").cum_sum().alias("cum_notional"),
            pl.col("sec_qty").cum_sum().alias("cum_qty"),
        ]
    )

    for h in HORIZONS:
        fwd_notional = pl.col("cum_notional").shift(-h) - pl.col("cum_notional")
        fwd_qty = pl.col("cum_qty").shift(-h) - pl.col("cum_qty")
        fwd_vwap = pl.when(fwd_qty > 0).then(fwd_notional / fwd_qty).otherwise(None)
        g = g.with_columns(
            [((fwd_vwap - pl.col("mid_price")) / pl.col("mid_price")).alias(f"ret_vwap_{h}s")]
        )

    g = g.with_columns(
        (
            pl.col("mid_price")
            .pct_change()
            .abs()
            .rolling_mean(window_size=REGIME_VOL_WINDOW, min_periods=REGIME_VOL_WINDOW // 2)
        ).alias("vol_proxy")
    )

    lo, hi = g["vol_proxy"].quantile(1 / 3), g["vol_proxy"].quantile(2 / 3)
    if lo is not None and hi is not None:
        g = g.with_columns(
            pl.when(pl.col("vol_proxy") <= lo)
            .then(pl.lit("low"))
            .when(pl.col("vol_proxy") <= hi)
            .then(pl.lit("normal"))
            .otherwise(pl.lit("high"))
            .alias("regime")
        )
    else:
        g = g.with_columns(pl.lit("normal").alias("regime"))

    return g


# Compute Spearman rank correlation.
def _safe_spearman(df: pl.DataFrame, xcol: str, ycol: str, min_n: int = MIN_ROWS_FOR_IC):
    sub = df.select([xcol, ycol]).drop_nulls()
    if sub.height < min_n or sub[xcol].n_unique() < 3 or sub[ycol].n_unique() < 3:
        return None, sub.height
    ic = sub.select(pl.corr(xcol, ycol, method="spearman")).item()
    if ic is None or (isinstance(ic, float) and np.isnan(ic)):
        return None, sub.height
    return float(ic), sub.height


# Block bootstrap for mean IC.
def _block_bootstrap_mean(values: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = v.size
    if n < 3:
        return (np.nan, np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = v[idx].mean(axis=1)
    mean = float(v.mean())
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    se = boot_means.std(ddof=1)
    return (mean, float(lo), float(hi), mean / se if se > 0 else np.nan)


# Compute Information Coefficient metrics.
def _ic_ir(values: np.ndarray):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = v.size
    if n < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    mean = v.mean()
    sd = v.std(ddof=1)
    return (
        float(mean),
        float(sd),
        (mean / sd) * np.sqrt(n) if sd > 0 else np.nan,
        float((v > 0).mean()),
    )


# Process one day of data.
def process_day(args):
    symbol, day_str, data_dir = args
    book_fp = os.path.join(
        data_dir, "processed", symbol, "bookTicker", f"{symbol}-bookTicker-{day_str}.parquet"
    )
    trade_fp = os.path.join(
        data_dir, "processed", symbol, "aggTrades", f"{symbol}-aggTrades-{day_str}.parquet"
    )

    if not (os.path.exists(book_fp) and os.path.exists(trade_fp)):
        return {"status": "missing", "symbol": symbol, "date": day_str}

    try:
        df_book = pl.read_parquet(
            book_fp, columns=["timestamp", "bid_price", "ask_price", "bid_qty", "ask_qty"]
        )
        df_trade = pl.read_parquet(trade_fp, columns=["timestamp", "quantity", "notional"])

        g = _build_second_grid(df_book, df_trade)
        if g.height < MIN_ROWS_FOR_IC:
            return {"status": "too_small", "symbol": symbol, "date": day_str}

        g = _add_signals_and_labels(g)
        ic_records, regime_records = [], []

        for sig in SIGNALS:
            for lab in LABELS:
                for h in HORIZONS:
                    ic, n = _safe_spearman(g, sig, f"ret_{lab}_{h}s")
                    if ic is not None:
                        ic_records.append((symbol, day_str, sig, lab, h, ic, n))

        for reg in ["low", "normal", "high"]:
            gr = g.filter(pl.col("regime") == reg)
            if gr.height < MIN_ROWS_FOR_IC:
                continue
            for h in HORIZONS:
                ic, n = _safe_spearman(gr, PRIMARY_SIGNAL, f"ret_{PRIMARY_LABEL}_{h}s", min_n=200)
                if ic is not None:
                    regime_records.append((symbol, day_str, reg, h, ic, n))

        return {
            "status": "ok",
            "symbol": symbol,
            "date": day_str,
            "ic_records": ic_records,
            "regime_records": regime_records,
            "ofi_mean": float(g["ofi_1s"].drop_nulls().mean()),
            "n_seconds": g.height,
        }
    except Exception as e:
        return {"status": "error", "symbol": symbol, "date": day_str, "error": repr(e)}


# Generate date sequence.
def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


# Aggregate IC decay stats.
def agg_decay(ic_df: pl.DataFrame, signal: str, label: str):
    rows = []
    for h in HORIZONS:
        vals = ic_df.filter(
            (pl.col("signal") == signal) & (pl.col("label") == label) & (pl.col("horizon") == h)
        )["ic"].to_numpy()
        mean, lo, hi, bt = _block_bootstrap_mean(vals)
        m, sd, ir, hit = _ic_ir(vals)
        rows.append(
            dict(
                horizon=h,
                mean_ic=mean,
                ci_lo=lo,
                ci_hi=hi,
                boot_t=bt,
                ic_ir=ir,
                hit_rate=hit,
                n_days=int(np.sum(~np.isnan(vals))),
            )
        )
    return pl.DataFrame(rows)


# Generate and save plots.
def make_plots(ic_df, regime_df, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. IC Decay Plot
    plt.figure(figsize=(10, 5.5))
    colors = {"ofi_1s": "steelblue", "ofi_clean": "crimson"}
    for sig in SIGNALS:
        d = agg_decay(ic_df, sig, PRIMARY_LABEL).sort("horizon")
        plt.plot(
            d["horizon"], d["mean_ic"], marker="o", color=colors.get(sig, "gray"), lw=2, label=sig
        )
        plt.fill_between(
            d["horizon"], d["ci_lo"], d["ci_hi"], color=colors.get(sig, "gray"), alpha=0.15
        )

    plt.axhline(0, color="black", ls="--", alpha=0.5)
    plt.title(f"OFI IC Decay (Label: {PRIMARY_LABEL})")
    plt.xlabel("Horizon (seconds)")
    plt.ylabel("Spearman Rank IC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ic_decay_curve.png"), dpi=200)
    plt.close()

    # 2. IC Distribution Plot
    vals = ic_df.filter(
        (pl.col("signal") == PRIMARY_SIGNAL)
        & (pl.col("label") == PRIMARY_LABEL)
        & (pl.col("horizon") == PRIMARY_HORIZON)
    )["ic"].to_numpy()
    if vals.size:
        plt.figure(figsize=(8, 5))
        plt.hist(vals, bins=max(5, vals.size // 2), color="crimson", alpha=0.75, edgecolor="white")
        plt.axvline(0, color="black", ls="--")
        plt.axvline(
            float(np.nanmean(vals)), color="navy", lw=2, label=f"mean={np.nanmean(vals):.4f}"
        )
        plt.title(f"Daily IC Dist ({PRIMARY_SIGNAL}, {PRIMARY_LABEL}, {PRIMARY_HORIZON}s)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ic_distribution.png"), dpi=200)
        plt.close()

    # 3. Regime Plot
    if regime_df.height:
        rg = (
            regime_df.filter(pl.col("horizon") == PRIMARY_HORIZON)
            .group_by("regime")
            .agg(pl.col("ic").mean().alias("mean_ic"))
        )
        rg = rg.with_columns(
            pl.col("regime")
            .replace_strict({"low": 0, "normal": 1, "high": 2}, default=3)
            .alias("o")
        ).sort("o")
        plt.figure(figsize=(7, 5))
        plt.bar(
            rg["regime"].to_list(), rg["mean_ic"].to_list(), color=["#8ecae6", "#219ebc", "#023047"]
        )
        plt.axhline(0, color="black", ls="--")
        plt.title(f"Mean IC by Vol Regime ({PRIMARY_SIGNAL}, {PRIMARY_HORIZON}s)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "regime_ic.png"), dpi=200)
        plt.close()


# Execute ML gate evaluation.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default="research/ml_gate_out")
    args = ap.parse_args()

    start, end = (
        datetime.strptime(args.start, "%Y-%m-%d").date(),
        datetime.strptime(args.end, "%Y-%m-%d").date(),
    )
    if start > end:
        sys.exit("Error: start date must be <= end date")

    os.makedirs(args.out, exist_ok=True)
    tasks = [
        (s, d.strftime("%Y-%m-%d"), args.data_dir)
        for s in args.symbols
        for d in daterange(start, end)
    ]

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(process_day, t) for t in tasks]):
            r = f.result()
            results.append(r)
            if r["status"] == "ok":
                log.info(f"Processed {r['symbol']} {r['date']} - OFI Mean: {r['ofi_mean']:.3g}")

    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        sys.exit("No valid days processed. Check your data dir.")

    ic_df = pl.DataFrame(
        [rec for r in ok for rec in r["ic_records"]],
        schema=["symbol", "date", "signal", "label", "horizon", "ic", "n"],
        orient="row",
    )
    reg_rows = [rec for r in ok for rec in r["regime_records"]]
    regime_df = (
        pl.DataFrame(
            reg_rows, schema=["symbol", "date", "regime", "horizon", "ic", "n"], orient="row"
        )
        if reg_rows
        else pl.DataFrame()
    )

    ic_df.write_csv(os.path.join(args.out, "daily_ic_records.csv"))
    if regime_df.height:
        regime_df.write_csv(os.path.join(args.out, "regime_ic_records.csv"))

    # Results readout
    print("\n--- ML Gate Results ---")
    print(f"Dates: {args.start} to {args.end} | Symbols: {args.symbols}")

    prim = ic_df.filter(
        (pl.col("signal") == PRIMARY_SIGNAL)
        & (pl.col("label") == PRIMARY_LABEL)
        & (pl.col("horizon") == PRIMARY_HORIZON)
    )["ic"].to_numpy()
    m, sd, ir, hit = _ic_ir(prim)
    b_mean, b_lo, b_hi, b_t = _block_bootstrap_mean(prim)

    print(
        f"\nTarget Signal: {PRIMARY_SIGNAL} | Label: {PRIMARY_LABEL} | Horizon: {PRIMARY_HORIZON}s"
    )
    print(f"Mean IC: {m:.4f} | IC-IR: {ir:.2f} | Hit Rate: {hit:.2f}")
    print(f"Bootstrap 95% CI: [{b_lo:.4f}, {b_hi:.4f}]")

    print("\nDecay Curve Stats:")
    dec = agg_decay(ic_df, PRIMARY_SIGNAL, PRIMARY_LABEL).sort("horizon")
    for row in dec.iter_rows(named=True):
        print(
            f"Horizon {row['horizon']:>3}s | "
            f"Mean IC: {row['mean_ic']:.4f} | "
            f"IC-IR: {row['ic_ir']:.2f}"
        )

    dec.write_csv(os.path.join(args.out, "decay_curve.csv"))
    make_plots(ic_df, regime_df, args.out)

    passed_all = (
        (m >= GATE["min_mean_ic"]) and (ir >= GATE["min_ic_ir"]) and (hit >= GATE["min_hit_rate"])
    )
    if GATE["require_ci_above_zero"]:
        passed_all = passed_all and (b_lo > 0)

    verdict = "PASS - Proceed to ML Layer" if passed_all else "FAIL - Needs tuning/more data"
    print(f"\nFinal Verdict: {verdict}")
    print(f"Plots and CSVs saved to {args.out}/")


if __name__ == "__main__":
    main()
