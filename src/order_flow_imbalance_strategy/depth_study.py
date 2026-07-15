"""M* depth study — empirically derive how many L2 levels the OFI signal needs.

Consumes the Stage-3 aligned Hyperliquid L2 Parquet and answers: *how deep into
the book does multi-level Order Flow Imbalance carry incremental predictive power
for the mid-price move?* The archive gives 20 levels/side; the "right" depth M*
is an empirical question, and committing/using more levels than M* just adds
noise and cost. This module derives M* with a statistically defensible procedure
rather than assuming a number.

Procedure (see also the ML-gate section of the design doc — Newey-West 1987 for
HAC, Cont-Kukanov-Stoikov 2014 for OFI):

1. **Per-level OFI** — for each level m, the piecewise Cont-Kukanov-Stoikov
   contribution ``e^(m) = e_bid^(m) - e_ask^(m)`` between consecutive 1s
   snapshots, keyed on whether that level's price moved. **Null-safe**: if level
   m is null (thin book / nulled defect) on t or t-1, or the row is stale /
   post-gap / day-boundary, that level's contribution is null (never 0 — zeroing
   fabricates flow). This honors the cross-stage contract from Stage 2/3.

2. **Nested-model marginal R²** — regress forward mid-return on the cumulative
   OFI vector (levels 1..M) for M = 1..20; track adjusted R² *in-sample* and
   **out-of-sample** R² on a held-out tail. OOS penalizes useless deep levels
   (in-sample R² is monotone; OOS turns over) — the OOS elbow is the honest M*.

3. **Per-level HAC t-stats** — Newey-West (HAC) standard errors on the full
   20-level regression; 1s book data is heavily autocorrelated so OLS SEs lie.
   The deepest level with a significant |t| bounds M*.

4. **PCA of the level-wise OFI** — eigenvalue spectrum of the 20-OFI correlation
   matrix. The literature expects the top 1-2 PCs to explain ~90%+, i.e. the
   effective dimensionality is small even if raw levels are kept.

**Decision rule:** M* = the smallest depth within 1 standard error of the best
OOS R². Reported alongside the HAC-significance and PCA evidence.

Run::

    python -m order_flow_imbalance_strategy.depth_study \\
        --symbol BTC --start 2024-01-01 --end 2024-01-31 \\
        --aligned-dir data/aligned_hl --horizon 10 --out depth_study_BTC.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from order_flow_imbalance_strategy.normalize_hyperliquid import DEPTH, px, sz

logger = logging.getLogger("depth_study")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- per-level OFI (null-safe Cont-Kukanov-Stoikov) ------------------------


def level_ofi_expr(m: int) -> pl.Expr:
    """Per-level-m OFI contribution between row t-1 and t, as a Polars expr.

    Bid (e_b):  P^b_t > P^b_{t-1} -> +q^b_t ; == -> q^b_t - q^b_{t-1} ; < -> -q^b_{t-1}
    Ask (e_a):  P^a_t > P^a_{t-1} -> -q^a_{t-1}; == -> q^a_t - q^a_{t-1}; < -> +q^a_t
    OFI^(m) = e_b - e_a.  Null if any input (this level, either side, t or t-1) is
    null — the caller further nulls stale/gap/day-boundary rows.
    """
    # Cast to Float64 so all-null (absent) levels have a numeric dtype — otherwise
    # negating a null-dtype column (the `.otherwise(-bsz0)` branch) errors.
    bpx, bsz = pl.col(px("bid", m)).cast(pl.Float64), pl.col(sz("bid", m)).cast(pl.Float64)
    apx, asz = pl.col(px("ask", m)).cast(pl.Float64), pl.col(sz("ask", m)).cast(pl.Float64)
    bpx0, bsz0 = bpx.shift(1), bsz.shift(1)
    apx0, asz0 = apx.shift(1), asz.shift(1)

    e_b = (
        pl.when(bpx > bpx0)
        .then(bsz)
        .when(bpx == bpx0)
        .then(bsz - bsz0)
        .otherwise(-bsz0)
    )
    e_a = (
        pl.when(apx > apx0)
        .then(-asz0)
        .when(apx == apx0)
        .then(asz - asz0)
        .otherwise(asz)
    )
    # Any null input -> null contribution (Polars arithmetic already propagates
    # null, but the when/then above could mask it, so guard explicitly).
    any_null = (
        bpx.is_null() | bsz.is_null() | apx.is_null() | asz.is_null()
        | bpx0.is_null() | bsz0.is_null() | apx0.is_null() | asz0.is_null()
    )
    return pl.when(any_null).then(None).otherwise(e_b - e_a).alias(f"ofi_{m:02d}")


def build_ofi_panel(df: pl.DataFrame, horizon_s: int) -> pl.DataFrame:
    """Add per-level ofi_01..20, null out invalid rows, and add the forward
    mid-return target over ``horizon_s`` seconds. Input is one aligned day (or a
    concatenation sorted by ts within day)."""
    df = df.sort("ts")
    df = df.with_columns([level_ofi_expr(m) for m in range(1, DEPTH + 1)])

    # Null OFI on rows where the diff t-1->t is invalid: stale book, the row after
    # a gap (book_stale_prev), or the first row of the frame (no t-1).
    invalid = (
        pl.col("book_stale")
        | pl.col("book_stale_prev")
        | (pl.int_range(pl.len()) == 0)
    )
    df = df.with_columns(
        [
            pl.when(invalid).then(None).otherwise(pl.col(f"ofi_{m:02d}")).alias(f"ofi_{m:02d}")
            for m in range(1, DEPTH + 1)
        ]
    )

    # Forward mid-return target: (mid[t+h] - mid[t]) / mid[t]. Uses shift(-h).
    df = df.with_columns(
        ((pl.col("mid_price").shift(-horizon_s) - pl.col("mid_price")) / pl.col("mid_price")).alias(
            "fwd_ret"
        )
    )
    return df


# --- statistics ------------------------------------------------------------


def _ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Plain OLS R² (with intercept) via lstsq; used for in/out-of-sample fit."""
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _oos_r2(X: np.ndarray, y: np.ndarray, train_frac: float) -> float:
    """Out-of-sample R²: fit on the first train_frac, score on the tail. A
    time-ordered split (no shuffle) — the honest test for useless deep levels."""
    n = len(X)
    k = int(n * train_frac)
    if k < 50 or n - k < 50:
        return float("nan")
    Xtr = np.column_stack([np.ones(k), X[:k]])
    beta, *_ = np.linalg.lstsq(Xtr, y[:k], rcond=None)
    Xte = np.column_stack([np.ones(n - k), X[k:]])
    pred = Xte @ beta
    resid = y[k:] - pred
    ss_res = float(resid @ resid)
    ss_tot = float(((y[k:] - y[k:].mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def run_depth_study(df: pl.DataFrame, horizon_s: int, train_frac: float = 0.7) -> dict:
    """Core analysis on an OFI panel. Returns the M* result dict."""
    import statsmodels.api as sm

    ofi_cols = [f"ofi_{m:02d}" for m in range(1, DEPTH + 1)]
    panel = df.select([*ofi_cols, "fwd_ret"]).drop_nulls()
    n = panel.height
    if n < 500:
        raise ValueError(f"too few complete rows for depth study: {n}")

    X_all = panel.select(ofi_cols).to_numpy()
    y = panel["fwd_ret"].to_numpy()

    # (1) Nested marginal R² curves.
    curve = []
    for M in range(1, DEPTH + 1):
        Xm = X_all[:, :M]
        adj_is = _adjusted_r2(_ols_r2(Xm, y), n, M)
        oos = _oos_r2(Xm, y, train_frac)
        curve.append({"M": M, "adj_r2_is": adj_is, "oos_r2": oos})

    # Decision rule: smallest M within 1 SE of the best OOS R².
    oos_vals = np.array([c["oos_r2"] for c in curve], dtype=float)
    best_M = int(np.nanargmax(oos_vals)) + 1
    best_oos = float(np.nanmax(oos_vals))
    # 1-SE band: SE of OOS R² approximated across the level curve's dispersion.
    se = float(np.nanstd(oos_vals)) / np.sqrt(max(1, np.isfinite(oos_vals).sum()))
    within = [c["M"] for c in curve if np.isfinite(c["oos_r2"]) and c["oos_r2"] >= best_oos - se]
    m_star = min(within) if within else best_M

    # (3) HAC (Newey-West) t-stats on the full 20-level model.
    Xc = sm.add_constant(X_all)
    lag = int(round(1.5 * n ** (1 / 3)))  # Newey-West rule-of-thumb lag length
    hac = sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    tvals = hac.tvalues[1:]  # drop intercept
    sig_levels = [m + 1 for m, t in enumerate(tvals) if abs(t) > 1.96]
    deepest_sig = max(sig_levels) if sig_levels else 0

    # (4) PCA of the level-wise OFI (correlation-matrix eigen-spectrum).
    Xz = (X_all - X_all.mean(0)) / (X_all.std(0) + 1e-12)
    corr = np.corrcoef(Xz, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(corr))[::-1]
    evr = (eigvals / eigvals.sum()).tolist()
    pc_cum = np.cumsum(evr).tolist()

    return {
        "n_rows": n,
        "horizon_s": horizon_s,
        "train_frac": train_frac,
        "hac_maxlags": lag,
        "curve": curve,
        "best_oos_M": best_M,
        "best_oos_r2": best_oos,
        "oos_se": se,
        "m_star": m_star,
        "hac_tvalues": [round(float(t), 3) for t in tvals],
        "deepest_significant_level": deepest_sig,
        "pca_explained_var_ratio": [round(float(v), 4) for v in evr],
        "pca_cum_var": [round(float(v), 4) for v in pc_cum],
        "pc1_explains": round(float(evr[0]), 4),
        "pc_for_90pct": int(np.searchsorted(pc_cum, 0.90) + 1),
    }


def _adjusted_r2(r2: float, n: int, k: int) -> float:
    if n - k - 1 <= 0:
        return float("nan")
    return 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)


# --- IO / CLI --------------------------------------------------------------


def load_aligned(aligned_dir, symbol: str, start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
    """Load and concatenate aligned per-day files, building the OFI panel per day
    (so shift/diff never crosses a day boundary), then stack."""
    frames = []
    cur = start
    while cur <= end:
        date_str = cur.strftime("%Y-%m-%d")
        p = Path(aligned_dir) / symbol / f"{symbol}-l2-aligned-{date_str}.parquet"
        cur += dt.timedelta(days=1)
        if p.exists():
            frames.append(pl.read_parquet(p))
    if not frames:
        raise FileNotFoundError("no aligned files found in range")
    return frames


def main():
    parser = argparse.ArgumentParser(description="M* depth study on aligned L2 data")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--start", type=dt.datetime.fromisoformat, required=True)
    parser.add_argument("--end", type=dt.datetime.fromisoformat, required=True)
    parser.add_argument("--aligned-dir", default="data/aligned_hl")
    parser.add_argument("--horizon", type=int, default=10, help="forward return horizon (seconds)")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--out", default=None, help="write result JSON here")
    args = parser.parse_args()

    day_frames = load_aligned(args.aligned_dir, args.symbol, args.start, args.end)
    logger.info("Loaded %s aligned day-files", len(day_frames))
    # Build the OFI panel per day (shift stays within a day), then concatenate.
    panels = [build_ofi_panel(d, args.horizon) for d in day_frames]
    panel = pl.concat(panels, how="vertical")

    result = run_depth_study(panel, args.horizon, args.train_frac)
    result["symbol"] = args.symbol
    result["start"] = args.start.strftime("%Y-%m-%d")
    result["end"] = args.end.strftime("%Y-%m-%d")

    print("\n=== M* DEPTH STUDY:", args.symbol, result["start"], "->", result["end"], "===")
    print(f"rows={result['n_rows']:,}  horizon={args.horizon}s  HAC lags={result['hac_maxlags']}")
    print("\n  M   adj_R2(IS)   OOS_R2")
    for c in result["curve"]:
        oos = c["oos_r2"]
        print(f"  {c['M']:2d}   {c['adj_r2_is']:+.5f}   {oos:+.5f}" if np.isfinite(oos)
              else f"  {c['M']:2d}   {c['adj_r2_is']:+.5f}     nan")
    print(f"\nBest OOS R² at M={result['best_oos_M']} ({result['best_oos_r2']:+.5f}); "
          f"1-SE band => M* = {result['m_star']}")
    print(f"Deepest HAC-significant level (|t|>1.96): {result['deepest_significant_level']}")
    print(f"PCA: PC1 explains {result['pc1_explains']:.1%}; "
          f"{result['pc_for_90pct']} PCs reach 90% variance")
    print(f"\n>>> M* = {result['m_star']} levels <<<")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
