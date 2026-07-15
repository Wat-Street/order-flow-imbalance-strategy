"""Tests for the M* depth study — OFI correctness, null-safety, and that the
selection procedure recovers a known injected depth.
"""

import numpy as np
import polars as pl
import pytest

import order_flow_imbalance_strategy.depth_study as D
import order_flow_imbalance_strategy.normalize_hyperliquid as N


def _aligned_row(bpx, bsz, apx, asz, valid=5, stale=False, stale_prev=False):
    d = {
        "book_stale": stale,
        "book_stale_prev": stale_prev,
        "mid_price": (bpx + apx) / 2,
    }
    for side in ("bid", "ask"):
        for k in range(1, N.DEPTH + 1):
            d[N.px(side, k)] = None
            d[N.sz(side, k)] = None
            d[N.cnt(side, k)] = None
    for k in range(1, valid + 1):
        d[N.px("bid", k)] = float(bpx - (k - 1))
        d[N.sz("bid", k)] = float(bsz)
        d[N.px("ask", k)] = float(apx + (k - 1))
        d[N.sz("ask", k)] = float(asz)
    return d


# --- OFI six-case correctness (the doc's mandatory unit tests) -------------


@pytest.mark.parametrize(
    "prev,curr,expect_sign",
    [
        ((100, 5, 101, 5), (100.5, 3, 101, 5), 1),  # bid price rises -> +
        ((100, 5, 101, 5), (99, 3, 101, 5), -1),  # bid price falls -> -
        ((100, 5, 101, 5), (100, 5, 101.5, 3), 1),  # ask price rises -> +
        ((100, 5, 101, 5), (100, 5, 100.5, 3), -1),  # ask price falls -> -
        ((100, 5, 101, 5), (100, 8, 101, 5), 1),  # bid qty up same px -> +
        ((100, 5, 101, 5), (100, 5, 101, 9), -1),  # ask qty up same px -> -
    ],
)
def test_ofi_six_cases(prev, curr, expect_sign):
    df = pl.DataFrame([_aligned_row(*prev), _aligned_row(*curr)])
    ofi = df.with_columns(D.level_ofi_expr(1))["ofi_01"].to_list()[1]
    assert np.sign(ofi) == expect_sign


def test_ofi_mean_near_zero_on_random_walk():
    # A symmetric random book should give OFI mean ~ 0 (sign convention sanity).
    rng = np.random.default_rng(0)
    rows = []
    bpx = 100.0
    for _ in range(2000):
        bpx += rng.choice([-1, 0, 1])
        rows.append(_aligned_row(bpx, rng.integers(1, 10), bpx + 1, rng.integers(1, 10)))
    df = pl.DataFrame(rows).with_columns(D.level_ofi_expr(1))
    mean_ofi = df["ofi_01"].drop_nulls().mean()
    assert abs(mean_ofi) < 3.0  # not a persistent drift


# --- null-safety -----------------------------------------------------------


def test_ofi_null_on_stale_and_first_row():
    import datetime as dt

    base = dt.datetime(2024, 1, 2, tzinfo=dt.UTC)
    rows = [
        _aligned_row(100, 5, 101, 5),  # row 0: no t-1 -> null
        _aligned_row(100, 8, 101, 5, stale=True),  # stale -> null
        _aligned_row(100, 8, 101, 5, stale_prev=True),  # stale_prev -> null
        _aligned_row(100, 8, 101, 5),  # valid
    ]
    for i, r in enumerate(rows):
        r["ts"] = base + dt.timedelta(seconds=i)
    df = pl.DataFrame(rows)
    panel = D.build_ofi_panel(df, horizon_s=1)
    ofi = panel["ofi_01"].to_list()
    assert ofi[0] is None  # first row
    assert ofi[1] is None  # stale
    assert ofi[2] is None  # stale_prev
    assert ofi[3] is not None  # valid


def test_ofi_null_when_level_absent():
    # Level 6 is absent (thin book valid=5) -> ofi_06 must be null, not 0.
    df = pl.DataFrame(
        [_aligned_row(100, 5, 101, 5, valid=5), _aligned_row(100, 8, 101, 5, valid=5)]
    )
    ofi6 = df.with_columns(D.level_ofi_expr(6))["ofi_06"].to_list()[1]
    assert ofi6 is None


# --- selection recovers an injected depth ----------------------------------


def test_selection_recovers_injected_depth():
    # Construct an OFI panel where only levels 1..3 carry signal into fwd_ret and
    # levels 4..20 are pure noise. m_star should be small (<= a few).
    rng = np.random.default_rng(42)
    n = 4000
    ofi = rng.normal(size=(n, N.DEPTH))
    signal = ofi[:, 0] * 0.5 + ofi[:, 1] * 0.3 + ofi[:, 2] * 0.2
    fwd = signal + rng.normal(scale=1.0, size=n)  # noisy target
    cols = {f"ofi_{m:02d}": ofi[:, m - 1] for m in range(1, N.DEPTH + 1)}
    cols["fwd_ret"] = fwd
    panel = pl.DataFrame(cols)
    res = D.run_depth_study(panel, horizon_s=1, train_frac=0.7)
    # Deep noise levels shouldn't improve OOS; M* should be modest.
    assert res["m_star"] <= 6
    # PC structure should be near-identity here (independent noise) -> many PCs;
    # just assert the field exists and is sane.
    assert 1 <= res["pc_for_90pct"] <= N.DEPTH
    assert res["deepest_significant_level"] >= 1
