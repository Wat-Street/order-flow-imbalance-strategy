from datetime import datetime, timedelta

import polars as pl
import pytest

from order_flow_imbalance_strategy.ofi import (
    build_tasks,
    compute_ofi,
    daterange,
    main,
    process_file,
    validate_ofi,
)

T0 = datetime(2024, 1, 1, 0, 0, 0)


def make_frame(rows: list[dict]) -> pl.DataFrame:
    out = []
    for i, r in enumerate(rows):
        rec = {
            "timestamp": T0 + timedelta(seconds=i),
            "bid_price": r["bid_price"],
            "ask_price": r["ask_price"],
            "bid_qty": r["bid_qty"],
            "ask_qty": r["ask_qty"],
            "book_stale": r.get("book_stale", False),
        }
        out.append(rec)
    return pl.DataFrame(out)


def ofi_of(rows: list[dict]) -> pl.DataFrame:
    return compute_ofi(make_frame(rows))


# Case 1: Bid up, ask unchanged => pos
# Case 2: Bid down, ask unchanged => neg
# Case 3: Ask up, bid unchanged => pos
# Case 4: Ask down, bid unchanged=> neg
# Case 5: Bid/ask unchanged, best bid vol increase => pos
# Case 6: Bid/ask unchanged, best ask vol increase => neg


def test_bid_price_rises_ask_unchanged_is_positive():
    # e_b = +bid_qty[t] = +5, e_a = 0, OFI = 5
    df = ofi_of(
        [
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
            {"bid_price": 100.5, "ask_price": 101.0, "bid_qty": 5.0, "ask_qty": 4.0},
        ]
    )
    assert df["bid_contribution"][1] == pytest.approx(5.0)
    assert df["ask_contribution"][1] == pytest.approx(0.0)
    assert df["ofi_1s"][1] == pytest.approx(5.0)
    assert df["ofi_1s"][1] > 0


def test_bid_price_falls_ask_unchanged_is_negative():
    # e_b = -bid_qty[t-1] = -3, e_a = 0 , OFI = -3
    df = ofi_of(
        [
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
            {"bid_price": 99.5, "ask_price": 101.0, "bid_qty": 5.0, "ask_qty": 4.0},
        ]
    )
    assert df["bid_contribution"][1] == pytest.approx(-3.0)
    assert df["ofi_1s"][1] == pytest.approx(-3.0)
    assert df["ofi_1s"][1] < 0


def test_ask_price_rises_bid_unchanged_is_positive():
    # e_a = -ask_qty[t-1] = -4, e_b = 0, OFI = 4
    df = ofi_of(
        [
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
            {"bid_price": 100.0, "ask_price": 101.5, "bid_qty": 3.0, "ask_qty": 7.0},
        ]
    )
    assert df["ask_contribution"][1] == pytest.approx(-4.0)
    assert df["ofi_1s"][1] == pytest.approx(4.0)
    assert df["ofi_1s"][1] > 0


def test_ask_price_falls_bid_unchanged_is_negative():
    # e_a = +ask_qty[t] = 7, e_b = 0, OFI =-7
    df = ofi_of(
        [
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
            {"bid_price": 100.0, "ask_price": 100.5, "bid_qty": 3.0, "ask_qty": 7.0},
        ]
    )
    assert df["ask_contribution"][1] == pytest.approx(7.0)
    assert df["ofi_1s"][1] == pytest.approx(-7.0)
    assert df["ofi_1s"][1] < 0


def test_prices_unchanged_bid_qty_increases_is_positive():
    # e_b = bid_qty[t] - bid_qty[t-1] = 8 - 3 = 5, e_a = 0, OFI = 5
    df = ofi_of(
        [
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 8.0, "ask_qty": 4.0},
        ]
    )
    assert df["bid_contribution"][1] == pytest.approx(5.0)
    assert df["ofi_1s"][1] == pytest.approx(5.0)
    assert df["ofi_1s"][1] > 0


def test_prices_unchanged_ask_qty_increases_is_negative():
    # e_a = ask_qty[t] - ask_qty[t-1] = 9 - 4 = 5, e_b = 0, OFI = -5
    df = ofi_of(
        [
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 9.0},
        ]
    )
    assert df["ask_contribution"][1] == pytest.approx(5.0)
    assert df["ofi_1s"][1] == pytest.approx(-5.0)
    assert df["ofi_1s"][1] < 0


## edge cases
# Case 7: First row of day (no t-1) => null
def test_first_row_of_day_is_null():
    df = ofi_of(
        [
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
            {"bid_price": 100.5, "ask_price": 101.0, "bid_qty": 5.0, "ask_qty": 4.0},
        ]
    )
    assert df["ofi_1s"][0] is None
    assert df["bid_contribution"][0] is None
    assert df["ask_contribution"][0] is None


# Case 8: Book stale on current row => null
def test_book_stale_current_row_is_null():
    df = ofi_of(
        [
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
            {
                "bid_price": 100.5,
                "ask_price": 101.0,
                "bid_qty": 5.0,
                "ask_qty": 4.0,
                "book_stale": True,
            },
        ]
    )
    assert df["ofi_1s"][1] is None
    assert df["bid_contribution"][1] is None
    assert df["ask_contribution"][1] is None


# Case 9: Book stale on previous row (t-1) => null
def test_book_stale_previous_row_is_null():
    # Row 1 is not stale, but row 0 (its t-1) is -> row 1 OFI must be null.
    df = ofi_of(
        [
            {
                "bid_price": 100.0,
                "ask_price": 101.0,
                "bid_qty": 3.0,
                "ask_qty": 4.0,
                "book_stale": True,
            },
            {"bid_price": 100.5, "ask_price": 101.0, "bid_qty": 5.0, "ask_qty": 4.0},
        ]
    )
    assert df["ofi_1s"][1] is None


# Case 10: Null input qty (current row and following row's t-1) => null
def test_null_input_propagates_to_null_output():
    df = make_frame(
        [
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
            {"bid_price": 100.5, "ask_price": 101.0, "bid_qty": 5.0, "ask_qty": 4.0},
            {"bid_price": 100.5, "ask_price": 101.0, "bid_qty": 6.0, "ask_qty": 4.0},
        ]
    )
    df = df.with_columns(
        pl.when(pl.arange(0, pl.len()) == 1)
        .then(None)
        .otherwise(pl.col("bid_qty"))
        .alias("bid_qty")
    )
    out = compute_ofi(df)
    # Row 1 has a null input -> null output.
    assert out["ofi_1s"][1] is None
    # Row 2 has a null *previous* value (row 1's bid_qty) -> null output too.
    assert out["ofi_1s"][2] is None


# Case 11: Row count & columns preserved, no inf/NaN among non-null ofi_1s
def test_row_count_and_columns_preserved():
    rows = [
        {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
        {"bid_price": 100.5, "ask_price": 101.0, "bid_qty": 5.0, "ask_qty": 4.0},
    ]
    src = make_frame(rows)
    out = compute_ofi(src)
    assert out.height == src.height
    for col in src.columns:
        assert col in out.columns
    for col in ("ofi_1s", "bid_contribution", "ask_contribution"):
        assert col in out.columns
    # No NaN/inf among non-null ofi_1s values.
    bad = out["ofi_1s"].is_finite().not_() & out["ofi_1s"].is_not_null()
    assert not bad.any()


# Case 12: Unsorted input is sorted by timestamp before compute
def test_unsorted_input_is_sorted_by_timestamp():
    rows = [
        {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
        {"bid_price": 100.5, "ask_price": 101.0, "bid_qty": 5.0, "ask_qty": 4.0},
    ]
    src = make_frame(rows)
    shuffled = src.reverse()
    out = compute_ofi(shuffled)
    # After internal sort, row 0 is warmup and row 1 is the rising-bid case (+5).
    assert out["ofi_1s"][0] is None
    assert out["ofi_1s"][1] == pytest.approx(5.0)


# Case 13: Missing required column => ValueError
def test_missing_required_column_raises():
    df = pl.DataFrame({"timestamp": [T0], "bid_price": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        compute_ofi(df)


# Case 13b: Input already carrying an output column (e.g. a rerun/fed-back
# signal frame) => ValueError, not an opaque DuplicateError from select().
def test_input_with_existing_output_column_raises():
    df = make_frame(
        [
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
            {"bid_price": 100.5, "ask_price": 101.0, "bid_qty": 5.0, "ask_qty": 4.0},
        ]
    ).with_columns(pl.lit(0.0).alias("ofi_1s"))
    with pytest.raises(ValueError, match="already contains output columns"):
        compute_ofi(df)


# --- worker / IO ------------------------------------------------------------


# Case 14: process_file writes output and is idempotent (skip on rerun)
def test_process_file_writes_and_is_idempotent(tmp_path):
    src = make_frame(
        [
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
            {"bid_price": 100.5, "ask_price": 101.0, "bid_qty": 5.0, "ask_qty": 4.0},
        ]
    )
    in_path = tmp_path / "BTCUSDT-aligned-2024-01-01.parquet"
    out_path = tmp_path / "BTCUSDT-ofi-A-2024-01-01.parquet"
    src.write_parquet(in_path)

    task = {
        "symbol": "BTCUSDT",
        "date": "2024-01-01",
        "in_path": str(in_path),
        "out_path": str(out_path),
    }

    r1 = process_file(task)
    assert r1["status"] == "ok"
    assert r1["rows"] == 2
    assert out_path.exists()

    written = pl.read_parquet(out_path)
    assert written.height == 2
    assert written["ofi_1s"][1] == pytest.approx(5.0)

    # Second run is a no-op skip.
    r2 = process_file(task)
    assert r2["status"] == "skipped"


# Case 15: process_file with missing input => no_input status
def test_process_file_missing_input_returns_no_input(tmp_path):
    task = {
        "symbol": "BTCUSDT",
        "date": "2024-01-01",
        "in_path": str(tmp_path / "nope.parquet"),
        "out_path": str(tmp_path / "out.parquet"),
    }
    assert process_file(task)["status"] == "no_input"


# Case 16: validate_ofi flags row-count mismatch vs input
def test_validate_ofi_flags_row_count_mismatch():
    out = ofi_of(
        [
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
            {"bid_price": 100.5, "ask_price": 101.0, "bid_qty": 5.0, "ask_qty": 4.0},
        ]
    )
    assert validate_ofi(out, expected_rows=2) == []
    problems = validate_ofi(out, expected_rows=999)
    assert problems and "row count" in problems[0]


# Case 17: validate_ofi flags non-finite ofi_1s (inf/-inf/NaN)
@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_validate_ofi_flags_non_finite(bad):
    # A clean frame, then poison a single non-null ofi_1s value with inf/-inf/NaN.
    out = ofi_of(
        [
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
            {"bid_price": 100.5, "ask_price": 101.0, "bid_qty": 5.0, "ask_qty": 4.0},
        ]
    )
    poisoned = out.with_columns(
        pl.when(pl.arange(0, pl.len()) == 1)
        .then(pl.lit(bad))
        .otherwise(pl.col("ofi_1s"))
        .alias("ofi_1s")
    )
    problems = validate_ofi(poisoned, expected_rows=poisoned.height)
    assert problems and "non-finite" in problems[0]


# Case 18: validate_ofi flags ofi/contribution null-state mismatch
def test_validate_ofi_flags_contribution_null_state_mismatch():
    # ofi_1s present on row 1 but bid_contribution nulled -> inconsistent null-state.
    out = ofi_of(
        [
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
            {"bid_price": 100.5, "ask_price": 101.0, "bid_qty": 5.0, "ask_qty": 4.0},
        ]
    )
    null_f64 = pl.lit(None, dtype=pl.Float64)
    mismatched = out.with_columns(
        pl.when(pl.arange(0, pl.len()) == 1)
        .then(null_f64)
        .otherwise(pl.col("bid_contribution"))
        .alias("bid_contribution")
    )
    problems = validate_ofi(mismatched, expected_rows=mismatched.height)
    assert problems and "null-state disagrees" in problems[0]


# Case 19: Full-day OFI mean ~ 0 (sign-convention sanity), validation clean
def test_full_day_ofi_mean_is_approximately_zero():
    # A long symmetric random walk in the best bid/ask should produce a roughly
    # mean-zero OFI; a persistent large drift would mean the sign convention is
    # flipped. validate_ofi does NOT block on drift, so the frame stays clean.
    import random

    random.seed(1234)
    price = 100.0
    rows: list[dict] = []
    for _ in range(2000):
        price = round(price + random.choice([-0.01, 0.0, 0.01]), 2)
        rows.append(
            {
                "bid_price": price,
                "ask_price": round(price + 0.02, 2),
                "bid_qty": round(random.uniform(1.0, 9.0), 3),
                "ask_qty": round(random.uniform(1.0, 9.0), 3),
            }
        )
    out = ofi_of(rows)

    mean = out["ofi_1s"].mean()
    std = out["ofi_1s"].std()
    assert std and std > 0
    # Drift well under the 0.1*std threshold compute_ofi flags as suspicious.
    assert abs(mean) < 0.1 * std
    # A mean-zero, finite, consistent frame must pass hard validation cleanly.
    assert validate_ofi(out, expected_rows=out.height) == []


# Case 20: Gap in the 1-second grid (missing second) => null across the gap
def test_gap_in_grid_nulls_ofi():
    # Rows at t, t+1s, then a jump to t+300s. The third row must be null because
    # OFI is only defined between adjacent seconds, never across a missing gap.
    df = make_frame(
        [
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
            {"bid_price": 105.0, "ask_price": 106.0, "bid_qty": 9.0, "ask_qty": 9.0},
        ]
    )
    df = df.with_columns(
        pl.when(pl.arange(0, pl.len()) == 2)
        .then(pl.lit(T0 + timedelta(seconds=300)))
        .otherwise(pl.col("timestamp"))
        .alias("timestamp")
    )
    out = compute_ofi(df)
    assert out["ofi_1s"][1] == pytest.approx(0.0)  # adjacent 1s row is fine
    assert out["ofi_1s"][2] is None  # across the 299s gap -> null


# Case 21: Crossed/locked book (ask <= bid, non-positive spread) => null
def test_crossed_book_nulls_ofi():
    # Row 1 is locked (bid == ask); row 2's t-1 is that locked book.
    df = make_frame(
        [
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 3.0, "ask_qty": 4.0},
            {"bid_price": 101.0, "ask_price": 101.0, "bid_qty": 5.0, "ask_qty": 4.0},
            {"bid_price": 100.0, "ask_price": 101.0, "bid_qty": 6.0, "ask_qty": 4.0},
        ]
    )
    out = compute_ofi(df)
    assert out["ofi_1s"][1] is None  # crossed/locked current book
    assert out["ofi_1s"][2] is None  # previous book was crossed/locked


# Case 21b: Frame omitting book_stale falls back to "not stale" (not all-null).
# ``pl.lit(False).shift(1)`` is all-null; a naive fallback would treat every
# t-1 as stale and null the whole output. The fallback must compute OFI normally.
def test_missing_book_stale_column_falls_back_to_not_stale():
    df = pl.DataFrame(
        {
            "timestamp": [T0, T0 + timedelta(seconds=1)],
            "bid_price": [100.0, 100.5],
            "ask_price": [101.0, 101.0],
            "bid_qty": [3.0, 5.0],
            "ask_qty": [4.0, 4.0],
        }
    )
    assert "book_stale" not in df.columns
    out = compute_ofi(df)
    # Row 0 is warmup; row 1 is the rising-bid case (+5) and MUST be non-null.
    assert out["ofi_1s"][0] is None
    assert out["ofi_1s"][1] == pytest.approx(5.0)
    assert out["bid_contribution"][1] == pytest.approx(5.0)
    assert out["ask_contribution"][1] == pytest.approx(0.0)


# --- CLI (merged into the ofi module) --------------------------------------


# Case 22: daterange is inclusive on both ends
def test_daterange_is_inclusive():
    from datetime import date

    assert daterange(date(2022, 1, 1), date(2022, 1, 1)) == [date(2022, 1, 1)]
    assert daterange(date(2022, 1, 1), date(2022, 1, 3)) == [
        date(2022, 1, 1),
        date(2022, 1, 2),
        date(2022, 1, 3),
    ]


# Case 23: build_tasks only emits days whose aligned input file exists
def test_build_tasks_only_existing_inputs(tmp_path):
    from datetime import date

    aligned = tmp_path / "aligned" / "BTCUSDT"
    aligned.mkdir(parents=True)
    # Create input for day 1 only.
    (aligned / "BTCUSDT-aligned-2022-01-01.parquet").write_bytes(b"")
    tasks = build_tasks(["BTCUSDT"], date(2022, 1, 1), date(2022, 1, 2), tmp_path)
    assert len(tasks) == 1
    assert tasks[0]["date"] == "2022-01-01"
    assert tasks[0]["out_path"].endswith("BTCUSDT-ofi-A-2022-01-01.parquet")


# Case 24: main() returns 2 when --end precedes --start
def test_main_end_before_start_returns_2(tmp_path):
    rc = main(
        [
            "--symbols",
            "BTCUSDT",
            "--start",
            "2022-01-02",
            "--end",
            "2022-01-01",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


# Case 25: main() returns 0 when there is no input to process
def test_main_no_inputs_returns_0(tmp_path):
    rc = main(
        [
            "--symbols",
            "BTCUSDT",
            "--start",
            "2022-01-01",
            "--end",
            "2022-01-01",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
