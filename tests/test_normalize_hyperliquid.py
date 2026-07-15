"""Tests for Stage-2 L2 normalization — the ladder-validation taxonomy.

Covers Categories A-E from the design doc: top-of-book drops, deep-level nulling,
monotonicity (equal-dup and inversion), thin-book tolerance, row-count guard,
derived fields, levels_valid counts, and the pre-write schema assertions.
No network; builds synthetic frames in the real archive shape.
"""

import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest

import order_flow_imbalance_strategy.normalize_hyperliquid as N


def mkrow(event_ms: int, bids: list[tuple], asks: list[tuple]) -> dict:
    """Per-snapshot flat dict. bids/asks are lists of (px, sz, n); padded to 20."""
    r: dict[str, object] = {"event_time_ms": event_ms}
    for side, book in (("bid", bids), ("ask", asks)):
        for i in range(N.DEPTH):
            k = i + 1
            if i < len(book):
                p, s, n = book[i]
                r[N.px(side, k)], r[N.sz(side, k)], r[N.cnt(side, k)] = p, s, n
            else:
                r[N.px(side, k)] = r[N.sz(side, k)] = r[N.cnt(side, k)] = None
    return r


def frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


# --- Category A: top-of-book drops ----------------------------------------


def test_cat_a_crossed_touch_dropped():
    df = frame([mkrow(1, [(100, 1, 1)], [(100, 1, 1)])])  # ask==bid crossed
    clean, stats = N.validate_ladder(df)
    assert clean.height == 0
    assert stats["after_touch"] == 0


def test_cat_a_nonpositive_touch_dropped():
    df = frame([mkrow(1, [(0, 1, 1)], [(101, 1, 1)]), mkrow(2, [(100, 1, 1)], [(101, 1, 1)])])
    clean, _ = N.validate_ladder(df)
    assert clean["event_time_ms"].to_list() == [2]


def test_cat_a_zero_touch_size_dropped():
    df = frame([mkrow(1, [(100, 0, 1)], [(101, 1, 1)]), mkrow(2, [(100, 1, 1)], [(101, 1, 1)])])
    clean, _ = N.validate_ladder(df)
    assert clean["event_time_ms"].to_list() == [2]


def test_cat_a_duplicate_event_time_keep_last():
    # Same event_ms twice; keep last (different bid px to tell them apart).
    df = frame([mkrow(5, [(100, 1, 1)], [(101, 1, 1)]), mkrow(5, [(200, 1, 1)], [(201, 1, 1)])])
    clean, stats = N.validate_ladder(df)
    assert clean.height == 1
    assert stats["dup_time_dropped"] == 1
    assert clean[N.px("bid", 1)].to_list() == [200.0]  # the last one


# --- Category B: deep-level individual defects ----------------------------


def test_cat_b_negative_deep_size_nulled_not_dropped():
    df = frame([mkrow(1, [(100, 1, 1), (99, -5, 1)], [(101, 1, 1)])])
    clean, _ = N.validate_ladder(df)
    assert clean.height == 1  # row kept
    assert clean[N.px("bid", 2)].to_list() == [None]  # level 2 nulled
    assert clean[N.sz("bid", 2)].to_list() == [None]


def test_cat_b_nonpositive_deep_price_nulled():
    df = frame([mkrow(1, [(100, 1, 1), (0, 2, 1)], [(101, 1, 1)])])
    clean, _ = N.validate_ladder(df)
    assert clean[N.px("bid", 2)].to_list() == [None]


# --- Category C: monotonicity ---------------------------------------------


def test_cat_c_eq_duplicate_price_nulls_deeper():
    # bid level 2 and 3 share price 99 -> null the deeper (level 3).
    df = frame([mkrow(1, [(100, 1, 1), (99, 2, 1), (99, 3, 1), (97, 1, 1)], [(101, 1, 1)])])
    clean, _ = N.validate_ladder(df)
    assert clean[N.px("bid", 2)].to_list() == [99.0]  # kept
    assert clean[N.px("bid", 3)].to_list() == [None]  # dup nulled


def test_cat_c_inv_bid_ascending_dropped():
    # bid prices ascending (100 then 101) = inversion -> drop row.
    df = frame(
        [
            mkrow(1, [(100, 1, 1), (101, 1, 1)], [(102, 1, 1)]),
            mkrow(2, [(100, 1, 1), (99, 1, 1)], [(101, 1, 1)]),
        ]
    )
    clean, stats = N.validate_ladder(df)
    assert stats["dropped_inverted"] == 1
    assert clean["event_time_ms"].to_list() == [2]


def test_cat_c_inv_ask_descending_dropped():
    df = frame([mkrow(1, [(100, 1, 1)], [(101, 1, 1), (100.5, 1, 1)])])  # ask descending
    clean, stats = N.validate_ladder(df)
    assert stats["dropped_inverted"] == 1
    assert clean.height == 0


# --- Category D: thin book is valid ---------------------------------------


def test_cat_d_thin_book_kept():
    # Only 3 levels populated; trailing nulls are NOT a defect.
    df = frame([mkrow(1, [(100, 1, 1), (99, 1, 1), (98, 1, 1)], [(101, 1, 1), (102, 1, 1)])])
    clean = N.add_derived(N.validate_ladder(df)[0])
    assert clean.height == 1
    assert clean["levels_valid_bid"].to_list() == [3]
    assert clean["levels_valid_ask"].to_list() == [2]


# --- derived fields + levels_valid ----------------------------------------


def test_derived_fields():
    df = frame([mkrow(1, [(100, 2, 1)], [(102, 4, 1)])])
    clean = N.add_derived(N.validate_ladder(df)[0])
    row = clean.to_dicts()[0]
    assert row["mid_price"] == 101.0
    assert row["spread"] == 2.0
    assert row["queue_imbalance"] == pytest.approx((2 - 4) / (2 + 4))


def test_levels_valid_truncates_at_null():
    # bid level 3 will be nulled (dup of level 2 price) -> valid depth = 2.
    df = frame([mkrow(1, [(100, 1, 1), (99, 1, 1), (99, 1, 1), (97, 1, 1)], [(101, 1, 1)])])
    clean = N.add_derived(N.validate_ladder(df)[0])
    assert clean["levels_valid_bid"].to_list() == [2]  # not 3 (null truncates run)


# --- schema assertions -----------------------------------------------------


def test_assert_schema_rejects_negative_spread():
    # Construct a frame that (illegally) has a negative spread to trip assertion.
    df = frame([mkrow(1, [(100, 1, 1)], [(101, 1, 1)])])
    clean = N.add_derived(N.validate_ladder(df)[0]).with_columns(
        pl.col("event_time_ms").cast(pl.Datetime("ms")).alias("timestamp"),
        pl.lit(-1.0).alias("spread"),
    )
    with pytest.raises(AssertionError, match="negative spread"):
        N.assert_schema(clean)


# --- end-to-end normalize_day ---------------------------------------------


def _archive_line(event_ms: int, bids: list[tuple], asks: list[tuple]) -> str:
    def lv(side):
        return [{"px": str(p), "sz": str(s), "n": n} for p, s, n in side]

    data = {"coin": "BTC", "time": event_ms, "levels": [lv(bids), lv(asks)]}
    return json.dumps(
        {"time": "2024-01-02T00:00:00.0", "ver_num": 1, "raw": {"channel": "l2Book", "data": data}}
    )


def _write_hour(raw_dir: Path, coin: str, date_str: str, hour: int, lines: list[str]) -> None:
    p = N.raw_hour_path(raw_dir, coin, date_str, hour)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_normalize_day_writes_parquet(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "proc"
    base = int(dt.datetime(2024, 1, 2, tzinfo=dt.UTC).timestamp() * 1000)
    # 150 valid snapshots so we clear the MIN_ROWS=100 guard.
    lines = [
        _archive_line(base + i * 100, [(100 + i * 0.1, 1, 1), (99, 1, 1)], [(101 + i * 0.1, 1, 1)])
        for i in range(150)
    ]
    _write_hour(raw, "BTC", "2024-01-02", 0, lines)
    res = N.normalize_day(raw, out, "BTC", "2024-01-02")
    assert res["status"] == "ok"
    dest = N.out_path(out, "BTC", "2024-01-02")
    df = pl.read_parquet(dest)
    assert df.height == 150
    assert set(N.CRITICAL_COLUMNS).issubset(df.columns)
    assert df["timestamp"].dtype == pl.Datetime("ms")
    # No nulls in critical columns.
    for c in N.CRITICAL_COLUMNS:
        assert df[c].null_count() == 0


def test_normalize_day_too_few_rows_not_written(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "proc"
    base = int(dt.datetime(2024, 1, 2, tzinfo=dt.UTC).timestamp() * 1000)
    lines = [_archive_line(base + i * 100, [(100, 1, 1)], [(101, 1, 1)]) for i in range(10)]
    _write_hour(raw, "BTC", "2024-01-02", 0, lines)
    res = N.normalize_day(raw, out, "BTC", "2024-01-02")
    assert res["status"] == "failed_validation"
    assert not N.out_path(out, "BTC", "2024-01-02").exists()


def test_normalize_day_missing_raw_is_empty(tmp_path):
    res = N.normalize_day(tmp_path / "raw", tmp_path / "proc", "BTC", "2024-01-02")
    assert res["status"] == "empty"
