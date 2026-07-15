"""Tests for Stage-3 L2 alignment — the 1-second grid + forward-fill contract.

Covers: 86,400-row grid, last-event-per-second, forward-fill with book_stale,
book_stale_prev propagation, gap_prev_s, levels_valid carry-through, and dropping
seconds before the first snapshot. No network.
"""

import datetime as dt

import polars as pl

import order_flow_imbalance_strategy.align_hyperliquid as A
import order_flow_imbalance_strategy.normalize_hyperliquid as N

DATE = "2024-01-02"
BASE_MS = int(dt.datetime(2024, 1, 2, tzinfo=dt.UTC).timestamp() * 1000)


def _norm_row(ms_off: int, bidpx: float, valid_levels: int = 5) -> dict:
    d: dict[str, object] = {
        "event_time_ms": BASE_MS + ms_off,
        "coin": "BTC",
        "mid_price": bidpx + 0.5,
        "spread": 1.0,
        "queue_imbalance": 0.0,
        "levels_valid_bid": valid_levels,
        "levels_valid_ask": valid_levels,
    }
    for side in ("bid", "ask"):
        for k in range(1, N.DEPTH + 1):
            d[N.px(side, k)] = float(bidpx) if k <= valid_levels else None
            d[N.sz(side, k)] = 1.0 if k <= valid_levels else None
            d[N.cnt(side, k)] = 1 if k <= valid_levels else None
    return d


def _normalized_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("event_time_ms").cast(pl.Datetime("ms")).alias("timestamp"),
        pl.lit("BTC").cast(pl.Categorical).alias("coin"),
    )


def test_grid_is_86400_rows_when_first_second_populated():
    df = _normalized_frame([_norm_row(0, 100), _norm_row(86_399_000, 200)])
    al = A.align_day_frame(df, DATE)
    assert al.height == 86_400  # full day, first native at second 0


def test_last_event_per_second_wins():
    # Two snapshots in second 0: at +100ms (100) and +900ms (101). Last wins.
    df = _normalized_frame([_norm_row(100, 100), _norm_row(900, 101)])
    al = A.align_day_frame(df, DATE)
    first = al.sort("ts").head(1).to_dicts()[0]
    assert first["mid_price"] == 101.5  # from the +900ms snapshot
    assert first["book_stale"] is False


def test_forward_fill_and_book_stale():
    # Snapshots at sec 0 and sec 3; secs 1,2 must be stale + forward-filled.
    df = _normalized_frame([_norm_row(0, 100), _norm_row(3200, 102)])
    al = A.align_day_frame(df, DATE).sort("ts").head(5)
    rows = al.to_dicts()
    assert rows[0]["book_stale"] is False and rows[0]["mid_price"] == 100.5
    assert rows[1]["book_stale"] is True and rows[1]["mid_price"] == 100.5  # ffilled
    assert rows[2]["book_stale"] is True and rows[2]["mid_price"] == 100.5
    assert rows[3]["book_stale"] is False and rows[3]["mid_price"] == 102.5  # native
    assert rows[1]["gap_prev_s"] == 1 and rows[2]["gap_prev_s"] == 2


def test_book_stale_prev_true_after_gap():
    df = _normalized_frame([_norm_row(0, 100), _norm_row(3200, 102)])
    al = A.align_day_frame(df, DATE).sort("ts").head(4)
    rows = al.to_dicts()
    # The native row at sec 3 follows a gap, so book_stale_prev must be True.
    assert rows[3]["book_stale"] is False
    assert rows[3]["book_stale_prev"] is True


def test_seconds_before_first_snapshot_dropped():
    # First snapshot at second 10 -> rows for seconds 0..9 should not exist.
    df = _normalized_frame([_norm_row(10_000, 100), _norm_row(20_000, 101)])
    al = A.align_day_frame(df, DATE)
    first_ts = al.sort("ts").head(1)["ts"].to_list()[0]
    expected = dt.datetime(2024, 1, 2, 0, 0, 10, tzinfo=dt.UTC)
    assert first_ts.replace(tzinfo=dt.UTC) == expected
    assert al.height == 86_400 - 10


def test_levels_valid_carried_through():
    df = _normalized_frame(
        [_norm_row(0, 100, valid_levels=7), _norm_row(5000, 101, valid_levels=3)]
    )
    al = A.align_day_frame(df, DATE).sort("ts")
    # Second 0 native -> 7; seconds 1..4 stale (carry 7); second 5 native -> 3.
    assert al.head(1)["levels_valid_bid"].to_list() == [7]
    sec5 = al.filter(pl.col("gap_prev_s") == 0).sort("ts")
    assert sec5["levels_valid_bid"].to_list()[-1] == 3


def test_all_expected_columns_present():
    df = _normalized_frame([_norm_row(0, 100), _norm_row(1000, 101)])
    al = A.align_day_frame(df, DATE)
    for col in ["ts", "coin", "mid_price", "spread", "queue_imbalance",
                "levels_valid_bid", "levels_valid_ask",
                "book_stale", "book_stale_prev", "gap_prev_s",
                N.px("bid", 1), N.px("ask", 20)]:
        assert col in al.columns
