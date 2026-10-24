from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from order_flow_imbalance_strategy.event_alignment import build_alignment_engine
from order_flow_imbalance_strategy.ofi import compute_ofi
from order_flow_imbalance_strategy.spoofing import (
    SpoofConfig,
    side_scores,
    trailing_stats,
    unexplained_removal,
)

T0 = int(datetime(2023, 1, 1, 9, 0, 0, tzinfo=UTC).timestamp() * 1000)
DAY = datetime(2023, 1, 1).date()
BASE_DT = datetime(2023, 1, 1, 9, 0, 0)
_TS = pl.col("timestamp").cast(pl.Datetime("ms"))

# Columns the spoofing filter will need to read. No trade_feed_missing: the
# real pipeline only distinguishes "no trades" (no_trades), not "no trades
# because the feed was down" vs "no trades because none happened".
SPOOFING_INPUT_COLUMNS = (
    "timestamp",
    "bid_price",
    "ask_price",
    "bid_qty",
    "ask_qty",
    "ofi_1s",
    "buy_qty",
    "sell_qty",
    "book_stale",
    "book_dead",
    "no_trades",
)


def _aligned_ofi(trades: pl.LazyFrame) -> pl.DataFrame:
    """Run the real align -> OFI path over a 10-second book with a native update
    every second (09:00:00 .. 09:00:09)."""
    n = 10
    book = pl.LazyFrame(
        {
            "timestamp": [T0 + i * 1000 for i in range(n)],
            "bid_price": [100.0 + i * 0.1 for i in range(n)],
            "ask_price": [100.5 + i * 0.1 for i in range(n)],
            "bid_qty": [10.0 + i for i in range(n)],
            "ask_qty": [20.0 + i for i in range(n)],
            "mid_price": [100.25 + i * 0.1 for i in range(n)],
            "spread": [0.5] * n,
            "queue_imbalance": [0.0] * n,
        }
    ).with_columns(_TS)
    klines = pl.LazyFrame({"timestamp": [T0], "close": [100.0]}).with_columns(_TS)
    aligned = build_alignment_engine(book, trades, klines, day=DAY).collect()
    return compute_ofi(aligned)


def make_frame(rows: list[dict]) -> pl.DataFrame:
    """One row per second from BASE_DT (or at ``sec`` if given). Defaults are a
    fresh, live book at 100.0 / 100.5 with no trades, so each test only spells
    out the fields it is about."""
    out = []
    for i, r in enumerate(rows):
        out.append(
            {
                "timestamp": BASE_DT + timedelta(seconds=r.get("sec", i)),
                "bid_price": r.get("bid_price", 100.0),
                "ask_price": r.get("ask_price", 100.5),
                "bid_qty": r["bid_qty"],
                "ask_qty": r["ask_qty"],
                "buy_qty": r.get("buy_qty", 0.0),
                "sell_qty": r.get("sell_qty", 0.0),
                "book_stale": r.get("book_stale", False),
                "book_dead": r.get("book_dead", False),
            }
        )
    return pl.DataFrame(out)


def score_frame(
    removal: list[float | None],
    median: list[float | None],
    std: list[float | None],
) -> pl.DataFrame:
    """side_scores input with identical bid and ask columns, so either side can be
    asserted on."""
    cols = {"removal": removal, "prior_median": median, "prior_std": std}
    return pl.DataFrame(
        {f"{side}_{k}": v for side in ("bid", "ask") for k, v in cols.items()},
        schema_overrides={f"{side}_{k}": pl.Float64 for side in ("bid", "ask") for k in cols},
    )


def removal_frame(
    bid: list[float | None],
    ask: list[float | None] | None = None,
    secs: list[int] | None = None,
    book_dead: list[bool] | None = None,
) -> pl.DataFrame:
    """Minimal trailing_stats input: removals per second from BASE_DT. ``ask``
    defaults to a copy of ``bid``; ``secs`` lets a test punch gaps in the grid."""
    n = len(bid)
    return pl.DataFrame(
        {
            "timestamp": [BASE_DT + timedelta(seconds=s) for s in (secs or range(n))],
            "book_dead": book_dead or [False] * n,
            "bid_removal": bid,
            "ask_removal": ask if ask is not None else list(bid),
        },
        schema_overrides={"bid_removal": pl.Float64, "ask_removal": pl.Float64},
    )


def test_aligned_ofi_output_has_the_columns_spoofing_needs():
    # A sell print at +5s should be the only thing that ever "explains away"
    # a bid-side depletion for this heuristic; a buy print must not.
    trades = pl.LazyFrame(
        {
            "timestamp": [T0 + 2_000, T0 + 5_000],
            "buy_qty": [1.0, 0.0],
            "sell_qty": [0.0, 2.0],
            "notional": [100.0, 200.0],
        }
    ).with_columns(_TS)

    out = _aligned_ofi(trades)

    missing = [c for c in SPOOFING_INPUT_COLUMNS if c not in out.columns]
    assert not missing, f"align-v2 output no longer has: {missing}"

    # Spot-check the actual values a sell print and a buy print leave behind,
    # since the spoofing filter nets bid depletion only against sell_qty.
    row5 = out.filter(pl.col("timestamp") == datetime(2023, 1, 1, 9, 0, 5)).row(0, named=True)
    assert row5["sell_qty"] == 2.0
    assert row5["buy_qty"] == 0.0
    assert row5["no_trades"] is False


def test_multiple_prints_in_one_second_are_summed_not_last_print():
    """Two sells and a buy inside 09:00:03 must sum. Keeping only the final
    print would leave sell_qty == 0.0 and make 3.5 of real selling look like
    unexplained bid depletion."""
    trades = pl.LazyFrame(
        {
            "timestamp": [T0 + 3_100, T0 + 3_400, T0 + 3_900],
            "buy_qty": [0.0, 0.0, 0.5],
            "sell_qty": [1.0, 2.5, 0.0],
            "notional": [100.0, 250.0, 50.0],
        }
    ).with_columns(_TS)

    out = _aligned_ofi(trades)

    row3 = out.filter(pl.col("timestamp") == datetime(2023, 1, 1, 9, 0, 3)).row(0, named=True)
    assert row3["sell_qty"] == 3.5
    assert row3["buy_qty"] == 0.5
    assert row3["trade_count"] == 3


# Unexplained removal: for an unchanged best price,
#   bid: max(0, prev_bid_qty - bid_qty - sell_qty)   (sells hit the bid)
#   ask: max(0, prev_ask_qty - ask_qty - buy_qty)    (buys lift the ask)


def test_same_price_bid_depletion_is_net_of_sells():
    # 10 -> 4 with 2 sold: 10 - 4 - 2 = 4 unexplained
    df = unexplained_removal(
        make_frame(
            [
                {"bid_qty": 10.0, "ask_qty": 8.0},
                {"bid_qty": 4.0, "ask_qty": 8.0, "sell_qty": 2.0},
            ]
        )
    )
    assert df["bid_removal"][1] == pytest.approx(4.0)
    assert df["ask_removal"][1] == pytest.approx(0.0)


def test_fully_explained_ask_removal_is_zero():
    # Hand-check: ask 8 -> 3 and exactly 5 bought, so 8 - 3 - 5 = 0. Every unit
    # that left the queue traded; nothing is left to call a cancellation.
    df = unexplained_removal(
        make_frame(
            [
                {"bid_qty": 4.0, "ask_qty": 8.0},
                {"bid_qty": 4.0, "ask_qty": 3.0, "buy_qty": 5.0},
            ]
        )
    )
    assert df["ask_removal"][1] == 0.0
    assert df["ask_identifiable"][1] is True


def test_opposing_trades_do_not_explain_depletion():
    # Buys can't fill resting bids and sells can't fill resting asks, so each side
    # nets only against its own aggressor.
    df = unexplained_removal(
        make_frame(
            [
                {"bid_qty": 4.0, "ask_qty": 5.0},
                {"bid_qty": 1.0, "ask_qty": 5.0, "buy_qty": 3.0},  # bid 4 - 1 - 0 = 3
                {"bid_qty": 1.0, "ask_qty": 2.0, "sell_qty": 3.0},  # ask 5 - 2 - 0 = 3
            ]
        )
    )
    assert df["bid_removal"][1] == pytest.approx(3.0)
    assert df["ask_removal"][2] == pytest.approx(3.0)


def test_queue_growth_and_overexplained_drops_floor_at_zero():
    # bid grows 4 -> 6: nothing removed. ask 5 -> 4 but 3 bought: 5 - 4 - 3 = -2 -> 0.
    df = unexplained_removal(
        make_frame(
            [
                {"bid_qty": 4.0, "ask_qty": 5.0},
                {"bid_qty": 6.0, "ask_qty": 4.0, "buy_qty": 3.0},
            ]
        )
    )
    assert df["bid_removal"][1] == 0.0
    assert df["ask_removal"][1] == 0.0


def test_price_change_is_unidentifiable_for_that_side_only():
    # The bid moved, so its old queue is out of view; the ask stayed put and is
    # still scored: 3 - 1 - 0 = 2.
    df = unexplained_removal(
        make_frame(
            [
                {"bid_qty": 4.0, "ask_qty": 3.0},
                {"bid_price": 99.9, "bid_qty": 6.0, "ask_qty": 1.0},
            ]
        )
    )
    assert df["bid_removal"][1] is None
    assert df["bid_identifiable"][1] is False
    assert df["ask_removal"][1] == pytest.approx(2.0)


@pytest.mark.parametrize("flag", ["book_stale", "book_dead"])
@pytest.mark.parametrize("row", [0, 1])
def test_stale_or_dead_book_on_either_row_is_unidentifiable(flag, row):
    # A forward-filled or dead quote on t-1 OR t means the diff isn't between two
    # real observations; that's unknown, not "nothing removed".
    rows = [{"bid_qty": 10.0, "ask_qty": 8.0}, {"bid_qty": 4.0, "ask_qty": 2.0}]
    rows[row][flag] = True
    df = unexplained_removal(make_frame(rows))
    assert df["bid_removal"][1] is None
    assert df["ask_removal"][1] is None
    assert df["bid_identifiable"][1] is False


def test_gap_in_grid_is_unidentifiable():
    # 09:00:00 -> 09:00:02: a missing second means the removal spans two intervals.
    df = unexplained_removal(
        make_frame(
            [
                {"sec": 0, "bid_qty": 10.0, "ask_qty": 8.0},
                {"sec": 2, "bid_qty": 4.0, "ask_qty": 2.0},
            ]
        )
    )
    assert df["bid_removal"][1] is None
    assert df["ask_removal"][1] is None


def test_missing_trade_context_is_unidentifiable_not_zero():
    # Unknown sells can't be netted out of the bid; treating them as 0 would call
    # all 6 units cancelled. The ask nets against buys, which are known: 8 - 2 - 0.
    df = unexplained_removal(
        make_frame(
            [
                {"bid_qty": 10.0, "ask_qty": 8.0},
                {"bid_qty": 4.0, "ask_qty": 2.0, "sell_qty": None},
            ]
        )
    )
    assert df["bid_removal"][1] is None
    assert df["bid_identifiable"][1] is False
    assert df["ask_removal"][1] == pytest.approx(6.0)


def test_first_row_has_no_previous_observation():
    df = unexplained_removal(make_frame([{"bid_qty": 10.0, "ask_qty": 8.0}]))
    assert df["bid_removal"][0] is None
    assert df["ask_removal"][0] is None


def test_source_columns_pass_through_unchanged():
    src = make_frame(
        [
            {"bid_qty": 10.0, "ask_qty": 8.0},
            {"bid_qty": 4.0, "ask_qty": 8.0, "sell_qty": 2.0},
        ]
    ).with_columns(pl.Series("ofi_1s", [None, -6.0]))
    df = unexplained_removal(src)
    assert df.select(src.columns).equals(src)
    assert df.columns == [
        *src.columns,
        "bid_removal",
        "ask_removal",
        "bid_identifiable",
        "ask_identifiable",
    ]


def test_missing_required_column_raises():
    with pytest.raises(ValueError, match="sell_qty"):
        unexplained_removal(make_frame([{"bid_qty": 1.0, "ask_qty": 1.0}]).drop("sell_qty"))


# Trailing stats: median / sample std of the previous ``window_s`` comparable
# removals, never the current one. window_s=4 keeps the arithmetic hand-checkable.
SMALL = SpoofConfig(window_s=4, min_obs=2)


def test_prior_stats_exclude_the_current_interval():
    # Row 6's window is rows 2..5 = [3, null, 2, 8] -> [3, 2, 8]: median 3,
    # mean 13/3, squared deviations sum to 20.667, / (3 - 1) -> std sqrt(10.333).
    # The 12.0 at row 6 itself must not appear anywhere in its own stats.
    df = trailing_stats(removal_frame([None, 1.0, 3.0, None, 2.0, 8.0, 12.0]), SMALL)
    assert df["bid_prior_n"][6] == 3
    assert df["bid_prior_median"][6] == pytest.approx(3.0)
    assert df["bid_prior_std"][6] == pytest.approx(3.214550, abs=1e-6)
    # Row 4: window [null, 1, 3, null] -> median 2, std sqrt(2)
    assert df["bid_prior_median"][4] == pytest.approx(2.0)
    assert df["bid_prior_std"][4] == pytest.approx(2**0.5)


def test_insufficient_history_until_the_window_is_full():
    # Rows 0..3 have fewer than window_s prior seconds in the segment.
    df = trailing_stats(removal_frame([None, 1.0, 3.0, None, 2.0, 8.0, 12.0]), SMALL)
    assert df["warm_up"].to_list() == [True] * 4 + [False] * 3
    assert df["bid_prior_median"][:4].null_count() == 4
    assert df["bid_prior_std"][:4].null_count() == 4


def test_fewer_than_min_obs_comparable_removals_is_null():
    # Warm, but the window [null, null, null, 5] holds only one comparable value,
    # and a sample std needs two.
    df = trailing_stats(removal_frame([None, None, None, 5.0, 1.0]), SMALL)
    assert df["warm_up"][4] is False
    assert df["bid_prior_n"][4] == 1
    assert df["bid_prior_median"][4] is None
    assert df["bid_prior_std"][4] is None


def test_sides_use_their_own_history():
    df = trailing_stats(
        removal_frame([0.0, 0.0, 0.0, 0.0, 9.0], ask=[1.0, 2.0, 3.0, 4.0, 0.0]), SMALL
    )
    assert df["bid_prior_median"][4] == pytest.approx(0.0)
    assert df["ask_prior_median"][4] == pytest.approx(2.5)


def test_no_variance_history_gives_zero_std():
    # A flat history is a real (if awkward) case: std is 0.0, not null. Scoring
    # handles it via the +1e-8 in the z denominator.
    df = trailing_stats(removal_frame([0.0, 0.0, 0.0, 0.0, 0.5]), SMALL)
    assert df["bid_prior_median"][4] == 0.0
    assert df["bid_prior_std"][4] == 0.0


def test_gap_in_grid_resets_the_warm_up():
    # 09:00:00..04, then a missing second, then 09:00:06..10. The second stretch
    # has to re-warm from scratch: its history must not bridge the hole.
    secs = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    df = trailing_stats(removal_frame([1.0] * 10, secs=secs), SMALL)
    assert df["warm_up"].to_list() == [True] * 4 + [False] + [True] * 4 + [False]
    assert df["bid_prior_n"][5] == 0  # first row after the gap sees no history


def test_daily_boundary_resets_the_warm_up():
    # 23:59:58 -> 00:00:00 is a 1-second step on the grid but a new trading day.
    start = datetime(2023, 1, 1, 23, 59, 55)
    base_offset = int((start - BASE_DT).total_seconds())
    df = trailing_stats(removal_frame([1.0] * 10, secs=[base_offset + i for i in range(10)]), SMALL)
    # 23:59:55..59 is one segment (5 rows), 00:00:00..04 starts another.
    assert df["warm_up"].to_list() == [True] * 4 + [False] + [True] * 4 + [False]


def test_dead_book_resets_the_warm_up():
    # A dead feed is a gap in the book even though the grid row exists.
    dead = [False] * 5 + [True] + [False] * 5
    df = trailing_stats(removal_frame([1.0] * 11, book_dead=dead), SMALL)
    assert df["warm_up"][4] is False
    assert df["warm_up"][5] is True
    assert df["warm_up"][6:10].to_list() == [True] * 4
    assert df["warm_up"][10] is False


def test_trailing_stats_do_not_depend_on_future_rows():
    # Changing everything after the cut must leave every stat up to the cut as is.
    base = [float(i % 7) for i in range(40)]
    cut = 25
    future_changed = base[: cut + 1] + [100.0 * v for v in base[cut + 1 :]]
    a = trailing_stats(removal_frame(base), SMALL).head(cut + 1)
    b = trailing_stats(removal_frame(future_changed), SMALL).head(cut + 1)
    assert a.equals(b)


def test_trailing_stats_preserve_source_columns():
    src = removal_frame([1.0, 2.0, 3.0, 4.0, 5.0])
    df = trailing_stats(src, SMALL)
    assert df.select(src.columns).equals(src)
    assert df.columns == [
        *src.columns,
        "warm_up",
        "bid_prior_n",
        "bid_prior_median",
        "bid_prior_std",
        "ask_prior_n",
        "ask_prior_median",
        "ask_prior_std",
    ]


def test_spoof_config_is_frozen():
    with pytest.raises(FrozenInstanceError):
        SMALL.window_s = 10  # type: ignore[misc]


# Scoring: z = (removal - prior_median) / (prior_std + eps), then a linear ramp
# score = clip((z - z_floor) / (z_cap - z_floor), 0, 1)  -- 2 and 6 by default.


def test_score_matches_hand_computed_values():
    # z = 0 -> 0; z = (8 - 2) / 1 = 6 -> 1; z = (12 - 3) / 3.21455 = 2.7998 -> 0.19994
    df = side_scores(
        score_frame([2.0, 8.0, 12.0], [2.0, 2.0, 3.0], [2**0.5, 1.0, 3.214550]),
        SpoofConfig(),
    )
    assert df["bid_score"].to_list() == pytest.approx([0.0, 1.0, 0.199942], abs=1e-6)
    assert df["bid_z"][2] == pytest.approx(2.79977, abs=1e-5)


def test_score_ramp_is_clipped_to_zero_and_one():
    # median 0, std 1, so z == removal: below the floor, mid-ramp, past the cap.
    df = side_scores(score_frame([1.0, 4.0, 50.0], [0.0] * 3, [1.0] * 3), SpoofConfig())
    assert df["bid_score"].to_list() == pytest.approx([0.0, 0.5, 1.0])


def test_no_variance_history_scores_any_excess_as_maximal():
    # std 0: the +1e-8 keeps z finite. An unchanged removal is 0 suspicion; any
    # excess at all is z ~ 5e7 -> score 1.
    df = side_scores(score_frame([0.0, 0.5], [0.0, 0.0], [0.0, 0.0]), SpoofConfig())
    assert df["bid_score"].to_list() == [0.0, 1.0]


def test_unknown_removal_or_baseline_scores_null():
    df = side_scores(
        score_frame([None, 5.0, 5.0], [1.0, None, 1.0], [1.0, 1.0, None]), SpoofConfig()
    )
    assert df["bid_score"].null_count() == 3
    assert df["bid_z"].null_count() == 3


def test_thresholds_are_configurable():
    # z = 2: on the default floor (score 0), but halfway up a 1..3 ramp.
    frame = score_frame([2.0], [0.0], [1.0])
    assert side_scores(frame, SpoofConfig())["bid_score"][0] == pytest.approx(0.0)
    narrow = SpoofConfig(z_floor=1.0, z_cap=3.0)
    assert side_scores(frame, narrow)["bid_score"][0] == pytest.approx(0.5)


def test_side_scores_preserve_source_columns():
    src = score_frame([1.0], [0.0], [1.0])
    df = side_scores(src, SpoofConfig())
    assert df.select(src.columns).equals(src)
    assert df.columns == [*src.columns, "bid_z", "bid_score", "ask_z", "ask_score"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_s": 0},
        {"min_obs": 1},  # a sample std needs two observations
        {"z_floor": 6.0, "z_cap": 6.0},
        {"z_floor": 6.0, "z_cap": 2.0},
        {"eps": 0.0},
    ],
)
def test_spoof_config_rejects_impossible_settings(kwargs):
    with pytest.raises(ValueError):
        SpoofConfig(**kwargs)


def test_fingerprint_identifies_the_exact_settings():
    # Same settings -> same id; any change (version included) -> a different one.
    assert SpoofConfig().fingerprint() == SpoofConfig().fingerprint()
    others = [
        SpoofConfig(window_s=1800),
        SpoofConfig(z_floor=2.5),
        SpoofConfig(version="l1-depletion-v2"),
    ]
    ids = {c.fingerprint() for c in [SpoofConfig(), *others]}
    assert len(ids) == 4
