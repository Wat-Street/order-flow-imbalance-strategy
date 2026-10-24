import json
import math
import random
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from order_flow_imbalance_strategy import spoofing as SP
from order_flow_imbalance_strategy.event_alignment import build_alignment_engine
from order_flow_imbalance_strategy.ofi import compute_ofi
from order_flow_imbalance_strategy.spoofing import (
    DEFAULT_CONFIG,
    NO_FILTER_CONFIG,
    VARIANTS,
    SpoofConfig,
    add_variants,
    bootstrap_mean_ci,
    build_tasks,
    clean_ofi,
    daily_ic,
    diagnostics,
    forward_return,
    main,
    process_file,
    rescore,
    run_ablation,
    score_spoofing,
    side_scores,
    spoof_path,
    trailing_stats,
    unexplained_removal,
)

T0 = int(datetime(2023, 1, 1, 9, 0, 0, tzinfo=UTC).timestamp() * 1000)
DAY = datetime(2023, 1, 1).date()
BASE_DT = datetime(2023, 1, 1, 9, 0, 0)
_TS = pl.col("timestamp").cast(pl.Datetime("ms"))
ABLATION_CFG = SpoofConfig(window_s=30)
EMPTY_TRADES = {
    "timestamp": pl.Datetime("ms"),
    "buy_qty": pl.Float64,
    "sell_qty": pl.Float64,
    "notional": pl.Float64,
}

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


# Run the real align -> OFI path over a 10-second book with a native update
# every second (09:00:00 .. 09:00:09).
def _aligned_ofi(trades: pl.LazyFrame) -> pl.DataFrame:
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


# One row per second from BASE_DT (or at `sec` if given). Defaults are a
# fresh, live book at 100.0 / 100.5 with no trades, so each test only spells
# out the fields it is about.
def make_frame(rows: list[dict]) -> pl.DataFrame:
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


# diagnostics input: valid OFI everywhere, unknown wherever the score is null.
# A synthetic aligned day: a live book that sometimes ticks, with random queues and
# flow, so every stage has real work to do. Seeded so it's reproducible.
def synthetic_day(n: int, seed: int, start: datetime = BASE_DT) -> pl.DataFrame:
    rng = random.Random(seed)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(seconds=i) for i in range(n)],
            "bid_price": [100.0 + 0.1 * rng.randint(0, 1) for _ in range(n)],
            "ask_price": [100.5 + 0.1 * rng.randint(0, 1) for _ in range(n)],
            "bid_qty": [rng.expovariate(1 / 5) for _ in range(n)],
            "ask_qty": [rng.expovariate(1 / 5) for _ in range(n)],
            "buy_qty": [rng.expovariate(1.0) for _ in range(n)],
            "sell_qty": [rng.expovariate(1.0) for _ in range(n)],
            "book_stale": [rng.random() < 0.05 for _ in range(n)],
            "book_dead": [False] * n,
        }
    )


# A quote frame from bid prices (ask = bid + 0.5) at the given seconds.
def quotes(bids: list[float], secs: list[int] | None = None, start: datetime = BASE_DT, **flags):
    n = len(bids)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(seconds=s) for s in (secs or range(n))],
            "bid_price": bids,
            "ask_price": flags.get("asks", [b + 0.5 for b in bids]),
            "book_stale": flags.get("book_stale", [False] * n),
            "book_dead": flags.get("book_dead", [False] * n),
        }
    )


# Scored days as the runner would write them, one per calendar day.
def scored_days(days: list[int], seed0: int, n: int = 300) -> pl.DataFrame:
    frames = []
    for i, d in enumerate(days):
        raw = synthetic_day(n, seed=seed0 + i, start=datetime(2023, 1, d, 9))
        frames.append(score_spoofing(compute_ofi(raw), ABLATION_CFG))
    return pl.concat(frames)


def scored_frame(scores: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ofi_1s": [1.0] * len(scores),
            "spoof_score": scores,
            "spoof_unknown": [s is None for s in scores],
            "spoof_config_version": ["l1-depletion-v1"] * len(scores),
        },
        schema_overrides={"spoof_score": pl.Float64},
    )


def combine_frame(
    ofi: list[float | None], bid: list[float | None], ask: list[float | None]
) -> pl.DataFrame:
    return pl.DataFrame(
        {"ofi_1s": ofi, "bid_score": bid, "ask_score": ask},
        schema_overrides={"ofi_1s": pl.Float64, "bid_score": pl.Float64, "ask_score": pl.Float64},
    )


# side_scores input with identical bid and ask columns, so either side can be
# asserted on.
def score_frame(
    removal: list[float | None],
    median: list[float | None],
    std: list[float | None],
) -> pl.DataFrame:
    cols = {"removal": removal, "prior_median": median, "prior_std": std}
    return pl.DataFrame(
        {f"{side}_{k}": v for side in ("bid", "ask") for k, v in cols.items()},
        schema_overrides={f"{side}_{k}": pl.Float64 for side in ("bid", "ask") for k in cols},
    )


# Minimal trailing_stats input: removals per second from BASE_DT. `ask`
# defaults to a copy of `bid`; `secs` lets a test punch gaps in the grid.
def removal_frame(
    bid: list[float | None],
    ask: list[float | None] | None = None,
    secs: list[int] | None = None,
    book_dead: list[bool] | None = None,
) -> pl.DataFrame:
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
    # Two sells and a buy inside 09:00:03 must sum. Keeping only the final
    # print would leave sell_qty == 0.0 and make 3.5 of real selling look like
    # unexplained bid depletion.
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


# Trailing stats: median / sample std of the previous `window_s` comparable
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


# Combining: spoof_score = max(bid_score, ask_score) only when BOTH are known.
# ofi_clean = ofi_1s * (1 - spoof_score); an unknown score passes valid OFI
# through flagged spoof_unknown; invalid (null) OFI stays null.


def test_known_score_down_weights_ofi_by_the_worse_side():
    # max(0.2, 0.5) = 0.5 -> 10 * 0.5 = 5; both 0 -> untouched; score 1 -> fully muted.
    df = clean_ofi(combine_frame([10.0, -4.0, 3.0], [0.2, 0.0, 1.0], [0.5, 0.0, 0.3]))
    assert df["spoof_score"].to_list() == pytest.approx([0.5, 0.0, 1.0])
    assert df["ofi_clean"].to_list() == pytest.approx([5.0, -4.0, 0.0])
    assert df["spoof_unknown"].to_list() == [False, False, False]


def test_one_unknown_side_makes_the_combined_score_unknown():
    # A 1.0 bid score doesn't let us call the second scored: the ask is unknown.
    # Valid OFI passes through untouched and is flagged rather than guessed at.
    df = clean_ofi(combine_frame([6.0, 8.0], [1.0, None], [None, None]))
    assert df["spoof_score"].null_count() == 2
    assert df["ofi_clean"].to_list() == [6.0, 8.0]
    assert df["spoof_unknown"].to_list() == [True, True]


def test_invalid_ofi_stays_null_even_with_a_known_score():
    df = clean_ofi(combine_frame([None, None], [0.3, None], [0.1, None]))
    assert df["ofi_clean"].null_count() == 2
    assert df["spoof_score"][0] == pytest.approx(0.3)
    # Nothing to be uncertain about when there's no OFI to pass through.
    assert df["spoof_unknown"].to_list() == [False, False]


def test_raw_ofi_column_is_left_untouched():
    src = combine_frame([10.0, -4.0, 6.0, None], [0.2, 0.0, 1.0, 0.3], [0.5, 0.0, None, 0.1])
    df = clean_ofi(src)
    assert df.select(src.columns).equals(src)
    assert df.columns == [*src.columns, "spoof_score", "ofi_clean", "spoof_unknown"]


# Pipeline: score_spoofing runs removal -> stats -> scores -> clean and stamps the
# config; NO_FILTER_CONFIG is the "filter off" arm of the ablation.


def test_score_spoofing_runs_on_real_aligned_ofi_and_stamps_the_config():
    trades = pl.LazyFrame(
        {"timestamp": [T0 + 3_000], "buy_qty": [0.0], "sell_qty": [1.0], "notional": [100.0]}
    ).with_columns(_TS)
    src = _aligned_ofi(trades)
    cfg = SpoofConfig(window_s=3)

    df = score_spoofing(src, cfg)

    assert df.select(src.columns).equals(src)  # nothing upstream is rewritten
    for c in ("bid_removal", "warm_up", "bid_score", "spoof_score", "ofi_clean"):
        assert c in df.columns
    assert df["spoof_config_version"].unique().to_list() == [cfg.version]
    assert df["spoof_config_id"].unique().to_list() == [cfg.fingerprint()]


def test_invalid_ofi_stays_null_through_the_whole_pipeline():
    # The real align -> OFI frame nulls OFI everywhere outside 09:00:01..09, so
    # nearly the whole day is invalid OFI that must stay invalid after cleaning.
    df = score_spoofing(_aligned_ofi(pl.LazyFrame(schema=EMPTY_TRADES)), SpoofConfig(window_s=3))
    invalid = df.filter(pl.col("ofi_1s").is_null())
    assert invalid.height > 0
    assert invalid["ofi_clean"].null_count() == invalid.height
    assert not invalid["spoof_unknown"].any()


def test_no_filter_baseline_scores_zero_only_where_ofi_is_valid():
    src = pl.DataFrame(
        {
            "timestamp": [BASE_DT + timedelta(seconds=i) for i in range(3)],
            "ofi_1s": [1.5, None, -2.0],
        }
    )
    df = score_spoofing(src, NO_FILTER_CONFIG)
    assert df["spoof_score"].to_list() == [0.0, None, 0.0]
    assert df["ofi_clean"].to_list() == [1.5, None, -2.0]
    assert df["spoof_unknown"].to_list() == [False, False, False]
    assert df["spoof_config_version"][0] == "no-filter-v1"


def test_no_filter_baseline_is_a_distinct_configuration():
    assert NO_FILTER_CONFIG.version != DEFAULT_CONFIG.version
    assert NO_FILTER_CONFIG.fingerprint() != DEFAULT_CONFIG.fingerprint()


def test_diagnostics_warns_when_over_ten_percent_of_scored_rows_exceed_half(caplog):
    # 8 scored rows (2 unknown), 2 above 0.5 -> 25% > 10%. Unknown rows are not
    # in the denominator: they aren't evidence either way.
    SP.logger.disabled = False  # same ingest_data dictConfig issue as test_event_alignment
    with caplog.at_level("WARNING", logger=SP.logger.name):
        out = diagnostics(scored_frame([0.0, 0.9, None, 0.1, 0.6, 0.0, 0.0, 0.2, None, 0.0]))
    assert out["n_scored"] == 8
    assert out["n_unknown"] == 2
    assert out["n_high"] == 2
    assert out["frac_high"] == pytest.approx(0.25)
    assert out["warned"] is True
    assert "25.0%" in caplog.text


def test_diagnostics_is_quiet_at_exactly_ten_percent(caplog):
    SP.logger.disabled = False
    with caplog.at_level("WARNING", logger=SP.logger.name):
        out = diagnostics(scored_frame([0.9] + [0.0] * 9))
    assert out["frac_high"] == pytest.approx(0.10)
    assert out["warned"] is False
    assert caplog.text == ""


def test_diagnostics_with_nothing_scored_has_no_fraction():
    out = diagnostics(scored_frame([None, None]))
    assert out["n_scored"] == 0
    assert out["frac_high"] is None
    assert out["warned"] is False


# Whole pipeline: future rows can't change earlier scores, and the per-day runner.


def test_scores_do_not_depend_on_future_rows():
    # Empty both queues and stop all trading after the cut, which turns the next
    # identifiable second into a large removal. Nothing up to the cut may change.
    cfg = SpoofConfig(window_s=30)
    base = synthetic_day(200, seed=0)
    for cut in range(60, 180, 10):
        later = pl.int_range(pl.len()) > cut
        perturbed = base.with_columns(
            [
                pl.when(later).then(0.0).otherwise(pl.col(c)).alias(c)
                for c in ("bid_qty", "ask_qty", "buy_qty", "sell_qty")
            ]
        )
        a = score_spoofing(compute_ofi(base), cfg).head(cut + 1)
        b = score_spoofing(compute_ofi(perturbed), cfg).head(cut + 1)
        assert a["spoof_score"].drop_nulls().len() > 0  # something was actually scored
        assert a.equals(b), f"rows up to {cut} changed"


def test_process_file_writes_scored_day_and_is_idempotent(tmp_path):
    src = compute_ofi(synthetic_day(50, seed=1))
    in_path = tmp_path / "BTCUSDT-ofi-A-2023-01-01.parquet"
    out_path = tmp_path / "BTCUSDT-ofi-spoof-x-2023-01-01.parquet"
    src.write_parquet(in_path)
    task = {
        "symbol": "BTCUSDT",
        "date": "2023-01-01",
        "in_path": str(in_path),
        "out_path": str(out_path),
        "cfg": SpoofConfig(window_s=10),
    }

    r1 = process_file(task)
    assert r1["status"] == "ok"
    assert r1["n_rows"] == 50
    assert "frac_high" in r1  # diagnostics ride along in the status
    out = pl.read_parquet(out_path)
    assert out["ofi_1s"].equals(src["ofi_1s"])  # raw OFI recoverable from the file
    assert process_file(task)["status"] == "skipped"


def test_process_file_missing_input_returns_no_input(tmp_path):
    task = {
        "symbol": "BTCUSDT",
        "date": "2023-01-01",
        "in_path": str(tmp_path / "nope.parquet"),
        "out_path": str(tmp_path / "out.parquet"),
        "cfg": DEFAULT_CONFIG,
    }
    assert process_file(task)["status"] == "no_input"


def test_build_tasks_reads_ofi_files_and_names_output_by_config(tmp_path):
    # Only days with an OFI file become tasks; the version in the output name keeps
    # filtered and baseline runs from overwriting each other.
    signals = tmp_path / "signals" / "BTCUSDT"
    signals.mkdir(parents=True)
    (signals / "BTCUSDT-ofi-A-2023-01-01.parquet").write_bytes(b"")
    tasks = build_tasks(["BTCUSDT"], date(2023, 1, 1), date(2023, 1, 2), tmp_path, DEFAULT_CONFIG)
    assert len(tasks) == 1
    assert tasks[0]["out_path"].endswith("BTCUSDT-ofi-spoof-l1-depletion-v1-2023-01-01.parquet")
    baseline = build_tasks(
        ["BTCUSDT"], date(2023, 1, 1), date(2023, 1, 1), tmp_path, NO_FILTER_CONFIG
    )
    assert baseline[0]["out_path"].endswith("BTCUSDT-ofi-spoof-no-filter-v1-2023-01-01.parquet")


def test_main_scores_existing_days_and_baseline_flag_writes_separately(tmp_path):
    signals = tmp_path / "signals" / "BTCUSDT"
    signals.mkdir(parents=True)
    compute_ofi(synthetic_day(50, seed=2)).write_parquet(
        signals / "BTCUSDT-ofi-A-2023-01-01.parquet"
    )
    args = ["score", "--symbols", "BTCUSDT", "--start", "2023-01-01", "--end", "2023-01-01"]
    args += ["--data-dir", str(tmp_path), "--workers", "1"]

    assert main(args) == 0
    assert main([*args, "--baseline"]) == 0
    assert (signals / "BTCUSDT-ofi-spoof-l1-depletion-v1-2023-01-01.parquet").exists()
    assert (signals / "BTCUSDT-ofi-spoof-no-filter-v1-2023-01-01.parquet").exists()


def test_main_end_before_start_returns_2(tmp_path):
    rc = main(
        [
            "score",
            "--symbols",
            "BTCUSDT",
            "--start",
            "2023-01-02",
            "--end",
            "2023-01-01",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


def test_main_no_inputs_returns_0(tmp_path):
    rc = main(
        [
            "score",
            "--symbols",
            "BTCUSDT",
            "--start",
            "2023-01-01",
            "--end",
            "2023-01-01",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0


# Forward midpoint return: log(mid[t+h] / mid[t]), only between two valid quotes
# exactly h seconds apart on the same UTC day.


def test_forward_return_is_the_log_mid_change_h_seconds_ahead():
    # mids 100.25, 100.25, 101.25, 101.25, 102.25 with h = 2
    df = forward_return(quotes([100.0, 100.0, 101.0, 101.0, 102.0]), 2)
    up = math.log(101.25 / 100.25)
    assert df["fwd_ret_2s"].to_list()[:3] == pytest.approx([up, up, math.log(102.25 / 101.25)])
    assert df["fwd_ret_2s"][3:].null_count() == 2  # no t+2 inside the frame


def test_flat_moves_are_kept_as_zero_returns():
    df = forward_return(quotes([100.0, 100.0]), 1)
    assert df["fwd_ret_1s"][0] == 0.0


@pytest.mark.parametrize("flag", ["book_stale", "book_dead"])
def test_stale_or_dead_endpoint_has_no_label(flag):
    df = forward_return(quotes([100.0, 101.0, 102.0], **{flag: [False, True, False]}), 1)
    assert df["fwd_ret_1s"][0] is None  # ends on the bad quote
    assert df["fwd_ret_1s"][1] is None  # starts on it


def test_crossed_endpoint_has_no_label():
    df = forward_return(quotes([100.0, 101.0], asks=[100.5, 100.5]), 1)
    assert df["fwd_ret_1s"][0] is None


def test_gap_or_day_boundary_has_no_label():
    gap = forward_return(quotes([100.0, 101.0, 102.0], secs=[0, 1, 3]), 1)
    assert gap["fwd_ret_1s"][1] is None
    midnight = forward_return(quotes([100.0, 101.0], start=datetime(2023, 1, 1, 23, 59, 59)), 1)
    assert midnight["fwd_ret_1s"][0] is None


def test_rescore_with_the_original_thresholds_reproduces_the_scores():
    scored = score_spoofing(compute_ofi(synthetic_day(300, seed=0)), ABLATION_CFG)
    again = rescore(scored, ABLATION_CFG)
    assert again["spoof_score"].equals(scored["spoof_score"])
    narrow = rescore(scored, replace(ABLATION_CFG, z_floor=0.5, z_cap=1.0))
    assert not narrow["spoof_score"].equals(scored["spoof_score"])


def test_variants_share_raw_ofi_and_differ_only_in_cleaning():
    df = add_variants(
        pl.DataFrame(
            {
                "ofi_1s": [10.0, 10.0, 10.0],
                "spoof_score": [0.6, 0.5, 0.0],
                "ofi_clean": [4.0, 5.0, 10.0],
            }
        )
    )
    assert df["feat_raw"].to_list() == [10.0, 10.0, 10.0]
    assert df["feat_down_weighted"].to_list() == [4.0, 5.0, 10.0]
    # trade-excluded zeroes OFI strictly above 0.5 and keeps the row in the cohort
    assert df["feat_trade_excluded"].to_list() == [0.0, 10.0, 10.0]


def test_daily_ic_is_a_rank_correlation_per_day():
    day1 = [BASE_DT + timedelta(seconds=i) for i in range(4)]
    day2 = [datetime(2023, 1, 2, 9) + timedelta(seconds=i) for i in range(4)]
    df = pl.DataFrame(
        {
            "timestamp": day1 + day2,
            "x": [1.0, 2.0, 3.0, 4.0] * 2,
            "y": [1.0, 4.0, 9.0, 16.0] + [4.0, 3.0, 2.0, 1.0],
        }
    )
    out = daily_ic(df, "x", "y")
    assert out["day"].to_list() == [date(2023, 1, 1), date(2023, 1, 2)]
    assert out["ic"].to_list() == pytest.approx([1.0, -1.0])


def test_bootstrap_ci_resamples_days_reproducibly():
    diffs = [0.012, -0.004, 0.020, 0.001, 0.015, -0.008, 0.009, 0.011]
    mean, lo, hi = bootstrap_mean_ci(diffs, n_boot=10_000, seed=7)
    assert mean == pytest.approx(0.007)
    assert (lo, hi) == pytest.approx((0.0005, 0.01312), abs=1e-5)


# The ablation: thresholds picked on selection days only, then the locked config
# is evaluated once on later confirmation days, every variant on one cohort.


def test_ablation_locks_a_grid_config_using_selection_days_only():
    selection = scored_days([1, 2, 3], seed0=10)
    confirmation = scored_days([4, 5], seed0=20)
    report = run_ablation(selection, confirmation)

    grid = report["selection"]["grid"]
    assert report["selection"]["configs_tried"] == len(grid) == 9
    locked = report["selection"]["locked_config"]
    best = max(grid, key=lambda g: g["mean_diff"])
    assert (locked["z_floor"], locked["z_cap"]) == (best["z_floor"], best["z_cap"])
    assert report["selection"]["locked_id"] != DEFAULT_CONFIG.fingerprint()

    # wrecking the confirmation days can't move anything chosen on selection days
    wrecked = confirmation.with_columns(pl.col("bid_price") * 3, pl.col("ask_price") * 3)
    again = run_ablation(selection, wrecked)
    assert again["selection"] == report["selection"]


def test_every_variant_is_scored_on_the_same_cohort():
    selection = scored_days([1, 2], seed0=30)
    confirmation = scored_days([3, 4], seed0=40)
    report = run_ablation(selection, confirmation)

    for horizon in report["confirmation"]["by_horizon"].values():
        assert set(horizon["variants"]) == set(VARIANTS)
        assert horizon["n_rows"] > 0
        raw = horizon["variants"]["raw"]
        assert raw["diff_vs_raw"] == 0.0
        assert raw["ci95"] == [0.0, 0.0]


def test_selection_must_end_before_confirmation_starts():
    early = scored_days([1, 2], seed0=50)
    with pytest.raises(ValueError, match="before confirmation"):
        run_ablation(early, early)


def test_main_writes_a_report_from_runner_files(tmp_path):
    for i, d in enumerate([1, 2, 3, 4]):
        day = date(2023, 1, d)
        path = spoof_path(tmp_path, "BTCUSDT", day, DEFAULT_CONFIG)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = synthetic_day(300, seed=60 + i, start=datetime(2023, 1, d, 9))
        score_spoofing(compute_ofi(raw), ABLATION_CFG).write_parquet(path)

    args = ["ablate", "--symbol", "BTCUSDT", "--data-dir", str(tmp_path)]
    args += ["--select-start", "2023-01-01", "--select-end", "2023-01-02"]
    args += ["--confirm-start", "2023-01-03", "--confirm-end", "2023-01-04"]
    assert main(args) == 0

    report_path = tmp_path / "reports" / "spoofing-ablation-BTCUSDT-2023-01-03-2023-01-04.json"
    report = json.loads(report_path.read_text())
    assert report["horizon_s"] == 30
    assert report["selection"]["days"] == ["2023-01-01", "2023-01-02"]


def test_main_rejects_overlapping_periods(tmp_path):
    args = ["ablate", "--symbol", "BTCUSDT", "--data-dir", str(tmp_path)]
    args += ["--select-start", "2023-01-01", "--select-end", "2023-01-03"]
    args += ["--confirm-start", "2023-01-03", "--confirm-end", "2023-01-04"]
    assert main(args) == 2
