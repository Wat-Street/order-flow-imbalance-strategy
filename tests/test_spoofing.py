from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from order_flow_imbalance_strategy.event_alignment import build_alignment_engine
from order_flow_imbalance_strategy.ofi import compute_ofi
from order_flow_imbalance_strategy.spoofing import unexplained_removal

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
