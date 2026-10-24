from datetime import UTC, datetime
import polars as pl

from order_flow_imbalance_strategy.event_alignment import build_alignment_engine
from order_flow_imbalance_strategy.ofi import compute_ofi

T0 = int(datetime(2023, 1, 1, 9, 0, 0, tzinfo=UTC).timestamp() * 1000)
DAY = datetime(2023, 1, 1).date()
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
