# L1 cancellation-suspicion filter layered on raw OFI.
#
# A best-level queue that shrinks while its price holds either traded or was
# pulled. Aggressive sells hit the bid and aggressive buys lift the ask, so the
# same-second opposing flow explains part of the shrink; what's left over is
# "unexplained removal" -- a cancellation proxy:
#   * bid: max(0, prev_bid_qty - bid_qty - sell_qty)
#   * ask: max(0, prev_ask_qty - ask_qty - buy_qty)
# This is only a proxy. An L1 decrease can also be a fill or a queue moving out
# of view, and nothing here sees order IDs, placement lifetimes or intent.
#
# Unidentifiable rules (removal is null -- never 0, which would claim "checked,
# nothing suspicious"):
#   * Row 0 has no ``t-1``.
#   * The gap to the previous row is not exactly 1 second (a missing grid second).
#   * The side's price changed, so the old queue is out of view.
#   * ``book_stale`` or ``book_dead`` on the current OR previous row: the diff is
#       not between two real observations.
#   * Either quantity, or the opposing flow, is null (no trade context to net).
#
# Known limitation: the aligned frame zero-fills flow on seconds without trades
# and has no flag for a missing trade feed, so a feed outage reads as "no trades"
# and its depletion as unexplained.

from __future__ import annotations

import polars as pl

#: Columns that must be present to compute unexplained removal.
REMOVAL_INPUT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "bid_price",
    "ask_price",
    "bid_qty",
    "ask_qty",
    "buy_qty",
    "sell_qty",
    "book_stale",
    "book_dead",
)

#: Columns appended by :func:`unexplained_removal`.
REMOVAL_OUTPUT_COLUMNS: tuple[str, ...] = (
    "bid_removal",
    "ask_removal",
    "bid_identifiable",
    "ask_identifiable",
)


def unexplained_removal(df: pl.DataFrame) -> pl.DataFrame:
    """Add per-side unexplained removal and identifiability flags.

    Returns a new frame with every original column preserved (same order, same
    values) plus ``REMOVAL_OUTPUT_COLUMNS``. Removal is null wherever the side is
    unidentifiable; see the module header for the rules.
    """
    missing = [c for c in REMOVAL_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input frame missing required columns: {missing}")
    conflicting = [c for c in REMOVAL_OUTPUT_COLUMNS if c in df.columns]
    if conflicting:
        raise ValueError(f"input frame already contains output columns: {conflicting}")

    original_columns = df.columns
    # shift(1) must mean "the previous second", whatever order the parquet was in.
    df = df.sort("timestamp")

    adjacent = (pl.col("timestamp").diff() == pl.duration(seconds=1)).fill_null(False)
    unusable = pl.col("book_stale") | pl.col("book_dead")
    # Row 0's t-1 doesn't exist, so it counts as unusable.
    clean_pair = adjacent & ~unusable & ~unusable.shift(1).fill_null(True)

    df = df.with_columns(
        _identifiable("bid_price", "bid_qty", "sell_qty", clean_pair).alias("bid_identifiable"),
        _identifiable("ask_price", "ask_qty", "buy_qty", clean_pair).alias("ask_identifiable"),
    )
    df = df.with_columns(
        pl.when(pl.col("bid_identifiable"))
        .then(_depletion("bid_qty", "sell_qty"))
        .alias("bid_removal"),
        pl.when(pl.col("ask_identifiable"))
        .then(_depletion("ask_qty", "buy_qty"))
        .alias("ask_removal"),
    )
    return df.select(*original_columns, *REMOVAL_OUTPUT_COLUMNS)


def _identifiable(price: str, qty: str, opposing_flow: str, clean_pair: pl.Expr) -> pl.Expr:
    """True where this side's t-1 -> t queue change can be attributed at all."""
    same_price = (pl.col(price) == pl.col(price).shift(1)).fill_null(False)
    known = (
        pl.col(qty).is_not_null()
        & pl.col(qty).shift(1).is_not_null()
        & pl.col(opposing_flow).is_not_null()
    )
    return clean_pair & same_price & known


def _depletion(qty: str, opposing_flow: str) -> pl.Expr:
    """max(0, prev_qty - qty - opposing_flow): the drop no trade accounts for."""
    return (pl.col(qty).shift(1) - pl.col(qty) - pl.col(opposing_flow)).clip(lower_bound=0.0)
