import polars as pl


def build_alignment_engine(
    book_lf: pl.LazyFrame, trades_lf: pl.LazyFrame, klines_lf: pl.LazyFrame
) -> pl.LazyFrame:

    # combines L1 book, filtered trades, and klines into a 1-second grid. Every
    # input carries the normalized ``timestamp`` column as Datetime("ms") (the
    # Stage-2 contract), so we operate on it directly rather than via from_epoch.
    # The output grid is emitted as ``timestamp`` (Datetime), which is exactly the
    # key ofi.py sorts on and diffs to detect gaps.

    # shift klines forward by 1 minute before forward-filling: a bar covering
    # 09:00:00 - 09:00:59 becomes valid exactly at 09:01:00 (no lookahead bias).
    klines = (
        klines_lf.with_columns(
            (pl.col("timestamp") + pl.duration(minutes=1)).dt.truncate("1s").alias("dt_1s")
        )
        .drop("timestamp")
        .sort("dt_1s")
    )

    # truncate book updates to 1s, keep the final state per second. The raw
    # per-update timestamp is kept only as ``exact_time`` (to pick the last update
    # in the second and to measure staleness) and is dropped before the grid join
    # so it never collides with the emitted ``timestamp`` grid key.
    book = (
        book_lf.with_columns(
            [
                pl.col("timestamp").alias("exact_time"),
                pl.col("timestamp").dt.truncate("1s").alias("dt_1s"),
            ]
        )
        .drop("timestamp")
        .group_by("dt_1s")
        .agg(pl.all().sort_by("exact_time").last())
        .sort("dt_1s")
    )

    # aggregate trades per second (volume, signed_volume, VWAP). The trade inputs
    # are the wash-filtered *cleaned-flow* columns from Stage-4: buy_qty / sell_qty
    # / notional carry the runner's renames of buy_qty_clean / sell_qty_clean /
    # notional_clean, so trades flagged as wash never reach the aligned volume,
    # VWAP, or signed volume. Volume and notional are derived from the cleaned
    # split rather than a raw price*quantity so all three agree on the same tape.
    trades = (
        trades_lf.with_columns(pl.col("timestamp").dt.truncate("1s").alias("dt_1s"))
        .drop("timestamp")
        .group_by("dt_1s")
        .agg(
            [
                pl.col("buy_qty").sum().alias("buy_vol"),
                pl.col("sell_qty").sum().alias("sell_vol"),
                pl.col("notional").sum().alias("quote_vol"),
            ]
        )
        .with_columns((pl.col("buy_vol") + pl.col("sell_vol")).alias("volume"))
        .with_columns(
            [
                (pl.col("quote_vol") / pl.col("volume")).alias("vwap"),
                (pl.col("buy_vol") - pl.col("sell_vol")).alias("signed_volume"),
            ]
        )
        .drop(["quote_vol", "buy_vol", "sell_vol"])
        .sort("dt_1s")
    )

    # 1-second base grid spanning the full UTC day of the data (00:00:00 ..
    # 23:59:59), so every per-day aligned file has the same 86,400-row grid
    # regardless of when the first/last book update landed. Seconds before the
    # first native book (nothing to carry yet) are dropped after the join below;
    # trailing seconds are forward-filled and flagged stale. datetime_ranges is
    # inclusive of both bounds.
    day_start = pl.col("dt_1s").min().dt.truncate("1d")
    grid = book.select(
        pl.datetime_ranges(
            day_start,
            day_start + pl.duration(days=1) - pl.duration(seconds=1),
            "1s",
        ).alias("dt_1s")
    ).explode("dt_1s", empty_as_null=True)

    # ASOF join handles the forward-filling
    aligned = (
        grid.join_asof(book, on="dt_1s", strategy="backward")
        .join_asof(klines, on="dt_1s", strategy="backward")
        .join(trades, on="dt_1s", how="left")
    )

    # boundary rules: stale quotes, zeroed volumes
    aligned = aligned.with_columns(
        [
            # book_stale: this grid second had NO native book update (its book was
            # forward-filled from an earlier second). exact_time is the carried
            # update's time; if its truncated second != this second, the quote is
            # stale. OFI computed from a stale quote is meaningless, so any fill is
            # flagged — not just fills older than a fixed tolerance. A null carried
            # book (before the first update of the day) is stale too.
            (pl.col("exact_time").dt.truncate("1s") != pl.col("dt_1s"))
            .fill_null(True)
            .alias("book_stale"),
            # gap_prev_s: whole seconds since the last native book update (0 on a
            # native second, >0 on a forward-filled one). Provenance for OFI/analysis.
            (pl.col("dt_1s") - pl.col("exact_time").dt.truncate("1s"))
            .dt.total_seconds()
            .alias("gap_prev_s"),
            # trade defaults
            pl.col("volume").is_null().alias("no_trades"),
            pl.col("volume").fill_null(0.0),
            pl.col("signed_volume").fill_null(0.0),
        ]
    )
    # Drop the seconds before the first native book update — there is nothing to
    # carry forward there (a null carried book). Trailing seconds after the last
    # update are retained (forward-filled, flagged stale) to keep full-day coverage.
    aligned = aligned.filter(pl.col("exact_time").is_not_null())

    # book_stale_prev: the previous grid second was stale (or absent). ofi.py
    # derives its own t-1 flag internally, but carry it for provenance and parity
    # with the L2 aligned schema. Grid rows are already in ascending time order;
    # computed after the leading-drop so the first retained row's prev is null->True.
    return (
        aligned.with_columns(pl.col("book_stale").shift(1).fill_null(True).alias("book_stale_prev"))
        .drop("exact_time")
        # emit the 1-second grid as ``timestamp`` (Datetime) — the key ofi.py sorts
        # on and diffs; the raw book timestamp was dropped above so there is no clash.
        .rename({"dt_1s": "timestamp"})
    )
