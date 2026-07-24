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

    # aggregate trades per second (volume, signed_volume, VWAP)
    trades = (
        trades_lf.with_columns(pl.col("timestamp").dt.truncate("1s").alias("dt_1s"))
        .drop("timestamp")
        .group_by("dt_1s")
        .agg(
            [
                pl.col("quantity").sum().alias("volume"),
                (pl.col("price") * pl.col("quantity")).sum().alias("quote_vol"),
                pl.col("buy_qty").sum().alias("buy_vol"),
                pl.col("sell_qty").sum().alias("sell_vol"),
            ]
        )
        .with_columns(
            [
                (pl.col("quote_vol") / pl.col("volume")).alias("vwap"),
                (pl.col("buy_vol") - pl.col("sell_vol")).alias("signed_volume"),
            ]
        )
        .drop(["quote_vol", "buy_vol", "sell_vol"])
        .sort("dt_1s")
    )

    # creating the 1-second base grid mapped strictly to the book ticker bounds
    grid = book.select(
        pl.datetime_ranges(pl.col("dt_1s").min(), pl.col("dt_1s").max(), "1s").alias("dt_1s")
    ).explode("dt_1s", empty_as_null=True)

    # ASOF join handles the forward-filling
    aligned = (
        grid.join_asof(book, on="dt_1s", strategy="backward")
        .join_asof(klines, on="dt_1s", strategy="backward")
        .join(trades, on="dt_1s", how="left")
    )

    # boundary rules: stale quotes, zeroed volumes
    return (
        aligned.with_columns(
            [
                # cap forward fill: flag if the exact quote time is > 5s behind current grid second
                ((pl.col("dt_1s") - pl.col("exact_time")).dt.total_seconds() > 5)
                .fill_null(True)
                .alias("book_stale"),
                # trade defaults
                pl.col("volume").is_null().alias("no_trades"),
                pl.col("volume").fill_null(0.0),
                pl.col("signed_volume").fill_null(0.0),
            ]
        )
        .drop("exact_time")
        # emit the 1-second grid as ``timestamp`` (Datetime) — the key ofi.py sorts
        # on and diffs; the raw book timestamp was dropped above so there is no clash.
        .rename({"dt_1s": "timestamp"})
    )
