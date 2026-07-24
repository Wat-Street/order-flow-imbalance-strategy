import polars as pl


def build_alignment_engine(
    book_lf: pl.LazyFrame, trades_lf: pl.LazyFrame, klines_lf: pl.LazyFrame
) -> pl.LazyFrame:

    # combines L1 book, filtered trades, and klines into a 1-second grid, assuming
    # time stamps come in ms

    # shift klines forward by 1 minute (60,000 ms) before forward-filling
    # a bar covering 09:00:00 - 09:00:59 becomes valid exactly at 09:01:00
    klines = (
        klines_lf.with_columns((pl.col("timestamp") + 60_000).alias("shifted_ts"))
        .with_columns(pl.from_epoch("shifted_ts", time_unit="ms").alias("dt_1s"))
        .drop(["timestamp", "shifted_ts"])
        .sort("dt_1s")
    )

    # truncate book updates to 1s, keep the final s
    # tate per second
    book = (
        book_lf.with_columns(
            [
                pl.from_epoch("timestamp", time_unit="ms").alias("exact_time"),
                pl.from_epoch("timestamp", time_unit="ms").dt.truncate("1s").alias("dt_1s"),
            ]
        )
        .group_by("dt_1s")
        .agg(pl.all().sort_by("exact_time").last())
        .sort("dt_1s")
    )

    # aggregate trades per second (volume, signed_volume, VWAP)
    trades = (
        trades_lf.with_columns(
            pl.from_epoch("timestamp", time_unit="ms").dt.truncate("1s").alias("dt_1s")
        )
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
    return aligned.with_columns(
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
    ).drop("exact_time")
