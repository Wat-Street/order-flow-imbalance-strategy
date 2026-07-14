import argparse
import concurrent.futures
import datetime as dt
import logging
import sys
from pathlib import Path

import polars as pl
from tqdm import tqdm

# set up logging for progress tracking and post-run inspection
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/filter_wash_trades.log", mode="a"),
    ],
)
logger = logging.getLogger("wash_trade_filter_logger")

# define constants
symbol_list = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
MIN_CSV_SIZE = 100
DATA_PROCESSED_DIR = Path("data/processed")


# Step 3 - build final list of all dates and symbols
def generate_tasks(args):
    output = []
    delta = dt.timedelta(days=1)
    for symbol in args.symbols:
        current_date = args.start
        while current_date <= args.end:
            date_str = current_date.strftime("%Y-%m-%d")
            output.append((symbol, date_str))
            current_date += delta
    return output


# Step 4 - match trades to price using asof join with polars
def match_trades_to_price(symbol, date_str):
    trades_path = DATA_PROCESSED_DIR / f"{symbol}/aggTrades/{symbol}-aggTrades-{date_str}.parquet"
    prices_path = DATA_PROCESSED_DIR / f"{symbol}/bookTicker/{symbol}-bookTicker-{date_str}.parquet"

    if not trades_path.exists() or not prices_path.exists():
        logger.warning("Missing input file(s) for %s on %s. Skipping.", symbol, date_str)
        return

    trades = pl.scan_parquet(trades_path)
    prices = pl.scan_parquet(prices_path)

    # normalized schema uses a shared "timestamp" column for both tables
    trades = trades.sort("timestamp")
    prices = prices.sort("timestamp")

    after_quotes = prices.select(["timestamp", "bid_price", "ask_price"]).rename(
        {"bid_price": "bid_after", "ask_price": "ask_after"}
    )

    matched_data = (
        trades.join_asof(prices, on="timestamp", strategy="backward", tolerance="100ms")
        .rename({"bid_price": "bid_before", "ask_price": "ask_before"})
        .join_asof(after_quotes, on="timestamp", strategy="forward", tolerance="100ms")
        .collect()
    )

    if matched_data.height < MIN_CSV_SIZE:
        logger.info(
            "Skipping %s on %s because file has only %s rows rows.",
            symbol,
            date_str,
            matched_data.height,
        )
        return None

    # return a polars dataframe with the matched data
    return matched_data


# Step 5 - calculate suspicion clues


# Zero Price-Impact
def zero_price_impact(df, impact_eps_bps, size_quantile, horizon):
    # price impact = relative mid-price move from the trade to `horizon` trades later.
    # a horizon (rather than the immediate next trade) is required because the mid
    # rarely changes between adjacent trades, so shift(-1) reports ~zero impact for
    # almost every large trade. mid_price is provided by the normalized schema.
    df = df.with_columns(
        pl.col("mid_price").shift(-horizon).alias("next_mid_price"),
    )

    # define a large trade as a trade larger than the size_quantile of the last 500 trades
    df = df.with_columns(
        [
            pl.col("quantity")
            .rolling_quantile(window_size=500, quantile=size_quantile)
            .alias("size_baseline")
        ]
    )
    large_trade = pl.col("quantity") > pl.col("size_baseline")

    # express the move in basis points so the threshold is scale-invariant across symbols
    price_change_bps = (
        (pl.col("next_mid_price") - pl.col("mid_price")).abs() / pl.col("mid_price") * 1e4
    )

    # calculate score for price impact
    # closer to 0 = trade was large and had a price impact
    # closer to 1 = trade was large and had little/no price impact, making it suspicious
    score = (
        pl.when(large_trade & (price_change_bps < impact_eps_bps))
        .then(1.0)
        .when(large_trade)
        .then(
            ((impact_eps_bps - price_change_bps) / impact_eps_bps).clip(
                lower_bound=0.0, upper_bound=1.0
            )
        )
        .otherwise(0.0)
    )

    return df.select(score).to_series()


# Off-Touch Execution


def off_touch_execution(df, touch_floor_bps=2.0):
    # bracket matching
    bracket_low = pl.min_horizontal(["bid_before", "bid_after"])
    bracket_high = pl.max_horizontal(["ask_before", "ask_after"])

    below = (bracket_low - pl.col("price")).clip(lower_bound=0.0)
    above = (pl.col("price") - bracket_high).clip(lower_bound=0.0)
    distance = pl.max_horizontal([below, above])

    # calculate a score from 0 to 1 for off touch execution
    # closer to 0 = trade price was inside the before/after bracket
    # closer to 1 = trade price was outside the bracket, making it suspicious
    score = (distance / (pl.col("price") * (touch_floor_bps / 1e4))).clip(
        lower_bound=0.0, upper_bound=1.0
    )

    return df.select(score.fill_null(0.0)).to_series()


# Ping-Pong Reversal
def ping_pong_reversal(df, ping_pong_window):
    # normalized schema stores time as Datetime(ms); work in integer epoch-ms so the
    # window arithmetic (ping_pong_window is in ms) matches the raw-schema behaviour

    # find the number of rows within the ping-pong window, i.e. how far back we need
    # to look to spot ping-pong behaviour
    times = df["timestamp"].dt.epoch("ms")
    time_ms = pl.col("timestamp").dt.epoch("ms")

    cutoff_times = times - ping_pong_window
    boundary_index = times.search_sorted(cutoff_times, side="left")
    row_index = pl.Series(range(len(times)))
    depth = row_index - boundary_index

    lookback = int(depth.max())
    lookback = max(lookback, 1)

    # calculate a score for ping-pong reversal
    # closer to 0 = trade was not part of a ping-pong reversal
    # closer to 1 = trade was likely part of a ping-pong reversal, making it suspicious
    # closer to 1 also means same-price/same-quantity trades on opposite sides occurred
    # closer together in time

    # fold the per-lookback scores into a running maximum in batches so peak memory
    # stays O(rows): lookback can exceed 1000 on liquid symbols, and materializing
    # that many full-length columns at once to max_horizontal would exhaust memory
    batch_size = 32
    running_max = None
    batch = []

    def fold(batch, running_max):
        if not batch:
            return running_max
        batch_max = df.select(pl.max_horizontal(batch).alias("score")).to_series()
        if running_max is None:
            return batch_max
        return (
            pl.DataFrame({"running": running_max, "batch": batch_max})
            .select(pl.max_horizontal("running", "batch").alias("score"))
            .to_series()
        )

    for k in range(1, lookback + 1):
        prev_price = pl.col("price").shift(k)
        prev_qty = pl.col("quantity").shift(k)
        prev_side = pl.col("side").shift(k)
        prev_time = time_ms.shift(k)
        delta_ms = time_ms - prev_time

        same_price = pl.col("price") == prev_price
        same_qty = pl.col("quantity") == prev_qty
        flipped = pl.col("side") != prev_side
        fastness = (1.0 - delta_ms / float(ping_pong_window)).clip(lower_bound=0.0, upper_bound=1.0)

        # calculate a score for the row that is "k" rows back from the current row
        # take the highest score found across all lookback distances
        row_score = (
            pl.when(
                (delta_ms > 0) & (delta_ms <= ping_pong_window) & flipped & same_price & same_qty
            )
            .then(fastness)
            .otherwise(0.0)
        )

        batch.append(row_score)
        if len(batch) >= batch_size:
            running_max = fold(batch, running_max)
            batch = []

    running_max = fold(batch, running_max)
    if running_max is None:
        running_max = pl.Series("score", [0.0] * df.height)
    return running_max


# Duplicate Prints
def duplicate_prints(df):
    # find the number of duplicate trades that have the same timestamp, price, quantity, and side
    df = df.with_columns(
        [pl.len().over(["timestamp", "price", "quantity", "side"]).alias("duplicate_count")]
    )

    # score for duplicate trades: no duplicates = 0, one duplicate = 0.5,
    # two duplicates = 0.67, and so on
    score = 1.0 - (1.0 / pl.col("duplicate_count"))

    return df.select(score).to_series()


# Size Clustering
def size_clustering(df, size_quantile):
    trailing_baseline = (
        pl.col("quantity")
        .rolling_quantile(window_size=500, quantile=size_quantile)
        .fill_null(pl.col("quantity"))
    )

    # how far above baseline this trade's size is, as a ratio, where a "small" trade
    # is 0 and a "large" trade is 1
    size_excess = ((pl.col("quantity") - trailing_baseline) / (trailing_baseline + 1e-12)).clip(
        lower_bound=0.0
    )
    size_score = size_excess / (size_excess + 1.0)

    # score for repeated trades: no repeats = 0, one repeat = 0.5,
    # two repeats = 0.67, and so on
    repeat_count = pl.len().over("quantity")
    repeat_score = 1.0 - (1.0 / repeat_count)

    # calculate a final size clustering score
    score = size_score * repeat_score

    return df.select(score.fill_null(0.0)).to_series()


# process each file
def process_task(
    symbol,
    date_str,
    impact_eps,
    impact_horizon,
    pingpong_window,
    size_quantile,
    touch_floor_bps,
    wash_score_cut,
    passthrough,
):
    out_dir = DATA_PROCESSED_DIR / symbol / "aggTrades_filtered"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}-aggTrades-{date_str}.parquet"

    # Checking if the output file already exists so failed runs can be restarted
    # is commented out because we may still want to rerun with different parameters
    # on the same files for testing/tuning purposes

    # if out_path.exists():
    #    logger.info("Skipping %s on %s because output file already exists.", symbol, date_str)
    #    return {"symbol": symbol, "date": date_str, "status": "skipped_existing"}

    matched_data = match_trades_to_price(symbol, date_str)

    # skip if file is smaller than MIN_CSV_SIZE or if the parquet file is missing
    if matched_data is None:
        return {"symbol": symbol, "date": date_str, "status": "missing"}

    if passthrough:
        matched_data.write_parquet(out_path)
        logger.info(
            "Passthrough wrote %s on %s with %s rows.", symbol, date_str, matched_data.height
        )
        return {
            "symbol": symbol,
            "date": date_str,
            "status": "passthrough",
            "rows": matched_data.height,
        }

    # all the scoring functions assume that the data is sorted by timestamp
    matched_data = matched_data.sort("timestamp", maintain_order=True)

    score_cols = [
        "impact_score",
        "touch_score",
        "ping_pong_score",
        "duplicate_score",
        "size_cluster_score",
    ]

    scored = (
        matched_data.with_columns(
            [
                zero_price_impact(matched_data, impact_eps, size_quantile, impact_horizon).alias(
                    "impact_score"
                ),
                off_touch_execution(matched_data, touch_floor_bps).alias("touch_score"),
                ping_pong_reversal(matched_data, pingpong_window).alias("ping_pong_score"),
                duplicate_prints(matched_data).alias("duplicate_score"),
                size_clustering(matched_data, size_quantile).alias("size_cluster_score"),
            ]
        )
        .with_columns(pl.max_horizontal(score_cols).alias("wash_score"))
        .with_columns(
            (pl.col("wash_score") >= wash_score_cut).alias("wash_suspect"),
            pl.when(pl.col("impact_score") == pl.col("wash_score"))
            .then(pl.lit("impact_score"))
            .when(pl.col("touch_score") == pl.col("wash_score"))
            .then(pl.lit("touch_score"))
            .when(pl.col("ping_pong_score") == pl.col("wash_score"))
            .then(pl.lit("ping_pong_score"))
            .when(pl.col("duplicate_score") == pl.col("wash_score"))
            .then(pl.lit("duplicate_score"))
            .otherwise(pl.lit("size_cluster_score"))
            .alias("wash_reason"),
            # cleaned-flow columns (buy_qty_clean, sell_qty_clean, notional_clean)
            # suspected prints are zeroed
        )
        .with_columns(
            [
                pl.when(pl.col("wash_suspect"))
                .then(0.0)
                .otherwise(pl.col("buy_qty"))
                .alias("buy_qty_clean"),
                pl.when(pl.col("wash_suspect"))
                .then(0.0)
                .otherwise(pl.col("sell_qty"))
                .alias("sell_qty_clean"),
                pl.when(pl.col("wash_suspect"))
                .then(0.0)
                .otherwise(pl.col("notional"))
                .alias("notional_clean"),
            ]
        )
    )

    reason_counts = (
        scored.filter(pl.col("wash_suspect"))
        .group_by("wash_reason")
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / pl.col("count").sum() * 100).alias("pct"))
        .sort("count", descending=True)
    )

    reason_breakdown = {
        row["wash_reason"]: {"count": row["count"], "pct": row["pct"]}
        for row in reason_counts.to_dicts()
    }

    clean_volume = scored["buy_qty_clean"].sum() + scored["sell_qty_clean"].sum()
    assert clean_volume <= (matched_data["buy_qty"].sum() + matched_data["sell_qty"].sum())

    flag_rate = scored["wash_suspect"].mean()
    if flag_rate > 0.15:
        logger.warning(
            "Flag rate (%.3f) is greater than 0.15 for %s on %s; filtering may be too sensitive",
            flag_rate,
            symbol,
            date_str,
        )

    scored.write_parquet(out_path)

    logger.info(
        "Completed %s on %s: rows=%s flagged=%s flag_rate=%.3f reason_breakdown=%s",
        symbol,
        date_str,
        scored.height,
        int(scored["wash_suspect"].sum()),
        flag_rate,
        reason_breakdown,
    )

    return {
        "symbol": symbol,
        "date": date_str,
        "status": "ok",
        "total": scored.height,
        "flagged": int(scored["wash_suspect"].sum()),
        "flag_rate": flag_rate,
        "reason_breakdown": reason_breakdown,
    }


# Step 1-2 - define CLI arguments
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", type=str, default=symbol_list)
    parser.add_argument("--start", type=dt.datetime.fromisoformat, required=True)
    parser.add_argument("--end", type=dt.datetime.fromisoformat, required=True)
    parser.add_argument("--workers", help="parallelism", type=int, default=4)
    parser.add_argument(
        "--wash-score-cut", help="threshold for wash trade detection", type=float, default=0.7
    )

    parser.add_argument(
        "--impact-eps",
        type=float,
        default=0.5,
        help="min post-trade mid move (bps) below which a large trade is zero-impact",
    )
    parser.add_argument(
        "--impact-horizon",
        type=int,
        default=50,
        help="number of trades ahead to measure price impact over",
    )
    parser.add_argument(
        "--touch-floor-bps",
        type=float,
        default=2.0,
        help="floor for the off-touch half-spread denominator (bps of price)",
    )
    parser.add_argument("--pingpong-window", help="ping-pong window size", type=int, default=500)
    parser.add_argument("--size-quantile", help="size quantile threshold", type=float, default=0.9)

    parser.add_argument("--passthrough", action="store_true", help="turn off filtering")

    args = parser.parse_args()
    if args.start > args.end:
        sys.exit("start date must be before end date")

    tasks = generate_tasks(args)
    results = {"ok": 0, "passthrough": 0, "missing": 0, "skipped_existing": 0}
    total_flagged = 0

    logger.info(
        "Generated %s wash-trade tasks for %s symbol(s) from %s to %s.",
        len(tasks),
        len(args.symbols),
        args.start,
        args.end,
    )
    logger.info("Submitting tasks to %s worker processes.", args.workers)

    # submit tasks to ProcessPoolExecutor to perform wash trade filtering in parallel

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_task,
                symbol,
                date_str,
                args.impact_eps,
                args.impact_horizon,
                args.pingpong_window,
                args.size_quantile,
                args.touch_floor_bps,
                args.wash_score_cut,
                args.passthrough,
            )
            for symbol, date_str in tasks
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Wash Trade Filtering",
        ):
            result = future.result()
            results[result["status"]] += 1
            if result["status"] == "ok":
                total_flagged += result["flagged"]

    logger.info("Wash-trade run complete: %s", results)
    logger.info("Total flagged trades: %s", total_flagged)


if __name__ == "__main__":
    main()
