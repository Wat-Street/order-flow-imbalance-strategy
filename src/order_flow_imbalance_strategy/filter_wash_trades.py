import argparse
import datetime as dt
import sys
from pathlib import Path
import concurrent.futures

import polars as pl 
from tqdm import tqdm

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
    trades_path = DATA_PROCESSED_DIR/f"{symbol}/aggTrades/{symbol}-aggTrades-{date_str}.parquet"
    prices_path = DATA_PROCESSED_DIR/f"{symbol}/bookTicker/{symbol}-bookTicker-{date_str}.parquet"

    if not trades_path.exists() or not prices_path.exists():
        print(f"Missing file for {symbol} on {date_str}. Skipping.")
        return

    trades = pl.scan_parquet(trades_path)
    prices = pl.scan_parquet(prices_path)

    trades = trades.sort("transact_time")
    prices = prices.sort("transaction_time")

    matched_data = trades.join_asof(prices, left_on="transact_time",right_on="transaction_time", strategy="backward").collect()
    
    if matched_data.height < MIN_CSV_SIZE:
      return None
    
    # return a polars dataframe with the matched data
    return matched_data

# Step 5 - calculate suspicion clues

# Zero Price-Impact
def zero_price_impact(df, impact_eps):
    # calculate price impact as the difference between the mid-price of the current trade and the next trade
    mid = (pl.col("best_bid_price") + pl.col("best_ask_price")) / 2

    df = df.with_columns(
        mid.alias("mid_price"),
        mid.shift(-1).alias("next_mid_price"),
    ) 

    # define a large trade as a trade larger than the 90th percentile of the last 500 trades
    df = df.with_columns([
        pl.col("quantity").rolling_quantile(window_size=500, quantile=0.9).alias("size_baseline")
    ])
    large_trade = pl.col("quantity") > pl.col("size_baseline")

    price_change = (pl.col("next_mid_price") - pl.col("mid_price")).abs()

    # calculate score for price impact
    # closer to 0 = trade was large and had a price impact
    # closer to 1 = trade was large and had little/no price impact, making it suspicious
    score = (pl.when(large_trade & (price_change < impact_eps)).then(1).when(large_trade).then((impact_eps - price_change) / impact_eps).otherwise(0))
    
    return df.select(score).to_series()

# Off-Touch Execution
def off_touch_execution(df):
    # compare the trade price to the best bid and ask prices
    spread = pl.col("best_ask_price") - pl.col("best_bid_price")

    distance = (pl.when(pl.col("is_buyer_maker")).then((pl.col("price") - pl.col("best_bid_price")).abs())
                                                       .otherwise((pl.col("price") - pl.col("best_ask_price")).abs())
    )

    # calculate a score from 0 to 1 for off touch execution
    # closer to 0 = trade price was close to the exected bid or ask
    # closer to 1 = trade price was far from the expected bid or ask, making it suspicious
    score = (distance / (spread/2)).clip(lower_bound=0.0, upper_bound=1.0)

    return df.select(score).to_series()

# Ping-Pong Reversal
def ping_pong_reversal(df, ping_pong_window):
    # find the number of rows that are within the ping-pong window to see how many rows we need to lookback into to spot ping pong behaviour
    times = df["transact_time"]

    cutoff_times = times - ping_pong_window
    boundary_index = times.search_sorted(cutoff_times, side="left")
    row_index = pl.Series(range(len(times)))
    depth = (row_index - boundary_index)

    lookback = int(depth.max())
    lookback = max(lookback, 1)

    # calculate a score for ping-pong reversal
    # closer to 0 = trade was not part of a ping-pong reversal
    # closer to 1 = trade was likely part of a ping-pong reversal, making it suspicious
    # closer to 1 means trades with the same price and quantity, but opposite sides, occurred closer together
    scores = []

    for k in range(1, lookback + 1):
        prev_price = pl.col("price").shift(k)
        prev_qty = pl.col("quantity").shift(k)
        prev_side = pl.col("is_buyer_maker").shift(k)
        prev_time = pl.col("transact_time").shift(k)
        delta_ms = pl.col("transact_time") - prev_time

        same_price = pl.col("price") == prev_price
        same_qty = pl.col("quantity") == prev_qty
        flipped = pl.col("is_buyer_maker") != prev_side
        fastness = (1.0 - delta_ms / float(ping_pong_window)).clip(lower_bound=0.0, upper_bound=1.0)

        # calculate a score for the row that is "k" rows back from the current row
        # take the highest score found across all lookback distances
        row_score = pl.when(
            (delta_ms > 0) & (delta_ms <= ping_pong_window) & flipped & same_price & same_qty
        ).then(fastness).otherwise(0.0)

        scores.append(row_score)

    max_score = pl.max_horizontal(scores)
    return df.select(max_score).to_series()

# Duplicate Prints
def duplicate_prints(df):
    # find the number of duplicate trades that have the same transact_time, price, quantity, and is_buyer_maker
    df = df.with_columns([
        pl.len().over(["transact_time", "price", "quantity", "is_buyer_maker"]).alias("duplicate_count")
    ])

    # calculate a score for duplicate trades, where no duplicates = 0, one duplicate = 0.5, two duplicates = 0.67, and so on
    score = 1.0 - (1.0 / pl.col("duplicate_count"))

    return df.select(score).to_series()

# Size Clustering
def size_clustering(df, size_quantile):
    trailing_baseline = (
        pl.col("quantity").rolling_quantile(window_size=500, quantile=size_quantile)
        .fill_null(pl.col("quantity"))
    )

    # how far above baseline this trade's size is, as a ratio, where a "small" trade is 0 and a "large" trade is 1
    size_excess = ((pl.col("quantity") - trailing_baseline) / (trailing_baseline + 1e-12)).clip(lower_bound=0.0)
    size_score = size_excess / (size_excess + 1.0)

    # calculate a score for repeated trades, where no repeats = 0, one repeat = 0.5, two repeats = 0.67, and so on
    repeat_count = pl.len().over("quantity")
    repeat_score = 1.0 - (1.0 / repeat_count)

    # calculate a final size clustering score
    score = size_score * repeat_score

    return df.select(score.fill_null(0.0)).to_series()


# process each file
def process_task(symbol, date_str, impact_eps, pingpong_window, size_quantile, wash_score_cut, passthrough):
    matched_data = match_trades_to_price(symbol, date_str)

    # skip if file is smaller than MIN_CSV_SIZE or if the parquet file is missing
    if matched_data is None:
        return {"symbol": symbol, "date": date_str, "status": "missing"}

    out_dir = DATA_PROCESSED_DIR / symbol / "aggTrades_filtered"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}-aggTrades-{date_str}.parquet"

    if passthrough:
        matched_data.write_parquet(out_path)
        return {"symbol": symbol, "date": date_str, "status": "passthrough", "rows": matched_data.height}

    # all the scoring functions assume that the data is sorted by transact_time
    matched_data = matched_data.sort("transact_time", maintain_order=True)

    scored = matched_data.with_columns([
        zero_price_impact(matched_data, impact_eps, size_quantile).alias("impact_score"),
        off_touch_execution(matched_data).alias("touch_score"),
        ping_pong_reversal(matched_data, pingpong_window).alias("ping_pong_score"),
        duplicate_prints(matched_data).alias("duplicate_score"),
        size_clustering(matched_data, size_quantile).alias("size_cluster_score"),
    ]).with_columns(

        # final wash score is the maximum of all the individual scores
        pl.max_horizontal([
            "impact_score", "touch_score", "ping_pong_score", "duplicate_score", "size_cluster_score"
        ]).alias("wash_score")
    ).with_columns(
        (pl.col("wash_score") >= wash_score_cut).alias("wash_suspect")
    )

    clean_volume = scored.filter(~pl.col("wash_suspect"))["quantity"].sum()
    assert clean_volume <= matched_data["quantity"].sum()

    flag_rate = scored["wash_suspect"].mean()

    scored.write_parquet(out_path)

    return {"symbol": symbol, "date": date_str, "status": "ok", "total": scored.height, "flagged": int(scored["wash_suspect"].sum()), "flag_rate": flag_rate}

# Step 1-2 - define CLI arguments
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", type=str, default=symbol_list)
    parser.add_argument("--start", type=dt.datetime.fromisoformat, required=True)
    parser.add_argument("--end", type=dt.datetime.fromisoformat, required=True)
    parser.add_argument("--workers", help="parallelism", type=int, default=4)
    parser.add_argument("--wash-score-cut", help="threshold for wash trade detection", type=float, default=0.7)
    
    # Should these have any default values?
    parser.add_argument("--impact-eps", help="impact threshold", type=float, default = 0.01) 
    parser.add_argument("--pingpong-window", help="ping-pong window size", type=int, default=500)
    parser.add_argument("--size-quantile", help="size quantile threshold", type=float, default=0.9)

    parser.add_argument("--passthrough", action="store_true", help="turn off filtering")

    args = parser.parse_args()
    if args.start > args.end:
        sys.exit("start date must be before end date")

    tasks = generate_tasks(args)
    results = {"ok": 0, "passthrough": 0, "missing": 0}
    total_flagged = 0

    # submit tasks to ProcessPoolExecutor to perform wash trade filtering in parallel

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_task, symbol, date_str, args.impact_eps, args.pingpong_window, args.size_quantile, args.wash_score_cut, args.passthrough)
            for symbol, date_str in tasks
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Wash Trade Filtering"
        ):
            result = future.result()
            results[result["status"]] += 1
            if result["status"] == "ok":
                total_flagged += result["flagged"]
 
    print(results)
    print(f"total flagged trades: {total_flagged}")
        

if __name__ == "__main__":
    main()