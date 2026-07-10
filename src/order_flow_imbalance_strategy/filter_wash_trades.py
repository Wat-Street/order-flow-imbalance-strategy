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
    
    # return a polars dataframe with the matched data
    return matched_data

# Step 5 - calculate suspicion clues

# Zero Price-Impact
def zero_price_impact(df, impact_eps):
    # calculate price impact as the difference between the mid-price of the current trade and the next trade
    mid = (pl.col("best_bid_price") + pl.col("best_ask_price")) / 2

    df = df.with_columns(
        mid.alias("mid-price"),
        mid.shift(-1).alias("next_mid-price"),
    ) 

    df = df.sort("transact_time").with_columns([
        pl.col("quantity").rolling_quantile(window_size=500, quantile=0.9).alias("size_baseline")
    ])

    large_trade = pl.col("quantity") > pl.col("size_baseline")
    price_change = (pl.col("next_mid-price") - pl.col("mid-price")).abs()

    # calculate score for price impact
    score = (pl.when(large_trade & (price_change < impact_eps)).then(1).then((impact_eps - price_change) / impact_eps).otherwise(0))
    
    return df.select(score.alias("zero_price_impact_score")).to_series()



# process each file
def process_task(symbol, date_str):
    matched_data = match_trades_to_price(symbol, date_str)


# Step 1-2 - define CLI arguments
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", type=str, default=symbol_list)
    parser.add_argument("--start", type=dt.datetime.fromisoformat, required=True)
    parser.add_argument("--end", type=dt.datetime.fromisoformat, required=True)
    parser.add_argument("--workers", help="parallelism", type=int, default=4)
    parser.add_argument("--wash-score-cut", help="threshold for wash trade detection", type=float, default=0.7)
    
    # Should these have any default values?
    parser.add_argument("--impact-eps", help="impact threshold", type=float) 
    parser.add_argument("--pingping-window", help="ping-pong window size", type=int)
    parser.add_argument("--size-quantile", help="size quantile threshold", type=float)

    parser.add_argument("--passthrough", action="store_true", help="turn off filtering")

    args = parser.parse_args()
    if args.start > args.end:
        sys.exit("start date must be before end date")

    tasks = generate_tasks(args)

    # submit tasks to ProcessPoolExecutor to perform wash trade filtering in parallel
    # results = {"ok": 0, "skipped": 0, "missing": 0, "checksum_failed": 0, "error": 0}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_task, symbol, date_str)
            for symbol, date_str in tasks
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Wash Trade Filtering"
        ):
            result = future.result()
            # status = result["status"]
            # results[status] += 1
        

if __name__ == "__main__":
    main()