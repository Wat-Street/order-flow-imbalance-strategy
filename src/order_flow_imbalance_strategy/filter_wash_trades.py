import argparse
import datetime as dt
import sys
from pathlib import Path
import concurrent.futures

import polars as pl 
import tqdm

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
    trades = pl.scan_parquet(f"DATA_PROCESSED_DIR/{symbol}/aggTrades/{symbol}-aggTrades-{date_str}.parquet")
    prices = pl.scan_parquet(f"DATA_PROCESSED_DIR/{symbol}/bookTicker/{symbol}-bookTicker-{date_str}.parquet")

    trades = trades.sort("transact_time")
    prices = prices.sort("transaction_time")

    matched_data = trades.join_asof(prices, left_on="transact_time",right_on="transaction_time", strategy="backward").collect()
    
    # return a polars dataframe with the matched data
    return matched_data

# Step 5 - calculate suspicion clues
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

    ######### up to here is in order ##########

    # submit tasks to ThreadPoolExecutor to download tasks in parallel
    results = {"ok": 0, "skipped": 0, "missing": 0, "checksum_failed": 0, "error": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_task, symbol, date_str)
            for symbol, date_str in tasks
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Wash Trade Filtering"
        ):
            result = future.result()
            status = result["status"]
            results[status] += 1
        logger.info("Download process complete. Summary: %s", results)
    if args.validate:
        run_validation(data_dir, args)
