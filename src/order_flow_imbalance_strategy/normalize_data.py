import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import polars.selectors as cs

# set up for logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(processName)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Normalizer")

# --- Configuration constants -------------------------------------------------
# Centralized here rather than scattered through the code. These describe the
# Binance file schema (a fixed data contract, not user-tunable) and the default
# on-disk layout (overridable per-run via CLI flags below).

# Default data directories; override with --raw-dir / --processed-dir.
DEFAULT_RAW_DIR = "data/raw"
DEFAULT_PROCESSED_DIR = "data/processed"

# Minimum rows that must survive filtering before a file is written.
MIN_VALID_ROWS = 100

# Binance ships bookTicker CSVs with descriptive headers; map them to our schema.
# (update_id and event_time already match, so they are left untouched.)
BOOK_TICKER_RENAME = {
    "best_bid_price": "bid_price",
    "best_bid_qty": "bid_qty",
    "best_ask_price": "ask_price",
    "best_ask_qty": "ask_qty",
    "transaction_time": "transact_time",
}
# aggTrades headers (agg_trade_id, price, quantity, first_trade_id, last_trade_id,
# transact_time, is_buyer_maker) already match our schema, so no rename is needed.
# -----------------------------------------------------------------------------


def generate_tasks(symbols, start_date, end_date, data_types, raw_dir, processed_dir):
    # tasks hold the data for one specific asset on one specific day
    tasks = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        for symbol in symbols:
            for data_type in data_types:
                # paths
                raw_file = (
                    Path(raw_dir) / f"{symbol}/{data_type}/{symbol}-{data_type}-{date_str}.csv"
                )
                output_dir = Path(processed_dir) / f"{symbol}/{data_type}"
                output_file = output_dir / f"{symbol}-{data_type}-{date_str}.parquet"

                if output_file.exists():
                    continue
                if not raw_file.exists():
                    continue
                tasks.append(
                    {
                        "data_type": data_type,
                        "symbol": symbol,
                        "date_str": date_str,
                        "raw_path": str(raw_file),
                        "output_path": str(output_file),
                        "output_dir": str(output_dir),
                    }
                )
        current_date += timedelta(days=1)
    return tasks


# normalizes bookticker data
def process_book_ticker(lf: pl.LazyFrame, symbol: str) -> pl.LazyFrame:
    # rename raw Binance headers to our normalized schema
    lf = lf.rename(BOOK_TICKER_RENAME)
    # convert data to correct types
    lf = lf.with_columns(
        [
            pl.col("transact_time").cast(pl.Datetime("ms")).alias("timestamp"),
            pl.lit(symbol).alias("asset"),
            pl.col("bid_price").cast(pl.Float64),
            pl.col("ask_price").cast(pl.Float64),
            pl.col("bid_qty").cast(pl.Float64),
            pl.col("ask_qty").cast(pl.Float64),
        ]
    )
    # filter for basic reqs and duplicates
    lf = lf.filter(
        (pl.col("bid_price") > 0)
        & (pl.col("ask_price") > 0)
        & (pl.col("bid_qty") > 0)
        & (pl.col("ask_qty") > 0)
        & (pl.col("ask_price") > pl.col("bid_price"))
    ).unique(subset=["transact_time"], keep="last")
    # calculating the extra columns
    lf = lf.with_columns(
        [
            ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid_price"),
            (pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
            ((pl.col("bid_qty") - pl.col("ask_qty")) / (pl.col("bid_qty") + pl.col("ask_qty")))
            .fill_nan(0.0)
            .alias("queue_imbalance"),
        ]
    )
    return lf.select(
        [
            "timestamp",
            "asset",
            "bid_price",
            "ask_price",
            "bid_qty",
            "ask_qty",
            "mid_price",
            "spread",
            "queue_imbalance",
        ]
    ).sort("timestamp")


# normalizes raw agg trades data
def process_agg_trades(lf: pl.LazyFrame, symbol: str) -> pl.LazyFrame:
    # aggTrades headers already match our schema (see AGG_TRADES note above),
    # so no rename is required.

    # converting data to right types and generating the directions column
    lf = lf.with_columns(
        [
            pl.col("transact_time").cast(pl.Datetime("ms")).alias("timestamp"),
            pl.lit(symbol).alias("asset"),
            pl.col("price").cast(pl.Float64),
            pl.col("quantity").cast(pl.Float64),
            pl.col("is_buyer_maker").cast(pl.Boolean),
            # logic: T -> sellor aggressor ('sell'), F -> buyer aggressor ('buy')
            pl.when(pl.col("is_buyer_maker"))
            .then(pl.lit("sell"))
            .otherwise(pl.lit("buy"))
            .alias("side"),
        ]
    )
    # clean data
    lf = lf.filter((pl.col("price") > 0) & (pl.col("quantity") > 0)).unique(subset=["agg_trade_id"])
    # calculate extra columns
    lf = lf.with_columns(
        [
            (pl.col("price") * pl.col("quantity")).alias("notional"),
            pl.when(pl.col("side") == "buy")
            .then(pl.col("quantity"))
            .otherwise(0.0)
            .alias("buy_qty"),
            pl.when(pl.col("side") == "sell")
            .then(pl.col("quantity"))
            .otherwise(0.0)
            .alias("sell_qty"),
        ]
    )
    return lf.select(
        ["timestamp", "asset", "price", "quantity", "side", "notional", "buy_qty", "sell_qty"]
    ).sort("timestamp")


def execute_normalization(task):
    """Normalize a single raw file. Returns True on success or intentional skip,
    False if the job failed (so the caller can count failures)."""
    raw_path = task["raw_path"]
    output_path = task["output_path"]
    output_dir = task["output_dir"]
    data_type = task["data_type"]
    symbol = task["symbol"]
    date_str = task["date_str"]

    try:
        # scan the raw data and call corresponding function
        lf = pl.scan_csv(raw_path, has_header=True, rechunk=False)

        if data_type == "bookTicker":
            processed_lf = process_book_ticker(lf, symbol)
        elif data_type == "aggTrades":
            processed_lf = process_agg_trades(lf, symbol)
        else:
            logger.warning(f"SKIPPED [{symbol} | {data_type} | {date_str}]: unknown data type.")
            return True
        # stream the collection to keep peak RAM down on large files
        df = processed_lf.collect(engine="streaming")
        # skip file if fewer than the minimum rows survive filtering
        if df.height < MIN_VALID_ROWS:
            logger.warning(
                f"SKIPPED [{symbol} | {data_type} | {date_str}]: "
                f"Only {df.height} valid rows survived filters."
            )
            return True

        # assertions to check that data has been correctly normalized
        assert df.select(pl.all().null_count()).sum_horizontal().item() == 0, "Null values"
        # inf/NaN checks only apply to float columns (datetime/str/int don't support them)
        float_cols = df.select(cs.float())
        inf_count = float_cols.select(pl.all().is_infinite().sum()).sum_horizontal().item()
        nan_count = float_cols.select(pl.all().is_nan().sum()).sum_horizontal().item()
        assert inf_count == 0, "Inf values"
        assert nan_count == 0, "NaN values"
        if data_type == "bookTicker":
            assert df["spread"].min() >= 0, "Negative spread"
            assert df["bid_price"].min() > 0 and df["ask_price"].min() > 0, (
                "Prices must be positive values"
            )
            assert len(df.columns) == 9, f"Column count mismatch. Expected 9, got {len(df.columns)}"
        elif data_type == "aggTrades":
            assert df["price"].min() > 0, "Trade execution price must be positive"
            assert set(df["side"].unique().to_list()).issubset({"buy", "sell"}), (
                "Invalid trade direction generated"
            )
            assert len(df.columns) == 8, f"Column count mismatch. Expected 8, got {len(df.columns)}"

        os.makedirs(output_dir, exist_ok=True)
        df.write_parquet(output_path, compression="snappy")
        logger.info(
            f"SUCCESS [{symbol} | {data_type} | {date_str}]: Transformed {df.height} records."
        )
        return True

    except Exception as e:
        logger.error(
            f"FAILED [{symbol} | {data_type} | {date_str}]: Execution crashed with error: {str(e)}"
        )
        return False


def main():
    parser = argparse.ArgumentParser(
        description="High-Throughput Binance Raw Data Normalization Engine."
    )
    parser.add_argument(
        "--symbols", nargs="+", required=True, help="List of trading pairs (e.g., BTCUSDT ETHUSDT)"
    )
    parser.add_argument("--start", required=True, help="Processing window start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Processing window end date (YYYY-MM-DD)")
    parser.add_argument(
        "--types",
        nargs="+",
        default=["bookTicker", "aggTrades"],
        choices=["bookTicker", "aggTrades"],
        help="Data targets to clean",
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="Total parallel processes to spin up"
    )
    parser.add_argument(
        "--raw-dir", default=DEFAULT_RAW_DIR, help="Root directory of raw input CSVs"
    )
    parser.add_argument(
        "--processed-dir",
        default=DEFAULT_PROCESSED_DIR,
        help="Root directory for normalized parquet output",
    )

    args = parser.parse_args()

    # date validation
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")

    logger.info("Initializing task parameters and pipeline states...")
    tasks = generate_tasks(
        args.symbols, start_dt, end_dt, args.types, args.raw_dir, args.processed_dir
    )

    if not tasks:
        logger.info("All selected files are up to date. No new work needed.")
        return

    logger.info(
        f"Generated {len(tasks)} target normalization jobs. "
        f"Executing across {args.workers} process workers..."
    )

    # processing pool
    failed_tasks = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(execute_normalization, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                succeeded = future.result()
            except Exception as e:
                # worker died unexpectedly (e.g. crash/pickling), not caught inside the task
                succeeded = False
                logger.error(f"Failed for symbol {task['symbol']} on {task['date_str']}: {e}")
            if not succeeded:
                failed_tasks += 1

    # exit with error message if any tasks failed
    if failed_tasks > 0:
        logger.error(f"Completed with {failed_tasks} failure(s). Exiting with error.")
        sys.exit(1)

    logger.info("All normalization jobs completed successfully")


if __name__ == "__main__":
    main()
