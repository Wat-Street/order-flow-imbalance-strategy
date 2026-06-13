import argparse
import concurrent.futures
import datetime as dt
import hashlib
import logging
import logging.config
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# step 8: configure logging
LOG_CONFIG = {
    "version": 1,
    "formatters": {"standard": {"format": "%(asctime)s - %(levelname)s - %(message)s"}},
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "stream": "ext://sys.stdout",
            "formatter": "standard",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "logs/download.log",
            "level": "DEBUG",
            "mode": "a",
            "formatter": "standard",
        },
    },
    "loggers": {
        "root_logger": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True,
        }
    },
}
# Make the logs folder directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger("root_logger")

# define CLI arguments and validate them
url = "https://data.binance.vision/data/futures/um/daily"
symbol_list = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
types = ["bookTicker", "aggTrades", "klines"]


# step 1, define CLI arguments and validate them
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", type=str, default=symbol_list)
    parser.add_argument("--start", type=dt.datetime.fromisoformat, required=True)
    parser.add_argument(
        "--types",
        nargs="+",
        help="to download only specific data types",
        default=types,
        choices=types,
    )
    parser.add_argument("--end", type=dt.datetime.fromisoformat, required=True)
    parser.add_argument("--workers", help="parallelism", type=int, default=4)
    parser.add_argument("--data-dir", help="output root", default="data/raw")
    parser.add_argument(
        "--validate", action="store_true", help="run validation test after downloading"
    )
    args = parser.parse_args()
    if args.start > args.end:
        sys.exit("start date must be before end date")
    data_dir = Path(args.data_dir)

    # step 2
    tasks = generate_tasks(args, data_dir)
    print(f"Generated {len(tasks)} tasks to process!")

    # step 9, submit tasks to ThreadPoolExecutor to download tasks in parallel
    results = {"ok": 0, "skipped": 0, "missing": 0, "checksum_failed": 0, "error": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_task, data_dir, data_type, symbol, date_str)
            for data_type, symbol, date_str in tasks
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading data"
        ):
            result = future.result()
            status = result["status"]
            results[status] += 1
        print(f"\nDownload process complete. Summary: {results}")
    if args.validate:
        run_validation(data_dir, args)


# step 2, list of tuples for requested range (data_type, symbol, date_str)
def generate_tasks(args, data_dir):
    output = []
    delta = dt.timedelta(days=1)
    for symbol in args.symbols:
        for data_type in args.types:
            current_date = args.start
            while current_date <= args.end:
                date_str = current_date.strftime("%Y-%m-%d")
                output.append((data_type, symbol, date_str))
                current_date += delta
    return output


# step 3, check if output CSV already exists
def check_task_exists(data_dir, data_type, symbol, date_str):
    base = Path(data_dir)
    if data_type == "klines":
        kline_path = base / symbol / "klines" / f"{symbol}-1m-{date_str}.csv"
        if kline_path.exists():
            return True
    else:
        file_path = base / symbol / data_type / f"{symbol}-{data_type}-{date_str}.csv"
        if file_path.exists():
            return True
    return False


# step 4
def download_checksum(data_dir, data_type, symbol, date_str):
    if data_type == "klines":
        path_url = f"{url}/klines/{symbol}/1m/{symbol}-1m-{date_str}.zip.CHECKSUM"
    else:
        path_url = f"{url}/{data_type}/{symbol}/{symbol}-{data_type}-{date_str}.zip.CHECKSUM"
    MAX_ATTEMPTS = 4
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            time.sleep(0.1)  # Added rate-limiting cushion before network call
            response = requests.get(path_url)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.text.strip().split()[0]
        except requests.RequestException:
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(2**attempt)


# step 5, download zip file and compute hash concurrently
def download_zip(data_dir, data_type, symbol, date_str):
    base = Path(data_dir) / symbol / data_type
    base.mkdir(parents=True, exist_ok=True)
    if data_type == "klines":
        zip_filename = f"{symbol}-1m-{date_str}.zip"
        zip_url = f"{url}/klines/{symbol}/1m/{zip_filename}"
    else:
        zip_filename = f"{symbol}-{data_type}-{date_str}.zip"
        zip_url = f"{url}/{data_type}/{symbol}/{zip_filename}"
    zip_path = base / zip_filename
    MAX_RETRIES = 4
    response = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(0.1)  # Added rate-limiting cushion before network call
            response = requests.get(zip_url, stream=True)
            if response.status_code == 404:
                return None, None
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2**attempt)
    sha256_hash = hashlib.sha256()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                f.write(chunk)
                sha256_hash.update(chunk)
    return zip_path, sha256_hash.hexdigest()


# step 6: evaluate calculated hash against the checksum file's expected hash
def validate_checksum(expected_hash, calculated_hash, zip_path):
    # get zip file size for logging
    size = Path(zip_path).stat().st_size

    if calculated_hash == expected_hash:
        logger.info(
            "Checksum PASSED: zip=%s expected=%s actual=%s zip_size=%s",
            zip_path,
            expected_hash,
            calculated_hash,
            size,
        )
    else:
        Path(zip_path).unlink()
        logger.error(
            "Checksum FAILED: zip=%s expected=%s actual=%s zip_size=%s; zip file deleted",
            zip_path,
            expected_hash,
            calculated_hash,
            size,
        )

    return calculated_hash == expected_hash


# step 7: extract the CSV from the zip file
def extract_csv(zip_path, output_directory):
    with zipfile.ZipFile(zip_path, "r") as zipped_file:
        files = zipped_file.namelist()

        if len(files) != 1:
            logger.error(
                "Zip contains multiple files, skipping extraction, zip=%s, files=%s",
                zip_path,
                files,
            )
            return False
        file = files[0]

        extracted_path = (Path(output_directory) / file).resolve()
        output = Path(output_directory).resolve()

        if output not in extracted_path.parents:
            logger.error(
                "Unsafe extraction path detected for zip=%s, extracted_path=%s, "
                "output_directory=%s; skipping extraction",
                zip_path,
                extracted_path,
                output_directory,
            )
            return False
        zipped_file.extract(file, output_directory)

    Path(zip_path).unlink()

    logger.info("Extraction successful: output_directory=%s", output_directory)
    return True


COLUMNS = {
    "klines": [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "num_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ],
    "aggTrades": [
        "agg_trade_id",
        "price",
        "qty",
        "first_trade_id",
        "last_trade_id",
        "transact_time",
        "is_buyer_maker",
    ],
    "bookTicker": [
        "update_id",
        "best_bid_price",
        "best_bid_qty",
        "best_ask_price",
        "best_ask_qty",
        "transaction_time",
        "event_time",
    ],
}


def run_validation(data_dir, args):

    issues = 0
    checked = 0
    delta = dt.timedelta(days=1)
    for symbol in args.symbols:
        for data_type in args.types:
            current = args.start
            while current <= args.end:
                date_str = current.strftime("%Y-%m-%d")
                current += delta
                if data_type == "klines":
                    csv_path = data_dir / symbol / "klines" / f"{symbol}-1m-{date_str}.csv"
                else:
                    csv_path = (
                        data_dir / symbol / data_type / f"{symbol}-{data_type}-{date_str}.csv"
                    )
                if not csv_path.exists():
                    continue
                cols = COLUMNS[data_type]
                try:
                    df_raw = pd.read_csv(csv_path, nrows=5, header=None)
                    assert len(df_raw.columns) == len(cols), (
                        f"Column count mismatch: expected {len(cols)}, got {len(df_raw.columns)}"
                    )
                    df = df_raw.copy()
                    df.columns = cols
                    assert len(df) > 0, "File is empty"
                    checked += 1
                    logger.debug("Validation OK: %s", csv_path)
                except Exception as e:
                    logger.error("Validation FAILED: %s — %s", csv_path, e)
                    issues += 1
    print(f"\nValidation complete: {checked} files checked, {issues} issues found.")


def process_task(data_dir, data_type, symbol, date_str):
    try:
        if check_task_exists(data_dir, data_type, symbol, date_str):
            logger.debug(f"Skipped {symbol} {data_type} {date_str} (exists)")
            return {"status": "skipped"}
        expected_hash = download_checksum(data_dir, data_type, symbol, date_str)
        if expected_hash is None:
            logger.debug(f"Missing {symbol} {data_type} {date_str} (404)")
            return {"status": "missing"}
        logger.info(f"Downloading {symbol} {data_type} {date_str}...")
        zip_path, calculated_hash = download_zip(data_dir, data_type, symbol, date_str)
        if zip_path is None:
            logger.warning(
                f"Zip 404'd after checksum passed for {symbol} {data_type} {date_str} — skipping"
            )
            return {"status": "missing"}
        if not validate_checksum(expected_hash, calculated_hash, zip_path):
            return {"status": "checksum_failed"}
        base = Path(data_dir) / symbol / data_type
        if extract_csv(zip_path, base):
            return {"status": "ok"}
        else:
            return {"status": "error"}
    except Exception as e:
        logger.error(f"Error processing {symbol} {data_type} {date_str}: {str(e)}", exc_info=True)
        return {"status": "error"}


if __name__ == "__main__":
    main()
