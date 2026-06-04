import argparse
import datetime as dt
import sys
from datetime import timedelta
from pathlib import Path
import hashlib
import zipfile
import logging
import logging.config
import concurrent.futures


LOG_CONFIG = {
    "version": 1,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(levelname)s - %(message)s"
        }
    },
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
            "formatter": "standard"
        }
    },
    "loggers": {
        "root_logger": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
        }
    }
}
# Make the logs folder directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger("root_logger")

# define CLI arguments and validate them
def main():
    url = "https://data.binance.vision/?prefix=data/futures/um/daily/"
    symbol_list = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs='+',type=str)
    parser.add_argument("--start", type=dt.datetime.fromisoformat)
    parser.add_argument("--types", help="to download only specific data types")
    parser.add_argument("--end", type=dt.datetime.fromisoformat)
    parser.add_argument("--workers", help = "parallelism", type=int, default=4)
    parser.add_argument("--data-dir", help = "output root")
    parser.add_argument("--validate", action="store_true",help = "run validation test after downloading")
    data_types = []
    args = parser.parse_args()
    if args.start > args.end:
        sys.exit("start date must be before end date")
    #ideas for extra validation to add later: validating if the symbol is part of symbol_list

    #step 2, list of tuples for requested range (data_type, symbol, date_str)
    output = []
    delta = dt.timedelta(days=1)
    for symbol in args.symbols:
        for data_type in args.types:
            current_date = args.start
            while current_date <= args.end:
                date_str = current_date.strftime("%Y-%m-%d")
                output.append((data_type, symbol, date_str))
                current_date += delta

    # step 9, submit tasks to ThreadPoolExecutor to download tasks in parallel
    # data_dir is the root output director, e.g. data/raw, and then the download task function creates the full download path
    # tasks is a list of tuples created in step 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(download_task_function_that_doesn_not_exist_yet, data_dir, data_type, symbol, date_str)
            for data_type, symbol, date_str in tasks
        ]
        
        # we can add tqdm progress bar here 
        for future in concurrent.futures.as_completed(futures):
            result = future.result()


#step 3, check if output CSV already exists
def check_task_exists(data_dir, data_type, symbol, date_str):
    if data_type == "klines":
        kline_path = base / symbol / "klines" / f"{symbol}-1m-{date_str}.csv"
        if kline_path.exists():
            return True
    else:
        file_path = base / symbol / data_type / f"{symbol}-{data_type}-{date_str}.csv"
        if file_path.exists():
            return True
    return False

#step 4, download checksum file before zip
# def download_checksum(data_type, symbol, date_str):
    
# step 6: hash the downloaded zip and compare against the checksum file's expected hash
def validate_checksum(checksum_path, zip_path):
    with open(checksum_path, "r") as checksum_file:
        expected_hash = checksum_file.read().strip().split()[0]
    
    sha256_hash = hashlib.sha256()
    with open(zip_path, "rb") as zip_file:
        while True:
            chunk = zip_file.read(8192)
            if not chunk:
                break
            sha256_hash.update(chunk)
    
    calculated_hash = sha256_hash.hexdigest()
    
    # get zip file size for logging
    size = Path(zip_path).stat().st_size

    if calculated_hash == expected_hash:
        logger.info("Checksum PASSED: zip=%s expected=%s actual=%s zip_size=%s", zip_path, expected_hash, calculated_hash, size)
    else:
        Path(zip_path).unlink()
        logger.error("Checksum FAILED: zip=%s expected=%s actual=%s zip_size=%s; zip file deleted", zip_path, expected_hash, calculated_hash, size)

    return calculated_hash == expected_hash

# step 7: extract the CSV from the zip file
def extract_csv(zip_path, output_directory):
    with zipfile.ZipFile(zip_path, 'r') as zip:
        files = zip.namelist()

        if len(files) == 1:
            file = files[0]

            extracted_path = (Path(output_directory) / file).resolve()
            output = Path(output_directory).resolve()

            if output in extracted_path.parents:
                # Wrap in try-except?
                zip.extract(file, output_directory)
                Path(zip_path).unlink()
                logger.info("Extraction successful: output_directory=%s", output_directory)
                return True
            else:
                logger.error("Unsafe extraction path detected for zip=%s, extracted_path=%s, output_directory=%s; skipping extraction", zip_path, extracted_path, output_directory)
                return False
        else:
            logger.error("Zip contains multiple files, skipping extraction")
            return False