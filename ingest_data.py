import argparse
import datetime as dt
import sys
from datetime import timedelta
from pathlib import Path
import requests

url = "https://data.binance.vision/data/futures/um/daily"
symbol_list = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
types = ["bookTicker", "aggTrades", "klines"]
#step 1, define CLI arguments and validate them
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+",type=str, default=symbol_list)
    parser.add_argument("--start", type=dt.datetime.fromisoformat, required=True)
    parser.add_argument("--types", nargs="+", help="to download only specific data types", default = types, choices=types)
    parser.add_argument("--end", type=dt.datetime.fromisoformat, required=True)
    parser.add_argument("--workers", help = "parallelism", type=int, default=4)
    parser.add_argument("--data-dir", help = "output root", default="data/raw" )
    parser.add_argument("--validate", action="store_true",help = "run validation test after downloading")
    data_types = []
    args = parser.parse_args()
    if args.start > args.end:
        sys.exit("start date must be before end date")
    #ideas for extra validation to add later: validating if the symbol is part of symbol_list
    data_dir = Path(args.data_dir)

    #step 2
    tasks = generate_tasks(args, data_dir)
    print(f"Generated {len(tasks)} tasks to process!")
    #step 3
    for data_type, symbol, date_str in tasks:
        if check_task_exists(data_dir, data_type, symbol, date_str):
            print(f"-> Skipping {symbol} {data_type} for {date_str} (File exists)")
            continue
        hashh = download_checksum(data_dir, data_type, symbol, date_str)
        if hashh is None:
            print(f"-> Skipping {symbol} {data_type} for {date_str} (404 Not Found)")
            continue
        print(f"-> Downloading {symbol} {data_type} for {date_str}...")
        success = download_zip(data_dir, data_type, symbol, date_str)
     
    
#step 2, list of tuples for requested range (data_type, symbol, date_str)
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

#step 3, check if output CSV already exists
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

#step 4, download checksum file before zip
def download_checksum(data_dir, data_type, symbol, date_str):
    base = Path(data_dir)
    if data_type == "klines":
        path_url = f"{url}/klines/{symbol}/1m/{symbol}-1m-{date_str}.zip.CHECKSUM"
    else:
        path_url = f"{url}/{data_type}/{symbol}/{symbol}-{data_type}-{date_str}.zip.CHECKSUM"
    response = requests.get(path_url)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.text.strip().split()[0]


#step 5, download zip file, extract
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
    response = requests.get(zip_url, stream=True)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8*1024*1024):
            if chunk:
                f.write(chunk)
    return zip_path


if __name__ == "__main__":
    main()
