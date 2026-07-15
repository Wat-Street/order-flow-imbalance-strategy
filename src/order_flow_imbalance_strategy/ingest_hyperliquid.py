"""Stage 1 (parallel track) — ingest raw Hyperliquid L2 order-book snapshots.

Sibling to :mod:`order_flow_imbalance_strategy.ingest_data` (the Binance
ingester). It clones the same stage skeleton — eager task list, idempotent
skip, status-dict-never-raise workers, ``ThreadPoolExecutor``, two log handlers
— but swaps the transport:

* **Source** — Hyperliquid's official requester-pays archive::

      s3://hyperliquid-archive/market_data/{YYYYMMDD}/{H}/l2Book/{COIN}.lz4

  Files are hourly, LZ4-frame-compressed newline-delimited JSON, one L2 snapshot
  per line. ``{H}`` is the *un-padded* hour (``.../9/...`` not ``.../09/...``).
  The bucket is requester-pays, so ``RequestPayer='requester'`` is mandatory and
  AWS credentials must be resolvable via the standard boto3 chain.

  Each line's schema (confirmed against a live sample — see
  ``scripts/probe_hyperliquid_format.py``) wraps the WebSocket ``l2Book`` message
  under ``raw`` and adds archive capture metadata::

      {"time": "<ISO-8601 ns capture ts>", "ver_num": 1,
       "raw": {"channel": "l2Book",
               "data": {"coin": "BTC", "time": <epoch-ms>,
                        "levels": [[{"px","sz","n"}, ...],    # bids
                                   [{"px","sz","n"}, ...]]}}}  # asks

* **Integrity** — Hyperliquid publishes no checksums, so the checksum gate of the
  Binance path is replaced by a light *structural* validation: the file
  decompresses cleanly and its first line parses as a non-empty JSON object.

* **Output — raw, full depth, untransformed**, matching the Binance raw layer's
  "store exactly as the source provides it" rule. We decompress (the analog of
  unzipping) and keep the newline-JSON verbatim; normalization to a flat,
  schema-enforced Parquet (``normalize_hyperliquid``, Stage 2) and resampling to
  the 1-second grid (``align_hyperliquid``, Stage 3) are deferred to those
  parallel stages::

      data/raw_hl/{COIN}/l2Book/{COIN}-l2Book-{YYYY-MM-DD}-{HH}.jsonl

Run::

    python -m order_flow_imbalance_strategy.ingest_hyperliquid \\
        --symbols BTC --start 2024-01-01 --end 2024-01-07 --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import logging
import logging.config
import sys
import time
from pathlib import Path

import boto3
import lz4.frame
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
)
from tqdm import tqdm

# Configure logging: console INFO + full-detail DEBUG file (append), mirroring
# the Binance ingester so both tracks log the same way.
LOG_CONFIG = {
    "version": 1,
    # See ingest_data.py: keep sibling modules' loggers alive on import.
    "disable_existing_loggers": False,
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
            "filename": "logs/hyperliquid_download.log",
            "level": "DEBUG",
            "mode": "a",
            "formatter": "standard",
        },
    },
    "loggers": {
        "hyperliquid_logger": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True,
        }
    },
}
Path("logs").mkdir(exist_ok=True)
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger("hyperliquid_logger")

# --- constants -------------------------------------------------------------

BUCKET = "hyperliquid-archive"
# The archive bucket lives in ap-northeast-1. Pin it so ingestion works
# regardless of whether a user has a default region configured.
BUCKET_REGION = "ap-northeast-1"
DATA_TYPE = "l2Book"
DEFAULT_COINS = ["BTC"]
HOURS = range(24)
MIN_JSONL_SIZE = 2  # bytes; a decompressed file smaller than this is empty/corrupt
MAX_ATTEMPTS = 4
CHUNK_SIZE = 8 * 1024 * 1024
# S3 object-not-found codes we treat as "missing" (an expected archive gap), not
# an error to retry. HeadObject reports 404; GetObject reports NoSuchKey.
MISSING_ERROR_CODES = {"NoSuchKey", "404", "NoSuchBucket"}


def make_s3_client():
    """Build an S3 client. Retries/timeouts are set defensively; the worker adds
    its own outer backoff loop for logging + missing-key classification."""
    return boto3.client(
        "s3",
        region_name=BUCKET_REGION,
        config=Config(
            connect_timeout=10,
            read_timeout=300,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


CREDS_HELP = (
    "AWS credentials are required — the Hyperliquid archive bucket is "
    "requester-pays, so each user must authenticate with their own AWS "
    "account (fetches bill to that account, ~$0.09/GB egress).\n"
    "Set up credentials via ANY one of:\n"
    "  1. aws login            (keyless SSO session; needs: pip install 'botocore[crt]')\n"
    "  2. aws configure        (an IAM user's access key; scope: AmazonS3ReadOnlyAccess)\n"
    "  3. env vars             AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY\n"
    "Then verify with:  aws sts get-caller-identity\n"
    "Only needed to *fetch* raw data — downstream stages read the committed "
    "Parquet and need no AWS."
)


def preflight_credentials() -> None:
    """Fail fast with an actionable message if AWS credentials are missing or
    invalid, instead of surfacing a raw boto3 traceback mid-download. Uses STS
    ``get_caller_identity`` (a free, permission-less identity echo)."""
    sts = boto3.client("sts", region_name=BUCKET_REGION)
    try:
        identity = sts.get_caller_identity()
    except NoCredentialsError:
        logger.error("No AWS credentials found.")
        sys.exit(f"\nNo AWS credentials found.\n\n{CREDS_HELP}")
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        logger.error("AWS credentials rejected: %s", code)
        sys.exit(f"\nAWS credentials were found but rejected ({code}).\n\n{CREDS_HELP}")
    except BotoCoreError as exc:
        # e.g. the botocore[crt] MissingDependencyException for `aws login`.
        logger.error("Could not resolve AWS credentials: %s", exc)
        sys.exit(f"\nCould not resolve AWS credentials:\n  {exc}\n\n{CREDS_HELP}")
    logger.info("AWS identity OK: %s", identity.get("Arn", "<unknown>"))


# --- path / key builders ---------------------------------------------------


def s3_key(coin: str, date_str: str, hour: int) -> str:
    """Archive object key. Note the date is compact (``YYYYMMDD``) and the hour is
    un-padded, per Hyperliquid's layout (``market_data/20230916/9/l2Book/SOL.lz4``)."""
    ymd = date_str.replace("-", "")
    return f"market_data/{ymd}/{hour}/{DATA_TYPE}/{coin}.lz4"


def output_path(data_dir, coin: str, date_str: str, hour: int) -> Path:
    """Local raw output. Hour is zero-padded here (our choice) for lexical sort."""
    return Path(data_dir) / coin / DATA_TYPE / f"{coin}-{DATA_TYPE}-{date_str}-{hour:02d}.jsonl"


# --- pipeline steps --------------------------------------------------------


def generate_tasks(args) -> list[tuple[str, str, int]]:
    """Eager list of ``(coin, date_str, hour)`` for every hour in the range, so
    tqdm has an accurate total from the start."""
    output: list[tuple[str, str, int]] = []
    delta = dt.timedelta(days=1)
    for coin in args.symbols:
        current_date = args.start
        while current_date <= args.end:
            date_str = current_date.strftime("%Y-%m-%d")
            for hour in HOURS:
                output.append((coin, date_str, hour))
            current_date += delta
    return output


def check_task_exists(data_dir, coin: str, date_str: str, hour: int) -> bool:
    """True if a non-trivial output file already exists (resumable skip). A
    smaller-than-``MIN_JSONL_SIZE`` file is a crash artifact and is re-downloaded."""
    path = output_path(data_dir, coin, date_str, hour)
    if path.exists() and path.stat().st_size >= MIN_JSONL_SIZE:
        return True
    if path.exists():
        logger.debug(
            "Stale/undersized output, will re-download: path=%s size=%s bytes",
            path,
            path.stat().st_size,
        )
    return False


def download_and_extract(client, data_dir, coin: str, date_str: str, hour: int):
    """Download the hourly ``.lz4`` object and stream-decompress it to ``.jsonl``.

    Returns the written :class:`~pathlib.Path` on success, or ``None`` when the
    object does not exist in the archive (an expected gap). Writes atomically via
    a ``.tmp`` sibling so a partial file never satisfies the idempotent skip.
    Raises after exhausting retries on genuine transient failures.
    """
    key = s3_key(coin, date_str, hour)
    out_path = output_path(data_dir, coin, date_str, hour)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            time.sleep(0.1)  # small rate-limiting cushion before the network call
            response = client.get_object(Bucket=BUCKET, Key=key, RequestPayer="requester")
            body = response["Body"]
            decompressor = lz4.frame.LZ4FrameDecompressor()
            with open(tmp_path, "wb") as f:
                for chunk in body.iter_chunks(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(decompressor.decompress(chunk))
            tmp_path.replace(out_path)
            logger.info(
                "Downloaded: coin=%s date=%s hour=%s size=%.2f MB",
                coin,
                date_str,
                hour,
                out_path.stat().st_size / (1024 * 1024),
            )
            return out_path
        except ClientError as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            code = exc.response.get("Error", {}).get("Code", "")
            if code in MISSING_ERROR_CODES:
                logger.info(
                    "Archive object not found: coin=%s date=%s hour=%s (%s)",
                    coin,
                    date_str,
                    hour,
                    code,
                )
                return None
            if attempt == MAX_ATTEMPTS:
                logger.exception(
                    "S3 download failed after retries: coin=%s date=%s hour=%s",
                    coin,
                    date_str,
                    hour,
                )
                raise
            logger.warning(
                "S3 client error, retrying: coin=%s date=%s hour=%s attempt=%s code=%s",
                coin,
                date_str,
                hour,
                attempt,
                code,
            )
            time.sleep(2**attempt)
        except (BotoCoreError, OSError):
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == MAX_ATTEMPTS:
                logger.exception(
                    "S3 download failed after retries: coin=%s date=%s hour=%s",
                    coin,
                    date_str,
                    hour,
                )
                raise
            logger.warning(
                "S3 request error, retrying: coin=%s date=%s hour=%s attempt=%s",
                coin,
                date_str,
                hour,
                attempt,
            )
            time.sleep(2**attempt)


def validate_extracted(path: Path) -> bool:
    """Structural integrity gate (the checksum analog): the file is non-empty and
    its first line matches the real Hyperliquid archive l2Book schema.

    Schema confirmed by fetching a live sample (see
    ``scripts/probe_hyperliquid_format.py``). Each line is::

        {"time": "<ISO-8601 ns capture ts>",   # archive capture timestamp
         "ver_num": 1,
         "raw": {"channel": "l2Book",
                 "data": {"coin": "<COIN>",
                          "time": <epoch-ms event time>,
                          "levels": [[{px,sz,n}, ...],   # bids
                                     [{px,sz,n}, ...]]}}} # asks

    The docs (and the info-API) describe a flatter shape; the archive actually
    nests the WebSocket message under ``raw`` and adds its own capture ``time``
    and ``ver_num``. We assert the load-bearing invariant: ``raw.data.levels`` is
    a 2-element ``[bids, asks]`` array. On failure the corrupt file is deleted so
    the next run re-downloads it.
    """
    try:
        with open(path, encoding="utf-8") as f:
            first_line = f.readline().strip()
        if not first_line:
            raise ValueError("empty file")
        record = json.loads(first_line)
        if not isinstance(record, dict) or not record:
            raise ValueError("first line is not a non-empty JSON object")
        data = record.get("raw", {}).get("data")
        if not isinstance(data, dict):
            raise ValueError("missing raw.data object")
        levels = data.get("levels")
        if not (isinstance(levels, list) and len(levels) == 2):
            raise ValueError("raw.data.levels is not a 2-element [bids, asks] array")
        return True
    except (ValueError, AttributeError, OSError) as exc:
        logger.error("Validation FAILED, deleting: path=%s — %s", path, exc)
        path.unlink(missing_ok=True)
        return False


def process_task(data_dir, coin: str, date_str: str, hour: int, client=None) -> dict:
    """Worker: ingest one coin-date-hour. Never raises; always returns a status
    dict. ``status`` is one of ``ok/skipped/missing/malformed/error``."""
    try:
        if check_task_exists(data_dir, coin, date_str, hour):
            logger.debug("Skipped %s %s %s (exists)", coin, date_str, hour)
            return {"status": "skipped"}
        client = client or make_s3_client()
        out_path = download_and_extract(client, data_dir, coin, date_str, hour)
        if out_path is None:
            return {"status": "missing"}
        if not validate_extracted(out_path):
            return {"status": "malformed"}
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001 - workers must never propagate
        logger.error("Error processing %s %s %s: %s", coin, date_str, hour, exc, exc_info=True)
        return {"status": "error"}


def run_validation(data_dir, args) -> None:
    """Re-check every expected output file structurally; report a summary."""
    checked = 0
    issues = 0
    delta = dt.timedelta(days=1)
    for coin in args.symbols:
        current = args.start
        while current <= args.end:
            date_str = current.strftime("%Y-%m-%d")
            current += delta
            for hour in HOURS:
                path = output_path(data_dir, coin, date_str, hour)
                if not path.exists():
                    continue
                if validate_extracted(path):
                    checked += 1
                else:
                    issues += 1
    print(f"\nValidation complete: {checked} files checked, {issues} issues found.")


# --- CLI -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Ingest raw Hyperliquid L2 order-book snapshots from the S3 archive"
    )
    parser.add_argument("--symbols", nargs="+", type=str, default=DEFAULT_COINS)
    parser.add_argument("--start", type=dt.datetime.fromisoformat, required=True)
    parser.add_argument("--end", type=dt.datetime.fromisoformat, required=True)
    parser.add_argument("--workers", help="parallelism", type=int, default=4)
    parser.add_argument("--data-dir", help="output root", default="data/raw_hl")
    parser.add_argument(
        "--validate", action="store_true", help="re-validate outputs after downloading"
    )
    args = parser.parse_args()
    if args.start > args.end:
        logger.error("Invalid date range: start=%s end=%s", args.start, args.end)
        sys.exit("start date must be before end date")

    # Fail fast on missing/invalid credentials before spawning the pool, so users
    # get an actionable message instead of a traceback after tasks start.
    preflight_credentials()

    data_dir = Path(args.data_dir)
    tasks = generate_tasks(args)
    logger.info("Generated %s tasks to process", len(tasks))

    results = {"ok": 0, "skipped": 0, "missing": 0, "malformed": 0, "error": 0}
    logger.info("Submitting tasks to thread pool with %s workers", args.workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_task, data_dir, coin, date_str, hour)
            for coin, date_str, hour in tasks
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Hyperliquid L2"
        ):
            results[future.result()["status"]] += 1
    logger.info("Download process complete. Summary: %s", results)

    if args.validate:
        run_validation(data_dir, args)


if __name__ == "__main__":
    main()
