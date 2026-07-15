"""Trivial end-to-end probe: fetch ONE real Hyperliquid archive file and print
its actual structure, so we can verify the on-disk JSON schema empirically
(rather than trusting docs about the channel/data envelope).

Requires resolvable AWS credentials (the bucket is requester-pays). Run after
`aws login` / `aws sso login` / `aws configure`:

    ./.venv/Scripts/python.exe scripts/probe_hyperliquid_format.py

Optionally override the sample it fetches:

    ... probe_hyperliquid_format.py --coin BTC --date 2024-01-02 --hour 9

It downloads a few MB, decompresses in memory, and prints:
  * the raw first line (truncated),
  * the parsed top-level keys (does it have {"channel","data"}? or bare?),
  * the shape of `levels` (2-element [bids, asks]? each level's keys),
  * line count for the hour.
Nothing is written to disk.
"""

from __future__ import annotations

import argparse
import json

import boto3
import lz4.frame
from botocore.config import Config

BUCKET = "hyperliquid-archive"


def build_key(coin: str, date_str: str, hour: int) -> str:
    ymd = date_str.replace("-", "")
    return f"market_data/{ymd}/{hour}/l2Book/{coin}.lz4"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coin", default="BTC")
    p.add_argument("--date", default="2024-01-02", help="YYYY-MM-DD")
    p.add_argument("--hour", type=int, default=9, help="un-padded hour 0-23")
    p.add_argument("--region", default="ap-northeast-1")
    args = p.parse_args()

    key = build_key(args.coin, args.date, args.hour)
    print(f"Fetching s3://{BUCKET}/{key}  (region={args.region}, requester-pays)\n")

    client = boto3.client(
        "s3",
        region_name=args.region,
        config=Config(connect_timeout=10, read_timeout=120, retries={"max_attempts": 3}),
    )
    resp = client.get_object(Bucket=BUCKET, Key=key, RequestPayer="requester")
    compressed = resp["Body"].read()
    raw = lz4.frame.decompress(compressed)
    text = raw.decode("utf-8")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    print(f"Decompressed: {len(compressed):,} B -> {len(raw):,} B, {len(lines)} non-empty lines\n")

    if not lines:
        print("!! File decompressed but has no lines.")
        return

    first = lines[0]
    print("=== RAW first line (first 500 chars) ===")
    print(first[:500])
    print()

    record = json.loads(first)
    print("=== Top-level type & keys ===")
    print(f"type: {type(record).__name__}")
    if isinstance(record, dict):
        print(f"keys: {sorted(record.keys())}")

    # Locate the payload whether it's enveloped ({channel,data}) or bare.
    data = record.get("data", record) if isinstance(record, dict) else record
    enveloped = isinstance(record, dict) and "data" in record and "channel" in record
    print(f"\nEnveloped ({{'channel','data'}} wrapper)? -> {enveloped}")
    if enveloped:
        print(f"channel value: {record.get('channel')!r}")

    print("\n=== data payload keys ===")
    if isinstance(data, dict):
        print(f"keys: {sorted(data.keys())}")
        print(f"coin: {data.get('coin')!r}   time: {data.get('time')!r}")
        levels = data.get("levels")
        if isinstance(levels, list):
            print(f"\nlevels: list of {len(levels)} arrays (expect 2 -> [bids, asks])")
            for i, side in enumerate(levels):
                name = {0: "bids", 1: "asks"}.get(i, f"side{i}")
                n = len(side) if isinstance(side, list) else "?"
                sample = side[0] if isinstance(side, list) and side else None
                print(
                    f"  levels[{i}] ({name}): {n} entries; first entry keys = "
                    f"{sorted(sample.keys()) if isinstance(sample, dict) else sample}"
                )
                if isinstance(sample, dict):
                    print(f"      sample: {json.dumps(sample)}")
        else:
            print(f"levels present? {levels is not None} (type {type(levels).__name__})")

    print("\n=== Verdict for validate_extracted() ===")
    ok = (
        isinstance(data, dict)
        and isinstance(data.get("levels"), list)
        and len(data["levels"]) == 2
    )
    print(
        "Schema matches expected l2Book (data.levels == [bids, asks])."
        if ok
        else "Schema DID NOT match the expected shape — inspect output above."
    )


if __name__ == "__main__":
    main()
