"""Fetch/upload the aligned Hyperliquid L2 dataset from external hosting.

The aligned Stage-3 Parquet (``data/aligned_hl/``) is too large for git/Git LFS
(~21 GB for full BTC history), so it is hosted externally and synced with this
script instead of committed. Downstream users need NO AWS *archive* credentials —
``download`` pulls over plain HTTPS from a public base URL.

Two roles:

* **download** (anyone) — mirror the published dataset locally. Reads a manifest
  (``manifest.json``) listing each per-day file + size + sha256, then downloads
  any missing/stale files over HTTPS. Verifies checksums.
* **upload** (maintainer) — publish local ``data/aligned_hl/`` to an S3 bucket
  (or any S3-compatible store) and (re)write the manifest. Requires write creds
  for the *hosting* bucket (NOT the requester-pays archive).

Configure the base URL / bucket via ``--base-url`` / ``--bucket`` or the env vars
``HL_DATA_BASE_URL`` / ``HL_DATA_BUCKET``.

Examples::

    # Anyone: fetch the published dataset (no AWS needed)
    python scripts/hl_data.py download --base-url https://<host>/hl-aligned

    # Maintainer: publish local aligned data + manifest to your bucket
    python scripts/hl_data.py upload --bucket my-hl-aligned --prefix v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ALIGNED_DIR = Path("data/aligned_hl")
MANIFEST_NAME = "manifest.json"
CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_parquets(root: Path):
    """Yield (relative_posix_path, absolute_path) for every aligned Parquet."""
    for p in sorted(root.rglob("*.parquet")):
        yield p.relative_to(root).as_posix(), p


# --- download (no AWS) -----------------------------------------------------


def cmd_download(args) -> int:
    base = (args.base_url or os.environ.get("HL_DATA_BASE_URL", "")).rstrip("/")
    if not base:
        sys.exit("Set --base-url or HL_DATA_BASE_URL to the published dataset root.")
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)

    manifest_url = f"{base}/{MANIFEST_NAME}"
    print(f"Fetching manifest: {manifest_url}")
    with urllib.request.urlopen(manifest_url) as resp:  # noqa: S310 - https only
        manifest = json.loads(resp.read().decode("utf-8"))

    files = manifest.get("files", [])
    print(f"Manifest lists {len(files)} files.")
    fetched = skipped = failed = 0
    for entry in files:
        rel, size, digest = entry["path"], entry.get("size"), entry.get("sha256")
        dest = ALIGNED_DIR / rel
        if dest.exists() and (size is None or dest.stat().st_size == size):
            if digest is None or sha256_file(dest) == digest:
                skipped += 1
                continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{base}/{rel}"
        try:
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            with urllib.request.urlopen(url) as r, open(tmp, "wb") as out:  # noqa: S310
                while True:
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
            if digest and sha256_file(tmp) != digest:
                tmp.unlink(missing_ok=True)
                print(f"  CHECKSUM MISMATCH: {rel}")
                failed += 1
                continue
            tmp.replace(dest)
            fetched += 1
            print(f"  fetched {rel}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {rel}: {exc}")
            failed += 1
    print(f"\nDone. fetched={fetched} skipped={skipped} failed={failed}")
    return 1 if failed else 0


# --- upload (maintainer, needs S3 write creds) -----------------------------


def cmd_upload(args) -> int:
    bucket = args.bucket or os.environ.get("HL_DATA_BUCKET", "")
    if not bucket:
        sys.exit("Set --bucket or HL_DATA_BUCKET to your hosting bucket.")
    if not ALIGNED_DIR.exists():
        sys.exit(f"No aligned data at {ALIGNED_DIR}; run the pipeline first.")

    import boto3  # local import so `download` needs no boto3

    s3 = boto3.client("s3")
    prefix = args.prefix.strip("/")

    files = []
    for rel, abspath in iter_parquets(ALIGNED_DIR):
        key = f"{prefix}/{rel}" if prefix else rel
        digest = sha256_file(abspath)
        size = abspath.stat().st_size
        if not args.dry_run:
            s3.upload_file(str(abspath), bucket, key)
        files.append({"path": rel, "size": size, "sha256": digest})
        print(f"  {'(dry) ' if args.dry_run else ''}uploaded {rel} ({size / 1e6:.1f} MB)")

    manifest = {"version": prefix or "root", "files": files}
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    manifest_key = f"{prefix}/{MANIFEST_NAME}" if prefix else MANIFEST_NAME
    if not args.dry_run:
        s3.put_object(Bucket=bucket, Key=manifest_key, Body=manifest_bytes)
    # Also write a local copy for reference.
    (ALIGNED_DIR / MANIFEST_NAME).write_bytes(manifest_bytes)
    print(f"\nManifest: {len(files)} files -> s3://{bucket}/{manifest_key}")
    print("Make the bucket/prefix public-read (or front with CloudFront) so "
          "`download --base-url https://<host>/<prefix>` works with no creds.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch published aligned data (no AWS)")
    d.add_argument("--base-url", default=None, help="public HTTPS root of the dataset")
    d.set_defaults(func=cmd_download)

    u = sub.add_parser("upload", help="publish local aligned data to S3 (maintainer)")
    u.add_argument("--bucket", default=None, help="hosting S3 bucket (write creds)")
    u.add_argument("--prefix", default="", help="key prefix / version, e.g. v1")
    u.add_argument("--dry-run", action="store_true", help="list without uploading")
    u.set_defaults(func=cmd_upload)

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
