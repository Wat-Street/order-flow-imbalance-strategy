# Order Flow Imbalance (OFI) Trading System

This project is a full-stack quantitative research pipeline that investigates whether Order Flow Imbalance (OFI), enhanced with machine learning, contains statistically significant predictive power in liquid crypto markets.

The system is designed end-to-end: from raw market microstructure data ingestion, through feature engineering and regime modeling, to ML-based prediction and walk-forward backtesting.

---

## Core Idea

Order Flow Imbalance captures short-term pressure between buyers and sellers in the order book. While OFI is known to have predictive power at very short horizons, it is noisy and regime-dependent.

This project builds a context-aware ML system that improves OFI using:

- Multi-horizon signal fusion (1m, 5m, 15m)
- Time-decay modeling of microstructure signals
- Market regime detection (volatility-based)
- Feature-rich order book + trade flow representation

## Goals

- Determine whether ML-enhanced OFI has predictive power in crypto markets
- Build a fully reproducible microstructure research pipeline
- Evaluate robustness across regimes and time periods

## Status

Early-stage development (Phase 1: Data Ingestion + Pipeline Foundation)

## Development

Use the dev container for a consistent environment (see [CONTRIBUTING.md](CONTRIBUTING.md)), or install locally with `pip install -e ".[dev]"`. Run `./scripts/check.sh` before opening a PR.

## Data & AWS setup

### Just running the pipeline? No AWS-archive creds needed.

The downstream-ready dataset (Hyperliquid BTC L2 book, normalized and aligned to a
1-second grid) is **hosted externally**, not committed to git (it is ~21 GB for
full history — too large for git/LFS). Fetch it over plain HTTPS:

```bash
python scripts/hl_data.py download --base-url https://<published-dataset-host>
```

That mirrors `data/aligned_hl/BTC/BTC-l2-aligned-{date}.parquet` (per-day files)
locally — verified by checksum against a manifest — which the deep-OFI feature and
depth-study stages read directly. **No AWS credentials are required** to download
(the requester-pays archive is only touched when *regenerating* the data).

### The Hyperliquid L2 pipeline (maintainers only)

The L2 track mirrors the Binance/L1 pipeline's stage structure (Ingestion →
Normalization → Event Alignment). Raw L2 snapshots come from Hyperliquid's official
S3 archive (`s3://hyperliquid-archive/market_data/...`). That bucket is
**requester-pays**, so fetching bills the caller's own AWS account (~$0.09/GB
egress; the full BTC history is roughly 1 GB compressed / a few cents). Only the
Stage-3 aligned output is committed; `data/raw_hl/` and `data/processed_hl/` are
gitignored intermediates.

**1. Set up AWS credentials** (any one of these; boto3 resolves them automatically):

```bash
aws login                 # keyless SSO session (needs botocore[crt], installed by our deps)
# or
aws configure             # an IAM user's access key; scope: AmazonS3ReadOnlyAccess
# or export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
aws sts get-caller-identity   # verify it resolves
```

Prefer a scoped IAM user or `aws login` over root access keys. The ingester runs
a preflight check and exits with setup instructions if credentials are missing.

**Stage 1 — Ingest** raw hourly snapshots (LZ4 JSONL, stored verbatim):

```bash
python -m order_flow_imbalance_strategy.ingest_hyperliquid \
    --symbols BTC --start 2024-01-01 --end 2024-01-07 --workers 6
```

**Stage 2 — Normalize** to schema-enforced per-day Parquet (casts types, validates
the 20-level ladder, drops defect rows, computes derived top-of-book fields):

```bash
python -m order_flow_imbalance_strategy.normalize_hyperliquid \
    --symbols BTC --start 2024-01-01 --end 2024-01-07
```

**Stage 3 — Align** to the 1-second grid (86,400 rows/day, forward-fill with
`book_stale` / `levels_valid` flags for the OFI stage). This is the published
artifact:

```bash
python -m order_flow_imbalance_strategy.align_hyperliquid \
    --symbols BTC --start 2024-01-01 --end 2024-01-07
```

This writes `data/aligned_hl/BTC/BTC-l2-aligned-{date}.parquet` (all 20 book levels
per side, plus derived fields and alignment flags). Publish it to external hosting
for downstream users with `python scripts/hl_data.py upload --bucket <your-bucket>`
(writes a checksummed manifest); it is not committed to git.

To empirically verify the archive file format at any time:
`python scripts/probe_hyperliquid_format.py`.

> **Windows ARM64 note:** install `polars-lts-cpu` instead of `polars` (the
> default wheel crashes on import with a missing-`sse3` error).
