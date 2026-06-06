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
