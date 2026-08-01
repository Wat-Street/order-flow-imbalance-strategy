#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
chmod +x scripts/check.sh

python -c "import polars; print(f'Dev container ready (Polars {polars.__version__}).')"
echo "Run ./scripts/check.sh to lint and test."
