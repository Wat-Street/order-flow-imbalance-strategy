#!/usr/bin/env bash
set -euo pipefail

pip install --upgrade pip
pip install -e ".[dev]"
chmod +x scripts/check.sh

echo "Dev container ready. Run ./scripts/check.sh to lint and test."
