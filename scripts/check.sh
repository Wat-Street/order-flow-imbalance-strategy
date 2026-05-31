#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Formatting..."
ruff format .

echo "Linting..."
ruff check .

echo "Testing..."
pytest

echo "Done."
