from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from .contracts import BacktestResult
from .risk import RiskMetrics, compute_risk_metrics


def result_document(
    result: BacktestResult,
    risk: RiskMetrics | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "strategy": result.strategy,
        "source": result.source,
        "symbol": result.symbol,
        "model_id": result.model_id,
        "backtest_config": asdict(result.config),
        "trading_config": asdict(result.trading_config),
        "account": asdict(result.account) if result.account is not None else None,
        "decisions": [asdict(item) for item in result.decisions],
        "orders": [asdict(item) for item in result.orders],
        "order_events": [asdict(item) for item in result.order_events],
        "fills": [asdict(item) for item in result.fills],
        "funding": [asdict(item) for item in result.funding],
        "equity": [asdict(item) for item in result.equity],
        "risk": asdict(risk if risk is not None else compute_risk_metrics(result.equity)),
        "metadata": metadata or {},
    }


def write_result_json(
    path: str | Path,
    result: BacktestResult,
    risk: RiskMetrics | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a self-contained report atomically so partial reports are never published."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = result_document(result, risk, metadata)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            delete=False,
            suffix=".tmp",
        ) as temporary:
            temporary_name = temporary.name
            json.dump(document, temporary, default=_json_default, indent=2, sort_keys=True)
            temporary.write("\n")
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def write_comparison_json(
    path: str | Path,
    results: dict[str, BacktestResult],
    metadata: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "metadata": metadata or {},
        "accounts": {
            name: result_document(result, metadata=metadata) for name, result in results.items()
        },
    }
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            delete=False,
            suffix=".tmp",
        ) as temporary:
            temporary_name = temporary.name
            json.dump(document, temporary, default=_json_default, indent=2, sort_keys=True)
            temporary.write("\n")
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def _json_default(value: Any) -> Any:
    if isinstance(value, timedelta):
        return value.total_seconds()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")
