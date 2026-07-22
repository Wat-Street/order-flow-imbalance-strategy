from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class HorizonSpec:
    """Decay parameters for one fused horizon column."""

    column: str
    half_life_seconds: float
    min_periods: int

    def __post_init__(self) -> None:
        if not self.column.strip():
            raise ValueError("horizon column must not be empty")
        if self.half_life_seconds <= 0:
            raise ValueError(f"{self.column}: half_life_seconds must be positive")
        if self.min_periods < 1:
            raise ValueError(f"{self.column}: min_periods must be >= 1")


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Full fusion configuration."""

    horizons: tuple[HorizonSpec, ...]
    signal_column: str = "ofi_clean"
    timestamp_column: str = "timestamp"
    grid_seconds: int = 1

    def __post_init__(self) -> None:
        if self.grid_seconds < 1:
            raise ValueError("grid_seconds must be >= 1")
        if not self.horizons:
            raise ValueError("horizons must not be empty")
        if not self.signal_column.strip():
            raise ValueError("signal_column must not be empty")
        if not self.timestamp_column.strip():
            raise ValueError("timestamp_column must not be empty")
        columns = [h.column for h in self.horizons]
        if len(columns) != len(set(columns)):
            raise ValueError(f"duplicate horizon column names: {columns}")
        reserved = {self.signal_column, self.timestamp_column}
        collisions = sorted(reserved.intersection(columns))
        if collisions:
            raise ValueError(f"horizon columns conflict with input columns: {collisions}")

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(h.column for h in self.horizons)


DEFAULT_CONFIG = FusionConfig(
    horizons=(
        HorizonSpec(column="ofi_1m", half_life_seconds=30.0, min_periods=30),
        HorizonSpec(column="ofi_5m", half_life_seconds=150.0, min_periods=150),
        HorizonSpec(column="ofi_15m", half_life_seconds=450.0, min_periods=450),
    ),
)


def _horizon_from_dict(raw: dict[str, Any]) -> HorizonSpec:
    missing = [k for k in ("column", "half_life_seconds", "min_periods") if k not in raw]
    if missing:
        raise ValueError(f"horizon entry missing keys: {missing}")
    return HorizonSpec(
        column=str(raw["column"]),
        half_life_seconds=float(raw["half_life_seconds"]),
        min_periods=int(raw["min_periods"]),
    )


def load_config(path: Path | None = None) -> FusionConfig:
    """Load fusion config from JSON, or return :data:`DEFAULT_CONFIG`."""
    if path is None:
        return DEFAULT_CONFIG

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load fusion config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: config root must be a JSON object")
    horizons_raw = data.get("horizons")
    if not isinstance(horizons_raw, list) or not horizons_raw:
        raise ValueError(f"{path}: missing or empty 'horizons' list")
    if not all(isinstance(horizon, dict) for horizon in horizons_raw):
        raise ValueError(f"{path}: every horizon must be a JSON object")

    return FusionConfig(
        horizons=tuple(_horizon_from_dict(h) for h in horizons_raw),
        signal_column=str(data.get("signal_column", "ofi_clean")),
        timestamp_column=str(data.get("timestamp_column", "timestamp")),
        grid_seconds=int(data.get("grid_seconds", 1)),
    )
