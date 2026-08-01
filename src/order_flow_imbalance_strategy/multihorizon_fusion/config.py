from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

FUSION_OUTPUT_COLUMNS: tuple[str, ...] = ("ofi_1m", "ofi_5m", "ofi_15m")
FUSION_SIGNAL_COLUMN = "ofi_clean"
FUSION_TIMESTAMP_COLUMN = "timestamp"
FUSION_GRID_SECONDS = 1


def _validate_column_name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} must not contain leading or trailing whitespace")
    return value


@dataclass(frozen=True, slots=True)
class HorizonSpec:
    """Decay parameters for one fused horizon column."""

    column: str
    half_life_seconds: float
    min_periods: int

    def __post_init__(self) -> None:
        _validate_column_name(self.column, "horizon column")
        if isinstance(self.half_life_seconds, bool) or not isinstance(self.half_life_seconds, Real):
            raise ValueError(f"{self.column}: half_life_seconds must be a finite number")
        if not math.isfinite(float(self.half_life_seconds)) or self.half_life_seconds <= 0:
            raise ValueError(f"{self.column}: half_life_seconds must be finite and positive")
        if isinstance(self.min_periods, bool) or not isinstance(self.min_periods, int):
            raise ValueError(f"{self.column}: min_periods must be an integer")
        if self.min_periods < 1:
            raise ValueError(f"{self.column}: min_periods must be >= 1")

        object.__setattr__(self, "half_life_seconds", float(self.half_life_seconds))


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Full fusion configuration."""

    horizons: tuple[HorizonSpec, ...]
    signal_column: str = FUSION_SIGNAL_COLUMN
    timestamp_column: str = FUSION_TIMESTAMP_COLUMN
    grid_seconds: int = FUSION_GRID_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.horizons, tuple) or not all(
            isinstance(horizon, HorizonSpec) for horizon in self.horizons
        ):
            raise ValueError("horizons must be a tuple of HorizonSpec values")
        if isinstance(self.grid_seconds, bool) or not isinstance(self.grid_seconds, int):
            raise ValueError("grid_seconds must be an integer")
        if self.grid_seconds < 1:
            raise ValueError("grid_seconds must be >= 1")
        if not self.horizons:
            raise ValueError("horizons must not be empty")
        _validate_column_name(self.signal_column, "signal_column")
        _validate_column_name(self.timestamp_column, "timestamp_column")
        if self.signal_column == self.timestamp_column:
            raise ValueError("signal_column and timestamp_column must be different")
        if self.signal_column != FUSION_SIGNAL_COLUMN:
            raise ValueError(f"signal_column must be {FUSION_SIGNAL_COLUMN!r}")
        if self.timestamp_column != FUSION_TIMESTAMP_COLUMN:
            raise ValueError(f"timestamp_column must be {FUSION_TIMESTAMP_COLUMN!r}")
        if self.grid_seconds != FUSION_GRID_SECONDS:
            raise ValueError(f"grid_seconds must be {FUSION_GRID_SECONDS}")
        columns = [h.column for h in self.horizons]
        if len(columns) != len(set(columns)):
            raise ValueError(f"duplicate horizon column names: {columns}")
        if tuple(columns) != FUSION_OUTPUT_COLUMNS:
            raise ValueError(
                f"horizon columns must be exactly {list(FUSION_OUTPUT_COLUMNS)}, got {columns}"
            )
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


def _unexpected_keys(raw: dict[str, Any], allowed: set[str]) -> list[str]:
    """Return unknown config keys, allowing underscore-prefixed metadata."""
    return sorted(key for key in raw if key not in allowed and not key.startswith("_"))


def _horizon_from_dict(raw: dict[str, Any], index: int) -> HorizonSpec:
    unexpected = _unexpected_keys(
        raw,
        {"column", "half_life_seconds", "min_periods"},
    )
    if unexpected:
        raise ValueError(f"horizons[{index}] has unknown keys: {unexpected}")

    missing = [k for k in ("column", "half_life_seconds", "min_periods") if k not in raw]
    if missing:
        raise ValueError(f"horizons[{index}] missing keys: {missing}")
    try:
        return HorizonSpec(
            column=raw["column"],
            half_life_seconds=raw["half_life_seconds"],
            min_periods=raw["min_periods"],
        )
    except ValueError as exc:
        raise ValueError(f"horizons[{index}]: {exc}") from exc


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
    unexpected = _unexpected_keys(
        data,
        {"horizons", "signal_column", "timestamp_column", "grid_seconds"},
    )
    if unexpected:
        raise ValueError(f"{path}: unknown keys: {unexpected}")
    horizons_raw = data.get("horizons")
    if not isinstance(horizons_raw, list) or not horizons_raw:
        raise ValueError(f"{path}: missing or empty 'horizons' list")
    if not all(isinstance(horizon, dict) for horizon in horizons_raw):
        raise ValueError(f"{path}: every horizon must be a JSON object")

    try:
        return FusionConfig(
            horizons=tuple(_horizon_from_dict(h, i) for i, h in enumerate(horizons_raw)),
            signal_column=data.get("signal_column", FUSION_SIGNAL_COLUMN),
            timestamp_column=data.get("timestamp_column", FUSION_TIMESTAMP_COLUMN),
            grid_seconds=data.get("grid_seconds", FUSION_GRID_SECONDS),
        )
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
