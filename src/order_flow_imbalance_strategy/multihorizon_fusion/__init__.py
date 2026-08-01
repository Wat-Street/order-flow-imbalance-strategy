"""Multi-horizon OFI fusion (Stage 7).

Reads spoof-adjusted ``ofi-B`` signal parquets and appends exactly three
exponentially decayed aggregates (``ofi_1m``, ``ofi_5m``, ``ofi_15m``)
computed from ``ofi_clean``. Every Stage-B row and column is preserved.
"""

from order_flow_imbalance_strategy.multihorizon_fusion.compute import (
    REQUIRED_INPUT_COLUMNS,
    STAGE_B_REQUIRED_COLUMNS,
    compute_fusion,
    validate_fusion,
)
from order_flow_imbalance_strategy.multihorizon_fusion.config import (
    DEFAULT_CONFIG,
    FUSION_OUTPUT_COLUMNS,
    FusionConfig,
    HorizonSpec,
    load_config,
)

__all__ = [
    "DEFAULT_CONFIG",
    "FUSION_OUTPUT_COLUMNS",
    "FusionConfig",
    "HorizonSpec",
    "REQUIRED_INPUT_COLUMNS",
    "STAGE_B_REQUIRED_COLUMNS",
    "compute_fusion",
    "load_config",
    "validate_fusion",
]
