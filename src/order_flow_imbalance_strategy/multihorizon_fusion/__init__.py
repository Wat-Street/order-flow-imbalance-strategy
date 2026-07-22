"""Multi-horizon OFI fusion (Stage 7).

Reads spoof-adjusted ``ofi-B`` signal parquets and appends exponentially
decayed horizon aggregates (``ofi_1m``, ``ofi_5m``, ``ofi_15m``) computed
from ``ofi_clean``. Raw ``ofi_1s`` is never modified.
"""

from order_flow_imbalance_strategy.multihorizon_fusion.compute import (
    FUSION_OUTPUT_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
    compute_fusion,
    validate_fusion,
)
from order_flow_imbalance_strategy.multihorizon_fusion.config import (
    DEFAULT_CONFIG,
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
    "compute_fusion",
    "load_config",
    "validate_fusion",
]
