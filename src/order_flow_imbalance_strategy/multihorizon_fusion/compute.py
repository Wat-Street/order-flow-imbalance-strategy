from __future__ import annotations

import polars as pl

from order_flow_imbalance_strategy.multihorizon_fusion.config import (
    DEFAULT_CONFIG,
    FusionConfig,
)

#: Columns that must be present in an ``ofi-B`` input frame.
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "ofi_1s",
    "bid_contribution",
    "ask_contribution",
    "spoof_score",
)

FUSION_OUTPUT_COLUMNS: tuple[str, ...] = DEFAULT_CONFIG.output_columns


def _validate_input_schema(df: pl.DataFrame, config: FusionConfig) -> None:
    required = tuple(
        dict.fromkeys((*REQUIRED_INPUT_COLUMNS, config.timestamp_column, config.signal_column))
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"input frame missing required columns: {missing}")

    conflicting = [c for c in config.output_columns if c in df.columns]
    if conflicting:
        raise ValueError(f"input frame already contains fusion output columns: {conflicting}")

    ts = df[config.timestamp_column]
    if ts.dtype.base_type() != pl.Datetime:
        raise ValueError(f"timestamp column must be Datetime, got {ts.dtype}")
    if ts.null_count():
        raise ValueError("timestamp column contains null values")
    if ts.n_unique() != df.height:
        raise ValueError("timestamp column contains duplicate values")

    numeric_columns = tuple(dict.fromkeys((*REQUIRED_INPUT_COLUMNS, config.signal_column)))
    nonnumeric = [column for column in numeric_columns if not df[column].dtype.is_numeric()]
    if nonnumeric:
        details = ", ".join(f"{column}={df[column].dtype}" for column in nonnumeric)
        raise ValueError(f"input columns must be numeric: {details}")

    signal = df[config.signal_column]
    non_finite = signal.is_finite().not_() & signal.is_not_null()
    if non_finite.any():
        raise ValueError(f"signal column contains {int(non_finite.sum())} non-finite values")


def _segment_id_expr(timestamp_col: str, signal_col: str, grid_seconds: int) -> pl.Expr:
    """Increment state whenever the time grid or valid signal stream is broken."""
    previous_signal_is_null = pl.col(signal_col).shift(1).is_null()
    return (
        (
            (pl.col(timestamp_col).diff() != pl.duration(seconds=grid_seconds))
            | pl.col(signal_col).is_null()
            | previous_signal_is_null
        )
        .fill_null(True)
        .cast(pl.UInt32)
        .cum_sum()
        .alias("_segment_id")
    )


def compute_fusion(
    df: pl.DataFrame,
    config: FusionConfig = DEFAULT_CONFIG,
    history: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Append multi-horizon exponentially decayed aggregates of ``ofi_clean``.

    Returns a new frame with every original column preserved (same order, same
    row count) plus one column per configured horizon. ``ofi_1s`` and all other
    input columns are copied verbatim — fusion reads only ``ofi_clean``.

    ``history`` may contain preceding rows (normally the previous day) used to
    warm the causal filters without being emitted. The current input is sorted
    by timestamp in the result.
    """
    _validate_input_schema(df, config)

    original_columns = df.columns
    ts = config.timestamp_column
    signal = config.signal_column

    current = df.select(ts, pl.col(signal).cast(pl.Float64))
    frames = [current]
    if history is not None and not history.is_empty():
        _validate_input_schema(history, config)
        if history.schema[ts] != df.schema[ts]:
            raise ValueError(
                "history timestamp dtype must match current input "
                f"({history.schema[ts]} != {df.schema[ts]})"
            )
        if not df.is_empty() and history[ts].max() >= df[ts].min():
            raise ValueError("history timestamps must be strictly earlier than current input")
        history_frame = history.select(ts, pl.col(signal).cast(pl.Float64))
        frames.insert(0, history_frame)

    working = pl.concat(frames).sort(ts)
    if working[ts].n_unique() != working.height:
        raise ValueError("current input and history contain overlapping timestamps")
    segment_column = "_fusion_segment_id"
    while segment_column in {ts, signal, *config.output_columns}:
        segment_column = f"_{segment_column}"
    working = working.with_columns(
        _segment_id_expr(ts, signal, config.grid_seconds).alias(segment_column)
    )

    horizon_exprs: list[pl.Expr] = []
    for spec in config.horizons:
        # Causal EWMA; grouping by segment resets state at time-grid gaps
        # and invalid signal samples so discontinuities never bridge horizons.
        horizon_exprs.append(
            pl.col(signal)
            .ewm_mean(
                half_life=spec.half_life_seconds / config.grid_seconds,
                adjust=False,
                min_samples=spec.min_periods,
                ignore_nulls=True,
            )
            .over(segment_column)
            .alias(spec.column)
        )

    fused = working.with_columns(horizon_exprs).select(ts, *config.output_columns)
    return (
        df.join(fused, on=ts, how="left", validate="1:1")
        .sort(ts)
        .select(*original_columns, *config.output_columns)
    )


def validate_fusion(
    input_df: pl.DataFrame,
    output_df: pl.DataFrame,
    config: FusionConfig = DEFAULT_CONFIG,
) -> list[str]:
    """Return hard validation problems; empty means the frame is safe to write."""
    problems: list[str] = []

    if output_df.height != input_df.height:
        problems.append(f"row count {output_df.height} != input {input_df.height}")

    expected_columns = [*input_df.columns, *config.output_columns]
    if output_df.columns != expected_columns:
        problems.append(
            f"output schema must preserve input columns and append horizons: {expected_columns}"
        )

    expected_input = input_df.sort(config.timestamp_column)
    for col in input_df.columns:
        if col not in output_df.columns:
            problems.append(f"missing preserved input column {col!r}")
            continue
        if not expected_input[col].equals(output_df[col]):
            problems.append(f"input column {col!r} was modified during fusion")

    missing_outputs = [c for c in config.output_columns if c not in output_df.columns]
    if missing_outputs:
        problems.append(f"missing fusion output columns: {missing_outputs}")

    validated_input_columns = tuple(dict.fromkeys((*REQUIRED_INPUT_COLUMNS, config.signal_column)))
    for col in validated_input_columns:
        if col not in input_df.columns:
            problems.append(f"missing required input column {col!r}")
            continue
        series = input_df[col]
        if not series.dtype.is_numeric():
            problems.append(f"{col} must be numeric, got {series.dtype}")
            continue
        non_finite = series.is_finite().not_() & series.is_not_null()
        if non_finite.any():
            problems.append(f"{int(non_finite.sum())} non-finite {col} values (inf/NaN)")

    if "spoof_score" in input_df.columns and input_df["spoof_score"].dtype.is_numeric():
        spoof = input_df["spoof_score"]
        spoof_finite = spoof.is_finite() & spoof.is_not_null()
        out_of_range = spoof_finite & ((spoof < 0.0) | (spoof > 1.0))
        if out_of_range.any():
            problems.append(f"{int(out_of_range.sum())} spoof_score values outside [0, 1]")

    for col in config.output_columns:
        if col not in output_df.columns:
            continue
        series = output_df[col]
        if not series.dtype.is_numeric():
            problems.append(f"{col} must be numeric, got {series.dtype}")
            continue
        non_finite = series.is_finite().not_() & series.is_not_null()
        if non_finite.any():
            problems.append(f"{int(non_finite.sum())} non-finite {col} values (inf/NaN)")

    return problems
