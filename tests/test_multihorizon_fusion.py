from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from order_flow_imbalance_strategy.multihorizon_fusion.cli import (
    build_tasks,
    daterange,
    history_path,
    main,
    parse_symbol,
    process_file,
)
from order_flow_imbalance_strategy.multihorizon_fusion.compute import (
    FUSION_OUTPUT_COLUMNS,
    compute_fusion,
    validate_fusion,
)
from order_flow_imbalance_strategy.multihorizon_fusion.config import (
    DEFAULT_CONFIG,
    FusionConfig,
    HorizonSpec,
    load_config,
)

T0 = datetime(2024, 1, 1, 0, 0, 0)
SIGNAL_SCHEMA = {
    "timestamp": pl.Datetime("us"),
    "ofi_1s": pl.Float64,
    "bid_contribution": pl.Float64,
    "ask_contribution": pl.Float64,
    "spoof_score": pl.Float64,
    "ofi_clean": pl.Float64,
}


def make_signal_frame(rows: list[dict]) -> pl.DataFrame:
    out: list[dict] = []
    for i, row in enumerate(rows):
        rec = {
            "timestamp": T0 + timedelta(seconds=i),
            "ofi_1s": row.get("ofi_1s", 0.0),
            "bid_contribution": row.get("bid_contribution", row.get("ofi_1s", 0.0)),
            "ask_contribution": row.get("ask_contribution", 0.0),
            "spoof_score": row.get("spoof_score", 0.0),
            "ofi_clean": row.get("ofi_clean", row.get("ofi_1s", 0.0)),
        }
        out.append(rec)
    return pl.DataFrame(out, schema=SIGNAL_SCHEMA)


def fuse(rows: list[dict], config: FusionConfig = DEFAULT_CONFIG) -> pl.DataFrame:
    return compute_fusion(make_signal_frame(rows), config)


# --- compute -----------------------------------------------------------------


def test_constant_ofi_clean_produces_same_horizons_after_warmup():
    rows = [{"ofi_clean": 2.0, "ofi_1s": 99.0}] * 500
    out = fuse(rows)
    assert out["ofi_1s"].to_list() == [99.0] * 500
    assert out["ofi_1m"][499] == pytest.approx(2.0, rel=1e-6)
    assert out["ofi_5m"][499] == pytest.approx(2.0, rel=1e-6)
    assert out["ofi_15m"][499] == pytest.approx(2.0, rel=1e-6)


def test_ofi_1s_is_never_modified():
    rows = [
        {"ofi_1s": 1.0, "ofi_clean": 10.0},
        {"ofi_1s": -2.0, "ofi_clean": -20.0},
        {"ofi_1s": 3.5, "ofi_clean": 30.0},
    ]
    src = make_signal_frame(rows)
    out = compute_fusion(src)
    assert out["ofi_1s"].to_list() == src["ofi_1s"].to_list()


def test_horizon_warmup_rows_are_null():
    out = fuse([{"ofi_clean": 1.0}] * 40)
    assert out["ofi_1m"][0] is None
    assert out["ofi_1m"][28] is None
    assert out["ofi_1m"][29] is not None


def test_ewma_matches_known_half_life_values():
    config = FusionConfig(horizons=(HorizonSpec("h", half_life_seconds=1, min_periods=1),))
    out = fuse([{"ofi_clean": 0.0}, {"ofi_clean": 1.0}, {"ofi_clean": 0.0}], config)

    assert out["h"].to_list() == pytest.approx([0.0, 0.5, 0.25])


def test_fusion_is_causal():
    config = FusionConfig(horizons=(HorizonSpec("h", half_life_seconds=2, min_periods=1),))
    rows = [{"ofi_clean": float(value)} for value in (0, 1, 2, 100, -100)]

    full = fuse(rows, config)
    prefix = fuse(rows[:3], config)

    assert full["h"].head(3).equals(prefix["h"])


def test_all_three_horizon_columns_appended():
    src = make_signal_frame([{"ofi_clean": 1.0}] * 1000)
    out = compute_fusion(src)
    assert out.height == src.height
    for col in src.columns:
        assert col in out.columns
    for col in FUSION_OUTPUT_COLUMNS:
        assert col in out.columns


def test_empty_input_preserves_schema_and_adds_horizons():
    src = make_signal_frame([])

    out = compute_fusion(src)

    assert out.height == 0
    assert out.columns == [*src.columns, *FUSION_OUTPUT_COLUMNS]
    assert all(out.schema[column] == pl.Float64 for column in FUSION_OUTPUT_COLUMNS)


def test_unsorted_input_is_sorted_by_timestamp():
    src = make_signal_frame(
        [
            {"ofi_clean": 1.0},
            {"ofi_clean": 3.0},
            {"ofi_clean": 2.0},
        ]
    )
    shuffled = src.sort("timestamp", descending=True)
    out = compute_fusion(shuffled)
    assert out["timestamp"].is_sorted()


def test_gap_resets_horizon_state():
    n = 120
    timestamps: list[datetime] = []
    for i in range(n):
        if i < 60:
            timestamps.append(T0 + timedelta(seconds=i))
        elif i == 60:
            timestamps.append(T0 + timedelta(seconds=360))
        else:
            timestamps.append(T0 + timedelta(seconds=360 + (i - 60)))

    df = make_signal_frame([{"ofi_clean": 10.0}] * n).with_columns(
        pl.Series("timestamp", timestamps)
    )
    out = compute_fusion(df)
    # After the gap, the 1m horizon must warm up again within the new segment.
    assert out["ofi_1m"][60] is None
    assert out["ofi_1m"][88] is None
    assert out["ofi_1m"][89] is not None


def test_null_ofi_clean_is_not_replaced_with_zero():
    df = make_signal_frame([{"ofi_clean": 4.0}] * 80)
    df = df.with_columns(
        pl.when(pl.arange(0, pl.len()) == 40)
        .then(None)
        .otherwise(pl.col("ofi_clean"))
        .alias("ofi_clean")
    )
    out = compute_fusion(df)
    assert out["ofi_1m"][79] == pytest.approx(4.0, rel=1e-6)


def test_null_signal_resets_horizon_state():
    df = make_signal_frame([{"ofi_clean": 4.0}] * 80).with_columns(
        pl.when(pl.arange(0, pl.len()) == 40)
        .then(None)
        .otherwise(pl.col("ofi_clean"))
        .alias("ofi_clean")
    )
    out = compute_fusion(df)
    assert out["ofi_1m"][40] is None
    assert out["ofi_1m"][69] is None
    assert out["ofi_1m"][70] is not None


def test_previous_day_history_warms_first_current_row():
    history = make_signal_frame([{"ofi_clean": 2.0}] * 60)
    current = make_signal_frame([{"ofi_clean": 2.0}] * 10).with_columns(
        pl.col("timestamp") + pl.duration(seconds=60)
    )
    out = compute_fusion(current, history=history)
    assert out["ofi_1m"][0] == pytest.approx(2.0)


def test_history_gap_does_not_warm_current_rows():
    history = make_signal_frame([{"ofi_clean": 2.0}] * 60)
    current = make_signal_frame([{"ofi_clean": 2.0}] * 30).with_columns(
        pl.col("timestamp") + pl.duration(hours=1)
    )
    out = compute_fusion(current, history=history)
    assert out["ofi_1m"][0] is None
    assert out["ofi_1m"][29] is not None


def test_history_must_strictly_precede_current_input():
    current = make_signal_frame([{"ofi_clean": 2.0}] * 3)
    future_history = current.with_columns(pl.col("timestamp") + pl.duration(days=1))

    with pytest.raises(ValueError, match="strictly earlier"):
        compute_fusion(current, history=future_history)


def test_internal_helper_names_in_input_are_preserved():
    src = make_signal_frame([{"ofi_clean": 2.0}] * 30).with_columns(
        pl.lit("upstream value").alias("_current_row")
    )

    out = compute_fusion(src)

    assert out["_current_row"].to_list() == ["upstream value"] * 30


def test_missing_required_column_raises():
    df = pl.DataFrame({"timestamp": [T0], "ofi_1s": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        compute_fusion(df)


def test_existing_output_column_raises():
    df = make_signal_frame([{"ofi_clean": 1.0}]).with_columns(pl.lit(0.0).alias("ofi_1m"))
    with pytest.raises(ValueError, match="already contains fusion output columns"):
        compute_fusion(df)


def test_duplicate_timestamp_raises():
    df = make_signal_frame([{"ofi_clean": 1.0}] * 2).with_columns(pl.lit(T0).alias("timestamp"))
    with pytest.raises(ValueError, match="duplicate"):
        compute_fusion(df)


def test_shorter_horizon_reacts_faster_than_longer():
    rows = [{"ofi_clean": 0.0}] * 200 + [{"ofi_clean": 10.0}] * 800
    out = fuse(rows)
    idx = 700
    assert out["ofi_1m"][idx] is not None
    assert out["ofi_5m"][idx] is not None
    assert out["ofi_15m"][idx] is not None
    assert out["ofi_1m"][idx] > out["ofi_5m"][idx]
    assert out["ofi_5m"][idx] > out["ofi_15m"][idx]


# --- validate ----------------------------------------------------------------


def test_validate_fusion_passes_clean_frame():
    src = make_signal_frame([{"ofi_clean": 1.0}] * 1000)
    out = compute_fusion(src)
    assert validate_fusion(src, out) == []


def test_validate_fusion_flags_modified_ofi_1s():
    src = make_signal_frame([{"ofi_1s": 1.0, "ofi_clean": 1.0}] * 50)
    out = compute_fusion(src).with_columns(pl.lit(999.0).alias("ofi_1s"))
    problems = validate_fusion(src, out)
    assert problems and "ofi_1s" in problems[0]


def test_validate_fusion_flags_row_count_mismatch():
    src = make_signal_frame([{"ofi_clean": 1.0}] * 50)
    out = compute_fusion(src)
    problems = validate_fusion(src, out.head(10))
    assert problems and "row count" in problems[0]


def test_validate_fusion_flags_invalid_spoof_score():
    src = make_signal_frame([{"spoof_score": 1.5, "ofi_clean": 1.0}] * 10)
    out = compute_fusion(src)
    problems = validate_fusion(src, out)
    assert problems and "spoof_score" in problems[0]


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_validate_fusion_flags_non_finite_spoof_score(bad):
    src = make_signal_frame([{"spoof_score": bad, "ofi_clean": 1.0}] * 10)
    out = compute_fusion(src)

    problems = validate_fusion(src, out)

    assert any("non-finite spoof_score" in problem for problem in problems)


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_validate_fusion_flags_non_finite_horizon(bad):
    src = make_signal_frame([{"ofi_clean": 1.0}] * 1000)
    out = compute_fusion(src).with_columns(
        pl.when(pl.arange(0, pl.len()) == 999)
        .then(pl.lit(bad))
        .otherwise(pl.col("ofi_1m"))
        .alias("ofi_1m")
    )
    problems = validate_fusion(src, out)
    assert problems and "non-finite" in problems[0]


def test_validate_fusion_reports_nonnumeric_horizon_without_raising():
    src = make_signal_frame([{"ofi_clean": 1.0}] * 30)
    out = compute_fusion(src).with_columns(pl.lit("bad").alias("ofi_1m"))

    problems = validate_fusion(src, out)

    assert any("ofi_1m must be numeric" in problem for problem in problems)


def test_validate_fusion_requires_exact_output_schema():
    src = make_signal_frame([{"ofi_clean": 1.0}] * 30)
    out = compute_fusion(src).with_columns(pl.lit(1).alias("unexpected"))

    problems = validate_fusion(src, out)

    assert any("output schema" in problem for problem in problems)


# --- config ------------------------------------------------------------------


def test_load_config_from_json(tmp_path: Path):
    cfg_path = tmp_path / "fusion.json"
    cfg_path.write_text(
        """
        {
          "horizons": [
            {"column": "ofi_1m", "half_life_seconds": 20, "min_periods": 10}
          ]
        }
        """,
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert len(cfg.horizons) == 1
    assert cfg.horizons[0].column == "ofi_1m"
    assert cfg.horizons[0].half_life_seconds == 20.0


def test_default_config_json_loads():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "configs" / "multihorizon_fusion.json")
    assert cfg == DEFAULT_CONFIG


def test_horizon_spec_rejects_invalid_half_life():
    with pytest.raises(ValueError, match="half_life_seconds"):
        HorizonSpec(column="x", half_life_seconds=0.0, min_periods=1)


@pytest.mark.parametrize(
    "raw, expected_error",
    [
        (
            '{"grid_seconds": 1.5, "horizons": '
            '[{"column": "x", "half_life_seconds": 1, "min_periods": 1}]}',
            "grid_seconds must be an integer",
        ),
        (
            '{"signal_column": null, "horizons": '
            '[{"column": "x", "half_life_seconds": 1, "min_periods": 1}]}',
            "signal_column must be a non-empty string",
        ),
        (
            '{"horizons": [{"column": "x", "half_life_seconds": "NaN", "min_periods": 1}]}',
            "half_life_seconds must be a finite number",
        ),
        (
            '{"horizons": [{"column": "x", "half_life_seconds": 1, "min_periods": 1.5}]}',
            "min_periods must be an integer",
        ),
        (
            '{"horizons": [{"column": "x", "half_life_seconds": 1, "min_periods": 1, "typo": 2}]}',
            "unknown keys",
        ),
    ],
)
def test_load_config_rejects_invalid_values(tmp_path: Path, raw: str, expected_error: str):
    config_path = tmp_path / "invalid.json"
    config_path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        load_config(config_path)


def test_fusion_config_requires_distinct_signal_and_timestamp_columns():
    with pytest.raises(ValueError, match="must be different"):
        FusionConfig(
            horizons=(HorizonSpec("h", half_life_seconds=1, min_periods=1),),
            signal_column="timestamp",
            timestamp_column="timestamp",
        )


# --- pipeline / IO -----------------------------------------------------------


def test_process_file_writes_and_is_idempotent(tmp_path: Path):
    src = make_signal_frame([{"ofi_clean": 2.0, "ofi_1s": 7.0}] * 1000)
    in_path = tmp_path / "BTCUSDT-ofi-B-2024-01-01.parquet"
    out_path = tmp_path / "BTCUSDT-ofi-C-2024-01-01.parquet"
    src.write_parquet(in_path)

    task = {
        "symbol": "BTCUSDT",
        "date": "2024-01-01",
        "in_path": str(in_path),
        "out_path": str(out_path),
    }

    r1 = process_file(task)
    assert r1["status"] == "ok"
    assert out_path.exists()

    written = pl.read_parquet(out_path)
    assert written.height == 1000
    assert "ofi_1m" in written.columns
    assert written["ofi_1s"][0] == pytest.approx(7.0)

    r2 = process_file(task)
    assert r2["status"] == "skipped"


def test_process_file_missing_input_returns_no_input(tmp_path: Path):
    task = {
        "symbol": "BTCUSDT",
        "date": "2024-01-01",
        "in_path": str(tmp_path / "missing.parquet"),
        "out_path": str(tmp_path / "out.parquet"),
    }
    assert process_file(task)["status"] == "no_input"


def test_process_file_returns_error_for_malformed_task():
    result = process_file({})

    assert result["status"] == "error"
    assert result["symbol"] == "<unknown>"
    assert "KeyError" in result["message"]


def test_process_file_cleans_up_temporary_file_after_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    src = make_signal_frame([{"ofi_clean": 2.0}] * 30)
    in_path = tmp_path / "BTCUSDT-ofi-B-2024-01-01.parquet"
    out_path = tmp_path / "BTCUSDT-ofi-C-2024-01-01.parquet"
    src.write_parquet(in_path)

    def fail_write(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", fail_write)
    result = process_file(
        {
            "symbol": "BTCUSDT",
            "date": "2024-01-01",
            "in_path": str(in_path),
            "out_path": str(out_path),
        }
    )

    assert result["status"] == "error"
    assert "simulated write failure" in result["message"]
    assert not out_path.exists()
    assert not list(tmp_path.glob(f".{out_path.name}.*.tmp"))


def test_build_tasks_includes_missing_inputs_and_deduplicates_symbols(tmp_path: Path):
    from datetime import date

    signals = tmp_path / "signals" / "BTCUSDT"
    signals.mkdir(parents=True)
    (signals / "BTCUSDT-ofi-B-2022-01-01.parquet").write_bytes(b"")
    tasks = build_tasks(["BTCUSDT", "BTCUSDT"], date(2022, 1, 1), date(2022, 1, 2), tmp_path)
    assert len(tasks) == 2
    assert tasks[0]["date"] == "2022-01-01"
    assert tasks[0]["out_path"].endswith("BTCUSDT-ofi-C-2022-01-01.parquet")
    assert tasks[0]["history_path"].endswith("BTCUSDT-ofi-B-2021-12-31.parquet")
    assert tasks[1]["date"] == "2022-01-02"


def test_history_path_is_previous_day(tmp_path: Path):
    from datetime import date

    path = history_path(tmp_path, "BTCUSDT", date(2024, 1, 1))
    assert path.name == "BTCUSDT-ofi-B-2023-12-31.parquet"


def test_daterange_is_inclusive():
    from datetime import date

    assert daterange(date(2022, 1, 1), date(2022, 1, 1)) == [date(2022, 1, 1)]


def test_parse_symbol_normalizes_case_and_rejects_paths():
    assert parse_symbol(" btcusdt ") == "BTCUSDT"
    with pytest.raises(argparse.ArgumentTypeError, match="invalid symbol"):
        parse_symbol("../BTCUSDT")


def test_main_end_before_start_returns_2(tmp_path: Path):
    rc = main(
        [
            "--symbols",
            "BTCUSDT",
            "--start",
            "2022-01-02",
            "--end",
            "2022-01-01",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


def test_main_reports_missing_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    rc = main(
        [
            "--symbols",
            "BTCUSDT",
            "--start",
            "2022-01-01",
            "--end",
            "2022-01-01",
            "--data-dir",
            str(tmp_path),
            "--workers",
            "1",
        ]
    )
    assert rc == 0
    assert "no_input: 1" in capsys.readouterr().out


def test_main_processes_input_with_single_worker(tmp_path: Path):
    signal_dir = tmp_path / "signals" / "BTCUSDT"
    signal_dir.mkdir(parents=True)
    source = make_signal_frame([{"ofi_clean": 2.0}] * 30)
    source.write_parquet(signal_dir / "BTCUSDT-ofi-B-2024-01-01.parquet")

    rc = main(
        [
            "--symbols",
            "btcusdt",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-01",
            "--data-dir",
            str(tmp_path),
            "--workers",
            "1",
        ]
    )

    assert rc == 0
    output = pl.read_parquet(signal_dir / "BTCUSDT-ofi-C-2024-01-01.parquet")
    assert output.height == source.height
    assert output["ofi_1m"][29] == pytest.approx(2.0)


def test_main_rejects_non_positive_workers(tmp_path: Path):
    rc = main(
        [
            "--symbols",
            "BTCUSDT",
            "--start",
            "2022-01-01",
            "--end",
            "2022-01-01",
            "--workers",
            "0",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


def test_main_rejects_invalid_config(tmp_path: Path):
    config_path = tmp_path / "invalid.json"
    config_path.write_text("not-json", encoding="utf-8")
    rc = main(
        [
            "--symbols",
            "BTCUSDT",
            "--start",
            "2022-01-01",
            "--end",
            "2022-01-01",
            "--config",
            str(config_path),
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2
