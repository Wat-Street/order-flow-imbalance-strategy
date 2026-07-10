from datetime import date

import polars as pl

from order_flow_imbalance_strategy.klines import (
    LOOKBACK_BARS,
    OUTPUT_COLUMNS,
    execute_normalization,
    load_day_with_context,
    process_klines,
)

# real Binance klines CSV header (only the columns the normalizer consumes)
KLINES_CSV_HEADER = "open_time,open,high,low,close,volume\n"

DAY = date(2024, 1, 1)
DAY_START_MS = 1704067200000  # 2024-01-01 00:00:00 UTC in ms
MINUTE_MS = 60_000


def _klines_rows(n, start_ms=DAY_START_MS, base=100.0):
    """Generate n valid 1m OHLCV rows (open_time in ms) as CSV text lines."""
    lines = []
    for i in range(n):
        ts = start_ms + i * MINUTE_MS
        o = base + i * 0.1
        lines.append(f"{ts},{o:.4f},{o + 1:.4f},{o - 1:.4f},{o + 0.5:.4f},10.0\n")
    return lines


def _write_klines_csv(path, n, start_ms=DAY_START_MS, base=100.0):
    path.write_text(KLINES_CSV_HEADER + "".join(_klines_rows(n, start_ms, base)))


def _sample_lazyframe(n, start_ms=DAY_START_MS, base=100.0):
    rows = [r.strip().split(",") for r in _klines_rows(n, start_ms, base)]
    return pl.DataFrame(
        {
            "open_time": [int(r[0]) for r in rows],
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "volume": [float(r[5]) for r in rows],
        }
    ).lazy()


def test_process_klines_transformations():
    res = process_klines(_sample_lazyframe(30), "BTCUSDT", thresholds=None, keep_date=DAY).collect()

    assert list(res.columns) == list(OUTPUT_COLUMNS)
    assert res.height == 30
    assert (res["asset"] == "BTCUSDT").all()
    # first bar has no prior close -> null log_return; later bars are populated
    assert res["log_return"][0] is None
    assert res["log_return"][1] is not None
    # rolling realized_vol is populated once the window (20) is filled
    assert res["realized_vol"][-1] is not None
    # no thresholds provided -> vol_regime stays null
    assert res["vol_regime"].null_count() == res.height


def test_process_klines_filters_invalid_ohlc():
    lf = _sample_lazyframe(5)
    # corrupt one row so high < low (must be filtered out)
    df = lf.collect()
    df[2, "high"] = 1.0
    df[2, "low"] = 99999.0
    res = process_klines(df.lazy(), "BTCUSDT", thresholds=None, keep_date=DAY).collect()
    assert res.height == 4


def test_vol_regime_labeling_with_thresholds():
    thresholds = {"BTCUSDT": {"p25": 0.0, "p75": 1e-9}}
    res = process_klines(_sample_lazyframe(30), "BTCUSDT", thresholds, keep_date=DAY).collect()
    labels = set(res["vol_regime"].drop_nulls().unique().to_list())
    assert labels <= {"low", "normal", "high"}


def test_load_day_with_context_prepends_prior_tail(tmp_path):
    raw_dir = tmp_path / "raw"
    kdir = raw_dir / "BTCUSDT" / "klines"
    kdir.mkdir(parents=True)

    prev = kdir / "BTCUSDT-1m-2023-12-31.csv"
    curr = kdir / "BTCUSDT-1m-2024-01-01.csv"
    _write_klines_csv(prev, 50, start_ms=DAY_START_MS - 50 * MINUTE_MS)
    _write_klines_csv(curr, 30, start_ms=DAY_START_MS)

    lf = load_day_with_context(curr, raw_dir, "BTCUSDT", DAY)
    # current day (30) + only the last LOOKBACK_BARS of the prior day
    assert lf.collect().height == 30 + LOOKBACK_BARS


def test_execute_normalization_writes_parquet(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed" / "BTCUSDT" / "klines"
    kdir = raw_dir / "BTCUSDT" / "klines"
    kdir.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    raw_file = kdir / "BTCUSDT-1m-2024-01-01.csv"
    out_file = out_dir / "BTCUSDT-klines-2024-01-01.parquet"
    _write_klines_csv(raw_file, 150)

    task = {
        "symbol": "BTCUSDT",
        "date_str": "2024-01-01",
        "keep_date": DAY.isoformat(),
        "raw_path": str(raw_file),
        "output_path": str(out_file),
        "output_dir": str(out_dir),
        "raw_dir": str(raw_dir),
        "thresholds": None,
    }

    assert execute_normalization(task) is True
    assert out_file.exists()

    result = pl.read_parquet(out_file)
    assert result.height == 150
    assert list(result.columns) == list(OUTPUT_COLUMNS)


def test_execute_normalization_skip_low_rowcount(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    kdir = raw_dir / "BTCUSDT" / "klines"
    kdir.mkdir(parents=True)
    out_dir.mkdir()

    raw_file = kdir / "BTCUSDT-1m-2024-01-01.csv"
    out_file = out_dir / "out.parquet"
    _write_klines_csv(raw_file, 5)

    task = {
        "symbol": "BTCUSDT",
        "date_str": "2024-01-01",
        "keep_date": DAY.isoformat(),
        "raw_path": str(raw_file),
        "output_path": str(out_file),
        "output_dir": str(out_dir),
        "raw_dir": str(raw_dir),
        "thresholds": None,
    }

    # a low-rowcount file is an intentional skip, not a failure
    assert execute_normalization(task) is True
    assert not out_file.exists()


def test_execute_normalization_reports_failure(tmp_path):
    """A malformed/unreadable input must return False (not silently 'succeed')."""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    kdir = raw_dir / "BTCUSDT" / "klines"
    kdir.mkdir(parents=True)
    out_dir.mkdir()

    raw_file = kdir / "BTCUSDT-1m-2024-01-01.csv"
    out_file = out_dir / "out.parquet"
    raw_file.write_text("foo,bar,baz\n1,2,3\n")  # wrong schema -> should fail, not skip

    task = {
        "symbol": "BTCUSDT",
        "date_str": "2024-01-01",
        "keep_date": DAY.isoformat(),
        "raw_path": str(raw_file),
        "output_path": str(out_file),
        "output_dir": str(out_dir),
        "raw_dir": str(raw_dir),
        "thresholds": None,
    }

    assert execute_normalization(task) is False
    assert not out_file.exists()
