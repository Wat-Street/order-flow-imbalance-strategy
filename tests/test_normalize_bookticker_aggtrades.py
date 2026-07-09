from datetime import datetime

import polars as pl
import pytest

from order_flow_imbalance_strategy.normalize_data import (
    execute_normalization,
    generate_tasks,
    process_agg_trades,
    process_book_ticker,
)


@pytest.fixture
def sample_book_ticker_lazyframe():
    """Generates a mock LazyFrame simulating raw bookTicker CSV input."""
    data = {
        "column_1": [100, 101, 102, 103],
        "column_2": [100.0, 100.5, 0.0, 101.0],  # Row 3 has 0 price (should filter)
        "column_3": [10.0, 5.0, 5.0, 10.0],
        "column_4": [101.0, 101.0, 102.0, 100.5],  # Row 4 has ask < bid (should filter)
        "column_5": [10.0, 5.0, 5.0, 10.0],
        "column_6": [1700000000000, 1700000001000, 1700000002000, 1700000003000],
        "column_7": [1700000000000, 1700000001000, 1700000002000, 1700000003000],
    }
    return pl.DataFrame(data).lazy()


@pytest.fixture
def sample_agg_trades_lazyframe():
    """Generates a mock LazyFrame simulating raw aggTrades CSV input."""
    data = {
        "column_1": [1, 2, 3, 2],
        "column_2": [100.0, 101.0, -5.0, 101.0],
        "column_3": [1.5, 2.0, 1.0, 2.0],
        "column_4": [10, 12, 14, 12],
        "column_5": [11, 13, 15, 13],
        "column_6": [1700000000000, 1700000001000, 1700000002000, 1700000001000],
        "column_7": [True, False, False, False],
    }
    return pl.DataFrame(data).lazy()


# tests for process book ticker
def test_process_book_ticker_transformations(sample_book_ticker_lazyframe):
    symbol = "BTCUSDT"
    res = process_book_ticker(sample_book_ticker_lazyframe, symbol).collect()

    # Verify column layout and count
    assert len(res.columns) == 9
    expected_cols = [
        "timestamp",
        "asset",
        "bid_price",
        "ask_price",
        "bid_qty",
        "ask_qty",
        "mid_price",
        "spread",
        "queue_imbalance",
    ]
    assert res.columns == expected_cols

    # Verify invalid rows (zero prices or negative spreads) were filtered out
    assert res.height == 2  # Only rows 1 and 2 should survive

    # Verify asset name assignment
    assert (res["asset"] == symbol).all()

    # Check calculated metrics for row 1 (bid=100.0, ask=101.0, bid_qty=10.0, ask_qty=10.0)
    assert res["mid_price"][0] == 100.5
    assert res["spread"][0] == 1.0
    assert res["queue_imbalance"][0] == 0.0  # (10 - 10) / (10 + 10) = 0.0


def test_process_book_ticker_queue_imbalance_zero_division():
    """Verify queue_imbalance handles 0/0 safely with fill_nan."""
    data = {
        "column_1": [1],
        "column_2": [100.0],
        "column_3": [0.001],  # Small positive to pass filter
        "column_4": [101.0],
        "column_5": [0.001],
        "column_6": [1700000000000],
        "column_7": [1700000000000],
    }
    lf = pl.DataFrame(data).lazy()
    res = process_book_ticker(lf, "BTCUSDT").collect()

    assert not res["queue_imbalance"].is_nan().any()


# tests for process agg_trades
def test_process_agg_trades_transformations(sample_agg_trades_lazyframe):
    symbol = "ETHUSDT"
    res = process_agg_trades(sample_agg_trades_lazyframe, symbol).collect()

    # Verify column count and layout
    assert len(res.columns) == 8
    expected_cols = [
        "timestamp",
        "asset",
        "price",
        "quantity",
        "side",
        "notional",
        "buy_qty",
        "sell_qty",
    ]
    assert res.columns == expected_cols

    # verify filters & deduplication
    assert res.height == 2

    # check side derivation logic
    assert res["side"].to_list() == ["sell", "buy"]
    assert res["sell_qty"][0] == 1.5
    assert res["buy_qty"][0] == 0.0
    assert res["buy_qty"][1] == 2.0

    # check calculated notional
    assert res["notional"][0] == 100.0 * 1.5


# tests for generation


def test_generate_tasks(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    symbol = "BTCUSDT"
    data_type = "bookTicker"
    date_str = "2024-01-01"

    file_dir = raw_dir / f"{symbol}/{data_type}"
    file_dir.mkdir(parents=True)
    raw_file = file_dir / f"{symbol}-{data_type}-{date_str}.csv"
    raw_file.write_text("header\n")

    start_date = datetime.strptime("2024-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2024-01-01", "%Y-%m-%d")

    tasks = generate_tasks(
        [symbol], start_date, end_date, [data_type], str(raw_dir), str(processed_dir)
    )

    assert len(tasks) == 1
    assert tasks[0]["symbol"] == symbol
    assert tasks[0]["date_str"] == date_str


def test_execute_normalization_skip_low_rowcount(tmp_path):
    """Verify that execute_normalization skips saving parquet if < 100 rows survive."""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir()
    out_dir.mkdir()

    raw_file = raw_dir / "sample.csv"
    out_file = out_dir / "sample.parquet"

    # just write 5 rows
    lines = ["column_1,column_2,column_3,column_4,column_5,column_6,column_7\n"]
    for i in range(5):
        lines.append(
            f"{i},100.0,1.0,101.0,1.0,{1700000000000 + i * 1000},{1700000000000 + i * 1000}\n"
        )
    raw_file.write_text("".join(lines))

    task = {
        "data_type": "bookTicker",
        "symbol": "BTCUSDT",
        "date_str": "2024-01-01",
        "raw_path": str(raw_file),
        "output_path": str(out_file),
        "output_dir": str(out_dir),
    }

    execute_normalization(task)

    # check no output file
    assert not out_file.exists()
