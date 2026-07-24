from datetime import UTC, datetime

import polars as pl
import pytest

from order_flow_imbalance_strategy.event_alignment import build_alignment_engine

T0 = int(datetime(2023, 1, 1, 9, 0, 0, tzinfo=UTC).timestamp() * 1000)

# The normalized Stage-2 schema stores ``timestamp`` as Datetime("ms"); an
# Int64 -> Datetime("ms") cast interprets the value as epoch-ms, so the T0
# offsets below stay readable while matching the real dtype the engine consumes.
_TS = pl.col("timestamp").cast(pl.Datetime("ms"))


@pytest.fixture
def mock_book_lf():
    return pl.LazyFrame(
        {
            "timestamp": [T0, T0 + 10_000, T0 + 65_000],
            "bid_price": [100.0, 101.0, 102.0],
            "ask_price": [100.5, 101.5, 102.5],
        }
    ).with_columns(_TS)


@pytest.fixture
def mock_klines_lf():
    return pl.LazyFrame({"timestamp": [T0 - 60_000, T0], "close": [99.0, 105.0]}).with_columns(_TS)


@pytest.fixture
def mock_trades_lf():
    return pl.LazyFrame(
        {
            "timestamp": [T0 + 2_000],
            "price": [100.2],
            "quantity": [5.0],
            "buy_qty": [5.0],
            "sell_qty": [0.0],
        }
    ).with_columns(_TS)


def test_alignment_engine_edge_cases(mock_book_lf, mock_trades_lf, mock_klines_lf):
    result = build_alignment_engine(mock_book_lf, mock_trades_lf, mock_klines_lf).collect()

    assert result.height == 66

    # Klines Lookahead Bias
    assert result.item(0, "close") == 99.0
    assert result.item(59, "close") == 99.0
    assert result.item(60, "close") == 105.0
    assert result.item(65, "close") == 105.0

    # book stale flag
    assert result.item(0, "book_stale") is False
    assert result.item(5, "book_stale") is False
    assert result.item(6, "book_stale") is True
    assert result.item(9, "book_stale") is True
    assert result.item(10, "book_stale") is False

    # checking trade defaults and vwap
    assert result.item(2, "no_trades") is False
    assert result.item(2, "volume") == 5.0
    assert result.item(2, "vwap") == 100.2
    assert result.item(3, "no_trades") is True
    assert result.item(3, "volume") == 0.0
    assert result.item(3, "vwap") is None
