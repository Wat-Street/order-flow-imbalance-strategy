from datetime import UTC, datetime

import polars as pl
import pytest

from order_flow_imbalance_strategy import event_alignment as EA
from order_flow_imbalance_strategy.event_alignment import align_one, build_alignment_engine
from order_flow_imbalance_strategy.ofi import REQUIRED_INPUT_COLUMNS, compute_ofi

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
            # bid_qty / ask_qty are part of the normalized bookTicker schema and
            # are required by ofi.py; they must survive the alignment untouched.
            "bid_qty": [10.0, 11.0, 12.0],
            "ask_qty": [20.0, 21.0, 22.0],
        }
    ).with_columns(_TS)


@pytest.fixture
def mock_klines_lf():
    return pl.LazyFrame({"timestamp": [T0 - 60_000, T0], "close": [99.0, 105.0]}).with_columns(_TS)


@pytest.fixture
def mock_trades_lf():
    # cleaned-flow columns (buy_qty/sell_qty/notional) as produced by the wash
    # filter (buy_qty_clean/sell_qty_clean/notional_clean, renamed by the runner).
    # notional = price * quantity = 100.2 * 5 = 501.0, so VWAP recovers 100.2.
    return pl.LazyFrame(
        {
            "timestamp": [T0 + 2_000],
            "buy_qty": [5.0],
            "sell_qty": [0.0],
            "notional": [501.0],
        }
    ).with_columns(_TS)


def test_alignment_engine_edge_cases(mock_book_lf, mock_trades_lf, mock_klines_lf):
    result = build_alignment_engine(mock_book_lf, mock_trades_lf, mock_klines_lf).collect()

    # Full UTC day (00:00:00-23:59:59) minus the seconds before the first native
    # book update at 09:00:00 => 09:00:00..23:59:59 = 15h * 3600 = 54,000 rows.
    assert result.height == 54_000
    # Row 0 is the first native second, so all index-based checks below still hold.

    # Klines Lookahead Bias
    assert result.item(0, "close") == 99.0
    assert result.item(59, "close") == 99.0
    assert result.item(60, "close") == 105.0
    assert result.item(65, "close") == 105.0

    # book_stale flag: True on ANY forward-filled (non-native) second. Native book
    # updates land at T0 (idx 0) and T0+10s (idx 10); every second between is filled.
    assert result.item(0, "book_stale") is False
    assert result.item(1, "book_stale") is True
    assert result.item(5, "book_stale") is True
    assert result.item(9, "book_stale") is True
    assert result.item(10, "book_stale") is False

    # gap_prev_s: whole seconds since the last native update (0 on native seconds).
    assert result.item(0, "gap_prev_s") == 0
    assert result.item(5, "gap_prev_s") == 5
    assert result.item(10, "gap_prev_s") == 0

    # book_stale_prev: whether the previous grid second was stale.
    assert result.item(1, "book_stale_prev") is False  # prev (idx 0) was native
    assert result.item(2, "book_stale_prev") is True  # prev (idx 1) was filled
    assert result.item(10, "book_stale_prev") is True  # prev (idx 9) was filled

    # checking trade defaults and vwap
    assert result.item(2, "no_trades") is False
    assert result.item(2, "volume") == 5.0
    assert result.item(2, "vwap") == 100.2
    assert result.item(3, "no_trades") is True
    assert result.item(3, "volume") == 0.0
    assert result.item(3, "vwap") is None

    # ofi.py requires bid_qty / ask_qty; they must pass through the alignment
    # (carried by the last book update in each second, forward-filled otherwise).
    assert {"bid_qty", "ask_qty"} <= set(result.columns)
    assert result.item(0, "bid_qty") == 10.0
    assert result.item(0, "ask_qty") == 20.0
    assert result.item(10, "bid_qty") == 11.0  # second book update at T0+10s


def test_aligned_output_feeds_compute_ofi():
    """Contract lock: the engine's output must drop straight into ofi.compute_ofi
    and produce real (non-null) OFI on native adjacent seconds. This is what the
    old dt_1s grid key silently broke (it nulled every row)."""
    n = 10  # ten consecutive native book seconds: 09:00:00 .. 09:00:09
    book = pl.LazyFrame(
        {
            "timestamp": [T0 + i * 1000 for i in range(n)],
            "asset": ["BTCUSDT"] * n,
            "bid_price": [100.0 + i * 0.1 for i in range(n)],
            "ask_price": [100.5 + i * 0.1 for i in range(n)],
            "bid_qty": [10.0 + i for i in range(n)],
            "ask_qty": [20.0 + i for i in range(n)],
        }
    ).with_columns(_TS)
    trades = pl.LazyFrame(
        {"timestamp": [T0 + 2000], "buy_qty": [1.0], "sell_qty": [0.0], "notional": [100.0]}
    ).with_columns(_TS)
    klines = pl.LazyFrame({"timestamp": [T0], "close": [100.0]}).with_columns(_TS)

    aligned = build_alignment_engine(book, trades, klines).collect()

    # every column ofi.py requires is present, and timestamp is the 1s grid.
    assert set(REQUIRED_INPUT_COLUMNS) <= set(aligned.columns)
    assert aligned.schema["timestamp"] == pl.Datetime("ms")

    out = compute_ofi(aligned)
    assert "ofi_1s" in out.columns
    # The first n rows are the native seconds; row 0 is OFI warmup (null), the
    # remaining native seconds compute a real value. Filled/stale rows stay null.
    native = out.head(n)
    assert native["ofi_1s"].drop_nulls().len() == n - 1
    assert out.item(0, "ofi_1s") is None  # warmup


def test_align_one_writes_aligned_parquet(tmp_path):
    """Runner I/O: reads normalized/wash-filtered parquet and writes the exact
    artifact ofi.py consumes, renaming the wash filter's cleaned-flow columns."""
    processed = tmp_path / "processed"
    data = tmp_path / "data"
    sym, day = "BTCUSDT", "2023-01-01"
    n = 200

    bp = processed / sym / "bookTicker"
    bp.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": [T0 + i * 1000 for i in range(n)],
            "asset": [sym] * n,
            "bid_price": [100.0] * n,
            "ask_price": [100.5] * n,
            "bid_qty": [10.0] * n,
            "ask_qty": [20.0] * n,
        }
    ).with_columns(_TS).write_parquet(bp / f"{sym}-bookTicker-{day}.parquet")

    tp = processed / sym / "aggTrades_filtered"
    tp.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": [T0 + 1000],
            "buy_qty_clean": [3.0],
            "sell_qty_clean": [1.0],
            "notional_clean": [400.0],
        }
    ).with_columns(_TS).write_parquet(tp / f"{sym}-aggTrades-{day}.parquet")

    kp = processed / sym / "klines"
    kp.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": [T0],
            "close": [100.0],
            "realized_vol": [0.2],
            "ATR": [0.1],
            "vol_regime": ["normal"],
        }
    ).with_columns(_TS).write_parquet(kp / f"{sym}-klines-{day}.parquet")

    res = align_one(str(processed), str(data), sym, day)
    assert res["status"] == "ok"

    out_path = EA.aligned_path(str(data), sym, day)
    assert out_path.exists()
    df = pl.read_parquet(out_path)
    assert {
        "timestamp",
        "bid_price",
        "ask_price",
        "bid_qty",
        "ask_qty",
        "book_stale",
        "signed_volume",  # cleaned trade flow flowed through under the generic name
    } <= set(df.columns)
    # re-running is idempotent (output already present).
    assert align_one(str(processed), str(data), sym, day)["status"] == "skipped"
