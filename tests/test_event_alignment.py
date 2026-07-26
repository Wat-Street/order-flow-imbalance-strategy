from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from order_flow_imbalance_strategy import event_alignment as EA
from order_flow_imbalance_strategy.event_alignment import (
    build_alignment_engine,
    build_day_grid,
    validate_aligned,
)
from order_flow_imbalance_strategy.ofi import REQUIRED_INPUT_COLUMNS, compute_ofi

T0 = int(datetime(2023, 1, 1, 9, 0, 0, tzinfo=UTC).timestamp() * 1000)
DAY = datetime(2023, 1, 1).date()
BASE_DT = datetime(2023, 1, 1, 9, 0, 0)  # naive, matches the naive UTC grid
FULL_DAY_ROWS = 86_400  # 1s grid; also == len(build_day_grid(DAY))

# The normalized Stage-2 schema stores ``timestamp`` as Datetime("ms"); an
# Int64 -> Datetime("ms") cast interprets the value as epoch-ms, so the T0
# offsets below stay readable while matching the real dtype the engine consumes.
_TS = pl.col("timestamp").cast(pl.Datetime("ms"))


def _at(df: pl.DataFrame, sec: int) -> dict:
    """Row at BASE_DT + ``sec`` seconds (by timestamp, not positional index)."""
    sub = df.filter(pl.col("timestamp") == BASE_DT + timedelta(seconds=sec))
    assert sub.height == 1, f"expected one row at +{sec}s, got {sub.height}"
    return sub.row(0, named=True)


@pytest.fixture
def mock_book_lf():
    # Full normalized bookTicker schema (incl. derived mid/spread/queue_imbalance).
    return pl.LazyFrame(
        {
            "timestamp": [T0, T0 + 10_000, T0 + 65_000],
            "bid_price": [100.0, 101.0, 102.0],
            "ask_price": [100.5, 101.5, 102.5],
            "bid_qty": [10.0, 11.0, 12.0],
            "ask_qty": [20.0, 21.0, 22.0],
            "mid_price": [100.25, 101.25, 102.25],
            "spread": [0.5, 0.5, 0.5],
            "queue_imbalance": [-10 / 30, -10 / 32, -10 / 34],
        }
    ).with_columns(_TS)


@pytest.fixture
def mock_klines_lf():
    # open_time-indexed 1-min bars: the bar opening at T0 (close 105) must not
    # attach to seconds inside its own minute (that would be lookahead).
    return pl.LazyFrame({"timestamp": [T0 - 60_000, T0], "close": [99.0, 105.0]}).with_columns(_TS)


@pytest.fixture
def mock_trades_lf():
    # cleaned-flow columns as fed by the runner (buy_qty/sell_qty/notional).
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
    result = build_alignment_engine(mock_book_lf, mock_trades_lf, mock_klines_lf, day=DAY).collect()

    # Full UTC-day grid, built from the calendar (missingness lives in flags).
    assert result.height == FULL_DAY_ROWS

    # Klines anti-lookahead: the bar opening at 09:00 (close 105) is only known at
    # 09:01; seconds inside 09:00 carry the *previous* completed bar (close 99).
    assert _at(result, 0)["close"] == 99.0
    assert _at(result, 30)["close"] == 99.0
    assert _at(result, 59)["close"] == 99.0
    assert _at(result, 60)["close"] == 105.0
    assert _at(result, 65)["close"] == 105.0

    # book_stale (strict): True on ANY forward-filled second. Native updates land
    # at 09:00:00 and 09:00:10; every second between is a fill.
    assert _at(result, 0)["book_stale"] is False
    assert _at(result, 1)["book_stale"] is True
    assert _at(result, 5)["book_stale"] is True
    assert _at(result, 10)["book_stale"] is False

    # gap_prev_s: whole seconds since the last native update (0 on native seconds).
    assert _at(result, 0)["gap_prev_s"] == 0
    assert _at(result, 5)["gap_prev_s"] == 5
    assert _at(result, 10)["gap_prev_s"] == 0

    # Capped forward-fill: within the 5s horizon the quote is carried (stale but
    # populated); beyond it the levels are nulled and book_dead is set.
    assert _at(result, 5)["book_dead"] is False
    assert _at(result, 5)["bid_price"] == 100.0  # carried from 09:00:00
    assert _at(result, 6)["book_dead"] is True
    assert _at(result, 6)["bid_price"] is None  # ancient quote not trusted

    # Trades (flow): summed, 0-filled on empty seconds, never carried.
    r2 = _at(result, 2)
    assert r2["no_trades"] is False
    assert r2["buy_qty"] == 5.0
    assert r2["sell_qty"] == 0.0
    assert r2["notional"] == 501.0
    assert r2["trade_count"] == 1
    assert r2["signed_volume"] == 5.0
    assert r2["volume"] == 5.0
    assert r2["vwap"] == 100.2
    r3 = _at(result, 3)
    assert r3["no_trades"] is True
    assert r3["buy_qty"] == 0.0
    assert r3["trade_count"] == 0
    assert r3["volume"] == 0.0
    assert r3["vwap"] is None  # undefined with no trades; NOT zero-filled

    # Full book schema passes through (ofi.py requires bid_qty/ask_qty).
    r0 = _at(result, 0)
    assert r0["bid_qty"] == 10.0
    assert r0["ask_qty"] == 20.0
    assert r0["mid_price"] == 100.25
    assert r0["spread"] == 0.5
    assert _at(result, 10)["bid_qty"] == 11.0


def test_unbounded_carry_when_horizon_disabled(mock_book_lf, mock_trades_lf, mock_klines_lf):
    # max_book_staleness_s <= 0 disables the cap: the quote is carried indefinitely
    # (still flagged book_stale) and never nulled/book_dead past a horizon.
    result = build_alignment_engine(
        mock_book_lf, mock_trades_lf, mock_klines_lf, day=DAY, max_book_staleness_s=0
    ).collect()
    assert _at(result, 6)["book_dead"] is False
    assert _at(result, 6)["bid_price"] == 100.0  # carried, not nulled
    assert _at(result, 6)["book_stale"] is True  # still a fill


def test_aligned_output_feeds_compute_ofi():
    """Contract lock: the engine's output drops straight into ofi.compute_ofi and
    yields real (non-null) OFI on native adjacent seconds -- what the old dt_1s
    grid key silently broke (it nulled every row)."""
    n = 10  # native book update every second, 09:00:00 .. 09:00:09
    book = pl.LazyFrame(
        {
            "timestamp": [T0 + i * 1000 for i in range(n)],
            "bid_price": [100.0 + i * 0.1 for i in range(n)],
            "ask_price": [100.5 + i * 0.1 for i in range(n)],
            "bid_qty": [10.0 + i for i in range(n)],
            "ask_qty": [20.0 + i for i in range(n)],
            "mid_price": [100.25 + i * 0.1 for i in range(n)],
            "spread": [0.5] * n,
            "queue_imbalance": [0.0] * n,
        }
    ).with_columns(_TS)
    trades = pl.LazyFrame(
        {"timestamp": [T0 + 2000], "buy_qty": [1.0], "sell_qty": [0.0], "notional": [100.0]}
    ).with_columns(_TS)
    klines = pl.LazyFrame({"timestamp": [T0], "close": [100.0]}).with_columns(_TS)

    aligned = build_alignment_engine(book, trades, klines, day=DAY).collect()

    assert set(REQUIRED_INPUT_COLUMNS) <= set(aligned.columns)
    assert aligned.schema["timestamp"] == pl.Datetime("ms")

    out = compute_ofi(aligned)
    assert "ofi_1s" in out.columns
    # Only the interior native seconds (09:00:01 .. 09:00:09) yield OFI: 09:00:00's
    # previous second is book_dead, and everything after 09:00:09 is a stale fill.
    assert out.filter(pl.col("ofi_1s").is_not_null()).height == n - 1


def test_align_one_writes_full_schema_parquet(tmp_path):
    """Runner I/O + validation: reads normalized/wash-filtered parquet and writes
    the exact artifact ofi.py consumes, with the full book+flow+context schema."""
    processed = tmp_path / "processed"
    data = tmp_path / "data"
    sym, day = "BTCUSDT", "2023-01-01"
    n = 300

    bp = processed / sym / "bookTicker"
    bp.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": [T0 + i * 1000 for i in range(n)],
            "bid_price": [100.0] * n,
            "ask_price": [100.5] * n,
            "bid_qty": [10.0] * n,
            "ask_qty": [20.0] * n,
            "mid_price": [100.25] * n,
            "spread": [0.5] * n,
            "queue_imbalance": [-1 / 3] * n,
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

    res = align_one_ok = EA.align_one(str(processed), str(data), sym, day)
    assert res["status"] == "ok"
    assert align_one_ok["rows"] == FULL_DAY_ROWS  # validation asserted the row count

    out_path = EA.aligned_path(str(data), sym, day)
    assert out_path.exists()
    df = pl.read_parquet(out_path)
    assert {
        "timestamp",
        "asset",
        "bid_price",
        "ask_price",
        "bid_qty",
        "ask_qty",
        "mid_price",
        "spread",
        "queue_imbalance",
        "buy_qty",
        "sell_qty",
        "notional",
        "trade_count",
        "signed_volume",
        "close",
        "realized_vol",
        "vol_regime",
        "book_stale",
        "book_dead",
        "no_trades",
    } <= set(df.columns)
    # cleaned trade flow flowed through under the generic names.
    assert df.filter(pl.col("trade_count") > 0)["signed_volume"].item() == 2.0  # 3 buy - 1 sell
    # re-running is idempotent (output already present).
    assert EA.align_one(str(processed), str(data), sym, day)["status"] == "skipped"


def test_align_one_falls_back_to_raw_trades_with_warning(tmp_path, caplog):
    """When the wash-filtered dir is absent, align falls back to raw aggTrades and
    warns -- rather than silently dropping every trade."""
    processed = tmp_path / "processed"
    data = tmp_path / "data"
    sym, day = "BTCUSDT", "2024-01-01"
    n = 300
    day_ms = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)

    bp = processed / sym / "bookTicker"
    bp.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": [day_ms + i * 1000 for i in range(n)],
            "bid_price": [100.0] * n,
            "ask_price": [100.5] * n,
            "bid_qty": [10.0] * n,
            "ask_qty": [20.0] * n,
            "mid_price": [100.25] * n,
            "spread": [0.5] * n,
            "queue_imbalance": [-1 / 3] * n,
        }
    ).with_columns(_TS).write_parquet(bp / f"{sym}-bookTicker-{day}.parquet")

    # raw normalized aggTrades only -- NO aggTrades_filtered dir.
    ap = processed / sym / "aggTrades"
    ap.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": [day_ms + 1000, day_ms + 2000],
            "buy_qty": [3.0, 0.0],
            "sell_qty": [0.0, 2.0],
            "notional": [300.0, 200.0],
        }
    ).with_columns(_TS).write_parquet(ap / f"{sym}-aggTrades-{day}.parquet")

    with caplog.at_level("WARNING"):
        res = EA.align_one(str(processed), str(data), sym, day)

    assert res["status"] == "ok"
    assert "falling back to UNWASHED" in caplog.text
    df = pl.read_parquet(EA.aligned_path(str(data), sym, day))
    assert df.filter(pl.col("trade_count") > 0).height == 2  # both raw trades survived


def _fake_aligned(nrows: int = 3) -> pl.DataFrame:
    """Minimal frame that passes validate_aligned (book_dead everywhere, so the
    null level columns are allowed)."""
    ts = [datetime(2023, 1, 1, 0, 0, i) for i in range(nrows)]
    return pl.DataFrame(
        {
            "timestamp": ts,
            **{c: [None] * nrows for c in EA.BOOK_LEVEL_COLUMNS},
            "buy_qty": [0.0] * nrows,
            "sell_qty": [0.0] * nrows,
            "notional": [0.0] * nrows,
            "trade_count": [0] * nrows,
            "volume": [0.0] * nrows,
            "book_dead": [True] * nrows,
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))


def test_validate_aligned_accepts_and_rejects():
    good = _fake_aligned(3)
    validate_aligned(good, expected_rows=3)  # no raise

    with pytest.raises(AssertionError, match="row count"):
        validate_aligned(good, expected_rows=999)

    negative = _fake_aligned(3).with_columns(
        pl.Series("buy_qty", [-1.0, 0.0, 0.0])  # a flow can never be negative
    )
    with pytest.raises(AssertionError, match="negative flow"):
        validate_aligned(negative, expected_rows=3)


def test_build_day_grid_count():
    assert len(build_day_grid(DAY)) == FULL_DAY_ROWS
    assert len(build_day_grid(DAY, grid="5s")) == FULL_DAY_ROWS // 5
