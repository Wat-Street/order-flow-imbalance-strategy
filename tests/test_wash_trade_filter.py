import datetime as dt
from types import SimpleNamespace

import polars as pl
import pytest

from order_flow_imbalance_strategy import filter_wash_trades
from order_flow_imbalance_strategy.filter_wash_trades import (
    generate_tasks,
    match_trades_to_price,
    zero_price_impact,
    off_touch_execution,
    ping_pong_reversal,
    duplicate_prints,
    size_clustering,
    process_task,
)

T0 = dt.datetime(2024, 1, 1)


def _ts(offset_ms):
    return T0 + dt.timedelta(milliseconds=offset_ms)


@pytest.fixture
def suspicion_fixture():
    # Shared dataframe for all 5 scoring functions: 500 'boring' background
    # rows (uniform price/qty/side, tight bid/ask bracket, 1s apart) so the
    # 500-row rolling windows have a baseline, plus 22 synthetic rows
    # (idx 500-521) that isolate one suspicion case
    offsets = []
    price = []
    quantity = []
    side = []
    mid_price = []
    bid_before = []
    ask_before = []
    bid_after = []
    ask_after = []

    def add(offset, p, q, s, mid, bb=99.5, ab=100.5, ba=99.5, aa=100.5):
        offsets.append(offset)
        price.append(p)
        quantity.append(q)
        side.append(s)
        mid_price.append(mid)
        bid_before.append(bb)
        ask_before.append(ab)
        bid_after.append(ba)
        ask_after.append(aa)

    # rows 0-499: background
    for i in range(500):
        add(i * 1000, 100.0, 1.0, "buy", 100.0)

    # row 500: zero_price_impact CASE A - large trade, no future move -> score 1.0
    add(500_000, 100.0, 3.0, "buy", 100.0)
    # row 501: zero_price_impact CASE C - small trade -> score 0.0; also
    # doubles as the "next" row for row 500's horizon shift
    add(501_000, 100.0, 1.0, "buy", 100.0)
    # row 502: filler - "next" row for row 501, sets a big mid jump
    add(502_000, 100.0, 1.0, "buy", 250.0)
    # row 503: zero_price_impact CASE B - large trade, real price move -> score 0.0
    add(503_000, 100.0, 3.0, "buy", 250.0)
    # row 504: filler - "next" row for row 503, sets a big mid drop
    add(504_000, 100.0, 1.0, "buy", 100.0)
    # row 505: off_touch CASE above bracket, clips to 1.0
    add(505_000, 102.0, 1.0, "buy", 100.0)
    # row 506: off_touch CASE below bracket, clips to 1.0
    add(506_000, 98.0, 1.0, "buy", 100.0)
    # row 507: off_touch CASE partial excursion, fractional unclipped score
    add(507_000, 100.51, 1.0, "buy", 100.0)
    # row 508: ping_pong predecessor row -> score 0.0
    add(508_000, 100.0, 1.0, "buy", 100.0)
    # row 509: ping_pong CASE match - flipped side, same price/qty, 100ms later
    add(508_100, 100.0, 1.0, "sell", 100.0)
    # row 510: duplicate CASE unique -> score 0.0
    add(510_000, 100.0, 1.0, "buy", 100.0)
    # rows 511-512: duplicate CASE pair -> score 0.5 each
    add(511_000, 100.0, 1.0, "buy", 100.0)
    add(511_000, 100.0, 1.0, "buy", 100.0)
    # rows 513-515: duplicate CASE triple -> score 0.6667 each
    add(512_000, 100.0, 1.0, "buy", 100.0)
    add(512_000, 100.0, 1.0, "buy", 100.0)
    add(512_000, 100.0, 1.0, "buy", 100.0)
    # rows 516-517: duplicate CASE side-mismatch - same ts/price/qty, flipped
    # side -> not counted as duplicates, and delta_ms=0 so not a ping-pong match either
    add(513_000, 100.0, 1.0, "buy", 100.0)
    add(513_000, 100.0, 1.0, "sell", 100.0)
    # row 518: size_clustering CASE large-but-unique -> score 0.0
    add(514_000, 100.0, 5.0, "buy", 100.0)
    # rows 519-520: size_clustering CASE large-and-repeated -> score ~0.4286 each
    add(515_000, 100.0, 7.0, "buy", 100.0)
    add(516_000, 100.0, 7.0, "buy", 100.0)
    # row 521: off_touch CASE null bracket component -> filled to 0.0
    add(517_000, 100.0, 1.0, "buy", 100.0, bb=None)

    return pl.DataFrame(
        {
            "timestamp": [_ts(o) for o in offsets],
            "price": price,
            "quantity": quantity,
            "side": side,
            "mid_price": mid_price,
            "bid_before": bid_before,
            "ask_before": ask_before,
            "bid_after": bid_after,
            "ask_after": ask_after,
        }
    )


# tests for generate_tasks
def test_generate_tasks_1():
    # builds (symbol, date_str) pairs for every symbol x every day in an inclusive date range
    args = SimpleNamespace(
        symbols=["BTCUSDT", "ETHUSDT"],
        start=dt.datetime(2024, 1, 1),
        end=dt.datetime(2024, 1, 3),
    )
    assert generate_tasks(args) == [
        ("BTCUSDT", "2024-01-01"),
        ("BTCUSDT", "2024-01-02"),
        ("BTCUSDT", "2024-01-03"),
        ("ETHUSDT", "2024-01-01"),
        ("ETHUSDT", "2024-01-02"),
        ("ETHUSDT", "2024-01-03"),
    ]


# tests for match_trades_to_price
def _write_agg_trades(path, n, start_ts=0):
    df = pl.DataFrame(
        {
            "timestamp": [_ts(start_ts + i * 1000) for i in range(n)],
            "price": [100.0] * n,
            "quantity": [1.0] * n,
            "side": ["buy"] * n,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _write_book_ticker(path, n, bid=99.5, ask=100.5, start_ts=0):
    df = pl.DataFrame(
        {
            "timestamp": [_ts(start_ts + i * 1000) for i in range(n)],
            "bid_price": [bid] * n,
            "ask_price": [ask] * n,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def test_match_trades_to_price_1(tmp_path, monkeypatch):
    # returns None when the input parquet files don't exist
    monkeypatch.setattr(filter_wash_trades, "DATA_PROCESSED_DIR", tmp_path)
    assert match_trades_to_price("BTCUSDT", "2024-01-01") is None


def test_match_trades_to_price_2(tmp_path, monkeypatch):
    # returns None when the joined result has fewer than MIN_CSV_SIZE (100) rows
    monkeypatch.setattr(filter_wash_trades, "DATA_PROCESSED_DIR", tmp_path)
    _write_agg_trades(tmp_path / "BTCUSDT/aggTrades/BTCUSDT-aggTrades-2024-01-01.parquet", 5)
    _write_book_ticker(tmp_path / "BTCUSDT/bookTicker/BTCUSDT-bookTicker-2024-01-01.parquet", 5)
    assert match_trades_to_price("BTCUSDT", "2024-01-01") is None


def test_match_trades_to_price_3(tmp_path, monkeypatch):
    # asof-joins bid/ask before and after the trade correctly
    monkeypatch.setattr(filter_wash_trades, "DATA_PROCESSED_DIR", tmp_path)
    _write_agg_trades(tmp_path / "BTCUSDT/aggTrades/BTCUSDT-aggTrades-2024-01-01.parquet", 120)
    _write_book_ticker(tmp_path / "BTCUSDT/bookTicker/BTCUSDT-bookTicker-2024-01-01.parquet", 120)

    res = match_trades_to_price("BTCUSDT", "2024-01-01")

    assert res.columns == [
        "timestamp",
        "price",
        "quantity",
        "side",
        "bid_before",
        "ask_before",
        "bid_after",
        "ask_after",
    ]
    assert res.height == 120
    assert res["bid_before"][0] == 99.5
    assert res["ask_before"][0] == 100.5
    assert res["bid_after"][0] == 99.5
    assert res["ask_after"][0] == 100.5


# tests for zero_price_impact
def test_zero_price_impact_1(suspicion_fixture):
    # a large trade with no price move afterward scores 1.0 (maximally suspicious)
    score = zero_price_impact(suspicion_fixture, impact_eps_bps=0.5, size_quantile=0.8, horizon=1)
    assert score[500] == 1.0


def test_zero_price_impact_2(suspicion_fixture):
    # a large trade followed by a real price move scores 0.0 (not suspicious)
    score = zero_price_impact(suspicion_fixture, impact_eps_bps=0.5, size_quantile=0.8, horizon=1)
    assert score[503] == 0.0


def test_zero_price_impact_3(suspicion_fixture):
    # a small trade scores 0.0 regardless of a subsequent price move - being
    # "large" is required, not just a flat market
    score = zero_price_impact(suspicion_fixture, impact_eps_bps=0.5, size_quantile=0.8, horizon=1)
    assert score[501] == 0.0


# tests for off_touch_execution
def test_off_touch_execution_1(suspicion_fixture):
    # a trade priced inside the bid/ask bracket scores 0.0
    score = off_touch_execution(suspicion_fixture, touch_floor_bps=2.0)
    assert score[0] == 0.0


def test_off_touch_execution_2(suspicion_fixture):
    # a trade priced far outside the bracket clips to 1.0, above and below
    score = off_touch_execution(suspicion_fixture, touch_floor_bps=2.0)
    assert score[505] == 1.0
    assert score[506] == 1.0


def test_off_touch_execution_3(suspicion_fixture):
    # a small excursion produces a fractional, unclipped score
    score = off_touch_execution(suspicion_fixture, touch_floor_bps=2.0)
    expected = (100.51 - 100.5) / (100.51 * 2.0 / 1e4)
    assert score[507] == pytest.approx(expected)


def test_off_touch_execution_4(suspicion_fixture):
    # a missing quote (null bid/ask) yields a filled 0.0, not a propagated null
    score = off_touch_execution(suspicion_fixture, touch_floor_bps=2.0)
    assert score[521] == 0.0


# tests for ping_pong_reversal
def test_ping_pong_reversal_1(suspicion_fixture):
    # a same-price/same-qty opposite-side trade within the window scores by
    # how fast the reversal happened
    score = ping_pong_reversal(suspicion_fixture, ping_pong_window=500)
    assert score[508] == 0.0
    assert score[509] == 0.8


def test_ping_pong_reversal_2(suspicion_fixture):
    # ordinary same-side flow never scores as a reversal
    score = ping_pong_reversal(suspicion_fixture, ping_pong_window=500)
    assert score[0:5].to_list() == [0.0] * 5


# tests for duplicate_prints
def test_duplicate_prints_1(suspicion_fixture):
    # score scales with group size - unique=0, a pair=0.5, a triple~0.667
    score = duplicate_prints(suspicion_fixture)
    assert score[510] == 0.0
    assert score[511] == 0.5
    assert score[512] == 0.5
    assert score[513] == pytest.approx(2 / 3)
    assert score[514] == pytest.approx(2 / 3)
    assert score[515] == pytest.approx(2 / 3)


def test_duplicate_prints_2(suspicion_fixture):
    # the grouping key is the full (timestamp, price, quantity, side) tuple -
    # a flipped side breaks the match even with everything else identical
    score = duplicate_prints(suspicion_fixture)
    assert score[516] == 0.0
    assert score[517] == 0.0


# tests for size_clustering
def test_size_clustering_1():
    # with fewer rows than the 500-row window, every row's baseline collapses
    # to its own quantity (fill_null(quantity)), so even a large, repeated
    # quantity scores 0.0 - needs its own small fixture to exercise the
    # sub-500-row cold-start path
    df = pl.DataFrame({"quantity": [1.0, 1.0, 100.0, 100.0, 5.0]})
    score = size_clustering(df, size_quantile=0.8)
    assert score.to_list() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_size_clustering_2(suspicion_fixture):
    # a large trade that never recurs scores 0.0 - magnitude alone isn't
    # suspicious (guards against flagging institutional order-splitting)
    score = size_clustering(suspicion_fixture, size_quantile=0.8)
    assert score[518] == 0.0


def test_size_clustering_3(suspicion_fixture):
    # a large quantity that repeats scores high - magnitude and recurrence
    # together are what's suspicious
    score = size_clustering(suspicion_fixture, size_quantile=0.8)
    assert score[519] == pytest.approx(0.4285714285713673)
    assert score[520] == pytest.approx(0.4285714285713673)


# tests for process_task
def test_process_task_1(tmp_path, monkeypatch):
    # missing input data returns a "missing" status and writes nothing
    monkeypatch.setattr(filter_wash_trades, "DATA_PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(filter_wash_trades, "match_trades_to_price", lambda symbol, date_str: None)

    result = process_task("BTCUSDT", "2024-01-01", 0.5, 1, 500, 0.8, 2.0, 0.5, False)

    assert result == {"symbol": "BTCUSDT", "date": "2024-01-01", "status": "missing"}
    out_path = tmp_path / "BTCUSDT" / "aggTrades_filtered" / "BTCUSDT-aggTrades-2024-01-01.parquet"
    assert not out_path.exists()


def test_process_task_2(tmp_path, monkeypatch):
    # passthrough mode writes the matched data unmodified, with no
    # score/wash/clean columns added
    monkeypatch.setattr(filter_wash_trades, "DATA_PROCESSED_DIR", tmp_path)
    fixture_df = pl.DataFrame(
        {
            "timestamp": [_ts(i * 1000) for i in range(4)],
            "price": [100.0] * 4,
            "quantity": [1.0] * 4,
            "side": ["buy"] * 4,
        }
    )
    monkeypatch.setattr(
        filter_wash_trades, "match_trades_to_price", lambda symbol, date_str: fixture_df
    )

    result = process_task("BTCUSDT", "2024-01-01", 0.5, 1, 500, 0.8, 2.0, 0.5, True)

    assert result == {"symbol": "BTCUSDT", "date": "2024-01-01", "status": "passthrough", "rows": 4}
    out_path = tmp_path / "BTCUSDT" / "aggTrades_filtered" / "BTCUSDT-aggTrades-2024-01-01.parquet"
    written = pl.read_parquet(out_path)
    assert written.columns == fixture_df.columns
    assert written.height == 4


def test_process_task_3(tmp_path, monkeypatch):
    # suspicious rows get flagged, and the reason breakdown attributes them
    # to the dominant clue
    monkeypatch.setattr(filter_wash_trades, "DATA_PROCESSED_DIR", tmp_path)
    # rows 0-2 are exact duplicates (same timestamp/price/quantity/side); row 3 is unique
    fixture_df = pl.DataFrame(
        {
            "timestamp": [_ts(0), _ts(0), _ts(0), _ts(1000)],
            "price": [100.0, 100.0, 100.0, 100.0],
            "quantity": [1.0, 1.0, 1.0, 1.0],
            "side": ["buy", "buy", "buy", "buy"],
            "buy_qty": [1.0, 1.0, 1.0, 1.0],
            "sell_qty": [0.0, 0.0, 0.0, 0.0],
            "notional": [100.0, 100.0, 100.0, 100.0],
            "mid_price": [100.0] * 4,
            "bid_before": [99.5] * 4,
            "ask_before": [100.5] * 4,
            "bid_after": [99.5] * 4,
            "ask_after": [100.5] * 4,
        }
    )
    monkeypatch.setattr(
        filter_wash_trades, "match_trades_to_price", lambda symbol, date_str: fixture_df
    )

    result = process_task("BTCUSDT", "2024-01-01", 0.5, 1, 500, 0.8, 2.0, 0.5, False)

    assert result == {
        "symbol": "BTCUSDT",
        "date": "2024-01-01",
        "status": "ok",
        "total": 4,
        "flagged": 3,
        "flag_rate": 0.75,
        "reason_breakdown": {"duplicate_score": {"count": 3, "pct": 100.0}},
    }
    out_path = tmp_path / "BTCUSDT" / "aggTrades_filtered" / "BTCUSDT-aggTrades-2024-01-01.parquet"
    written = pl.read_parquet(out_path)
    assert written["wash_suspect"].to_list() == [True, True, True, False]


def test_process_task_4(tmp_path, monkeypatch):
    # cleaned-flow columns are hard-zeroed for suspect rows and left
    # unchanged otherwise
    monkeypatch.setattr(filter_wash_trades, "DATA_PROCESSED_DIR", tmp_path)
    # row0 price inside bracket (not suspect); row1 price far outside bracket
    # (touch_score clips to 1.0, suspect given wash_score_cut=0.5)
    fixture_df = pl.DataFrame(
        {
            "timestamp": [_ts(0), _ts(1000)],
            "price": [100.0, 110.0],
            "quantity": [1.0, 1.0],
            "side": ["buy", "buy"],
            "buy_qty": [1.0, 1.0],
            "sell_qty": [0.0, 0.0],
            "notional": [100.0, 110.0],
            "mid_price": [100.0, 100.0],
            "bid_before": [99.5, 99.5],
            "ask_before": [100.5, 100.5],
            "bid_after": [99.5, 99.5],
            "ask_after": [100.5, 100.5],
        }
    )
    monkeypatch.setattr(
        filter_wash_trades, "match_trades_to_price", lambda symbol, date_str: fixture_df
    )

    process_task("BTCUSDT", "2024-01-01", 0.5, 1, 500, 0.8, 2.0, 0.5, False)

    out_path = tmp_path / "BTCUSDT" / "aggTrades_filtered" / "BTCUSDT-aggTrades-2024-01-01.parquet"
    written = pl.read_parquet(out_path)

    assert written["buy_qty_clean"][0] == 1.0
    assert written["sell_qty_clean"][0] == 0.0
    assert written["notional_clean"][0] == 100.0

    assert written["buy_qty_clean"][1] == 0.0
    assert written["sell_qty_clean"][1] == 0.0
    assert written["notional_clean"][1] == 0.0