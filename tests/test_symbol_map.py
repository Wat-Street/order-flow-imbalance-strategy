import pytest

from order_flow_imbalance_strategy import symbol_map


def test_btc_round_trips():
    assert symbol_map.to_hyperliquid("BTCUSDT") == "BTC"
    assert symbol_map.to_binance("BTC") == "BTCUSDT"


def test_reverse_map_is_consistent_with_forward_map():
    for binance_symbol, coin in symbol_map.BINANCE_TO_HYPERLIQUID.items():
        assert symbol_map.HYPERLIQUID_TO_BINANCE[coin] == binance_symbol


def test_unknown_binance_symbol_raises_with_hint():
    with pytest.raises(KeyError, match="symbol_map.py"):
        symbol_map.to_hyperliquid("DOGEUSDT")


def test_unknown_coin_raises_with_hint():
    with pytest.raises(KeyError, match="symbol_map.py"):
        symbol_map.to_binance("DOGE")
