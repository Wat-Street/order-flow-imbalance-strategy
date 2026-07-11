"""Skeleton cross-exchange symbol map.

Hyperliquid identifies perpetuals by a bare coin ticker (``"BTC"``) while the
Binance side of this project uses pair symbols (``"BTCUSDT"``). Any analysis
that joins the two feeds — e.g. testing whether Hyperliquid full-book depth
imbalance leads Binance L1 OFI — must first reconcile the two naming schemes.

This module is intentionally a thin, explicit lookup rather than a clever
derivation (``BTCUSDT``[:-4] would break on ``USDC`` pairs, index products,
etc.). Extend the table below as symbols are added to the study.
"""

from __future__ import annotations

#: Canonical Binance-symbol -> Hyperliquid-coin mapping. BTC is the only pair
#: wired up for the initial L2 study; ETH/SOL are placeholders kept in sync with
#: the Binance ``symbol_list`` so the table is obviously incomplete-by-design
#: rather than silently missing.
BINANCE_TO_HYPERLIQUID: dict[str, str] = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",  # TODO: confirm Hyperliquid has matching history before use
    "SOLUSDT": "SOL",  # TODO: confirm Hyperliquid has matching history before use
}

#: Reverse lookup, derived so the two never drift apart.
HYPERLIQUID_TO_BINANCE: dict[str, str] = {
    coin: symbol for symbol, coin in BINANCE_TO_HYPERLIQUID.items()
}


def to_hyperliquid(binance_symbol: str) -> str:
    """Return the Hyperliquid coin for a Binance pair symbol.

    Raises ``KeyError`` with an actionable message for an unmapped symbol so a
    typo or a not-yet-supported pair fails loudly instead of silently skipping.
    """
    try:
        return BINANCE_TO_HYPERLIQUID[binance_symbol]
    except KeyError:
        raise KeyError(
            f"no Hyperliquid coin mapped for Binance symbol {binance_symbol!r}; "
            f"add it to BINANCE_TO_HYPERLIQUID in symbol_map.py"
        ) from None


def to_binance(coin: str) -> str:
    """Return the Binance pair symbol for a Hyperliquid coin."""
    try:
        return HYPERLIQUID_TO_BINANCE[coin]
    except KeyError:
        raise KeyError(
            f"no Binance symbol mapped for Hyperliquid coin {coin!r}; "
            f"add it to BINANCE_TO_HYPERLIQUID in symbol_map.py"
        ) from None
