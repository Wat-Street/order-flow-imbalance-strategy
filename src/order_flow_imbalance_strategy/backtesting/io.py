from __future__ import annotations

import hashlib
import math
from pathlib import Path

import polars as pl

from .contracts import FundingPayment, Opportunity, Prediction, Quote


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _first_present(row: dict, *names: str) -> object:
    for name in names:
        if name in row:
            return row[name]
    return None


def _book_flag(row: dict, name: str) -> bool:
    """Missing column -> not flagged; present-but-null -> conservatively flagged."""
    if name not in row:
        return False
    value = row[name]
    return value is None or bool(value)


def load_opportunities(
    path: str | Path,
    model_id: str | None = None,
) -> list[Opportunity]:
    """Load one joined, full-clock Parquet file without filtering unknown labels."""
    return _rows_from_frame(pl.read_parquet(path), model_id)


def _rows_from_frame(frame: pl.DataFrame, model_id: str | None = None) -> list[Opportunity]:
    """Build Opportunity rows from an already-loaded feature-matrix frame.

    Shared by the eager loader and the bounded-memory streaming reader so the
    column handling, validity rules, and score derivation stay identical.
    """
    required = {
        "source",
        "timestamp",
        "available_at",
        "bid_price",
        "ask_price",
    }
    missing = sorted(required - set(frame.columns))
    if "symbol" not in frame.columns and "asset" not in frame.columns:
        missing.append("symbol or asset")
    if not ({"bid_size", "bid_qty"} & set(frame.columns)):
        missing.append("bid_size or bid_qty")
    if not ({"ask_size", "ask_qty"} & set(frame.columns)):
        missing.append("ask_size or ask_qty")
    if missing:
        raise ValueError(f"input parquet is missing required columns: {missing}")

    rows: list[Opportunity] = []
    for record in frame.to_dicts():
        source = str(record["source"])
        symbol = str(record.get("symbol", record.get("asset")))
        timestamp = record["timestamp"]
        available_at = record["available_at"]
        bid_price = _as_float(record.get("bid_price"))
        ask_price = _as_float(record.get("ask_price"))
        bid_size = _as_float(_first_present(record, "bid_size", "bid_qty"))
        ask_size = _as_float(_first_present(record, "ask_size", "ask_qty"))
        quote_valid = bool(record.get("quote_valid", True)) and all(
            value is not None for value in (bid_price, ask_price, bid_size, ask_size)
        )
        quote = Quote(
            source=source,
            symbol=symbol,
            timestamp=timestamp,
            available_at=available_at,
            bid_price=bid_price if bid_price is not None else math.nan,
            ask_price=ask_price if ask_price is not None else math.nan,
            bid_size=bid_size if bid_size is not None else math.nan,
            ask_size=ask_size if ask_size is not None else math.nan,
            # A missing column means the feed carries no stale/dead signal, so
            # default to "not stale/dead". A present-but-null value is genuinely
            # unknown; treat that conservatively as stale/dead so fills are skipped.
            book_stale=_book_flag(record, "book_stale"),
            book_dead=_book_flag(record, "book_dead"),
            valid=quote_valid,
        )

        raw_probability = _as_float(record.get("up_probability"))
        raw_score = _as_float(record.get("prediction_score", record.get("score")))
        score = (
            raw_score
            if raw_score is not None
            else (2 * raw_probability - 1 if raw_probability is not None else None)
        )
        prediction = None
        if model_id is not None or "up_probability" in record or "prediction_score" in record:
            row_model_id = record.get("model_id") or model_id
            if not row_model_id:
                raise ValueError("model_id is required when prediction columns are present")
            prediction = Prediction(
                source=source,
                symbol=symbol,
                timestamp=timestamp,
                available_at=record.get("prediction_available_at") or available_at,
                score=score,
                feature_ready=bool(record.get("feature_ready", False)),
                model_id=str(row_model_id),
            )
        rows.append(
            Opportunity(
                quote=quote,
                prediction=prediction,
                raw_ofi=_as_float(_first_present(record, "raw_ofi", "ofi_1s")),
                momentum_score=_as_float(record.get("momentum_score")),
                feature_ready=bool(record.get("feature_ready", True)),
            )
        )
    return rows


def load_funding(path: str | Path) -> list[FundingPayment]:
    frame = pl.read_parquet(path) if str(path).lower().endswith(".parquet") else pl.read_csv(path)
    required = {"source", "settled_at", "rate", "mark_price", "mark_available_at"}
    missing = sorted(required - set(frame.columns))
    if "symbol" not in frame.columns and "asset" not in frame.columns:
        missing.append("symbol or asset")
    if missing:
        raise ValueError(f"funding input is missing required columns: {missing}")
    return [
        FundingPayment(
            source=str(row["source"]),
            symbol=str(row.get("symbol", row.get("asset"))),
            settled_at=row["settled_at"],
            rate=float(row["rate"]),
            mark_price=float(row["mark_price"]),
            mark_available_at=row["mark_available_at"],
            mark_kind=row.get("mark_kind", "observed"),
        )
        for row in frame.to_dicts()
    ]
