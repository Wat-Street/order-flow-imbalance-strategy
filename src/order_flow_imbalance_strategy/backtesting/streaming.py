"""Bounded-memory streaming reader for the joined feature matrix.

``io.load_opportunities`` reads a whole Parquet file into memory at once, which
is fine for the synthetic demo and short studies. For long real-data runs the
spec requires bounded memory: process one causal partition at a time, release
it, then read the next. This module streams the feature matrix in day-sized
batches (ordered by the UTC calendar day of each row's ``timestamp``), so peak
memory stays flat regardless of how many days the study spans.

Batching by day also preserves the causal daily-partition discipline used
throughout the pipeline: earlier days are yielded before later ones, and a
batch never straddles a day boundary.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import polars as pl

from .contracts import Opportunity
from .io import _rows_from_frame


def stream_opportunity_days(
    path: str | Path,
    model_id: str | None = None,
) -> Iterator[tuple[date, list[Opportunity]]]:
    """Yield ``(utc_day, opportunities)`` one calendar day at a time, in order.

    Only one day's rows are materialized as ``Opportunity`` objects at a time.
    The lazy scan reads the ``timestamp`` column to enumerate the distinct days
    first, then collects each day's slice on demand.
    """
    lazy = pl.scan_parquet(path)
    if "timestamp" not in lazy.collect_schema().names():
        raise ValueError("feature matrix must contain a 'timestamp' column to stream by day")

    days = (
        lazy.select(pl.col("timestamp").dt.date().alias("day"))
        .unique()
        .sort("day")
        .collect()
        .get_column("day")
        .to_list()
    )
    for day in days:
        frame = (
            lazy.filter(pl.col("timestamp").dt.date() == pl.lit(day)).sort("timestamp").collect()
        )
        yield day, _rows_from_frame(frame, model_id)


def stream_opportunities(
    path: str | Path,
    model_id: str | None = None,
) -> Iterator[Opportunity]:
    """Flatten the day batches into a single ordered opportunity stream."""
    for _, rows in stream_opportunity_days(path, model_id):
        yield from rows
