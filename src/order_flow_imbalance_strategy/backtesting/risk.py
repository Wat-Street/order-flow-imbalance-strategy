from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal

from .contracts import EquityPoint

MetricStatus = Literal[
    "ok", "incomplete_valuation", "insufficient_days", "zero_variance", "invalid_input"
]


@dataclass(frozen=True, slots=True)
class Metric:
    value: float | None
    status: MetricStatus


@dataclass(frozen=True, slots=True)
class RiskMetrics:
    daily_returns: tuple[tuple[str, float], ...]
    sharpe: Metric
    max_drawdown: Metric
    drawdown_peak: datetime | None
    drawdown_trough: datetime | None
    drawdown_recovery_seconds: float | None
    unrecovered_drawdown: bool
    complete_days: int
    total_days: int


def _complete_daily_returns(points: tuple[EquityPoint, ...]) -> tuple[tuple[str, float], ...]:
    grouped: dict[date, list[EquityPoint]] = {}
    for point in points:
        clock_timestamp = point.clock_timestamp or point.timestamp
        grouped.setdefault(clock_timestamp.astimezone(UTC).date(), []).append(point)

    returns: list[tuple[str, float]] = []
    for day, day_points in sorted(grouped.items()):
        day_points.sort(key=lambda point: point.clock_timestamp or point.timestamp)
        opening = day_points[0]
        closing = day_points[-1]
        opening_time = opening.clock_timestamp or opening.timestamp
        closing_time = closing.clock_timestamp or closing.timestamp
        clock_times = [point.clock_timestamp or point.timestamp for point in day_points]
        consecutive_clock = all(
            later - earlier == timedelta(seconds=1)
            for earlier, later in zip(clock_times, clock_times[1:], strict=False)
        )
        if (
            len(day_points) != 86_400
            or not consecutive_clock
            or opening_time.astimezone(UTC).time() != time(0, 0)
            or closing_time.astimezone(UTC).time() != time(23, 59, 59)
            or any(not point.valuation_complete or point.equity is None for point in day_points)
            or opening.equity is None
            or opening.equity <= 0
            or closing.equity is None
        ):
            continue
        returns.append((day.isoformat(), closing.equity / opening.equity - 1.0))
    return tuple(returns)


def compute_risk_metrics(points: tuple[EquityPoint, ...] | list[EquityPoint]) -> RiskMetrics:
    path = tuple(points)
    if not path:
        invalid = Metric(None, "invalid_input")
        return RiskMetrics((), invalid, invalid, None, None, None, False, 0, 0)

    if any(path[index].timestamp >= path[index + 1].timestamp for index in range(len(path) - 1)):
        invalid = Metric(None, "invalid_input")
        return RiskMetrics((), invalid, invalid, None, None, None, False, 0, 0)

    daily_returns = _complete_daily_returns(path)
    total_days = len(
        {(point.clock_timestamp or point.timestamp).astimezone(UTC).date() for point in path}
    )
    incomplete_path = any(not point.valuation_complete or point.equity is None for point in path)

    if incomplete_path:
        sharpe = Metric(None, "incomplete_valuation")
        max_drawdown = Metric(None, "incomplete_valuation")
        peak_time = trough_time = None
        recovery_seconds = None
        unrecovered = False
    else:
        if len(daily_returns) < 2:
            sharpe = Metric(None, "insufficient_days")
        else:
            returns = [value for _, value in daily_returns]
            deviation = statistics.stdev(returns)
            sharpe = (
                Metric(None, "zero_variance")
                if deviation == 0
                else Metric(statistics.mean(returns) / deviation * math.sqrt(365), "ok")
            )

        peak = path[0].equity
        peak_time = path[0].timestamp
        worst_drawdown = 0.0
        trough_time = None
        worst_peak_time = None
        recovery_seconds = None
        unrecovered = False
        for point in path:
            value = point.equity
            if value is None:
                continue
            if value > peak:
                if trough_time is not None and recovery_seconds is None:
                    recovery_seconds = (point.timestamp - trough_time).total_seconds()
                peak = value
                peak_time = point.timestamp
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            if drawdown > worst_drawdown:
                worst_drawdown = drawdown
                trough_time = point.timestamp
                worst_peak_time = peak_time
                recovery_seconds = None
        if trough_time is not None and recovery_seconds is None:
            unrecovered = True
        peak_time = worst_peak_time
        max_drawdown = Metric(worst_drawdown, "ok")

    return RiskMetrics(
        daily_returns=daily_returns,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        drawdown_peak=peak_time,
        drawdown_trough=trough_time,
        drawdown_recovery_seconds=recovery_seconds,
        unrecovered_drawdown=unrecovered,
        complete_days=len(daily_returns),
        total_days=total_days,
    )


@dataclass(frozen=True, slots=True)
class PairedBootstrap:
    mean_difference: float
    confidence_low: float
    confidence_high: float
    two_sided_p_value: float
    block_length: int
    resamples: int
    seed: int


def paired_stationary_bootstrap(
    first: list[float] | tuple[float, ...],
    second: list[float] | tuple[float, ...],
    block_length: int = 7,
    resamples: int = 2_000,
    seed: int = 0,
) -> PairedBootstrap:
    if len(first) != len(second) or not first:
        raise ValueError("paired samples must have equal nonzero lengths")
    if block_length < 1 or resamples < 1:
        raise ValueError("block_length and resamples must be positive")
    differences = [left - right for left, right in zip(first, second, strict=True)]
    if any(not math.isfinite(value) for value in differences):
        raise ValueError("paired samples must be finite")

    rng = random.Random(seed)
    size = len(differences)
    boot_means: list[float] = []
    restart_probability = 1.0 / block_length
    for _ in range(resamples):
        index = rng.randrange(size)
        sample: list[float] = []
        for position in range(size):
            if position and rng.random() < restart_probability:
                index = rng.randrange(size)
            sample.append(differences[index])
            index = (index + 1) % size
        boot_means.append(statistics.mean(sample))

    boot_means.sort()
    low_index = max(0, math.floor(0.025 * resamples))
    high_index = min(resamples - 1, math.ceil(0.975 * resamples) - 1)
    below = sum(value <= 0 for value in boot_means) / resamples
    above = sum(value >= 0 for value in boot_means) / resamples
    return PairedBootstrap(
        mean_difference=statistics.mean(differences),
        confidence_low=boot_means[low_index],
        confidence_high=boot_means[high_index],
        two_sided_p_value=min(1.0, 2 * min(below, above)),
        block_length=block_length,
        resamples=resamples,
        seed=seed,
    )


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    if any(not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("p-values must be in [0, 1]")
    ranked = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    count = len(ranked)
    for index, (name, value) in enumerate(ranked):
        running_max = max(running_max, (count - index) * value)
        adjusted[name] = min(1.0, running_max)
    return adjusted
