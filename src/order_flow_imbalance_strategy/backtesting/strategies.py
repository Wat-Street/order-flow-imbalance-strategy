from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Protocol

from .contracts import Opportunity, TradingConfig


class Strategy(Protocol):
    name: str

    def decide(self, opportunity: Opportunity, trading: TradingConfig) -> int: ...


def _threshold_direction(score: float | None, trading: TradingConfig) -> int:
    if score is None or not math.isfinite(score):
        return 0
    if score >= trading.long_threshold:
        return 1
    if score <= trading.short_threshold:
        return -1
    return 0


@dataclass(frozen=True, slots=True)
class MLFilteredOFI:
    name: str = "ml_filtered_ofi"

    def decide(self, opportunity: Opportunity, trading: TradingConfig) -> int:
        prediction = opportunity.prediction
        if (
            prediction is None
            or not prediction.feature_ready
            or prediction.score is None
            or not opportunity.feature_ready
        ):
            return 0
        return _threshold_direction(prediction.score, trading)


@dataclass(frozen=True, slots=True)
class RawOFI:
    threshold: float = 0.0
    name: str = "raw_ofi"

    def __post_init__(self) -> None:
        if self.threshold < 0 or not math.isfinite(self.threshold):
            raise ValueError("raw OFI threshold must be finite and nonnegative")

    def decide(self, opportunity: Opportunity, trading: TradingConfig) -> int:
        del trading
        score = opportunity.raw_ofi
        if score is None or not math.isfinite(score) or not opportunity.feature_ready:
            return 0
        if score > self.threshold:
            return 1
        if score < -self.threshold:
            return -1
        return 0


@dataclass(slots=True)
class RandomEntry:
    attempt_probability: float
    long_probability: float
    seed: int
    name: str = "random"
    _random: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.attempt_probability <= 1:
            raise ValueError("attempt_probability must be in [0, 1]")
        if not 0 <= self.long_probability <= 1:
            raise ValueError("long_probability must be in [0, 1]")
        self._random = random.Random(self.seed)

    @classmethod
    def calibrate(
        cls,
        training: list[Opportunity],
        threshold: float = 0.2,
        seed: int = 0,
    ) -> RandomEntry:
        """Match the training opportunity rate and long/short mix, never test rows."""
        if not training:
            raise ValueError("training opportunities are required for random calibration")
        eligible_scores = [
            row.prediction.score
            for row in training
            if row.feature_ready
            and row.prediction is not None
            and row.prediction.feature_ready
            and row.prediction.score is not None
            and math.isfinite(row.prediction.score)
            and abs(row.prediction.score) >= threshold
        ]
        attempt_probability = len(eligible_scores) / len(training)
        long_probability = (
            sum(score > 0 for score in eligible_scores) / len(eligible_scores)
            if eligible_scores
            else 0.5
        )
        return cls(attempt_probability, long_probability, seed)

    def decide(self, opportunity: Opportunity, trading: TradingConfig) -> int:
        del trading
        if not opportunity.feature_ready:
            return 0
        if self._random.random() >= self.attempt_probability:
            return 0
        return 1 if self._random.random() < self.long_probability else -1


@dataclass(slots=True)
class Momentum:
    """Trailing-price momentum: buy if price rose over the lookback, else sell.

    The signal is computed here from the quotes the strategy observes, not read
    from a precomputed column. For a lookback of ``L`` seconds the momentum
    return is ``midpoint_now / midpoint_{L seconds ago} - 1``; a value above
    ``+threshold`` goes long, below ``-threshold`` goes short. The lookback and
    threshold must be chosen on earlier training data only (see ``select``).

    A bounded ring of recent midpoints keyed by decision time is maintained so
    the window is exact. State resets at a UTC day boundary and whenever the
    required earlier midpoint is missing, so an incomplete window never trades.
    """

    lookback: timedelta = timedelta(seconds=60)
    threshold: float = 0.0
    name: str = "momentum"
    _history: deque[tuple[datetime, float]] = field(init=False, repr=False)
    _current_day: date | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.threshold < 0 or not math.isfinite(self.threshold):
            raise ValueError("momentum threshold must be finite and nonnegative")
        if self.lookback <= timedelta(0):
            raise ValueError("momentum lookback must be positive")
        self._history = deque()

    def _reset(self) -> None:
        self._history.clear()

    def decide(self, opportunity: Opportunity, trading: TradingConfig) -> int:
        del trading
        quote = opportunity.quote
        now = quote.timestamp
        day = now.date()
        if day != self._current_day:
            self._reset()
            self._current_day = day

        tradable = quote.valid and not quote.book_stale and not quote.book_dead
        if not tradable or not opportunity.feature_ready:
            # A gap in usable data breaks the trailing window; drop stale history
            # rather than comparing across an unobserved hole.
            self._reset()
            return 0

        midpoint = quote.midpoint
        reference_time = now - self.lookback
        # Drop points strictly older than the window start, keeping the most
        # recent of them as the window's left endpoint (the price `lookback`
        # ago). History is append-ordered oldest-first.
        reference_price: float | None = None
        while len(self._history) >= 2 and self._history[1][0] <= reference_time:
            reference_price = self._history.popleft()[1]
        if self._history and self._history[0][0] <= reference_time:
            reference_price = self._history[0][1]

        decision = 0
        if reference_price is not None and reference_price > 0:
            momentum_return = midpoint / reference_price - 1.0
            if momentum_return > self.threshold:
                decision = 1
            elif momentum_return < -self.threshold:
                decision = -1

        self._history.append((now, midpoint))
        return decision

    @classmethod
    def select(
        cls,
        training: list[Opportunity],
        lookbacks: tuple[timedelta, ...] = (
            timedelta(seconds=30),
            timedelta(seconds=60),
            timedelta(seconds=120),
        ),
        thresholds: tuple[float, ...] = (0.0, 0.0001, 0.0005),
    ) -> Momentum:
        """Pick the lookback/threshold with the best training-only hit rate.

        For each candidate, we replay the trailing-momentum signal over the
        training rows and measure how often the predicted direction matched the
        realized next-step midpoint move. Only training data is consulted; the
        confirmation period is never touched here.
        """
        if not training:
            raise ValueError("training opportunities are required to select momentum settings")
        best: Momentum | None = None
        best_hits = -1.0
        for lookback in lookbacks:
            for threshold in thresholds:
                candidate = cls(lookback=lookback, threshold=threshold)
                hits = 0
                total = 0
                previous_mid: float | None = None
                for row in training:
                    decision = candidate.decide(row, TradingConfig())
                    quote = row.quote
                    mid = quote.midpoint if (quote.valid and not quote.book_stale) else None
                    if previous_mid is not None and mid is not None and decision != 0:
                        realized = 1 if mid > previous_mid else -1 if mid < previous_mid else 0
                        if realized != 0:
                            total += 1
                            if decision == realized:
                                hits += 1
                    previous_mid = mid if mid is not None else previous_mid
                hit_rate = hits / total if total else 0.0
                if hit_rate > best_hits:
                    best_hits = hit_rate
                    best = cls(lookback=lookback, threshold=threshold)
        assert best is not None
        return best


@dataclass(frozen=True, slots=True)
class PrecomputedMomentum:
    """Momentum baseline that reads a precomputed ``momentum_score`` column.

    Retained for inputs that already carry a trusted momentum feature; the
    default momentum baseline is the self-computing ``Momentum`` above.
    """

    threshold: float = 0.0
    name: str = "momentum"

    def __post_init__(self) -> None:
        if self.threshold < 0 or not math.isfinite(self.threshold):
            raise ValueError("momentum threshold must be finite and nonnegative")

    def decide(self, opportunity: Opportunity, trading: TradingConfig) -> int:
        del trading
        score = opportunity.momentum_score
        if score is None or not math.isfinite(score) or not opportunity.feature_ready:
            return 0
        if score > self.threshold:
            return 1
        if score < -self.threshold:
            return -1
        return 0


@dataclass(frozen=True, slots=True)
class Cash:
    name: str = "cash"

    def decide(self, opportunity: Opportunity, trading: TradingConfig) -> int:
        del opportunity, trading
        return 0
