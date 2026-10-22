"""Shared entry gatekeeping and coverage diagnostics.

Every baseline must face the same opportunity restrictions so the comparison
across strategies is fair. This module factors the "may a strategy open a new
position on this row, and if not why" decision out of the engine loop into one
inspectable, independently testable place, and tallies per-row coverage so the
report can show how much of the clock was actually tradable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import CoverageDiagnostics, Opportunity

# Reasons an entry is blocked before the strategy's own buy/sell opinion is
# consulted. These are stable strings so reports and tests can assert on them.
ENTRY_BLOCKED_STALE = "invalid_or_stale_quote"
ENTRY_BLOCKED_POSITION_OPEN = "position_open"
ENTRY_BLOCKED_PENDING_ORDER = "pending_order"
ENTRY_BLOCKED_FEATURE_NOT_READY = "feature_not_ready"
ENTRY_ALLOWED = "eligible"


def quote_tradable(opportunity: Opportunity) -> bool:
    """True when the row's quote is valid and neither stale nor dead."""
    quote = opportunity.quote
    return quote.valid and not quote.book_stale and not quote.book_dead


def entry_block_reason(
    opportunity: Opportunity,
    *,
    position: float,
    has_pending_entry: bool,
    has_pending_exit: bool,
) -> str:
    """Return ENTRY_ALLOWED, or the first reason a new entry is not permitted.

    The order of checks is deliberate: market-data usability is reported before
    account-state reasons so a stale row is never mislabeled as merely "already
    in a position".
    """
    if not quote_tradable(opportunity):
        return ENTRY_BLOCKED_STALE
    if not opportunity.feature_ready:
        return ENTRY_BLOCKED_FEATURE_NOT_READY
    if position != 0:
        return ENTRY_BLOCKED_POSITION_OPEN
    if has_pending_entry or has_pending_exit:
        return ENTRY_BLOCKED_PENDING_ORDER
    return ENTRY_ALLOWED


def entry_allowed(
    opportunity: Opportunity,
    *,
    position: float,
    has_pending_entry: bool,
    has_pending_exit: bool,
) -> bool:
    """Convenience wrapper returning whether a new entry may be considered."""
    return (
        entry_block_reason(
            opportunity,
            position=position,
            has_pending_entry=has_pending_entry,
            has_pending_exit=has_pending_exit,
        )
        == ENTRY_ALLOWED
    )


@dataclass(slots=True)
class CoverageTracker:
    """Accumulate per-row coverage counts over a single backtest replay."""

    total_rows: int = 0
    tradable_rows: int = 0
    stale_rows: int = 0
    dead_rows: int = 0
    invalid_quote_rows: int = 0
    feature_ready_rows: int = 0
    prediction_ready_rows: int = 0

    def observe(self, opportunity: Opportunity) -> None:
        self.total_rows += 1
        quote = opportunity.quote
        if quote_tradable(opportunity):
            self.tradable_rows += 1
        if quote.book_stale:
            self.stale_rows += 1
        if quote.book_dead:
            self.dead_rows += 1
        if not quote.valid:
            self.invalid_quote_rows += 1
        if opportunity.feature_ready:
            self.feature_ready_rows += 1
        if (
            opportunity.prediction is not None
            and opportunity.prediction.feature_ready
            and opportunity.prediction.score is not None
        ):
            self.prediction_ready_rows += 1

    def diagnostics(self) -> CoverageDiagnostics:
        tradable_fraction = self.tradable_rows / self.total_rows if self.total_rows else 0.0
        return CoverageDiagnostics(
            total_rows=self.total_rows,
            tradable_rows=self.tradable_rows,
            stale_rows=self.stale_rows,
            dead_rows=self.dead_rows,
            invalid_quote_rows=self.invalid_quote_rows,
            feature_ready_rows=self.feature_ready_rows,
            prediction_ready_rows=self.prediction_ready_rows,
            tradable_fraction=tradable_fraction,
        )
