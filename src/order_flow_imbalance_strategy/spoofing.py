# L1 cancellation-suspicion filter layered on raw OFI.
#
# A best-level queue that shrinks while its price holds either traded or was
# pulled. Aggressive sells hit the bid and aggressive buys lift the ask, so the
# same-second opposing flow explains part of the shrink; what's left over is
# "unexplained removal" -- a cancellation proxy:
#   * bid: max(0, prev_bid_qty - bid_qty - sell_qty)
#   * ask: max(0, prev_ask_qty - ask_qty - buy_qty)
# This is only a proxy. An L1 decrease can also be a fill or a queue moving out
# of view, and nothing here sees order IDs, placement lifetimes or intent.
#
# Unidentifiable rules (removal is null -- never 0, which would claim "checked,
# nothing suspicious"):
#   * Row 0 has no ``t-1``.
#   * The gap to the previous row is not exactly 1 second (a missing grid second).
#   * The side's price changed, so the old queue is out of view.
#   * ``book_stale`` or ``book_dead`` on the current OR previous row: the diff is
#       not between two real observations.
#   * Either quantity, or the opposing flow, is null (no trade context to net).
#
# Trailing reference stats (per side, from the previous ``window_s`` seconds only;
# the current second never enters its own baseline):
#   * median and sample std (ddof=1) of the *comparable* removals in the window --
#       unidentifiable seconds are skipped, not counted as 0.
#   * Usable only once the segment has run a full window (``warm_up`` False) AND
#       the window holds at least ``min_obs`` comparable removals; null otherwise.
#   * A segment restarts on a missing grid second, a new UTC date, or a dead book
#       (current or previous row): none of those is an uninterrupted history.
#       ``book_stale`` does not restart it -- a carried quote is a quiet book, not a
#       broken feed, and those seconds are already unidentifiable.
#
# Scoring (per side): how unusual is this second's removal against its baseline?
#   * z = (removal - prior_median) / (prior_std + eps); eps keeps a flat history
#       (std 0) finite -- there, any excess over the median scores as maximal.
#   * score = clip((z - z_floor) / (z_cap - z_floor), 0, 1): 0 up to z_floor,
#       1 from z_cap. Null removal or baseline -> null score, never 0.
# Window and thresholds live in a frozen ``SpoofConfig``; pick them on selection
# data, then evaluate that exact config (``fingerprint()``) on untouched data.
#
# Known limitation: the aligned frame zero-fills flow on seconds without trades
# and has no flag for a missing trade feed, so a feed outage reads as "no trades"
# and its depletion as unexplained.

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import polars as pl

#: Columns that must be present to compute unexplained removal.
REMOVAL_INPUT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "bid_price",
    "ask_price",
    "bid_qty",
    "ask_qty",
    "buy_qty",
    "sell_qty",
    "book_stale",
    "book_dead",
)

#: Columns appended by :func:`unexplained_removal`.
REMOVAL_OUTPUT_COLUMNS: tuple[str, ...] = (
    "bid_removal",
    "ask_removal",
    "bid_identifiable",
    "ask_identifiable",
)

#: Columns that must be present to compute trailing stats.
STATS_INPUT_COLUMNS: tuple[str, ...] = ("timestamp", "book_dead", "bid_removal", "ask_removal")

#: Columns appended by :func:`trailing_stats`.
STATS_OUTPUT_COLUMNS: tuple[str, ...] = (
    "warm_up",
    "bid_prior_n",
    "bid_prior_median",
    "bid_prior_std",
    "ask_prior_n",
    "ask_prior_median",
    "ask_prior_std",
)

#: Columns that must be present to score.
SCORE_INPUT_COLUMNS: tuple[str, ...] = tuple(
    f"{side}_{c}" for side in ("bid", "ask") for c in ("removal", "prior_median", "prior_std")
)

#: Columns appended by :func:`side_scores`.
SCORE_OUTPUT_COLUMNS: tuple[str, ...] = ("bid_z", "bid_score", "ask_z", "ask_score")


@dataclass(frozen=True)
class SpoofConfig:
    """Scoring parameters. Frozen so a run's settings can't drift mid-evaluation;
    bump ``version`` whenever a choice is re-made."""

    version: str = "l1-depletion-v1"
    window_s: int = 3600
    min_obs: int = 2
    z_floor: float = 2.0
    z_cap: float = 6.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.window_s < 1:
            raise ValueError(f"window_s must be >= 1, got {self.window_s}")
        if self.min_obs < 2:
            raise ValueError(f"min_obs must be >= 2 (a sample std needs two), got {self.min_obs}")
        if self.z_cap <= self.z_floor:
            raise ValueError(f"z_cap ({self.z_cap}) must exceed z_floor ({self.z_floor})")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")

    def fingerprint(self) -> str:
        """Short, stable id of these exact settings, for stamping outputs."""
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


#: The production settings (window 3600s, min_obs 2, ramp z 2..6).
DEFAULT_CONFIG = SpoofConfig()


def unexplained_removal(df: pl.DataFrame) -> pl.DataFrame:
    """Add per-side unexplained removal and identifiability flags.

    Returns a new frame with every original column preserved (same order, same
    values) plus ``REMOVAL_OUTPUT_COLUMNS``. Removal is null wherever the side is
    unidentifiable; see the module header for the rules.
    """
    _check_columns(df, REMOVAL_INPUT_COLUMNS, REMOVAL_OUTPUT_COLUMNS)

    original_columns = df.columns
    # shift(1) must mean "the previous second", whatever order the parquet was in.
    df = df.sort("timestamp")

    adjacent = (pl.col("timestamp").diff() == pl.duration(seconds=1)).fill_null(False)
    unusable = pl.col("book_stale") | pl.col("book_dead")
    # Row 0's t-1 doesn't exist, so it counts as unusable.
    clean_pair = adjacent & ~unusable & ~unusable.shift(1).fill_null(True)

    df = df.with_columns(
        _identifiable("bid_price", "bid_qty", "sell_qty", clean_pair).alias("bid_identifiable"),
        _identifiable("ask_price", "ask_qty", "buy_qty", clean_pair).alias("ask_identifiable"),
    )
    df = df.with_columns(
        pl.when(pl.col("bid_identifiable"))
        .then(_depletion("bid_qty", "sell_qty"))
        .alias("bid_removal"),
        pl.when(pl.col("ask_identifiable"))
        .then(_depletion("ask_qty", "buy_qty"))
        .alias("ask_removal"),
    )
    return df.select(*original_columns, *REMOVAL_OUTPUT_COLUMNS)


def trailing_stats(df: pl.DataFrame, cfg: SpoofConfig = DEFAULT_CONFIG) -> pl.DataFrame:
    """Add each side's baseline from the previous ``cfg.window_s`` seconds.

    Expects :func:`unexplained_removal` output. Returns every original column
    unchanged plus ``STATS_OUTPUT_COLUMNS``. ``{side}_prior_n`` is always the raw
    count of comparable removals in the window; median and std are null until the
    segment is warm and that count reaches ``cfg.min_obs``.
    """
    _check_columns(df, STATS_INPUT_COLUMNS, STATS_OUTPUT_COLUMNS)

    original_columns = df.columns
    df = df.sort("timestamp")

    t = pl.col("timestamp")
    dead = pl.col("book_dead")
    starts_segment = (
        (t.diff() != pl.duration(seconds=1)).fill_null(True)
        | (t.dt.date() != t.dt.date().shift(1)).fill_null(True)
        | dead
        | dead.shift(1).fill_null(False)
    )
    df = df.with_columns(starts_segment.cast(pl.Int64).cum_sum().alias("_segment"))
    df = df.with_columns((pl.int_range(pl.len()).over("_segment") < cfg.window_s).alias("warm_up"))

    for side in ("bid", "ask"):
        # shift(1) first so the window ends at t-1; over() keeps it inside the segment.
        prev = pl.col(f"{side}_removal").shift(1)
        df = df.with_columns(
            prev.is_not_null()
            .cast(pl.Int64)
            .rolling_sum(window_size=cfg.window_s, min_samples=1)
            .over("_segment")
            .alias(f"{side}_prior_n"),
            prev.rolling_median(window_size=cfg.window_s, min_samples=1)
            .over("_segment")
            .alias(f"{side}_prior_median"),
            prev.rolling_std(window_size=cfg.window_s, min_samples=1, ddof=1)
            .over("_segment")
            .alias(f"{side}_prior_std"),
        )
        usable = ~pl.col("warm_up") & (pl.col(f"{side}_prior_n") >= cfg.min_obs)
        df = df.with_columns(
            pl.when(usable).then(pl.col(f"{side}_prior_median")).alias(f"{side}_prior_median"),
            pl.when(usable).then(pl.col(f"{side}_prior_std")).alias(f"{side}_prior_std"),
        )

    return df.select(*original_columns, *STATS_OUTPUT_COLUMNS)


def side_scores(df: pl.DataFrame, cfg: SpoofConfig = DEFAULT_CONFIG) -> pl.DataFrame:
    """Add each side's z-score and bounded [0, 1] suspicion score.

    Expects :func:`trailing_stats` output. Returns every original column unchanged
    plus ``SCORE_OUTPUT_COLUMNS``; any null input gives a null z and score.
    """
    _check_columns(df, SCORE_INPUT_COLUMNS, SCORE_OUTPUT_COLUMNS)

    for side in ("bid", "ask"):
        z = (pl.col(f"{side}_removal") - pl.col(f"{side}_prior_median")) / (
            pl.col(f"{side}_prior_std") + cfg.eps
        )
        df = df.with_columns(z.alias(f"{side}_z"))
        ramp = (pl.col(f"{side}_z") - cfg.z_floor) / (cfg.z_cap - cfg.z_floor)
        df = df.with_columns(ramp.clip(0.0, 1.0).alias(f"{side}_score"))
    return df


def _check_columns(df: pl.DataFrame, required: tuple[str, ...], produced: tuple[str, ...]) -> None:
    """Fail fast on a missing input, or on a frame that already carries our outputs
    (e.g. a scored frame fed back in), before either turns into an opaque error."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"input frame missing required columns: {missing}")
    conflicting = [c for c in produced if c in df.columns]
    if conflicting:
        raise ValueError(f"input frame already contains output columns: {conflicting}")


def _identifiable(price: str, qty: str, opposing_flow: str, clean_pair: pl.Expr) -> pl.Expr:
    """True where this side's t-1 -> t queue change can be attributed at all."""
    same_price = (pl.col(price) == pl.col(price).shift(1)).fill_null(False)
    known = (
        pl.col(qty).is_not_null()
        & pl.col(qty).shift(1).is_not_null()
        & pl.col(opposing_flow).is_not_null()
    )
    return clean_pair & same_price & known


def _depletion(qty: str, opposing_flow: str) -> pl.Expr:
    """max(0, prev_qty - qty - opposing_flow): the drop no trade accounts for."""
    return (pl.col(qty).shift(1) - pl.col(qty) - pl.col(opposing_flow)).clip(lower_bound=0.0)
