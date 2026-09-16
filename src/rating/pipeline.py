"""Scoped end-to-end wiring for "rate this one player" -- PRD stages 5-6 against a
warehouse too large to fuse whole.

The rating engine's arithmetic never needed the whole warehouse. Two things do the
work: a **cohort baseline** (a pooled aggregate) and a **player's last-N-match window**
(a few hundred deliveries). The app used to obtain both by fusing every match in the
warehouse into `FusedDelivery` objects and then filtering the objects -- correct, and
quadratic, because selecting one match's rows costs a scan of the whole frame and it
did that once per match. At 8,026 matches / 1.8M deliveries that measured ~550 s before
any rating was computed.

This module inverts the order: **select, read, then fuse**, instead of fuse, then
filter.

1. `cohort.cohort_and_window` decides which match ids matter -- a bounded, like-for-like
   cohort plus the player's window -- without touching a delivery row.
2. `ParquetStore.read_deliveries(match_ids=...)` pushes that selection down to the
   Parquet reader, so rows outside it are never decoded.
3. The cohort baseline is computed **vectorised** (`engine.baseline_from_deliveries`),
   because every field on `RatingBaseline` is a pooled total and the per-delivery
   objects cancel out of the sums.
4. Only the player's window -- typically five matches, ~600 deliveries -- is fused into
   `FusedDelivery` objects, because that is the only place per-delivery identity is
   actually used (phase weighting, and the `DeliveryRef` citations a flag has to carry).

The measured effect at 8,026 matches is in the README; the structural claim is the one
worth stating here: **the cost of rating one player is now a function of that player's
window and the cohort cap, not of warehouse size.** `test_rating_pipeline_scale.py`
asserts that as a property rather than as a stopwatch reading, by checking that the same
request against a warehouse an order of magnitude larger fuses the same number of rows.

Contracts are unchanged. `rate_player`, `suggest_for_player` and `compute_baseline`
have the same signatures and the same semantics they had before; `rate_player` still
takes a history and windows it internally, and is simply handed a history that has
already been narrowed to the matches it was going to keep. The one deliberate change of
*behaviour* is which matches end up in the cohort -- see `cohort`'s module docstring for
why pooling a men's IPL final with a women's associate T20I was a correctness problem
before it was a performance one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import pandas as pd

from fusion.contracts import FusedDelivery, PlayerMatchStats
from fusion.pipeline import fuse_and_aggregate

from .cohort import CohortSelection, CohortSpec, cohort_and_window, match_dates, matches_to_read
from .contracts import CoachingSuggestion, PlayerRating, PlayerRole, RatingBaseline
from .engine import baseline_from_deliveries, rate_player
from .llm import NoteWriter
from .suggestions import FLAG_RELATIVE_THRESHOLD, MIN_BALLS_FOR_FLAG, suggest_for_player

__all__ = [
    "DeliverySource",
    "ScopedRatingInputs",
    "cohort_baseline",
    "player_window",
    "prepare_rating",
    "rate_and_suggest",
    "resolve_scope",
]


class DeliverySource(Protocol):
    """The slice of `ingestion.storage.ParquetStore` this module needs.

    A `Protocol` rather than an import of the concrete store so `rating` doesn't take a
    hard dependency on `ingestion`'s storage backend -- the stated direction is "build
    toward a real database", and the point of pushing selection down to the reader is
    that a SQL-backed store can satisfy exactly this interface with a `WHERE` clause.
    Tests pass a real `ParquetStore` over a temporary directory rather than a mock, in
    keeping with the rest of this repo's test style.
    """

    def read_matches(self, match_ids=None, competitions=None, columns=None) -> pd.DataFrame: ...

    def read_deliveries(self, match_ids=None, columns=None) -> pd.DataFrame: ...


@dataclass(frozen=True)
class ScopedRatingInputs:
    """Everything one player's rating needs, and nothing else.

    `window_history` is deliberately *only* the player's window rather than the full
    history `rate_player`'s signature accepts. `rate_player` windows internally, so
    handing it a pre-narrowed history returns the identical rating -- it is the same
    last-N matches either way, picked by the same `(match_date, match_id)` tie-break.
    Narrowing first is what stops the caller from having to build a `PlayerMatchStats`
    for every player in every match in the warehouse to rate one of them.

    `deliveries_read` and `deliveries_fused` are not debug counters. They are the two
    numbers that say whether the select-then-fuse ordering actually held, and
    `tests/rating/test_rating_pipeline_scale.py` asserts on them directly -- a wall-clock
    assertion would be flaky on shared CI while saying less about why.
    """

    player: str
    baseline: Optional[RatingBaseline]
    window_history: tuple[PlayerMatchStats, ...]
    window_fused: tuple[FusedDelivery, ...]
    window_match_ids: tuple[str, ...]
    cohort: CohortSelection
    deliveries_read: int
    deliveries_fused: int

    @property
    def has_data(self) -> bool:
        return self.baseline is not None and bool(self.window_match_ids)


def cohort_baseline(
    store: DeliverySource, cohort: CohortSelection, describe: bool = True
) -> Optional[RatingBaseline]:
    """The pooled `RatingBaseline` for `cohort`, read and aggregated without fusing.

    Reads only the cohort's matches (predicate pushdown) and only the columns the pooled
    sums touch, then aggregates vectorised. No `FusedDelivery` is constructed: a baseline
    is a set of totals, and totals don't need per-delivery identity.

    `describe` puts `CohortSelection.source` onto the baseline, which is what carries the
    scope into every flag and coaching note. Turning it off leaves `compute_baseline`'s
    generic "cohort mean over N matches, M players" wording, which is honest but says
    nothing about *which* cohort -- only useful when the caller is about to supply its
    own description.

    Returns `None` for an empty cohort, matching `compute_baseline`'s contract: there is
    no baseline over no matches, and a zero-filled one would make every downstream ratio
    undefined in a way the caller couldn't see.
    """
    if not cohort:
        return None
    rows = store.read_deliveries(match_ids=cohort.match_ids, columns=_BASELINE_COLUMNS)
    if rows.empty:
        return None
    return baseline_from_deliveries(rows, source=cohort.source if describe else None)


# The columns the pooled baseline sums actually read. Narrowing the projection matters at
# warehouse scale for the same reason narrowing the rows does -- Parquet is columnar, so
# the eight columns nothing sums are simply never decoded.
_BASELINE_COLUMNS = (
    "match_id", "phase", "striker", "bowler", "runs_batter", "runs_extras",
    "runs_total", "extras_type", "wicket_player_out", "wicket_kind",
)


def resolve_scope(
    store: DeliverySource,
    matches: pd.DataFrame,
    player: str,
    window_size: int = 5,
    spec: CohortSpec = CohortSpec(),
) -> tuple[CohortSelection, tuple[str, ...]]:
    """`(cohort, window_match_ids)` for one player -- which matches, before any are read.

    Public and separate from `prepare_rating` because the cohort is what the baseline
    caches on. A baseline depends on the cohort and the warehouse, never on the player,
    so an interactive caller wants to resolve the scope, look the baseline up by it, and
    only then do the per-player work. Folding this into `prepare_rating` would force
    every player switch to recompute a baseline that hasn't changed.

    The only read here is this player's own delivery rows, four columns wide, pushed
    down to the reader as a disjunction (see `ParquetStore.read_deliveries`) -- so
    answering "which matches has this player played in" costs a filtered read rather
    than a scan of three columns of the whole warehouse.
    """
    appearances = store.read_deliveries(
        columns=("match_id", "striker", "non_striker", "bowler"), involving_player=player
    )
    return cohort_and_window(appearances, matches, player, window_size, spec)


def player_window(
    store: DeliverySource,
    matches: pd.DataFrame,
    player: str,
    window_match_ids: Sequence[str],
) -> tuple[list[PlayerMatchStats], list[FusedDelivery], int]:
    """`(history, fused, rows_read)` for just the matches in `window_match_ids`.

    This is the only place `FusedDelivery` objects are built on the rating path, and it
    builds at most `window_size` matches' worth of them -- a few hundred rows. Video
    analyses aren't passed because none exist (`BroadcastFeedAdapter` is unimplemented),
    so every row comes out with `video_available=False`, exactly as before.
    """
    if not window_match_ids:
        return [], [], 0
    rows = store.read_deliveries(match_ids=window_match_ids)
    dates = match_dates(matches[matches["match_id"].isin(list(window_match_ids))]).to_dict()
    history, fused = fuse_and_aggregate(rows, dates)
    return history, fused, len(rows)


def prepare_rating(
    store: DeliverySource,
    matches: pd.DataFrame,
    player: str,
    window_size: int = 5,
    spec: CohortSpec = CohortSpec(),
    baseline: Optional[RatingBaseline] = None,
    scope: Optional[tuple[CohortSelection, Sequence[str]]] = None,
) -> ScopedRatingInputs:
    """Select, read and fuse exactly what rating `player` needs.

    `matches` is passed in rather than read here because the match table is small (one
    row per match, ~13 KB for 230 matches) and every caller already holds it -- re-reading
    it per player would be the one full-table read this module is trying to avoid.

    `baseline` lets a caller supply an already-computed cohort baseline. That is the
    difference between an interactive app that recomputes the cohort on every widget
    interaction and one that computes it once per cohort: the baseline depends only on
    the `CohortSpec` and the warehouse, never on which player is selected.

    `scope` lets that same caller hand back the `(cohort, window)` pair it already got
    from `resolve_scope` to look the baseline up with. Without it, a caller doing the
    cache-the-baseline dance resolves the scope twice per click -- once to key the cache
    and once in here -- which is a second filtered read for an answer it is already
    holding. Passing `scope` and `baseline` together is the intended interactive path.
    """
    cohort, window_ids = scope if scope is not None else resolve_scope(
        store, matches, player, window_size, spec
    )
    if baseline is None:
        baseline = cohort_baseline(store, cohort)

    history, fused, rows_read = player_window(store, matches, player, window_ids)
    return ScopedRatingInputs(
        player=player,
        baseline=baseline,
        window_history=tuple(history),
        window_fused=tuple(fused),
        window_match_ids=tuple(window_ids),
        cohort=cohort,
        deliveries_read=rows_read,
        deliveries_fused=len(fused),
    )


def rate_and_suggest(
    inputs: ScopedRatingInputs,
    role: PlayerRole,
    window_size: int = 5,
    writer: Optional[NoteWriter] = None,
    threshold: float = FLAG_RELATIVE_THRESHOLD,
    min_balls: int = MIN_BALLS_FOR_FLAG,
    limit: Optional[int] = None,
    concerns_only: bool = True,
) -> tuple[Optional[PlayerRating], list[CoachingSuggestion]]:
    """Run the unchanged stage 5-6 functions over already-scoped inputs.

    A thin seam on purpose: `rate_player` and `suggest_for_player` are called with
    exactly the arguments they always took. Nothing about the scoring, flagging, ranking
    or note-writing changed in this pass, and this function exists so that stays visibly
    true rather than being asserted in a commit message.

    Both halves are driven off the *same* `window_fused`, which fixes a real
    inconsistency in the app: it derived the rating window inside `rate_player` and then
    re-derived a window for the suggestions panel separately, so a change to either
    sort could have had the two views of "this player, recently" quietly disagree.
    """
    if inputs.baseline is None:
        return None, []

    rating = rate_player(
        list(inputs.window_history),
        inputs.player,
        role,
        inputs.baseline,
        list(inputs.window_fused),
        window_size=window_size,
    )
    suggestions = suggest_for_player(
        list(inputs.window_fused),
        inputs.player,
        inputs.baseline,
        writer=writer,
        threshold=threshold,
        min_balls=min_balls,
        limit=limit,
        concerns_only=concerns_only,
    )
    return rating, suggestions
