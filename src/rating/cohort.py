"""Which matches a rating is computed *against* -- the selection layer in front of
`engine.compute_baseline`.

This module exists because the warehouse stopped being one competition. `ParquetStore`
used to overwrite on every ingest, so whatever was on disk was implicitly a single
competition and `RatingBaseline`'s "feed it one competition's matches and it is that
competition's season average" was true by accident. Once writes started merging, the
same call pooled ICC T20 World Cups, the IPL, the BBL, the PSL, the NPL and the full
men's and women's T20I record into one number. That is not a worse-resolution baseline,
it is a **different and largely meaningless one**: a "cohort mean" spanning a men's IPL
final and a women's associate-nation T20I describes no player who has ever existed, and
every `PerformanceFlag` scored against it inherits that.

So the correctness fix and the performance fix are the same fix, and it is worth being
explicit that the correctness half came first. Scoping the cohort to matches comparable
to the ones the player actually played (`cohort_for_player`) makes the baseline mean
something; it also happens to reduce the rows that have to be read and fused from the
whole warehouse to a bounded slice. If it were only a speed argument, the right answer
would have been to cache the whole-warehouse pooling; it isn't, so it isn't.

**Three deliberate biases, all stated rather than hidden**, because a cohort is a
judgement and `RatingBaseline.source` is the only place an analyst can see which one
was made:

1. *Like-for-like scoping.* `cohort_for_player` reads the competition, gender and match
   type off the player's own window and scopes the cohort to those. It does not guess a
   "peer group" from anything the data doesn't carry -- the same refusal
   `prediction.contracts.MatchupFeatures` and `contracts.PlayerRole` make about
   batting/bowling style.
2. *Recency bound.* Cricket from 2009 is not the reference for a 2026 innings. The
   default window is `DEFAULT_SEASONS_BACK` seasons ending at the player's most recent
   match. This is a picked number, not a fitted one, in the same sense as
   `contracts.PHASE_IMPACT_WEIGHTS`.
3. *A cap, applied most-recent-first.* `DEFAULT_MAX_COHORT_MATCHES` bounds the work a
   single rating can trigger no matter how large the warehouse grows. The cap is a real
   sampling bias toward recent matches within the already-scoped set -- which is the
   direction you want for a "current average" reference, but it is still a bias, so
   `CohortSelection.source` says when it bit and `matches_available` says what it was
   sampled from.

Nothing here computes a statistic. It selects `match_id`s and describes the selection;
`engine.compute_baseline` / `engine.baseline_from_deliveries` do the arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import pandas as pd

__all__ = [
    "DEFAULT_SEASONS_BACK",
    "DEFAULT_MAX_COHORT_MATCHES",
    "CohortSpec",
    "CohortSelection",
    "WHOLE_WAREHOUSE",
    "match_dates",
    "player_match_ids",
    "select_cohort",
    "cohort_for_player",
    "cohort_and_window",
    "matches_to_read",
]

# Seasons of history a baseline pools over by default, counted back from the most recent
# match in scope. Picked, not fitted: long enough that a cohort rate isn't one tournament's
# pitch conditions, short enough that it describes how the format is currently played.
# Nothing in this repo has the data to fit it -- that would mean measuring when cohort
# rates stop predicting the next season's, across competitions, which needs the very
# multi-season warehouse this module was written to make usable.
DEFAULT_SEASONS_BACK = 3

# The most matches one baseline will ever pool, applied most-recent-first. Bounds the work
# a single `rate_player` call can trigger independently of warehouse size -- the property
# that stops this from regressing the next time the warehouse grows an order of magnitude.
# ~600 T20s is on the order of 140k deliveries, which is a sample far past the point where
# a pooled rate is still moving.
DEFAULT_MAX_COHORT_MATCHES = 600


@dataclass(frozen=True)
class CohortSpec:
    """Which matches are allowed into a baseline.

    Empty tuple means "don't filter on this dimension", not "match nothing" -- the same
    None-vs-empty distinction `ingestion.storage` draws, spelled the other way round
    because a spec is a value object that has to be hashable (it is used as a cache key
    by the app layer) and `None` and `()` reading differently in that role would be a
    trap. `seasons_back=None` and `max_matches=None` genuinely do mean "unbounded", and
    `WHOLE_WAREHOUSE` is the spec that turns all of it off at once.
    """

    competitions: tuple[str, ...] = ()
    genders: tuple[str, ...] = ()
    match_types: tuple[str, ...] = ()
    seasons_back: Optional[int] = DEFAULT_SEASONS_BACK
    max_matches: Optional[int] = DEFAULT_MAX_COHORT_MATCHES

    def describe(self) -> str:
        """The scope in words, for `RatingBaseline.source`.

        Short on purpose: this string is quoted inline in coaching notes, not just
        logged, and `engine.compute_baseline` already appends the match/player counts.
        """
        parts: list[str] = []
        if self.competitions:
            parts.append(" + ".join(self.competitions) if len(self.competitions) <= 2
                         else f"{len(self.competitions)} competitions")
        if self.genders:
            parts.append("/".join(self.genders))
        if self.match_types:
            parts.append("/".join(self.match_types))
        if self.seasons_back:
            parts.append(f"last {self.seasons_back} seasons")
        return ", ".join(parts) if parts else "whole warehouse"


# The pre-fix behaviour, kept reachable and named. `compute_baseline(history, fused)` over
# everything is still the right cohort for a genuinely single-competition warehouse, and a
# caller who wants it should be able to say so out loud rather than by omission.
WHOLE_WAREHOUSE = CohortSpec(seasons_back=None, max_matches=None)


@dataclass(frozen=True)
class CohortSelection:
    """The matches a baseline will pool, plus how they were chosen.

    `matches_available` is the count *before* `CohortSpec.max_matches` was applied, so a
    reader can tell a cohort that used everything in scope from one that sampled the most
    recent slice of a much larger set. Without it, `source` would say "600 matches" for
    both and the sampling would be invisible -- which is exactly the sort of unstated
    choice PRD 2.4 rules out.
    """

    match_ids: tuple[str, ...]
    source: str
    matches_available: int
    spec: CohortSpec

    def __bool__(self) -> bool:
        return bool(self.match_ids)

    @property
    def was_capped(self) -> bool:
        return self.matches_available > len(self.match_ids)


def _first_date(dates) -> str:
    """The match's start date as a plain string, or "" when it has none.

    `dates` arrives as a list/tuple/ndarray depending on how the frame was built
    (parquet round-trips the tuple as an ndarray), so this normalises rather than
    assuming -- an ndarray is truthy-ambiguous and `if dates` on one raises.
    """
    if dates is None:
        return ""
    try:
        return str(dates[0]) if len(dates) else ""
    except (TypeError, IndexError):
        return ""


def match_dates(matches: pd.DataFrame) -> pd.Series:
    """`match_id -> start date` for every row of `matches`.

    One place, because four call sites used to each write
    `matches.set_index("match_id")["dates"].apply(lambda d: d[0] if len(d) else "")`
    and an empty-`dates` match would have raised in whichever of them forgot the guard.
    """
    if matches.empty:
        return pd.Series(dtype=object, name="date")
    return (
        matches.set_index("match_id")["dates"].apply(_first_date).rename("date")
    )


def player_match_ids(
    deliveries: pd.DataFrame, matches: pd.DataFrame, player: str, n: Optional[int] = None
) -> list[str]:
    """Match ids `player` batted, bowled or was non-striker in, most recent first.

    Same selection and the same deterministic `(date, match_id)` descending tie-break as
    `player_reports.last_n_match_ids` -- duplicated here rather than imported to keep
    `rating` from depending on `player_reports`, and pinned identical by
    `tests/rating/test_cohort.py`. `n=None` returns all of them, which is what
    `cohort_for_player` needs to see the player's full competition footprint.
    """
    if deliveries.empty or matches.empty:
        return []
    played_in = deliveries[
        (deliveries["striker"] == player)
        | (deliveries["non_striker"] == player)
        | (deliveries["bowler"] == player)
    ]["match_id"].unique()

    dates = match_dates(matches)
    played = dates.loc[dates.index.intersection(played_in)]
    ordered = played.reset_index().sort_values(["date", "match_id"], ascending=False)
    ids = ordered["match_id"].tolist()
    return ids if n is None else ids[:n]


def _season(date: str) -> Optional[int]:
    """The calendar year a date string starts with, or `None` if it isn't one.

    Calendar year rather than a split "2025/26" season label: Cricsheet dates are ISO,
    and inventing a hemisphere-aware season boundary would be a guess that changes which
    matches land in a cohort. A BBL season straddling New Year is therefore split across
    two of this function's seasons, which widens `DEFAULT_SEASONS_BACK` by at most one
    season's worth of matches and never silently drops any.
    """
    if not date or len(date) < 4 or not date[:4].isdigit():
        return None
    return int(date[:4])


def select_cohort(
    matches: pd.DataFrame,
    spec: CohortSpec = CohortSpec(),
    reference_date: Optional[str] = None,
) -> CohortSelection:
    """The matches `spec` selects out of `matches`, most recent first.

    `reference_date` anchors the recency window; it defaults to the most recent match
    that survived the categorical filters. Anchoring to the *data* rather than to today's
    clock is what makes a baseline reproducible: the same warehouse and the same spec
    must produce the same cohort in six months' time, or a stored `CoachingSuggestion`
    can no longer be checked against the baseline it cites.

    Returns an empty selection (falsy) rather than raising when nothing matches -- a spec
    naming a competition the warehouse doesn't hold is a legitimate question with the
    answer "no data", and the caller can say that far better than a traceback can.
    """
    if matches.empty:
        return CohortSelection((), spec.describe(), 0, spec)

    frame = matches
    for column, allowed in (
        ("competition", spec.competitions),
        ("gender", spec.genders),
        ("match_type", spec.match_types),
    ):
        if allowed and column in frame.columns:
            frame = frame[frame[column].isin(list(allowed))]
    if frame.empty:
        return CohortSelection((), spec.describe(), 0, spec)

    dated = match_dates(frame).reset_index().sort_values(["date", "match_id"], ascending=False)

    if spec.seasons_back:
        anchor = _season(reference_date) if reference_date else None
        if anchor is None:
            anchor = next(
                (s for s in (_season(d) for d in dated["date"]) if s is not None), None
            )
        if anchor is not None:
            earliest = anchor - spec.seasons_back + 1
            seasons = dated["date"].map(_season)
            # Undated matches are kept, not dropped: a missing date is a metadata gap, and
            # silently excluding those matches would shrink the cohort for a reason that
            # has nothing to do with the analyst's question.
            dated = dated[seasons.isna() | ((seasons >= earliest) & (seasons <= anchor))]

    available = len(dated)
    if spec.max_matches is not None:
        dated = dated.head(spec.max_matches)

    source = spec.describe()
    if available > len(dated):
        source = f"{source}, most recent {len(dated)} of {available}"
    return CohortSelection(tuple(dated["match_id"]), source, available, spec)


def cohort_for_player(
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
    player: str,
    window_size: int = 5,
    spec: CohortSpec = CohortSpec(),
) -> CohortSelection:
    """The like-for-like cohort for rating `player` over their last `window_size` matches.

    Reads the competition, gender and match type off the player's own window and narrows
    `spec` to those before selecting, so a player rated on their last five IPL matches is
    scored against IPL cricket rather than against a pooled average of every competition
    the warehouse happens to hold. Dimensions the caller already pinned in `spec` are left
    alone -- an explicit request beats an inferred one.

    The recency anchor is the player's most recent match, not the warehouse's, so a player
    who last played two seasons ago is compared against the cricket they actually played
    in rather than against seasons they were absent for.

    A player with no matches yields an empty selection; there is no sensible cohort for a
    player the warehouse has never seen, and inventing one (the whole warehouse, say)
    would produce a confident rating for someone with no data behind it.
    """
    window = player_match_ids(deliveries, matches, player, n=window_size)
    if not window:
        return CohortSelection((), spec.describe(), 0, spec)

    window_rows = matches[matches["match_id"].isin(window)]
    inferred: dict[str, tuple[str, ...]] = {}
    for field, column in (
        ("competitions", "competition"),
        ("genders", "gender"),
        ("match_types", "match_type"),
    ):
        if getattr(spec, field) or column not in window_rows.columns:
            continue
        values = sorted({v for v in window_rows[column].dropna().tolist()})
        if values:
            inferred[field] = tuple(values)

    scoped = replace(spec, **inferred) if inferred else spec
    anchor = match_dates(window_rows).max() if not window_rows.empty else None
    return select_cohort(matches, scoped, reference_date=anchor)


def cohort_and_window(
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
    player: str,
    window_size: int = 5,
    spec: CohortSpec = CohortSpec(),
) -> tuple[CohortSelection, tuple[str, ...]]:
    """`(cohort, window_match_ids)` -- the two match sets a single rating needs.

    Returned together because they must be derived from the same call: the app previously
    computed the player's window twice (once inside `rate_player`, once again for the
    suggestion panel) and the two could disagree the moment either sort changed. The
    union of these two tuples is the complete set of matches that has to be read and
    fused to rate one player -- which is the whole point of the selection layer.
    """
    selection = cohort_for_player(deliveries, matches, player, window_size, spec)
    window = tuple(player_match_ids(deliveries, matches, player, n=window_size))
    return selection, window


def matches_to_read(cohort: CohortSelection, window: Sequence[str]) -> tuple[str, ...]:
    """The match ids to pull out of storage for one rating, deduplicated and ordered.

    Ordered so the resulting read and fuse are reproducible call to call; a set would be
    correct and would let the same inputs produce differently-ordered `FusedDelivery`
    lists, which `rating.suggestions` documents it will not tolerate in citation choice.
    """
    return tuple(sorted(set(cohort.match_ids) | set(window)))
