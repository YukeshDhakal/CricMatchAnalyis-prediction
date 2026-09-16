"""Normalised storage for the ball-by-ball table and match metadata.

Parquet stands in for the PRD's "columnar warehouse" (2.3) at MVP scale --
same columnar-on-disk shape, no server to run. Swap for a real warehouse
(e.g. a managed columnar store) later without changing the write call sites.

**Writes merge, they don't replace.** `write_matches`/`write_deliveries` upsert into
whatever's already on disk (keyed by `match_id` for matches, by `(match_id, innings,
over, ball)` -- the same join key `Delivery`'s docstring documents -- for deliveries)
rather than overwriting the whole file. Before this, `ingest-stats --competition X`
followed by `ingest-stats --competition Y` would silently wipe X's matches the moment
Y was written, because the old implementation was a plain `to_parquet` with no read of
what was already there -- every warehouse was implicitly single-competition, whether
or not that was ever the intent. Re-ingesting a match_id that's already stored replaces
its old row (last-write-wins), so a genuine re-fetch of the same competition still
picks up upstream corrections rather than duplicating rows.

**Reads can be scoped, and at this warehouse's size they should be.** Once merging
made a multi-competition warehouse possible, `read_deliveries()` stopped being a cheap
call: the ball-by-ball table grows by roughly 230 rows per match, so a full-history
warehouse is millions of rows and materialising all of them to answer a question about
one player's last five matches is most of the work for none of the answer. Both readers
therefore take optional filters that are pushed down to the Parquet reader
(`pyarrow`'s `filters=`) rather than applied to an already-materialised DataFrame --
row groups that can't match are never decoded. `read_deliveries(match_ids=[...])` is
the call the rating path uses; the no-argument form still returns everything, so
existing callers are unaffected.

Predicate pushdown is also the migration seam for the stated direction of travel ("a
real database"): a SQL-backed store swaps the `filters=` translation for a `WHERE`
clause and every call site keeps working, whereas a call site that reads everything and
filters in pandas has to be rewritten.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from .contracts import Delivery, MatchMeta


class ParquetStore:
    def __init__(self, root: Path | str = "data/warehouse"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _merge_write(self, path: Path, new_df: pd.DataFrame, key_cols: list[str]) -> None:
        """Shared upsert logic for both parquet files -- see the module docstring for
        why this merges instead of overwriting. `keep="last"` means the rows just
        passed in win over whatever was already on disk for the same key."""
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=key_cols, keep="last")
        else:
            combined = new_df
        combined.to_parquet(path, index=False)

    def write_matches(self, matches: list[MatchMeta]) -> Path:
        rows = [dict(asdict(m), teams=list(m.teams), dates=list(m.dates)) for m in matches]
        path = self.root / "matches.parquet"
        self._merge_write(path, pd.DataFrame(rows), key_cols=["match_id"])
        return path

    def write_deliveries(self, deliveries: list[Delivery]) -> Path:
        path = self.root / "deliveries.parquet"
        rows = [asdict(d) for d in deliveries]
        self._merge_write(path, pd.DataFrame(rows), key_cols=["match_id", "innings", "over", "ball"])
        return path

    def read_matches(
        self,
        match_ids: Optional[Iterable[str]] = None,
        competitions: Optional[Iterable[str]] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Match metadata, optionally narrowed before any row group is decoded.

        Passing `None` for a filter means "don't filter on it" -- so the no-argument
        call is exactly the old behaviour. An *empty* iterable is not the same thing
        and never has been: it means "no matches were selected", and returns an empty
        frame with the right columns rather than the whole warehouse. Getting that
        backwards would turn a cohort that legitimately selected nothing into a silent
        full-warehouse scan, which is the specific failure this module is trying to
        stop being possible.
        """
        return self._read(
            self.root / "matches.parquet",
            filters=_and_filters(("match_id", match_ids), ("competition", competitions)),
            columns=columns,
        )

    def read_deliveries(
        self,
        match_ids: Optional[Iterable[str]] = None,
        columns: Optional[Sequence[str]] = None,
        involving_player: Optional[str] = None,
    ) -> pd.DataFrame:
        """Ball-by-ball rows, optionally restricted to `match_ids`.

        The filter is handed to the Parquet reader, not applied afterwards, so the cost
        is proportional to the matches asked for rather than to the warehouse. Same
        empty-vs-None rule as `read_matches`.

        `involving_player` keeps only deliveries where that name appears as striker,
        non-striker or bowler -- the same "was in the middle" definition
        `player_reports.last_n_match_ids` and `rating.cohort.player_match_ids` use, so
        the three can't disagree about which matches a player played in. It is expressed
        as a disjunction so the Parquet reader can evaluate it (pyarrow takes filters in
        disjunctive normal form: a list of AND-clauses, OR-ed together), which is what
        makes "which matches has this player played in" cost a filtered read instead of
        a scan of three columns of the whole warehouse.

        Combining `involving_player` with `match_ids` ANDs the match restriction into
        each of the three disjuncts, because a DNF filter has no other way to express
        "(A or B or C) and D".
        """
        return self._read(
            self.root / "deliveries.parquet",
            filters=_delivery_filters(match_ids, involving_player),
            columns=columns,
        )

    @staticmethod
    def _read(path: Path, filters: Optional[list], columns: Optional[Sequence[str]]) -> pd.DataFrame:
        cols = list(columns) if columns is not None else None
        if filters is None:
            return pd.read_parquet(path, columns=cols)
        if not filters:
            # At least one filter selected nothing -- see read_matches' docstring. Read
            # the schema only (a zero-row predicate) so the caller still gets the right
            # columns to operate on.
            return pd.read_parquet(path, columns=cols).iloc[0:0]
        return pd.read_parquet(path, columns=cols, filters=filters)


def _and_filters(*specs: tuple[str, Optional[Iterable[str]]]) -> Optional[list]:
    """Translate `(column, allowed_values)` pairs into pyarrow's conjunctive filter form.

    Returns `None` when every spec is unfiltered (read everything) and `[]` when any
    spec selected nothing (read nothing) -- the two cases `ParquetStore._read`
    distinguishes. Values are de-duplicated and sorted so the same logical selection
    always produces the same filter expression, which keeps a filtered read
    cache-friendly and reproducible.
    """
    clauses = []
    for column, values in specs:
        if values is None:
            continue
        allowed = sorted(set(values))
        if not allowed:
            return []
        clauses.append((column, "in", allowed))
    return clauses or None


# The three columns a player can appear in on one ball-by-ball row. Bowler included
# because a bowling-only appearance is still an appearance -- a death-overs bowler who
# never batted has played the match.
_PLAYER_COLUMNS = ("striker", "non_striker", "bowler")


def _delivery_filters(
    match_ids: Optional[Iterable[str]], involving_player: Optional[str]
) -> Optional[list]:
    """Filters for `read_deliveries`, in pyarrow's disjunctive normal form where needed.

    Without `involving_player` this is the plain conjunctive form `_and_filters`
    produces. With it, the result is one AND-clause per player column, each carrying the
    match restriction too -- `[[player-is-striker, match-in-X], [player-is-non-striker,
    match-in-X], [player-is-bowler, match-in-X]]`. Repeating the match clause in every
    disjunct is not redundancy that could be factored out; it is how DNF expresses a
    conjunction over a disjunction, and dropping it from any one clause would silently
    widen that branch to the whole warehouse.
    """
    base = _and_filters(("match_id", match_ids))
    if involving_player is None:
        return base
    if base == []:
        return []
    shared = base or []
    return [[(column, "=", involving_player), *shared] for column in _PLAYER_COLUMNS]
