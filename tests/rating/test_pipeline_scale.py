"""Scale tests for the scoped rating path (`rating.pipeline`).

These exist because the bug they cover was invisible at the size the original tests ran
at. `build_rating_inputs` fused every match in the warehouse to rate one player, which
is O(matches x deliveries): fine at 230 matches (~6 ms a match), and 557 s at 8,026
matches / 1.8M deliveries. Shrinking the fixture back down would make these tests pass
and tell nobody anything, so they don't.

**They assert a property, not a stopwatch reading.** The claim that matters is "the work
done to rate one player does not grow with the warehouse", and that is checked directly:
build two warehouses an order of magnitude apart, rate the same player against both, and
assert the same number of deliveries were read and fused. A wall-clock threshold would be
flaky on shared CI while telling you less about *why* it regressed -- and a regression
here would be a silent 500x, which a timing assertion generous enough not to flake would
happily let through.

One genuinely large case is included and marked `slow` (deselect with `-m "not slow"`),
because a property test over two small warehouses can't catch a constant factor that only
hurts at size. It builds its warehouse on the fly rather than committing a 25 MB fixture.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fusion.pipeline import fuse_and_aggregate
from ingestion.storage import ParquetStore
from rating.cohort import CohortSpec, match_dates
from rating.contracts import PlayerRole
from rating.engine import baseline_from_deliveries, compute_baseline
from rating.pipeline import prepare_rating, rate_and_suggest, resolve_scope

# Deliberately more than one competition, more than one gender, and several seasons:
# the whole reason the cohort layer exists is that the warehouse stopped being uniform.
_COMPETITIONS = ("Indian Premier League", "Big Bash League", "ICC Women's T20 World Cup")
_GENDERS = {"Indian Premier League": "male", "Big Bash League": "male",
            "ICC Women's T20 World Cup": "female"}

_BATTERS = tuple(f"Batter {i}" for i in range(6))
_BOWLERS = tuple(f"Bowler {i}" for i in range(6))


def _match_rows(match_id: str, competition: str, season: int) -> tuple[dict, list[dict]]:
    """One synthetic match: metadata plus two innings of six overs.

    Runs vary deterministically with the match and ball index rather than randomly, so
    every assertion in this file is reproducible and any failure is reproducible too.
    Six overs rather than twenty keeps a large warehouse buildable in test time while
    preserving the thing under test -- the number of *matches* is what used to hurt.
    """
    seed = abs(hash(match_id)) % 7
    deliveries = []
    for innings in (1, 2):
        for over in range(6):
            phase = "powerplay" if over < 2 else ("death" if over >= 4 else "middle")
            for ball in range(1, 7):
                index = over * 6 + ball
                batter = _BATTERS[(seed + innings + index) % len(_BATTERS)]
                bowler = _BOWLERS[(seed + over) % len(_BOWLERS)]
                runs = (seed + index + innings) % 5
                deliveries.append(
                    {
                        "match_id": match_id, "innings": innings, "over": over, "ball": ball,
                        "batting_team": f"T{innings}", "bowling_team": f"T{3 - innings}",
                        "striker": batter,
                        "non_striker": _BATTERS[(seed + innings + index + 1) % len(_BATTERS)],
                        "bowler": bowler,
                        "runs_batter": runs, "runs_extras": 0, "runs_total": runs,
                        "extras_type": None,
                        "wicket_player_out": batter if index == 17 and innings == 1 else None,
                        "wicket_kind": "bowled" if index == 17 and innings == 1 else None,
                        "phase": phase,
                    }
                )
    meta = {
        "match_id": match_id, "competition": competition, "match_type": "T20",
        "gender": _GENDERS[competition], "teams": ["T1", "T2"], "venue": "Ground",
        "city": "City", "dates": [f"{season}-03-{(seed % 27) + 1:02d}"],
        "toss_winner": "T1", "toss_decision": "bat", "outcome_winner": "T1",
        "outcome_by": "runs",
    }
    return meta, deliveries


def build_warehouse(root, n_matches: int) -> ParquetStore:
    """A warehouse of `n_matches` synthetic matches, spread over competitions and seasons."""
    metas, rows = [], []
    for i in range(n_matches):
        competition = _COMPETITIONS[i % len(_COMPETITIONS)]
        season = 2021 + (i // max(1, n_matches // 5)) % 5
        meta, deliveries = _match_rows(f"m{i:06d}", competition, season)
        metas.append(meta)
        rows.extend(deliveries)

    store = ParquetStore(root)
    pd.DataFrame(metas).to_parquet(store.root / "matches.parquet", index=False)
    pd.DataFrame(rows).to_parquet(store.root / "deliveries.parquet", index=False)
    return store


def test_work_to_rate_one_player_does_not_grow_with_the_warehouse(tmp_path):
    """The core regression. Twenty times the warehouse, the same work per rating.

    This is the property `build_rating_inputs` violated: it fused every match before it
    could answer anything, so a 20x warehouse was 20x the work (worse, actually --
    per-match cost grew too, because selecting one match's rows scanned the whole frame).
    """
    small = build_warehouse(tmp_path / "small", 30)
    large = build_warehouse(tmp_path / "large", 600)
    player = _BATTERS[0]

    small_inputs = prepare_rating(small, small.read_matches(), player, window_size=5)
    large_inputs = prepare_rating(large, large.read_matches(), player, window_size=5)

    assert small_inputs.window_match_ids and large_inputs.window_match_ids
    # Five matches' worth of deliveries either way -- not 30 matches' and 600 matches'.
    assert small_inputs.deliveries_fused == large_inputs.deliveries_fused
    assert small_inputs.deliveries_read == large_inputs.deliveries_read
    assert large_inputs.deliveries_fused <= 5 * 72  # window_size x deliveries per match


def test_rating_fuses_only_the_window_not_the_cohort(tmp_path):
    """The cohort is pooled, never fused.

    A baseline is a set of totals, so the cohort's rows never need to become
    `FusedDelivery` objects. If they ever do, the cohort cap becomes the thing bounding
    memory rather than a sampling decision, and this number jumps by two orders of
    magnitude -- which is exactly the regression to catch.
    """
    store = build_warehouse(tmp_path / "w", 300)
    matches = store.read_matches()
    cohort, window = resolve_scope(store, matches, _BATTERS[0], window_size=5)

    inputs = prepare_rating(store, matches, _BATTERS[0], window_size=5)

    assert len(cohort.match_ids) > len(window) * 5  # the cohort really is much larger
    assert inputs.deliveries_fused <= 5 * 72
    assert inputs.baseline is not None
    assert inputs.baseline.matches_in_cohort == len(cohort.match_ids)


def test_scoped_rating_matches_the_whole_warehouse_fuse_for_the_same_cohort(tmp_path):
    """The rewrite changed *which* matches are pooled, not *how*.

    Pinned by computing the baseline both ways over an identical cohort: the old path
    (fuse everything, then `compute_baseline`) and the new one (read the cohort's rows,
    pool them vectorised). Equal field-for-field, including rounding. Without this, "the
    fast path is the same answer" would be an assertion in a docstring.
    """
    store = build_warehouse(tmp_path / "w", 40)
    matches = store.read_matches()
    deliveries = store.read_deliveries()

    history, fused = fuse_and_aggregate(deliveries, match_dates(matches).to_dict())
    old_way = compute_baseline(history, fused)
    new_way = baseline_from_deliveries(deliveries, source=old_way.source)

    assert new_way == old_way
    # The precondition the equality rests on -- see `baseline_from_deliveries`' docstring.
    # Pitch geometry can't come out of the ball-by-ball frame, so the two agree only while
    # no fused row carries any. This fails loudly once a detector lands.
    assert not any(d.has_pitch_geometry() for d in fused)


def test_a_rating_is_unchanged_by_matches_outside_the_cohort_and_window(tmp_path):
    """Adding unrelated matches to the warehouse must not move an existing rating.

    The behavioural consequence of scoping, and the one an analyst would notice: before,
    ingesting another competition silently changed every stored rating, because the
    baseline pooled whatever happened to be on disk. Here the extra matches are a
    different competition and gender, so they're outside the cohort and must not count.
    """
    base = build_warehouse(tmp_path / "base", 60)
    matches = base.read_matches()
    player = _BATTERS[0]
    spec = CohortSpec(competitions=("Indian Premier League",), genders=("male",))

    before = prepare_rating(base, matches, player, window_size=5, spec=spec)
    rating_before, _ = rate_and_suggest(before, PlayerRole.TOP_ORDER_BATTER, window_size=5)

    # Same warehouse plus 200 women's-competition matches, which the spec excludes.
    grown = build_warehouse(tmp_path / "grown", 60)
    extra_meta, extra_rows = [], []
    for i in range(200):
        meta, rows = _match_rows(f"x{i:06d}", "ICC Women's T20 World Cup", 2024)
        extra_meta.append(meta)
        extra_rows.extend(rows)
    pd.concat([grown.read_matches(), pd.DataFrame(extra_meta)], ignore_index=True).to_parquet(
        grown.root / "matches.parquet", index=False
    )
    pd.concat([grown.read_deliveries(), pd.DataFrame(extra_rows)], ignore_index=True).to_parquet(
        grown.root / "deliveries.parquet", index=False
    )

    after = prepare_rating(grown, grown.read_matches(), player, window_size=5, spec=spec)
    rating_after, _ = rate_and_suggest(after, PlayerRole.TOP_ORDER_BATTER, window_size=5)

    assert before.baseline == after.baseline
    assert rating_before.composite == rating_after.composite


def test_cohort_is_capped_and_says_so(tmp_path):
    """A capped cohort reports the cap rather than presenting itself as the full set."""
    store = build_warehouse(tmp_path / "w", 400)
    matches = store.read_matches()
    spec = CohortSpec(seasons_back=None, max_matches=50)
    cohort, _ = resolve_scope(store, matches, _BATTERS[0], window_size=5, spec=spec)

    assert len(cohort.match_ids) == 50
    assert cohort.was_capped
    assert cohort.matches_available > 50
    assert "most recent 50 of" in cohort.source


@pytest.mark.slow
def test_full_scale_warehouse_rates_a_player_promptly(tmp_path):
    """One end-to-end run at the real warehouse's order of magnitude.

    The property tests above can't see a constant factor that only bites at size, so this
    builds ~8,000 matches and rates a player through the real entry point. The bound is
    deliberately loose (30 s against a path that measures well under a second here): it is
    there to catch a return to the 557 s behaviour, not to police a few hundred
    milliseconds of CI noise.
    """
    store = build_warehouse(tmp_path / "big", 8000)
    matches = store.read_matches()
    assert len(matches) == 8000

    import time

    started = time.perf_counter()
    inputs = prepare_rating(store, matches, _BATTERS[0], window_size=5)
    rating, suggestions = rate_and_suggest(inputs, PlayerRole.TOP_ORDER_BATTER, window_size=5)
    elapsed = time.perf_counter() - started

    assert rating is not None
    assert inputs.deliveries_fused <= 5 * 72
    assert elapsed < 30, f"rating one player took {elapsed:.1f}s at 8,000 matches"
