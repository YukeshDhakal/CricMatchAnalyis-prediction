"""Tests for cohort selection (`rating.cohort`).

Cohort selection is a judgement expressed as code, so these tests are mostly about the
judgement being the documented one and staying visible: which matches get in, which
don't, and whether the resulting `source` string tells an analyst what happened. The
performance consequence of scoping is covered separately in `test_pipeline_scale.py`.

Frames are built by hand here rather than sampled from the real warehouse, for the same
reason `helpers.py` builds rows by hand: a test that can state its input in a few lines
can also state exactly which match ids it expects back.
"""
from __future__ import annotations

import pandas as pd
import pytest

from player_reports.stats import last_n_match_ids
from rating.cohort import (
    WHOLE_WAREHOUSE,
    CohortSpec,
    cohort_for_player,
    match_dates,
    matches_to_read,
    player_match_ids,
    select_cohort,
)


def _matches(*rows: tuple[str, str, str, str]) -> pd.DataFrame:
    """`(match_id, competition, gender, date)` tuples into a match table."""
    return pd.DataFrame(
        [
            {"match_id": m, "competition": c, "gender": g, "match_type": "T20", "dates": [d]}
            for m, c, g, d in rows
        ]
    )


def _deliveries(*rows: tuple[str, str, str, str]) -> pd.DataFrame:
    """`(match_id, striker, non_striker, bowler)` tuples into a ball-by-ball table."""
    return pd.DataFrame(
        [{"match_id": m, "striker": s, "non_striker": n, "bowler": b} for m, s, n, b in rows]
    )


def test_select_cohort_filters_on_every_categorical_dimension():
    matches = _matches(
        ("m1", "IPL", "male", "2025-04-01"),
        ("m2", "BBL", "male", "2025-04-02"),
        ("m3", "IPL", "female", "2025-04-03"),
    )
    selection = select_cohort(
        matches, CohortSpec(competitions=("IPL",), genders=("male",), seasons_back=None)
    )
    assert selection.match_ids == ("m1",)


def test_an_empty_dimension_means_do_not_filter_on_it():
    """`()` is "any", not "none" -- the distinction `CohortSpec`'s docstring turns on."""
    matches = _matches(("m1", "IPL", "male", "2025-04-01"), ("m2", "BBL", "male", "2025-04-02"))
    assert set(select_cohort(matches, CohortSpec(seasons_back=None)).match_ids) == {"m1", "m2"}


def test_recency_bound_is_anchored_to_the_data_not_to_the_clock():
    """A stored suggestion has to stay checkable against the baseline it cites.

    If the recency window were anchored to `today`, the same warehouse and the same spec
    would select a different cohort next season, and a `CoachingSuggestion` citing
    "last 3 seasons" could no longer be recomputed. Anchoring to the most recent match in
    scope makes the selection a function of the data alone.
    """
    matches = _matches(
        ("old", "IPL", "male", "2015-04-01"),
        ("mid", "IPL", "male", "2023-04-01"),
        ("new", "IPL", "male", "2025-04-01"),
    )
    selection = select_cohort(matches, CohortSpec(seasons_back=3))
    assert set(selection.match_ids) == {"mid", "new"}  # 2023..2025, not 2015

    # Explicitly anchoring elsewhere moves the window, and only the anchor moves it.
    earlier = select_cohort(matches, CohortSpec(seasons_back=3), reference_date="2016-01-01")
    assert set(earlier.match_ids) == {"old"}


def test_undated_matches_are_kept_rather_than_silently_dropped():
    """A missing date is a metadata gap, not a reason to shrink an analyst's cohort."""
    matches = _matches(("dated", "IPL", "male", "2025-04-01"))
    matches = pd.concat(
        [matches, pd.DataFrame([{"match_id": "undated", "competition": "IPL",
                                 "gender": "male", "match_type": "T20", "dates": []}])],
        ignore_index=True,
    )
    selection = select_cohort(matches, CohortSpec(seasons_back=1))
    assert set(selection.match_ids) == {"dated", "undated"}


def test_cap_takes_the_most_recent_and_reports_that_it_did():
    matches = _matches(*[(f"m{i}", "IPL", "male", f"2025-04-{i:02d}") for i in range(1, 11)])
    selection = select_cohort(matches, CohortSpec(seasons_back=None, max_matches=3))

    assert selection.match_ids == ("m10", "m9", "m8")
    assert selection.matches_available == 10
    assert selection.was_capped
    # The sampling has to be visible: `source` is what ends up on every flag.
    assert "most recent 3 of 10" in selection.source


def test_whole_warehouse_spec_restores_the_unbounded_pre_fix_behaviour():
    matches = _matches(*[(f"m{i}", "IPL", "male", f"20{10 + i}-04-01") for i in range(1, 9)])
    selection = select_cohort(matches, WHOLE_WAREHOUSE)
    assert len(selection.match_ids) == 8
    assert not selection.was_capped
    assert selection.source == "whole warehouse"


def test_cohort_for_player_scopes_to_the_competitions_the_player_actually_played():
    """The correctness half: an IPL batter is not scored against women's T20Is.

    This is the behaviour change the rewrite makes, stated as a test rather than only in
    a docstring -- before, every player was scored against everything on disk.
    """
    matches = _matches(
        ("ipl1", "IPL", "male", "2025-04-01"),
        ("ipl2", "IPL", "male", "2025-04-02"),
        ("ipl3", "IPL", "male", "2025-04-03"),
        ("wc1", "Women's T20 WC", "female", "2025-04-04"),
        ("wc2", "Women's T20 WC", "female", "2025-04-05"),
    )
    deliveries = _deliveries(
        ("ipl1", "Bob", "Ann", "Carl"),
        ("ipl2", "Bob", "Ann", "Carl"),
        ("ipl3", "Dee", "Ann", "Carl"),
        ("wc1", "Eve", "Fay", "Gia"),
        ("wc2", "Eve", "Fay", "Gia"),
    )
    selection = cohort_for_player(deliveries, matches, "Bob", window_size=5)

    assert set(selection.match_ids) == {"ipl1", "ipl2", "ipl3"}
    assert "IPL" in selection.source


def test_an_explicit_spec_dimension_beats_the_inferred_one():
    """A caller who names a competition means it; inference only fills gaps."""
    matches = _matches(
        ("ipl1", "IPL", "male", "2025-04-01"),
        ("bbl1", "BBL", "male", "2025-04-02"),
    )
    deliveries = _deliveries(("ipl1", "Bob", "Ann", "Carl"))
    selection = cohort_for_player(
        deliveries, matches, "Bob", spec=CohortSpec(competitions=("BBL",), seasons_back=None)
    )
    assert selection.match_ids == ("bbl1",)


def test_cohort_for_an_unknown_player_is_empty_rather_than_the_whole_warehouse():
    """There is no cohort for a player with no matches, and inventing one would produce
    a confident rating for someone with no data behind it."""
    matches = _matches(("m1", "IPL", "male", "2025-04-01"))
    selection = cohort_for_player(_deliveries(("m1", "Bob", "Ann", "Carl")), matches, "Nobody")
    assert selection.match_ids == ()
    assert not selection


def test_player_match_ids_agrees_with_player_reports_last_n_match_ids():
    """`rating.cohort` duplicates this selection rather than importing it, so pin them
    together -- two definitions of "which matches has this player played in" that drifted
    would put a player's window and their cohort out of step."""
    matches = _matches(
        ("m1", "IPL", "male", "2025-04-01"),
        ("m2", "IPL", "male", "2025-04-01"),  # same date: the tie-break case
        ("m3", "IPL", "male", "2025-04-02"),
    )
    deliveries = _deliveries(
        ("m1", "Bob", "Ann", "Carl"),
        ("m2", "Ann", "Bob", "Carl"),   # Bob as non-striker still counts
        ("m3", "Ann", "Dee", "Bob"),    # Bob as bowler still counts
    )
    assert player_match_ids(deliveries, matches, "Bob", n=3) == last_n_match_ids(
        deliveries, matches, "Bob", n=3
    )
    assert player_match_ids(deliveries, matches, "Bob", n=2) == ["m3", "m2"]


def test_match_dates_tolerates_a_match_with_no_dates():
    matches = pd.DataFrame(
        [
            {"match_id": "a", "dates": ["2025-01-01"]},
            {"match_id": "b", "dates": []},
            {"match_id": "c", "dates": None},
        ]
    )
    assert match_dates(matches).to_dict() == {"a": "2025-01-01", "b": "", "c": ""}


def test_matches_to_read_is_the_deduplicated_union_in_a_stable_order():
    matches = _matches(*[(f"m{i}", "IPL", "male", f"2025-04-{i:02d}") for i in range(1, 5)])
    cohort = select_cohort(matches, CohortSpec(seasons_back=None))
    assert matches_to_read(cohort, ["m2", "m2", "m9"]) == ("m1", "m2", "m3", "m4", "m9")


@pytest.mark.parametrize("frame", [pd.DataFrame(), pd.DataFrame(columns=["match_id"])])
def test_empty_inputs_return_an_empty_selection_rather_than_raising(frame):
    assert select_cohort(frame, CohortSpec()).match_ids == ()
