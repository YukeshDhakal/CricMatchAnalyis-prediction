from dataclasses import replace

import pytest

from rating.contracts import Pillar, PillarWeights, PlayerRole, ROLE_PROFILES
from rating.engine import compute_baseline, rate_player, rate_players

from .helpers import fused, match_lines, over_of, stats


def _level_match(match_id: str, date: str):
    """A match where everyone performs identically: 12 balls each to Bob and Dan off
    Carl, one run a ball, all in the middle overs. Every rate in it equals every other
    rate, so a player rated against this cohort sits exactly on the baseline -- which
    makes 50 the expected pillar score and any deviation from it a real signal."""
    rows = over_of([1] * 12, match_id=match_id, striker="Bob", bowler="Carl") + over_of(
        [1] * 12, match_id=match_id, start_ball=12, striker="Dan", bowler="Carl"
    )
    return rows, match_lines(rows, match_date=date)


def _level_cohort():
    rows_1, lines_1 = _level_match("m1", "2026-01-01")
    rows_2, lines_2 = _level_match("m2", "2026-01-02")
    deliveries = rows_1 + rows_2
    history = lines_1 + lines_2
    return deliveries, history, compute_baseline(history, deliveries)


def test_role_weight_table_matches_the_prd_and_every_row_sums_to_one():
    assert ROLE_PROFILES[PlayerRole.TOP_ORDER_BATTER].weights == PillarWeights(0.30, 0.30, 0.25, 0.15)
    assert ROLE_PROFILES[PlayerRole.DEATH_OVERS_BOWLER].weights == PillarWeights(0.20, 0.35, 0.35, 0.10)
    for profile in ROLE_PROFILES.values():
        assert profile.weights.total() == pytest.approx(1.0)


def test_a_player_exactly_on_the_cohort_baseline_scores_fifty_on_every_measured_pillar():
    deliveries, history, baseline = _level_cohort()

    rating = rate_player(history, "Bob", PlayerRole.TOP_ORDER_BATTER, baseline, deliveries)

    assert rating.pillars.execution == 50.0
    assert rating.pillars.game_impact == 50.0
    assert rating.pillars.consistency == 100.0  # identical in both matches: zero spread


def test_technique_is_never_scored_and_its_weight_redistributes_across_the_other_pillars():
    deliveries, history, baseline = _level_cohort()

    rating = rate_player(history, "Bob", PlayerRole.TOP_ORDER_BATTER, baseline, deliveries)

    assert rating.pillars.technique is None, "no calibrated biomechanics extractor exists"
    assert rating.unmeasured_pillars == (Pillar.TECHNIQUE,)
    assert rating.insufficient_data is False

    # The PRD's 30/30/25/15 with technique removed and the rest rescaled to sum to 1.
    assert rating.weights_used.technique == 0.0
    assert rating.weights_used.execution == pytest.approx(0.30 / 0.70)
    assert rating.weights_used.game_impact == pytest.approx(0.25 / 0.70)
    assert rating.weights_used.consistency == pytest.approx(0.15 / 0.70)
    assert rating.weights_used.total() == pytest.approx(1.0)

    # (0.30*50 + 0.25*50 + 0.15*100) / 0.70 -- NOT scored as if technique were 0, which
    # would give (0.30*50 + 0.25*50 + 0.15*100) / 1.00 = 42.5 and silently mark an
    # exactly-average player as well below average.
    assert rating.composite == 60.71


def test_populated_pixel_space_biomechanics_still_do_not_produce_a_technique_score():
    """The regression guard on the Technique refusal.

    `video_engine`'s heuristic segmenter can put *numbers* in these fields -- wrist
    displacement in raw pixel space -- and those numbers are not a technique
    measurement without camera calibration. If a future change makes the engine read
    them, the pillar starts looking measured and the composite silently changes
    meaning, which is precisely the failure `_technique_score` refuses.
    """
    deliveries, history, baseline = _level_cohort()
    with_biomechanics = [
        replace(d, video_available=True, wrist_speed_ms=31.4, bowling_elbow_extension_deg=11.0)
        for d in deliveries
    ]

    rating = rate_player(history, "Bob", PlayerRole.TOP_ORDER_BATTER, baseline, with_biomechanics)

    assert rating.pillars.technique is None
    assert Pillar.TECHNIQUE in rating.unmeasured_pillars


def test_composite_is_none_and_flagged_rather_than_guessed_when_no_pillar_is_measurable():
    history = [stats(player="Bob", match_id="m1", runs_scored=2, balls_faced=3)]
    baseline = compute_baseline(history)

    rating = rate_player(history, "Bob", PlayerRole.TOP_ORDER_BATTER, baseline)

    assert rating is not None, "a known player with unscorable data is not an unknown player"
    assert rating.composite is None
    assert rating.insufficient_data is True
    assert set(rating.unmeasured_pillars) == set(Pillar)
    assert rating.weights_used.total() == 0.0


def test_game_impact_separates_two_players_with_identical_strike_rates_by_when_they_scored():
    """Both players score 12 off 12 at a cohort baseline of a run a ball, so Execution
    can't tell them apart. One scores in the middle overs and goes quiet at the death;
    the other does the reverse. Phase weighting is the only thing that separates them.
    """
    cohort = over_of([1] * 12, match_id="m0", striker="Ann", bowler="Carl", phase="middle") + over_of(
        [1] * 12, match_id="m0", start_ball=12, striker="Ann", bowler="Carl", phase="death"
    )
    front_loaded = over_of([2] * 6, match_id="m1", striker="Bob", bowler="Carl", phase="middle") + over_of(
        [0] * 6, match_id="m1", start_ball=6, striker="Bob", bowler="Carl", phase="death"
    )
    back_loaded = over_of([0] * 6, match_id="m1", striker="Dan", bowler="Carl", phase="middle") + over_of(
        [2] * 6, match_id="m1", start_ball=6, striker="Dan", bowler="Carl", phase="death"
    )

    deliveries = cohort + front_loaded + back_loaded
    history = match_lines(cohort, "2026-01-01") + match_lines(front_loaded + back_loaded, "2026-01-02")
    baseline = compute_baseline(history, deliveries)

    bob = rate_player(history, "Bob", PlayerRole.TOP_ORDER_BATTER, baseline, deliveries)
    dan = rate_player(history, "Dan", PlayerRole.TOP_ORDER_BATTER, baseline, deliveries)

    assert bob.pillars.execution == dan.pillars.execution, "identical strike rates"
    # weights are 1.0 (middle) and 1.5 (death), so total weight is 15 over 12 balls:
    # Bob's phase-weighted rate is 12/15 = 0.8 against an expectation of 1.0, Dan's is
    # 18/15 = 1.2 against the same 1.0.
    assert bob.pillars.game_impact == 30.0
    assert dan.pillars.game_impact == 70.0


def test_consistency_scores_spread_not_level():
    steady = [
        stats(player="Steady", match_id="m1", match_date="2026-01-01", runs_scored=30, balls_faced=24),
        stats(player="Steady", match_id="m2", match_date="2026-01-02", runs_scored=30, balls_faced=24),
    ]
    streaky = [
        stats(player="Streaky", match_id="m1", match_date="2026-01-01", runs_scored=60, balls_faced=24),
        stats(player="Streaky", match_id="m2", match_date="2026-01-02", runs_scored=0, balls_faced=24),
    ]
    history = steady + streaky
    baseline = compute_baseline(history)

    assert rate_player(history, "Steady", PlayerRole.TOP_ORDER_BATTER, baseline).pillars.consistency == 100.0
    # spread equals the mean (cv == 1), the bottom of the scale
    assert rate_player(history, "Streaky", PlayerRole.TOP_ORDER_BATTER, baseline).pillars.consistency == 0.0


def test_consistency_needs_two_innings_to_have_a_spread():
    history = [stats(player="Bob", match_id="m1", runs_scored=30, balls_faced=24)]
    baseline = compute_baseline(history)

    assert rate_player(history, "Bob", PlayerRole.TOP_ORDER_BATTER, baseline).pillars.consistency is None


def test_bowling_execution_blends_economy_and_wickets_rather_than_economy_alone():
    cohort = [
        stats(player="Carl", match_id="m1", match_date="2026-01-01", balls_bowled=24, runs_conceded=24, wickets_taken=2),
        stats(player="Eve", match_id="m1", match_date="2026-01-01", balls_bowled=24, runs_conceded=24, wickets_taken=2),
    ]
    # Same wicket haul as the cohort, half the runs: economy scores 100, wickets 50.
    miser = [
        stats(player="Miser", match_id="m1", match_date="2026-01-01", balls_bowled=24, runs_conceded=12, wickets_taken=2)
    ]
    history = cohort + miser
    # Baseline over the cohort alone, so the player being rated isn't part of the
    # expectation he's scored against -- otherwise an outlier drags the bar toward
    # itself and every pillar reads closer to 50 than it should.
    baseline = compute_baseline(cohort)

    at_baseline = rate_player(history, "Carl", PlayerRole.DEATH_OVERS_BOWLER, baseline)
    ahead = rate_player(history, "Miser", PlayerRole.DEATH_OVERS_BOWLER, baseline)

    assert at_baseline.pillars.execution == 50.0
    assert ahead.pillars.execution == 75.0


def test_rate_player_returns_none_for_a_player_with_no_history():
    history = [stats(player="Bob", match_id="m1", runs_scored=30, balls_faced=24)]

    assert rate_player(history, "Nobody", PlayerRole.TOP_ORDER_BATTER, compute_baseline(history)) is None


def test_rate_player_window_breaks_same_date_ties_deterministically():
    history = [
        stats(player="Bob", match_id="m1", match_date="2026-01-05", runs_scored=10, balls_faced=20),
        stats(player="Bob", match_id="m2", match_date="2026-01-05", runs_scored=20, balls_faced=20),
        stats(player="Bob", match_id="m3", match_date="2026-01-01", runs_scored=30, balls_faced=20),
    ]
    baseline = compute_baseline(history)

    results = [
        rate_player(history, "Bob", PlayerRole.TOP_ORDER_BATTER, baseline, window_size=2) for _ in range(20)
    ]

    assert all(r.composite == results[0].composite for r in results), "must not vary across repeated calls"
    assert results[0].matches_in_sample == 2
    # m2 (same date, higher match_id) then m1 -> runs [20, 10], mean 15, sd 5, cv 1/3.
    # Picking m3 instead would give a different spread, so this pins the window itself.
    assert results[0].pillars.consistency == 66.67


def test_game_impact_is_unmeasured_rather_than_guessed_without_delivery_rows():
    _, history, baseline = _level_cohort()

    rating = rate_player(history, "Bob", PlayerRole.TOP_ORDER_BATTER, baseline)

    assert rating.pillars.game_impact is None, "phase lives on FusedDelivery, not PlayerMatchStats"
    assert Pillar.GAME_IMPACT in rating.unmeasured_pillars
    assert rating.weights_used.execution == pytest.approx(0.30 / 0.45)


def test_compute_baseline_pools_totals_rather_than_averaging_per_player_rates():
    history = [
        stats(player="Bulk", match_id="m1", runs_scored=100, balls_faced=100),  # SR 100
        stats(player="Cameo", match_id="m1", runs_scored=12, balls_faced=4),  # SR 300
    ]

    baseline = compute_baseline(history)

    # A mean of the two rates would be 200 and let a four-ball cameo set the cohort.
    assert baseline.batting_strike_rate == 107.69


def test_compute_baseline_returns_none_for_an_empty_history():
    assert compute_baseline([]) is None


def test_compute_baseline_leaves_delivery_level_fields_none_without_delivery_rows():
    history = [stats(player="Bob", match_id="m1", runs_scored=30, balls_faced=24)]

    baseline = compute_baseline(history)

    assert baseline.dot_ball_percent is None
    assert baseline.phases == ()
    # The description is quoted inline in coaching notes, so it has to read as prose.
    assert baseline.source == "cohort mean over 1 matches, 1 players"


def test_compute_baseline_keeps_batting_and_bowling_phase_rates_apart():
    """The two per-ball phase rates are not the same number, and conflating them would
    score one side of the ball against the other side's expectation.

    A no-ball counts as a ball faced but not as a legal ball bowled; leg-byes score for
    the batting side but aren't charged to the bowler. Both conventions come straight
    from `fusion.pipeline`.
    """
    rows = (
        over_of([1] * 6, match_id="m1", striker="Bob", bowler="Carl")
        + [fused(match_id="m1", over=1, ball=1, striker="Bob", bowler="Carl", runs_batter=2, runs_extras=1,
                 extras_type="noballs")]
        + [fused(match_id="m1", over=1, ball=2, striker="Bob", bowler="Carl", runs_extras=4, extras_type="legbyes")]
    )

    baseline = compute_baseline(match_lines(rows, "2026-01-01"), rows)
    middle = baseline.phase_baseline("middle")

    # 8 runs off the bat over 8 balls faced (the no-ball counts, nothing was a wide).
    assert middle.runs_per_ball == pytest.approx(1.0)
    # 6 runs charged over 7 legal balls: the no-ball isn't a legal ball and the four
    # leg-byes aren't Carl's to answer for.
    assert middle.runs_conceded_per_ball == pytest.approx(6 / 7, abs=1e-4)


def test_rate_players_covers_every_assigned_role_and_skips_players_with_no_history():
    deliveries, history, baseline = _level_cohort()

    ratings = rate_players(
        history,
        {"Bob": PlayerRole.TOP_ORDER_BATTER, "Carl": PlayerRole.DEATH_OVERS_BOWLER, "Ghost": PlayerRole.TOP_ORDER_BATTER},
        baseline,
        deliveries,
    )

    assert [r.player for r in ratings] == ["Bob", "Carl"]
    assert all(r.baseline_source == baseline.source for r in ratings)


def test_restricted_weights_collapse_to_zero_when_no_pillar_survives():
    weights = ROLE_PROFILES[PlayerRole.TOP_ORDER_BATTER].weights

    assert weights.restricted_to(frozenset()) == PillarWeights(0.0, 0.0, 0.0, 0.0)
