"""Tests for the pitch-length/line metrics (`rating.contracts`, `engine`, `suggestions`).

**Nothing in this file can happen on real data today.** No ball or stumps detector
exists, so no `FusedDelivery` carries a bounce point and every pitch metric is inert --
see `video_engine.calibration` and the README's "Ball and stumps detection". These
tests construct rows that *do* carry geometry, by hand, the same way
`tests/rating/helpers.py` constructs every other row.

Two distinct things are being pinned, and the second matters as much as the first:

1. The arithmetic is right *when* geometry exists, so a detector landing later is a data
   change rather than a debugging session.
2. The arithmetic stays **silent** when geometry doesn't exist. A contract that can't
   fire is only safe if it provably doesn't fire -- a pitch flag appearing on
   video-less rows would be a fabricated finding with a real citation attached, which is
   the precise failure mode PRD 2.4 and this repo's whole design are organised against.
"""
from __future__ import annotations

import pytest

from rating.contracts import (
    METRIC_SPECS,
    PITCH_METRIC_BANDS,
    STUMP_LINE_TOLERANCE_M,
    Metric,
    PitchBaseline,
)
from rating.engine import cohort_pitch_baseline, compute_baseline
from rating.llm import TemplateNoteWriter
from rating.suggestions import flag_player
from video_engine.calibration import STUMP_HALF_WIDTH_M
from video_engine.contracts import PitchLength

from .helpers import fused, match_lines

_PITCH_METRICS = (
    Metric.GOOD_LENGTH_PERCENT,
    Metric.SHORT_BALL_PERCENT,
    Metric.FULL_BALL_PERCENT,
    Metric.STUMP_LINE_PERCENT,
)


def bowled(length: PitchLength, line_m: float | None = 0.0, **overrides):
    """One legal delivery bowled by "Carl" carrying real pitch geometry.

    `pitch_length_m` is set to the midpoint of the band so the row is internally
    consistent -- a row whose band and metres disagreed couldn't come out of a real
    calibration and would let a test pass against an impossible input.
    """
    midpoints = {
        PitchLength.YORKER: 1.0, PitchLength.FULL: 3.0, PitchLength.GOOD: 5.5,
        PitchLength.BACK_OF_A_LENGTH: 8.0, PitchLength.SHORT: 11.0,
    }
    return fused(
        bowler="Carl",
        pitch_length=length,
        pitch_length_m=midpoints.get(length),
        pitch_line_m=line_m,
        video_available=True,
        **overrides,
    )


def cohort_with(*rows):
    """A baseline over `rows`, built through the real `compute_baseline`."""
    return compute_baseline(match_lines(rows, "2026-01-01"), rows)


# --- contract shape -------------------------------------------------------------------

@pytest.mark.parametrize("metric", _PITCH_METRICS)
def test_every_pitch_metric_has_a_spec(metric):
    """The closed-vocabulary guarantee `Metric`'s docstring makes: flagging, ranking and
    the note templates all key off these, and a metric without a spec would raise in the
    middle of a batch rather than at import."""
    assert metric in METRIC_SPECS
    assert METRIC_SPECS[metric].metric is metric
    assert METRIC_SPECS[metric].unit == "% of legal deliveries"
    assert METRIC_SPECS[metric].is_batting is False


@pytest.mark.parametrize("metric", _PITCH_METRICS)
def test_every_pitch_metric_has_template_phrasing_in_both_directions(metric):
    """A metric that reached the note-writer without a template would silently degrade to
    generic prose. That is the documented fallback, but for metrics shipped deliberately
    it would be an oversight, so both directions are pinned."""
    writer = TemplateNoteWriter()
    for is_concern in (True, False):
        assert (metric, is_concern) in writer._TITLES
        assert (metric, is_concern) in writer._BODIES


def test_length_bands_partition_the_pitch_length_enum():
    """Every band a calibration can produce is counted by exactly one length metric.

    A band in no metric would be measured by nothing; a band in two would be
    double-counted, and the three length percentages would sum past 100.
    """
    banded = [b for bands in PITCH_METRIC_BANDS.values() for b in bands]
    assert len(banded) == len(set(banded)), "a PitchLength band is claimed by two metrics"

    classifiable = {l.value for l in PitchLength} - {PitchLength.UNKNOWN.value}
    assert set(banded) == classifiable


# --- the baseline ---------------------------------------------------------------------

def test_cohort_pitch_baseline_is_none_without_any_geometry():
    """The case that holds for every real row today."""
    rows = [bowled(PitchLength.GOOD) for _ in range(10)]
    plain = [fused(bowler="Carl") for _ in range(10)]

    assert cohort_pitch_baseline(plain) is None
    assert compute_baseline(match_lines(plain, "2026-01-01"), plain).pitch is None
    assert cohort_pitch_baseline(rows) is not None


def test_pitch_baseline_denominator_is_covered_deliveries_not_all_deliveries():
    """Mixing the two would report a bowler's good-length share as near zero whenever
    most of their deliveries simply weren't filmed -- a coverage artefact presented as a
    bowling problem."""
    rows = [bowled(PitchLength.GOOD) for _ in range(4)]
    rows += [fused(bowler="Carl") for _ in range(96)]  # bowled, no footage

    pitch = cohort_pitch_baseline(rows)
    assert pitch.deliveries_with_geometry == 4
    assert pitch.good_length_percent == 100.0  # 4/4, not 4/100


def test_pitch_baseline_rates_are_percentages_of_the_covered_set():
    rows = (
        [bowled(PitchLength.GOOD) for _ in range(5)]
        + [bowled(PitchLength.SHORT) for _ in range(3)]
        + [bowled(PitchLength.FULL) for _ in range(2)]
    )
    pitch = cohort_pitch_baseline(rows)

    assert pitch.good_length_percent == 50.0
    assert pitch.short_ball_percent == 30.0
    assert pitch.full_ball_percent == 20.0
    assert pitch.deliveries_with_geometry == 10


def test_illegal_deliveries_are_excluded_from_the_pitch_baseline():
    """Same convention as every other bowling figure in this repo: a wide or a no-ball is
    not a legal delivery bowled."""
    rows = [bowled(PitchLength.GOOD) for _ in range(6)]
    rows += [bowled(PitchLength.SHORT, extras_type="wides") for _ in range(6)]
    rows += [bowled(PitchLength.SHORT, extras_type="noballs") for _ in range(6)]

    pitch = cohort_pitch_baseline(rows)
    assert pitch.deliveries_with_geometry == 6
    assert pitch.good_length_percent == 100.0


def test_a_full_toss_counts_for_length_but_not_for_line():
    """A ball that never bounced has a real length classification and no bounce point, so
    it is evidence about length and no evidence at all about line."""
    rows = [bowled(PitchLength.FULL_TOSS, line_m=None) for _ in range(2)]
    rows += [bowled(PitchLength.GOOD, line_m=0.0) for _ in range(8)]

    pitch = cohort_pitch_baseline(rows)
    assert pitch.deliveries_with_geometry == 10
    assert pitch.full_ball_percent == 20.0      # full tosses counted as overpitched
    assert pitch.stump_line_percent == 100.0    # 8/8 with a line, not 8/10


# --- flagging -------------------------------------------------------------------------

def test_no_pitch_flag_is_produced_without_a_pitch_baseline():
    """Precondition one. This is the state of every cohort today."""
    rows = [bowled(PitchLength.SHORT) for _ in range(20)]
    plain_baseline = cohort_with(*[fused(bowler="Carl") for _ in range(20)])
    assert plain_baseline.pitch is None

    flags = flag_player(rows, "Carl", plain_baseline)
    assert not [f for f in flags if f.metric in _PITCH_METRICS]


def test_no_pitch_flag_when_the_cohort_has_geometry_but_the_player_does_not():
    """Precondition two, and the one that would do real damage if it were missing.

    A cohort can be covered while this player's own matches weren't filmed. Flagging
    their "0% good length" against a real cohort rate would turn missing footage into a
    damning, citation-backed finding.
    """
    cohort = cohort_with(*[bowled(PitchLength.GOOD) for _ in range(20)])
    assert cohort.pitch is not None

    uncovered = [fused(bowler="Carl") for _ in range(20)]
    flags = flag_player(uncovered, "Carl", cohort)
    assert not [f for f in flags if f.metric in _PITCH_METRICS]


def test_a_bowler_well_off_the_cohort_length_is_flagged_as_a_concern():
    cohort = cohort_with(
        *([bowled(PitchLength.GOOD) for _ in range(8)] + [bowled(PitchLength.SHORT) for _ in range(2)])
    )
    assert cohort.pitch.good_length_percent == 80.0

    # This bowler lands a good length on 2 of 10 -- far below the cohort's 80%.
    player_rows = [bowled(PitchLength.GOOD) for _ in range(2)]
    player_rows += [bowled(PitchLength.SHORT, over=1) for _ in range(8)]

    flags = {f.metric: f for f in flag_player(player_rows, "Carl", cohort)}
    good = flags[Metric.GOOD_LENGTH_PERCENT]

    assert good.actual == 20.0
    assert good.baseline == 80.0
    assert good.is_concern            # less good length is worse
    assert good.sample_size == 10     # covered deliveries, not all deliveries
    assert flags[Metric.SHORT_BALL_PERCENT].is_concern  # more short balls is also worse


def test_beating_the_cohort_length_is_a_strength_not_a_concern():
    cohort = cohort_with(
        *([bowled(PitchLength.GOOD) for _ in range(2)] + [bowled(PitchLength.SHORT) for _ in range(8)])
    )
    player_rows = [bowled(PitchLength.GOOD) for _ in range(9)]
    player_rows += [bowled(PitchLength.SHORT, over=1)]

    good = {f.metric: f for f in flag_player(player_rows, "Carl", cohort)}[Metric.GOOD_LENGTH_PERCENT]
    assert good.actual == 90.0
    assert not good.is_concern


def test_line_is_scored_against_the_stump_corridor():
    corridor = STUMP_HALF_WIDTH_M + STUMP_LINE_TOLERANCE_M
    cohort = cohort_with(*[bowled(PitchLength.GOOD, line_m=0.0) for _ in range(10)])
    assert cohort.pitch.stump_line_percent == 100.0

    # Six of ten well outside the corridor.
    player_rows = [bowled(PitchLength.GOOD, line_m=0.0) for _ in range(4)]
    player_rows += [bowled(PitchLength.GOOD, line_m=corridor + 0.4, over=1) for _ in range(6)]

    line = {f.metric: f for f in flag_player(player_rows, "Carl", cohort)}[Metric.STUMP_LINE_PERCENT]
    assert line.actual == 40.0
    assert line.is_concern


def test_a_pitch_flag_cites_real_deliveries():
    """PRD 2.4: a finding without a checkable citation isn't a finding. The pitch metrics
    must earn citations the same way every other metric does."""
    cohort = cohort_with(*[bowled(PitchLength.GOOD) for _ in range(10)])
    player_rows = [bowled(PitchLength.SHORT, over=i // 6, ball=i % 6 + 1, runs_batter=i % 5)
                   for i in range(10)]

    good = {f.metric: f for f in flag_player(player_rows, "Carl", cohort)}[Metric.GOOD_LENGTH_PERCENT]
    assert good.citations
    assert all(c.match_id == "m1" for c in good.citations)
    assert good.citations_have_video  # these rows are covered, by construction
    assert good.citation_tokens()[0].startswith("[clip m1 ")


def test_a_thin_covered_sample_produces_no_pitch_flag():
    """Below `MIN_BALLS_FOR_FLAG` covered deliveries, a rate is noise, and a flag on it
    is a false alarm dressed up with a citation."""
    cohort = cohort_with(*[bowled(PitchLength.GOOD) for _ in range(10)])
    player_rows = [bowled(PitchLength.SHORT) for _ in range(3)]

    flags = flag_player(player_rows, "Carl", cohort)
    assert not [f for f in flags if f.metric in _PITCH_METRICS]


def test_pitch_flags_do_not_disturb_the_existing_stat_flags():
    """The extension is additive: adding geometry to a row must not change what the
    strike-rate/economy flags say about it."""
    rows_plain = [fused(bowler="Carl", striker="Bob", runs_batter=2) for _ in range(12)]
    rows_geo = [
        bowled(PitchLength.GOOD, striker="Bob", runs_batter=2) for _ in range(12)
    ]
    baseline = cohort_with(*[fused(bowler="Carl", striker="Bob", runs_batter=1) for _ in range(12)])

    plain = {(f.metric, f.phase): f for f in flag_player(rows_plain, "Bob", baseline)}
    geo = {(f.metric, f.phase): f for f in flag_player(rows_geo, "Bob", baseline)}

    assert plain, "expected at least one batting flag to compare"
    for key, flag in plain.items():
        assert geo[key].actual == flag.actual
        assert geo[key].is_concern == flag.is_concern


def test_pitch_baseline_is_frozen_like_every_other_contract():
    pitch = PitchBaseline(50.0, 20.0, 30.0, 80.0, 10)
    with pytest.raises(AttributeError):
        pitch.good_length_percent = 0.0
