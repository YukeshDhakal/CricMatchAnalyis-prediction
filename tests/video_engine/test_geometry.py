"""Tests for stump-end resolution and bounce-point projection.

The important cases here are the refusals. Every clip in `data/uploads/videos/` shows only
one stump set, so the path these tests exercise most is the one that declines to produce a
line and a length and says why -- and a refusal that silently became a number would be the
most damaging regression this module could have.
"""
from video_engine.calibration import PITCH_LENGTH_M, STUMP_HALF_WIDTH_M
from video_engine.contracts import (
    BoundingBox,
    Detection,
    GeometryConfidence,
    ObjectClass,
    PitchLength,
)
from video_engine.geometry import pitch_geometry, resolve_stump_ends


def _stumps(frame: int, x: float, y_base: float, width: float, height: float) -> Detection:
    return Detection(
        frame,
        ObjectClass.STUMPS,
        BoundingBox(x - width / 2, y_base - height, x + width / 2, y_base),
        0.6,
    )


def _two_ends(frames: int = 3) -> list[Detection]:
    """A near set (large) and a far set (small), as a usable camera placement would see."""
    out: list[Detection] = []
    for f in range(frames):
        out.append(_stumps(f, 400.0, 500.0, 60.0, 160.0))  # near / striker's end
        out.append(_stumps(f, 430.0, 260.0, 20.0, 55.0))  # far / bowler's end
    return out


def test_one_stump_set_refuses_rather_than_assuming_the_other_end():
    detections = [_stumps(f, 400.0, 500.0, 60.0, 160.0) for f in range(5)]

    assert resolve_stump_ends(detections) is None


def test_no_stumps_at_all_refuses():
    assert resolve_stump_ends([]) is None


def test_two_well_separated_sets_of_different_size_resolve_to_ends():
    ends = resolve_stump_ends(_two_ends())

    assert ends is not None
    # The nearer (taller) set is the striker's end.
    assert ends.strikers_end.y2 - ends.strikers_end.y1 > ends.bowlers_end.y2 - ends.bowlers_end.y1
    assert "assumes" in ends.assumption


def test_two_similar_sized_sets_refuse_because_near_far_would_be_a_coin_flip():
    detections = [
        _stumps(0, 300.0, 400.0, 40.0, 100.0),
        _stumps(0, 800.0, 400.0, 40.0, 104.0),
    ]

    assert resolve_stump_ends(detections) is None


def test_no_bounce_point_reports_no_geometry_with_a_reason():
    result = pitch_geometry(_two_ends(), bounce_point_px=None, track_confidence=0.8)

    assert result.point is None
    assert result.confidence is GeometryConfidence.NONE
    assert "ground contact" in result.notes


def test_an_undetected_bounce_is_unknown_and_never_a_full_toss():
    """A tracking gap must not be reported as a cricketing fact.

    Found by running the real pipeline on real_bowling_clip_full.mp4: passing
    `bounced=False` through to `classify_length` turned "no bounce found in seven tracked
    frames" into `FULL_TOSS` for a ball that was visibly rolling along the ground.
    """
    result = pitch_geometry(
        _two_ends(), bounce_point_px=None, track_confidence=0.8, bounced=False
    )

    assert result.length is PitchLength.UNKNOWN
    assert result.length is not PitchLength.FULL_TOSS
    assert "not evidence of a full toss" in result.notes


def test_one_stump_set_plus_a_bounce_still_reports_no_line_or_length():
    """The case every real clip in this repo hits."""
    detections = [_stumps(f, 400.0, 500.0, 60.0, 160.0) for f in range(5)]

    result = pitch_geometry(detections, bounce_point_px=(410.0, 430.0), track_confidence=0.9)

    assert result.point is None
    assert result.length is PitchLength.UNKNOWN
    assert result.confidence is GeometryConfidence.NONE
    assert "both stump sets" in result.notes


def test_a_bounce_between_the_ends_projects_onto_the_pitch():
    detections = _two_ends()
    ends = resolve_stump_ends(detections)
    assert ends is not None

    # A point on the near stumps' baseline must map to roughly the origin, which is the
    # property that says the homography is oriented the way the contract claims.
    near_mid_x = (ends.strikers_end.x1 + ends.strikers_end.x2) / 2
    result = pitch_geometry(
        detections, bounce_point_px=(near_mid_x, ends.strikers_end.y2), track_confidence=0.8
    )

    assert result.calibration is not None
    assert result.point is not None
    assert abs(result.point.length_m) < 0.5
    assert abs(result.point.line_m) < STUMP_HALF_WIDTH_M + 0.2


def test_a_bounce_off_the_pitch_is_reported_unknown_not_snapped_to_a_band():
    detections = _two_ends()

    # Far above the far stumps in the image: past the bowler's end of the pitch.
    result = pitch_geometry(detections, bounce_point_px=(430.0, 200.0), track_confidence=0.8)

    assert result.length is PitchLength.UNKNOWN
    assert result.confidence is GeometryConfidence.NONE


def test_confidence_never_reaches_high_from_a_single_camera():
    detections = _two_ends()
    ends = resolve_stump_ends(detections)
    assert ends is not None
    near_mid_x = (ends.strikers_end.x1 + ends.strikers_end.x2) / 2

    result = pitch_geometry(
        detections, bounce_point_px=(near_mid_x, ends.strikers_end.y2 - 40), track_confidence=1.0
    )

    assert result.confidence is not GeometryConfidence.HIGH


def test_a_projected_result_says_it_is_an_estimate():
    detections = _two_ends()
    ends = resolve_stump_ends(detections)
    assert ends is not None
    near_mid_x = (ends.strikers_end.x1 + ends.strikers_end.x2) / 2

    result = pitch_geometry(
        detections, bounce_point_px=(near_mid_x, ends.strikers_end.y2 - 60), track_confidence=0.8
    )

    if result.point is not None:
        assert "estimate" in result.notes.lower()
        assert "line" in result.notes.lower()


def test_pitch_length_constant_is_used_as_the_far_anchor():
    """Guards the origin convention: the far stumps sit at PITCH_LENGTH_M, not at 0."""
    detections = _two_ends()
    ends = resolve_stump_ends(detections)
    assert ends is not None
    far_mid_x = (ends.bowlers_end.x1 + ends.bowlers_end.x2) / 2

    result = pitch_geometry(
        detections, bounce_point_px=(far_mid_x, ends.bowlers_end.y2), track_confidence=0.8
    )

    assert result.point is not None
    assert abs(result.point.length_m - PITCH_LENGTH_M) < 1.0
