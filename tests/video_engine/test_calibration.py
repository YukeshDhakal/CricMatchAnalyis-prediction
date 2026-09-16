"""Tests for pitch-plane calibration (`video_engine.calibration`).

The module has no caller in the pipeline yet -- it needs stumps detections that no model
produces (see its docstring and the README's "Ball and stumps detection"). That makes
testing it *more* important rather than less: an untested implementation of geometry
nobody currently runs is exactly the kind of code that quietly turns out to be wrong the
day a detector lands and somebody trusts its output.

The approach is round-tripping through a known projection. Rather than asserting magic
pixel values, these tests build a synthetic camera (a real 3x3 projective transform),
push the four known stump-base points through it to get "detections", recover a
calibration from those, and check that mapping other points back gives the real-world
coordinates they started from. That tests the actual mathematical property the module
claims -- not that the code returns what it returned the day it was written.
"""
from __future__ import annotations

import numpy as np
import pytest

from video_engine.calibration import (
    LENGTH_BANDS,
    MAX_REPROJECTION_ERROR_PX,
    PITCH_LENGTH_M,
    STUMP_BLOCK_WIDTH_M,
    STUMP_HALF_WIDTH_M,
    STUMP_HEIGHT_M,
    calibrate_from_stumps,
    classify_length,
    in_stump_corridor,
)
from video_engine.contracts import BoundingBox, PitchLength

# A plausible fixed elevated camera behind the bowler's arm, as a homography from pitch
# metres to image pixels. Perspective terms in the bottom row are what make the far
# stumps smaller than the near ones; without them this would be an affine warp and would
# not exercise the projective part of the solve at all.
_WORLD_TO_IMAGE = np.array(
    [
        [42.0, 0.0, 640.0],
        [0.0, -18.0, 980.0],
        [0.0, -0.035, 1.0],
    ]
)


def _project(x_m: float, y_m: float) -> tuple[float, float]:
    u, v, w = _WORLD_TO_IMAGE @ np.array([x_m, y_m, 1.0])
    return u / w, v / w


def _stump_box(y_m: float) -> BoundingBox:
    """The image-space box a set of stumps at `y_m` down the pitch would occupy.

    Built from the projected positions of the outer stump bases plus the projected top,
    so the box has a realistic height -- and so `calibrate_from_stumps` is exercised on
    its documented behaviour of using only the *base* corners. A box whose top edge is
    the stumps' top is the box a detector would actually emit.
    """
    left_x, base_y = _project(-STUMP_HALF_WIDTH_M, y_m)
    right_x, _ = _project(STUMP_HALF_WIDTH_M, y_m)
    # Stump top is above the plane; approximate its image height by scaling with the
    # local perspective factor. Only the base row matters to the fit under test.
    scale = (right_x - left_x) / STUMP_BLOCK_WIDTH_M
    return BoundingBox(x1=left_x, y1=base_y - STUMP_HEIGHT_M * scale, x2=right_x, y2=base_y)


def test_calibration_recovers_known_pitch_coordinates():
    """The central property: pixels in, correct metres out.

    Points are projected through the synthetic camera and mapped back through the
    recovered calibration. Agreement to a millimetre means the homography really is the
    inverse of the projection, not merely something that fits the four anchors.
    """
    calibration = calibrate_from_stumps(_stump_box(0.0), _stump_box(PITCH_LENGTH_M))
    assert calibration is not None

    for length_m, line_m in [(0.0, 0.0), (5.0, 0.3), (8.5, -0.45), (17.0, 0.1), (2.0, -0.2)]:
        px, py = _project(line_m, length_m)
        point = calibration.pitch_point(px, py)
        assert point is not None
        assert point.length_m == pytest.approx(length_m, abs=1e-3)
        assert point.line_m == pytest.approx(line_m, abs=1e-3)


def test_calibration_anchors_are_the_stump_bases_not_the_box_tops():
    """Stump tops are 0.711 m above the pitch plane, so fitting to them would fit points
    that aren't on the plane the homography claims to map.

    Checked by giving two boxes with identical bases and very different heights: the
    resulting calibration must be the same, because only the base row is a plane point.
    """
    short_box = _stump_box(0.0)
    tall_box = BoundingBox(short_box.x1, short_box.y1 - 200, short_box.x2, short_box.y2)

    a = calibrate_from_stumps(short_box, _stump_box(PITCH_LENGTH_M))
    b = calibrate_from_stumps(tall_box, _stump_box(PITCH_LENGTH_M))
    assert a is not None and b is not None
    assert np.allclose(a.as_array(), b.as_array())


def test_origin_is_the_strikers_middle_stump():
    calibration = calibrate_from_stumps(_stump_box(0.0), _stump_box(PITCH_LENGTH_M))
    origin = calibration.pitch_point(*_project(0.0, 0.0))
    assert origin.length_m == pytest.approx(0.0, abs=1e-3)
    assert origin.line_m == pytest.approx(0.0, abs=1e-3)

    far = calibration.pitch_point(*_project(0.0, PITCH_LENGTH_M))
    assert far.length_m == pytest.approx(PITCH_LENGTH_M, abs=1e-3)


def test_missing_or_degenerate_stumps_yield_no_calibration():
    """Every honest-failure path returns `None` rather than a usable-looking matrix."""
    good = _stump_box(0.0)
    far = _stump_box(PITCH_LENGTH_M)

    assert calibrate_from_stumps(None, far) is None
    assert calibrate_from_stumps(good, None) is None
    assert calibrate_from_stumps(None, None) is None
    # Zero-width and zero-height boxes are degenerate: no two distinct base points.
    assert calibrate_from_stumps(BoundingBox(10, 10, 10, 40), far) is None
    assert calibrate_from_stumps(BoundingBox(10, 40, 50, 40), far) is None


def test_identical_boxes_at_both_ends_are_refused():
    """Two coincident stump sets give four points that can't span the pitch.

    This is the case that most needs to fail loudly rather than quietly: it solves
    numerically, and the resulting map would report every bounce at a nonsense length.
    """
    box = _stump_box(0.0)
    assert calibrate_from_stumps(box, box) is None


def test_a_reported_fit_is_within_the_documented_error_bound():
    calibration = calibrate_from_stumps(_stump_box(0.0), _stump_box(PITCH_LENGTH_M))
    assert calibration.rms_error_px <= MAX_REPROJECTION_ERROR_PX
    assert calibration.rms_error_px == pytest.approx(0.0, abs=1e-6)


def test_points_on_the_horizon_map_to_none_rather_than_being_clamped():
    """A pixel above the plane's horizon has no corresponding point on the pitch.

    "No answer" is the geometrically correct report; clamping to the nearest finite value
    would invent a bounce position for a pixel that can't be one.
    """
    calibration = calibrate_from_stumps(_stump_box(0.0), _stump_box(PITCH_LENGTH_M))
    matrix = calibration.as_array()
    # Solve for an image point whose third homogeneous coordinate vanishes.
    a, b, c = matrix[2]
    horizon_x = 0.0
    horizon_y = -c / b if abs(b) > 1e-12 else 0.0
    assert calibration.pitch_point(horizon_x, horizon_y) is None


def test_calibration_is_immutable_and_hashable_like_every_other_contract():
    calibration = calibrate_from_stumps(_stump_box(0.0), _stump_box(PITCH_LENGTH_M))
    assert isinstance(calibration.matrix, tuple)
    assert hash(calibration) is not None
    with pytest.raises(AttributeError):
        calibration.rms_error_px = 5.0  # frozen


# --- length classification ------------------------------------------------------------

@pytest.mark.parametrize(
    "length_m,expected",
    [
        (0.5, PitchLength.YORKER),
        (2.0, PitchLength.FULL),
        (3.9, PitchLength.FULL),
        (4.0, PitchLength.GOOD),
        (6.9, PitchLength.GOOD),
        (7.0, PitchLength.BACK_OF_A_LENGTH),
        (9.0, PitchLength.SHORT),
        (15.0, PitchLength.SHORT),
    ],
)
def test_length_bands_are_the_documented_table(length_m, expected):
    """States the mapping it depends on rather than hard-coding outputs, so retuning
    `LENGTH_BANDS` shows up here as a deliberate change."""
    assert classify_length(length_m) is expected


def test_bands_are_contiguous_and_non_overlapping():
    """A gap would silently classify a real length as UNKNOWN; an overlap would make the
    result depend on table order."""
    for (_, _, upper), (_, lower, _) in zip(LENGTH_BANDS, LENGTH_BANDS[1:]):
        assert upper == lower
    assert LENGTH_BANDS[0][1] == 0.0
    assert LENGTH_BANDS[-1][2] == PITCH_LENGTH_M


def test_a_full_toss_is_a_trajectory_fact_not_a_length_band():
    """`bounced=False` wins over any length, because a full toss never made ground."""
    assert classify_length(5.0, bounced=False) is PitchLength.FULL_TOSS
    assert classify_length(None, bounced=False) is PitchLength.FULL_TOSS


def test_impossible_or_missing_lengths_report_unknown_not_a_nearest_band():
    assert classify_length(None) is PitchLength.UNKNOWN
    assert classify_length(-1.0) is PitchLength.UNKNOWN  # behind the striker's stumps
    assert classify_length(PITCH_LENGTH_M + 1) is PitchLength.UNKNOWN  # past the bowler's


def test_stump_corridor_uses_the_real_stump_width():
    assert in_stump_corridor(0.0) is True
    assert in_stump_corridor(STUMP_HALF_WIDTH_M) is True
    assert in_stump_corridor(STUMP_HALF_WIDTH_M + 0.01) is False
    assert in_stump_corridor(STUMP_HALF_WIDTH_M + 0.01, tolerance_m=0.05) is True
    assert in_stump_corridor(None) is None


def test_reference_geometry_matches_the_laws():
    """Pinned because these are the numbers every derived measurement scales by, and a
    plausible-looking typo in one would be invisible in the output."""
    assert PITCH_LENGTH_M == 20.12  # 22 yards, stumps to stumps
    assert STUMP_HEIGHT_M == 0.711  # 28 in
    assert STUMP_BLOCK_WIDTH_M == 0.2286  # 9 in
    assert STUMP_HALF_WIDTH_M == pytest.approx(0.1143)
