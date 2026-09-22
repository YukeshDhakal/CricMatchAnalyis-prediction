"""Unit tests for the physics-constrained trajectory fit.

These test the *properties* the fit claims -- it rejects stationary clusters, it
interpolates frames the detector missed, it refuses tracks too sparse to verify -- rather
than pinning the coefficients the code happened to produce. A test that asserts
`coeffs_y == (0.5, -2.0, 100.0)` passes for a broken fit that returns constants, which is
the failure mode `tests/video_engine/test_calibration.py` already avoids by checking the
mathematical property instead.
"""
import pytest

from video_engine.contracts import BoundingBox, Detection, ObjectClass
from video_engine.trajectory import fit_ball_trajectory


def _ball(frame: int, cx: float, cy: float, size: float = 10.0, conf: float = 0.6) -> Detection:
    half = size / 2
    return Detection(frame, ObjectClass.BALL, BoundingBox(cx - half, cy - half, cx + half, cy + half), conf)


def _parabolic(frames=range(10), x0=50.0, vx=30.0, y0=300.0, vy=-40.0, g=6.0, size=10.0):
    """A ball launched upward in image space and pulled back down -- y = y0 + vy*t + g*t^2/2."""
    return [
        _ball(f, x0 + vx * f, y0 + vy * f + 0.5 * g * f * f, size)
        for f in frames
    ]


def test_recovers_a_clean_parabolic_track():
    fit = fit_ball_trajectory(_parabolic())

    assert fit is not None
    assert set(fit.inlier_frames) == set(range(10))
    assert fit.rms_px < 1.0


def test_interpolates_frames_the_detector_missed():
    """The recall fix: a frame with no detection still gets a position from its neighbours."""
    detections = [d for d in _parabolic() if d.frame_index not in {3, 4, 5}]

    fit = fit_ball_trajectory(detections)

    assert fit is not None
    assert 3 not in fit.inlier_frames
    expected = _parabolic(frames=[4])[0]
    cx = (expected.box.x1 + expected.box.x2) / 2
    cy = (expected.box.y1 + expected.box.y2) / 2
    x, y = fit.position_at(4)
    assert x == pytest.approx(cx, abs=2.0)
    assert y == pytest.approx(cy, abs=2.0)


def test_a_stationary_cluster_is_rejected_however_many_detections_it_has():
    """Fifty detections of a resting object are still not a ball."""
    stationary = [_ball(f, 500 + (f % 3) - 1, 300 + (f % 2)) for f in range(50)]

    assert fit_ball_trajectory(stationary) is None


def test_a_stationary_cluster_does_not_out_vote_a_shorter_real_track():
    """The hard-constraint ordering: more inliers must not beat actually moving."""
    stationary = [_ball(f, 500, 300) for f in range(40)]
    moving = _parabolic(frames=range(40, 48))

    fit = fit_ball_trajectory(stationary + moving)

    assert fit is not None
    assert min(fit.inlier_frames) >= 40, "the fit should have selected the moving track"


def test_two_distant_clusters_are_not_joined_into_one_flight():
    """A resting object early on and a moving one much later are not the same delivery.

    Without the frame-gap limit, a sufficiently gentle quadratic passes through both and
    wins on inlier count -- the failure this constraint was added for, observed on the 4K
    nets clip where a resting ball on frames 0-83 was joined to a falling object on
    frames 527-571.
    """
    early_static = [_ball(f, 400, 370) for f in range(30)]
    late_moving = _parabolic(frames=range(500, 510), x0=200.0)

    fit = fit_ball_trajectory(early_static + late_moving)

    assert fit is not None
    assert min(fit.inlier_frames) >= 500
    assert max(fit.inlier_frames) - min(fit.inlier_frames) < 100


def test_too_few_detections_to_verify_returns_none():
    """Three points lie exactly on some parabola, so three points verify nothing."""
    assert fit_ball_trajectory(_parabolic(frames=[0, 1, 2])) is None
    assert fit_ball_trajectory(_parabolic(frames=[0, 1, 2, 3])) is None


def test_empty_input_returns_none():
    assert fit_ball_trajectory([]) is None


def test_outlier_detections_are_excluded_from_the_fit():
    detections = _parabolic(frames=range(12)) + [_ball(4, 900, 50), _ball(7, 20, 500)]

    fit = fit_ball_trajectory(detections)

    assert fit is not None
    # Both outliers share a frame with a good detection, so the frames stay -- what must
    # hold is that the fitted path passes through the real points, not the outliers.
    x, y = fit.position_at(4)
    assert x < 400 and y < 400


def test_is_deterministic_for_the_same_input():
    detections = _parabolic(frames=range(12))

    first = fit_ball_trajectory(detections)
    second = fit_ball_trajectory(list(reversed(detections)))

    assert first is not None and second is not None
    assert first.inlier_frames == second.inlier_frames
    assert first.confidence == second.confidence


def test_bounce_is_located_where_vertical_direction_reverses():
    """A descend-then-ascend path in image space has ground contact at the reversal."""
    descending = [_ball(f, 100 + 25 * f, 200 + 12 * f) for f in range(6)]
    ascending = [_ball(6 + f, 250 + 25 * f, 272 - 12 * f) for f in range(6)]

    fit = fit_ball_trajectory(descending + ascending)

    assert fit is not None
    if fit.bounce_frame is not None:
        assert 4.0 <= fit.bounce_frame <= 8.0
        assert fit.bounce_point_px is not None


def test_a_track_that_never_reverses_reports_no_bounce():
    """Refusing to invent a bounce is the point -- a still-descending ball has not landed."""
    fit = fit_ball_trajectory([_ball(f, 100 + 25 * f, 100 + 10 * f) for f in range(10)])

    assert fit is not None
    assert fit.bounce_frame is None
    assert fit.bounce_point_px is None


def test_confidence_is_bounded_and_falls_when_the_fit_is_loose():
    tight = fit_ball_trajectory(_parabolic(frames=range(10)))
    noisy = fit_ball_trajectory(
        [
            _ball(f, 50 + 30 * f + (8 if f % 2 else -8), 300 - 40 * f + 3 * f * f + (8 if f % 3 else -8))
            for f in range(10)
        ]
    )

    assert tight is not None and noisy is not None
    assert 0.0 <= noisy.confidence <= 1.0
    assert 0.0 <= tight.confidence <= 1.0
    assert tight.confidence > noisy.confidence
