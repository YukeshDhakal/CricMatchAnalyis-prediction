"""The segmenter's use of frame-level motion to drop resting ball detections.

Three properties matter and each has a failure mode worth pinning: the filter helps when
frames are informative, it does nothing when they are not, and it never reduces the
evidence below what the trajectory fit needs -- because it is a second line of defence,
not the only one.
"""
import numpy as np

from video_engine.contracts import BoundingBox, Detection, ObjectClass, Track
from video_engine.events.heuristic_segmenter import _drop_resting_detections


def _ball(frame: int, cx: float, cy: float, half: float = 5.0) -> Detection:
    return Detection(
        frame, ObjectClass.BALL, BoundingBox(cx - half, cy - half, cx + half, cy + half), 0.5
    )


def _scene(n: int = 12, size: int = 120) -> list[np.ndarray]:
    """A textured background with one patch that moves and one that never does."""
    frames = []
    rng = np.random.default_rng(0)
    background = rng.integers(0, 90, size=(size, size, 3), dtype=np.uint8)
    for f in range(n):
        frame = background.copy()
        frame[20:30, 20:30] = 240  # resting object, identical every frame
        x = 40 + f * 6
        frame[70:80, x : x + 10] = 240  # moving object
        frames.append(frame)
    return frames


def test_resting_detections_are_dropped_and_moving_ones_kept():
    frames = _scene()
    moving = [_ball(f, 45 + f * 6, 75) for f in range(1, 11)]
    resting = [_ball(f, 25, 25) for f in range(1, 11)]

    kept = _drop_resting_detections(moving + resting, frames)

    assert len(kept) >= 5
    # Everything kept should be from the moving cluster.
    assert all(d.box.y1 > 50 for d in kept), [(d.frame_index, d.box.y1) for d in kept]


def test_without_frames_the_filter_is_a_no_op():
    detections = [_ball(f, 25, 25) for f in range(10)]

    assert _drop_resting_detections(detections, None) == detections


def test_uninformative_frames_fall_back_instead_of_deleting_the_track():
    """Blank frames make motion energy meaningless; the filter must defer, not delete.

    Guards the failure `tests/test_pipeline_smoke.py` hit: all-zero frames scored zero
    energy for every detection, so a filter that trusted the signal removed the entire
    candidate set and the pipeline reported no ball at all.
    """
    blank = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(12)]
    detections = [_ball(f, 20 + f * 4, 30) for f in range(10)]

    assert _drop_resting_detections(detections, blank) == detections


def test_a_moving_camera_disables_the_filter_rather_than_trusting_it():
    """When most of the frame is changing, per-detection energy no longer discriminates."""
    stripes = np.tile((np.arange(120) % 2 * 220).astype(np.uint8)[None, :, None], (120, 1, 3))
    panning = [np.roll(stripes, shift, axis=1) for shift in range(12)]
    resting = [_ball(f, 25, 25) for f in range(10)]

    # Every detection survives: the filter declines to act rather than discarding real
    # detections on a signal it knows is invalid here.
    assert _drop_resting_detections(resting, panning) == resting


def test_the_filter_leaves_a_genuine_track_intact_end_to_end():
    from video_engine.events.heuristic_segmenter import HeuristicEventSegmenter

    frames = _scene()
    moving = [_ball(f, 45 + f * 6, 75) for f in range(1, 11)]
    resting = [_ball(f, 25, 25) for f in range(1, 11)]
    track = Track(track_id=0, obj_class=ObjectClass.BALL, detections=moving + resting)

    event = HeuristicEventSegmenter().segment([track], [], frames)

    assert event.release_frame is not None
    assert event.track_confidence > 0.0
