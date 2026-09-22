"""Regression test against detections recorded from real footage, not a synthetic ideal.

The fixture below is **verbatim output** of `weights/ball_stumps_n.pt` run over
`data/uploads/videos/real_bowling_clip_full.mp4` (960x540, 90 frames) at a 0.15
confidence threshold. It is checked in as data so this stays a real-footage regression
test without needing the clip or the checkpoint on the machine running it -- the video is
gitignored and the weights are a local training artifact.

What the fixture contains, established by looking at the actual frames rather than by
assuming:

* frames 1-89 at roughly (735, 333), box ~60x47 px -- a **bag on the ground by the fence**,
  detected as "ball" on 32 frames at 0.15-0.34.
* frames 80-86 travelling (136, 418) -> (481, 442), box ~24x23 px -- the **real ball**.

This is the exact configuration that broke the previous pooled-spread gate: it pooled both
objects, found the spread between them large, passed, and reported `release_frame=1` --
the bag. See MISTAKES.md.
"""
from video_engine.contracts import BoundingBox, Detection, ObjectClass, ShotType, Track
from video_engine.events.heuristic_segmenter import HeuristicEventSegmenter
from video_engine.trajectory import fit_ball_trajectory

# (frame, x1, y1, x2, y2, confidence)
REAL_BALL_DETECTIONS = [
    (1, 706.0, 308.7, 766.8, 356.1, 0.17),
    (2, 706.1, 308.7, 767.2, 356.1, 0.21),
    (3, 706.1, 308.7, 767.3, 356.1, 0.21),
    (4, 706.2, 308.5, 767.3, 356.1, 0.21),
    (5, 706.1, 308.2, 767.3, 356.2, 0.20),
    (6, 705.5, 309.5, 768.3, 357.4, 0.16),
    (8, 705.5, 309.5, 768.3, 357.2, 0.23),
    (9, 705.2, 309.3, 768.1, 357.7, 0.19),
    (10, 705.5, 309.5, 768.4, 357.4, 0.22),
    (11, 705.6, 309.7, 768.1, 357.5, 0.15),
    (12, 705.5, 309.7, 768.6, 357.5, 0.19),
    (14, 705.3, 309.8, 769.2, 357.5, 0.15),
    (16, 706.1, 308.4, 766.6, 355.3, 0.19),
    (23, 705.0, 309.6, 768.0, 357.5, 0.17),
    (70, 705.0, 309.9, 764.5, 356.9, 0.24),
    (71, 705.0, 309.8, 764.5, 356.9, 0.24),
    (72, 705.1, 310.4, 764.0, 356.7, 0.19),
    (74, 705.0, 310.4, 763.7, 356.6, 0.15),
    (76, 704.8, 310.0, 764.0, 356.7, 0.18),
    (77, 705.0, 310.3, 764.0, 356.6, 0.18),
    (78, 704.8, 310.2, 764.0, 356.7, 0.16),
    (79, 704.7, 309.7, 763.3, 357.0, 0.15),
    (80, 124.0, 407.1, 147.7, 429.8, 0.65),
    (80, 704.6, 309.6, 763.5, 357.0, 0.17),
    (81, 197.2, 418.4, 219.4, 438.9, 0.44),
    (81, 704.3, 308.8, 762.8, 356.6, 0.22),
    (82, 262.8, 424.5, 287.3, 448.0, 0.48),
    (82, 704.3, 308.7, 764.0, 356.2, 0.28),
    (83, 316.6, 418.2, 340.4, 440.8, 0.54),
    (83, 704.5, 308.9, 762.8, 356.4, 0.21),
    (84, 369.9, 430.1, 394.0, 453.2, 0.38),
    (84, 704.6, 309.6, 763.2, 357.8, 0.17),
    (85, 419.7, 423.0, 443.1, 445.5, 0.57),
    (85, 704.6, 309.5, 766.8, 359.8, 0.34),
    (86, 469.2, 430.7, 492.5, 454.1, 0.56),
    (86, 704.6, 311.3, 766.7, 360.6, 0.18),
    (87, 703.9, 309.8, 766.0, 359.1, 0.19),
    (88, 704.0, 309.6, 766.4, 359.2, 0.21),
    (89, 705.2, 310.5, 765.3, 358.6, 0.17),
]

# Where the bag sits, and where the real ball's flight begins and ends.
BAG_X = 735.0
REAL_BALL_FRAMES = {80, 81, 82, 83, 84, 85, 86}


def _detections() -> list[Detection]:
    return [
        Detection(f, ObjectClass.BALL, BoundingBox(x1, y1, x2, y2), conf)
        for f, x1, y1, x2, y2, conf in REAL_BALL_DETECTIONS
    ]


def test_fit_selects_the_real_ball_and_not_the_bag():
    fit = fit_ball_trajectory(_detections())

    assert fit is not None, "the real ball track should be recoverable from these detections"
    assert set(fit.inlier_frames) == REAL_BALL_FRAMES, (
        "the fit must select exactly the seven frames the real ball was visible on; "
        f"got {fit.inlier_frames}"
    )
    # Nothing from the bag's cluster may be an inlier: every inlier is far from its x.
    for frame in fit.inlier_frames:
        x, _ = fit.position_at(frame)
        assert abs(x - BAG_X) > 200, f"frame {frame} landed on the bag at x={x:.0f}"


def test_segmenter_reports_the_ball_release_not_the_bags_first_frame():
    """The precise regression: the old gate returned release_frame=1, the bag's first frame."""
    ball_track = Track(track_id=0, obj_class=ObjectClass.BALL, detections=_detections())

    event = HeuristicEventSegmenter().segment(tracks=[ball_track], poses=[])

    assert event.release_frame == 80, (
        "release must come from the real ball's first frame (80), not the bag's (1)"
    )


def test_a_clip_with_only_the_bag_yields_no_track_at_all():
    """Strip the seven real-ball frames and the remainder must produce nothing.

    This is the half the old gate got right and the new one must not lose: 32 detections of
    a stationary object are still not a ball, however many of them there are.
    """
    bag_only = [d for d in _detections() if d.frame_index not in REAL_BALL_FRAMES]

    assert fit_ball_trajectory(bag_only) is None

    event = HeuristicEventSegmenter().segment(
        tracks=[Track(track_id=0, obj_class=ObjectClass.BALL, detections=bag_only)], poses=[]
    )
    assert event.release_frame is None
    assert event.shot_type is ShotType.UNKNOWN
    assert "static object" in event.notes


def test_the_bag_outnumbers_the_ball_so_inlier_count_alone_would_pick_wrong():
    """Documents *why* the minimum-displacement check is a filter and not a tiebreak.

    If hypotheses were ranked on inlier count alone, the stationary cluster would win this
    clip outright -- it has more than four times as many detections as the real ball.
    """
    # Split by position, not by frame: the bag is detected on the ball's frames too, so
    # filtering by frame would undercount it.
    bag = [d for d in _detections() if d.box.x1 > 600]
    ball = [d for d in _detections() if d.box.x1 <= 600]

    assert len(ball) == 7
    assert len(bag) > 4 * len(ball), f"bag={len(bag)} ball={len(ball)}"
