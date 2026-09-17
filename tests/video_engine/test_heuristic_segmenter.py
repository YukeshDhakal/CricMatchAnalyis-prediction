from video_engine.contracts import BoundingBox, Detection, Keypoint, ObjectClass, PoseFrame, ShotType, Track
from video_engine.events.heuristic_segmenter import HeuristicEventSegmenter


def _box(cx: float, cy: float, size: float = 10.0) -> BoundingBox:
    return BoundingBox(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


def _player(track_id: int, positions: dict[int, tuple[float, float]]) -> Track:
    return Track(
        track_id=track_id,
        obj_class=ObjectClass.PLAYER,
        detections=[
            Detection(frame_index=frame, obj_class=ObjectClass.PLAYER, box=_box(x, y), confidence=0.9)
            for frame, (x, y) in positions.items()
        ],
    )


def test_no_ball_track_reports_the_honest_no_track_note():
    segmenter = HeuristicEventSegmenter()
    event = segmenter.segment(tracks=[_player(0, {0: (100, 100)})], poses=[])

    assert event.shot_type == ShotType.UNKNOWN
    assert event.release_frame is None
    assert event.notes == "No ball track available for this clip."


def test_a_stationary_ball_track_is_rejected_as_a_false_positive():
    """The real-footage failure mode this guards against: a static background object (a bag,
    a sign) gets detected as 'ball' every frame at the same spot, jittering only within a
    fraction of its own box size. That must not be treated as a real ball -- it should fall
    back to the honest 'unknown' with a note explaining why, not silently drive a shot call."""
    stationary_ball = Track(
        track_id=9,
        obj_class=ObjectClass.BALL,
        detections=[
            Detection(frame_index=f, obj_class=ObjectClass.BALL, box=_box(500 + jitter, 300), confidence=0.55)
            for f, jitter in enumerate([0, 1, -1, 0, 2, -2, 1, 0])
        ],
    )
    batter = _player(1, {f: (505, 300) for f in range(8)})  # sits right next to the static object

    segmenter = HeuristicEventSegmenter()
    event = segmenter.segment(tracks=[stationary_ball, batter], poses=[])

    assert event.shot_type == ShotType.UNKNOWN
    assert event.release_frame is None
    assert "rejected" in event.notes
    assert "static object" in event.notes


def test_a_genuinely_moving_ball_track_is_still_accepted():
    """Guards against the fix being too aggressive: a ball that travels many multiples of
    its own box size across the clip must still be picked up and drive contact detection."""
    moving_ball = Track(
        track_id=9,
        obj_class=ObjectClass.BALL,
        detections=[
            Detection(frame_index=f, obj_class=ObjectClass.BALL, box=_box(100 + f * 60, 300), confidence=0.6)
            for f in range(6)
        ],
    )
    batter = _player(1, {f: (100 + f * 60, 300) for f in range(6)})  # co-located with the ball each frame
    poses = [
        PoseFrame(frame_index=f, track_id=1, keypoints=[Keypoint("right_wrist", 100 + f * 60, 300, 0.9)])
        for f in range(5, 5 + 6)
    ]

    segmenter = HeuristicEventSegmenter()
    event = segmenter.segment(tracks=[moving_ball, batter], poses=poses)

    assert event.release_frame == 0
    assert event.contact_frame is not None
    assert event.shot_type != ShotType.UNKNOWN or event.notes == ""


def test_prefers_a_moving_ball_track_over_a_stationary_one_when_both_exist():
    stationary_ball = Track(
        track_id=8,
        obj_class=ObjectClass.BALL,
        detections=[
            Detection(frame_index=f, obj_class=ObjectClass.BALL, box=_box(500, 300), confidence=0.5)
            for f in range(5)
        ],
    )
    moving_ball = Track(
        track_id=9,
        obj_class=ObjectClass.BALL,
        detections=[
            Detection(frame_index=f, obj_class=ObjectClass.BALL, box=_box(50 + f * 80, 100), confidence=0.6)
            for f in range(5)
        ],
    )
    batter = _player(1, {f: (50 + f * 80, 100) for f in range(5)})

    segmenter = HeuristicEventSegmenter()
    event = segmenter.segment(tracks=[stationary_ball, moving_ball, batter], poses=[])

    assert event.release_frame == 0
    assert event.notes != "Ball never came within range of a tracked player."
