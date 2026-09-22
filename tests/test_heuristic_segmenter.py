from video_engine.contracts import (
    BoundingBox,
    Detection,
    Keypoint,
    ObjectClass,
    PoseFrame,
    ShotType,
    Track,
)
from video_engine.events.heuristic_segmenter import HeuristicEventSegmenter


def _ball_track(positions: dict[int, BoundingBox]) -> Track:
    return Track(
        track_id=99,
        obj_class=ObjectClass.BALL,
        detections=[Detection(f, ObjectClass.BALL, box, 0.9) for f, box in positions.items()],
    )


def _approaching_ball(frames: int = 6, step: float = 20.0) -> Track:
    """A ball travelling steadily toward (100, 100), arriving on the last frame.

    These tests are about contact detection and shot classification, not about whether a
    ball is real -- but since `HeuristicEventSegmenter` now refuses to report anything
    from a track too sparse to verify, they need a track that clears that bar. Six frames
    of steady approach is the smallest honest fixture: it ends at the same place the old
    two-point fixtures did, so the assertions below still test what they always tested.
    """
    last = frames - 1
    return _ball_track(
        {
            f: BoundingBox(
                100 - (last - f) * step,
                100 - (last - f) * step,
                105 - (last - f) * step,
                105 - (last - f) * step,
            )
            for f in range(frames)
        }
    )


def _player_track(track_id: int, positions: dict[int, BoundingBox]) -> Track:
    return Track(
        track_id=track_id,
        obj_class=ObjectClass.PLAYER,
        detections=[Detection(f, ObjectClass.PLAYER, box, 0.9) for f, box in positions.items()],
    )


def test_returns_unknown_when_no_ball_track():
    segmenter = HeuristicEventSegmenter()

    event = segmenter.segment(tracks=[], poses=[])

    assert event.shot_type == ShotType.UNKNOWN
    assert event.contact_frame is None


def test_finds_contact_frame_at_closest_ball_player_approach():
    ball = _approaching_ball()
    batter = _player_track(1, {5: BoundingBox(98, 98, 130, 180)})
    segmenter = HeuristicEventSegmenter()

    event = segmenter.segment(tracks=[ball, batter], poses=[])

    assert event.release_frame == 0
    assert event.contact_frame == 5


def test_classifies_small_wrist_movement_as_defend():
    ball = _approaching_ball()
    batter = _player_track(1, {5: BoundingBox(98, 98, 130, 180)})
    poses = [
        PoseFrame(5, 1, [Keypoint("right_wrist", 110, 150, 0.9)]),
        PoseFrame(6, 1, [Keypoint("right_wrist", 112, 151, 0.9)]),
    ]
    segmenter = HeuristicEventSegmenter()

    event = segmenter.segment(tracks=[ball, batter], poses=poses)

    assert event.shot_type == ShotType.DEFEND


def test_classifies_wide_horizontal_wrist_swing_as_drive():
    ball = _approaching_ball()
    batter = _player_track(1, {5: BoundingBox(98, 98, 130, 180)})
    poses = [
        PoseFrame(5, 1, [Keypoint("right_wrist", 100, 150, 0.9)]),
        PoseFrame(6, 1, [Keypoint("right_wrist", 160, 152, 0.9)]),
    ]
    segmenter = HeuristicEventSegmenter()

    event = segmenter.segment(tracks=[ball, batter], poses=poses)

    assert event.shot_type == ShotType.DRIVE
