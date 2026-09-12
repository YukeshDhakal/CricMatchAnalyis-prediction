from video_engine.contracts import BoundingBox, Detection, ObjectClass
from video_engine.tracking.iou_tracker import IouTracker


def test_links_overlapping_detections_across_frames_into_one_track():
    tracker = IouTracker(iou_threshold=0.3)
    detections = [
        Detection(0, ObjectClass.PLAYER, BoundingBox(10, 10, 50, 90), 0.9),
        Detection(1, ObjectClass.PLAYER, BoundingBox(12, 10, 52, 90), 0.9),
        Detection(2, ObjectClass.PLAYER, BoundingBox(14, 11, 54, 91), 0.9),
    ]

    tracks = tracker.track(detections)

    assert len(tracks) == 1
    assert [d.frame_index for d in tracks[0].detections] == [0, 1, 2]


def test_separates_non_overlapping_detections_into_different_tracks():
    tracker = IouTracker(iou_threshold=0.3)
    detections = [
        Detection(0, ObjectClass.PLAYER, BoundingBox(0, 0, 20, 20), 0.9),
        Detection(0, ObjectClass.PLAYER, BoundingBox(500, 500, 520, 520), 0.9),
    ]

    tracks = tracker.track(detections)

    assert len(tracks) == 2


def test_drops_track_after_missing_too_many_frames():
    tracker = IouTracker(iou_threshold=0.3, max_frames_missing=1)
    detections = [
        Detection(0, ObjectClass.PLAYER, BoundingBox(10, 10, 50, 90), 0.9),
        # frames 1-2 missing, frame 3 reappears in roughly the same place
        Detection(3, ObjectClass.PLAYER, BoundingBox(10, 10, 50, 90), 0.9),
    ]

    tracks = tracker.track(detections)

    assert len(tracks) == 2
