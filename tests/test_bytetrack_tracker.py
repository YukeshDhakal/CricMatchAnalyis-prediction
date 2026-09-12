from video_engine.contracts import BoundingBox, Detection, ObjectClass
from video_engine.tracking.bytetrack_tracker import ByteTrackTracker


def test_links_overlapping_detections_across_frames_into_one_track():
    tracker = ByteTrackTracker()
    detections = [
        Detection(0, ObjectClass.PLAYER, BoundingBox(10, 10, 50, 90), 0.9),
        Detection(1, ObjectClass.PLAYER, BoundingBox(12, 10, 52, 90), 0.9),
        Detection(2, ObjectClass.PLAYER, BoundingBox(14, 11, 54, 91), 0.9),
    ]

    tracks = tracker.track(detections)

    assert len(tracks) == 1
    # frame 0 is the unconfirmed warm-up match; a track only gets a stable id afterwards
    assert [d.frame_index for d in tracks[0].detections] == [1, 2]


def test_track_ids_stay_unique_across_object_classes():
    tracker = ByteTrackTracker()
    detections = [
        Detection(0, ObjectClass.PLAYER, BoundingBox(10, 10, 50, 90), 0.9),
        Detection(1, ObjectClass.PLAYER, BoundingBox(12, 10, 52, 90), 0.9),
        Detection(2, ObjectClass.PLAYER, BoundingBox(14, 11, 54, 91), 0.9),
        Detection(0, ObjectClass.BALL, BoundingBox(200, 200, 210, 210), 0.9),
        Detection(1, ObjectClass.BALL, BoundingBox(202, 201, 212, 211), 0.9),
        Detection(2, ObjectClass.BALL, BoundingBox(204, 202, 214, 212), 0.9),
    ]

    tracks = tracker.track(detections)

    track_ids = [t.track_id for t in tracks]
    assert len(track_ids) == len(set(track_ids)), "track ids must be unique across classes"
    classes_present = {t.obj_class for t in tracks}
    assert classes_present == {ObjectClass.PLAYER, ObjectClass.BALL}
