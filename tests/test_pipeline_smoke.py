import numpy as np

from video_engine.contracts import (
    BoundingBox,
    DeliveryClip,
    DeliveryRef,
    Detection,
    ObjectClass,
    Source,
)
from video_engine.events.heuristic_segmenter import HeuristicEventSegmenter
from video_engine.pipeline import VideoEngine
from video_engine.tracking.iou_tracker import IouTracker

from .fakes import FakeDetector, FakePoseEstimator


def test_pipeline_wires_all_stages_end_to_end(monkeypatch):
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(4)]
    monkeypatch.setattr("video_engine.pipeline.load_frames", lambda clip: frames)

    detections = [
        Detection(0, ObjectClass.PLAYER, BoundingBox(10, 10, 40, 90), 0.9),
        Detection(1, ObjectClass.PLAYER, BoundingBox(11, 10, 41, 90), 0.9),
        Detection(0, ObjectClass.BALL, BoundingBox(0, 0, 5, 5), 0.9),
        Detection(1, ObjectClass.BALL, BoundingBox(20, 20, 25, 25), 0.9),
    ]

    engine = VideoEngine(
        detector=FakeDetector(detections),
        tracker=IouTracker(),
        pose_estimator=FakePoseEstimator([]),
        event_segmenter=HeuristicEventSegmenter(),
    )

    clip = DeliveryClip(
        delivery=DeliveryRef(match_id="m1", innings=1, over=12, ball=3),
        video_path="unused.mp4",
        fps=25.0,
        width=64,
        height=64,
        source=Source.USER_UPLOAD,
    )

    analysis = engine.analyze(clip)

    assert analysis.delivery.match_id == "m1"
    assert {t.obj_class for t in analysis.tracks} == {ObjectClass.PLAYER, ObjectClass.BALL}
    assert analysis.event.release_frame == 0
