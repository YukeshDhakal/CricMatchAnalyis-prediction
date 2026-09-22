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
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(8)]
    monkeypatch.setattr("video_engine.pipeline.load_frames", lambda clip: frames)

    # The ball spans six frames rather than two. Two detections are no longer enough to
    # produce a ball path: `trajectory.fit_ball_trajectory` needs at least five so that a
    # fit has residual degrees of freedom, because any three points lie exactly on some
    # parabola and would "verify" anything. A smoke test that wires the stages together
    # therefore has to hand the segmenter a track it can actually verify.
    detections = [
        Detection(f, ObjectClass.PLAYER, BoundingBox(10 + f, 10, 40 + f, 90), 0.9)
        for f in range(6)
    ] + [
        Detection(f, ObjectClass.BALL, BoundingBox(f * 9, f * 4, f * 9 + 5, f * 4 + 5), 0.9)
        for f in range(6)
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
