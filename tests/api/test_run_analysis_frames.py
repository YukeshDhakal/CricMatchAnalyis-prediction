"""`run_analysis`'s own wiring for `include_frames`, with the CV stack faked out.

What this covers is the glue, not the models: that the flag defaults to off, that turning
it on attaches both `tracks` and `sample_frames`, and -- the one most likely to rot -- that
the frames are built from the *same* detections and poses the summary counted, rather than
from a separately-recomputed set that could drift out of step with it.

Real models on real footage are not exercised here (a single clip is minutes of CPU). The
images themselves are verified pixel-by-pixel in tests/video_engine/test_overlay.py, and
this change was additionally run end to end against real footage during development.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from api import pipeline_runner
from video_engine.contracts import (
    BoundingBox,
    Detection,
    DeliveryEvent,
    Keypoint,
    ObjectClass,
    PoseFrame,
    ShotType,
    Track,
)

FRAME_COUNT = 6
WIDTH, HEIGHT = 320, 240


def _detection(frame_index: int) -> Detection:
    return Detection(
        frame_index=frame_index,
        obj_class=ObjectClass.PLAYER,
        box=BoundingBox(30.0, 30.0, 130.0, 190.0),
        confidence=0.8,
    )


@pytest.fixture
def fake_pipeline(monkeypatch):
    frames = [np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8) for _ in range(FRAME_COUNT)]
    detections = [_detection(i) for i in range(FRAME_COUNT)]
    poses = [
        PoseFrame(frame_index=i, track_id=1, keypoints=[Keypoint("nose", 80.0, 60.0, 0.9)])
        for i in range(FRAME_COUNT)
    ]
    tracks = [Track(track_id=1, obj_class=ObjectClass.PLAYER, detections=detections)]
    event = DeliveryEvent(
        release_frame=1, contact_frame=4, shot_type=ShotType.UNKNOWN, notes="faked"
    )

    class _FakeDetector:
        def detect(self, _frames):
            return detections

    class _FakeTracker:
        def track(self, _detections):
            return tracks

    class _FakePose:
        def estimate(self, _frames, _tracks):
            return poses

    class _FakeSegmenter:
        def segment(self, _tracks, _poses, _frames):
            return event

    # probe_video now also reports the frame count, because `api.frame_budget` needs it
    # to decide whether a clip fits in memory before anything is decoded.
    monkeypatch.setattr(
        pipeline_runner, "probe_video", lambda path: (WIDTH, HEIGHT, 25.0, FRAME_COUNT)
    )
    monkeypatch.setattr(
        pipeline_runner._Components,
        "get",
        classmethod(
            lambda cls: {
                "detector": _FakeDetector(),
                "tracker": _FakeTracker(),
                "pose_estimator": _FakePose(),
                "event_segmenter": _FakeSegmenter(),
            }
        ),
    )
    import video_engine.io.clip_loader as clip_loader

    # `load_frames` now takes the decode bound as keyword arguments; swallow them here --
    # that the real decoder honours them is tests/video_engine/test_clip_loader_budget.py.
    monkeypatch.setattr(clip_loader, "load_frames", lambda clip, **_kwargs: frames)
    return frames


def test_the_default_response_carries_no_frames_or_tracks(fake_pipeline, tmp_path):
    result = pipeline_runner.run_analysis(tmp_path / "clip.mp4", "clip.mp4")

    assert "sample_frames" not in result
    assert "tracks" not in result
    # The cheap summary is unaffected by the new flag.
    assert result["frames_processed"] == FRAME_COUNT
    assert result["people_tracked"] == 1


def test_include_frames_attaches_tracks_and_real_decodable_images(fake_pipeline, tmp_path):
    result = pipeline_runner.run_analysis(
        tmp_path / "clip.mp4", "clip.mp4", include_frames=True
    )

    assert result["tracks"] == [{"track_id": 1, "obj_class": "player", "frames_seen": 6}]
    assert len(result["sample_frames"]) == 4

    for frame in result["sample_frames"]:
        decoded = cv2.imdecode(
            np.frombuffer(base64.b64decode(frame["image_base64"]), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert decoded is not None, "sample_frames carried something that isn't an image"
        assert decoded.shape[:2] == (HEIGHT, WIDTH)
        # The overlay drew on it: the source frame was a single flat grey.
        assert len(np.unique(decoded.reshape(-1, 3), axis=0)) > 1


def test_the_sampled_frames_agree_with_the_counts_in_the_summary(fake_pipeline, tmp_path):
    result = pipeline_runner.run_analysis(
        tmp_path / "clip.mp4", "clip.mp4", include_frames=True
    )

    assert sum(f["detections"] for f in result["sample_frames"]) <= sum(
        v["count"] for v in result["detection_confidence"].values()
    )
    assert all(f["poses"] == 1 for f in result["sample_frames"])
    assert result["pose_frames"] == FRAME_COUNT
