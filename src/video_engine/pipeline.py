from __future__ import annotations

from .contracts import DeliveryAnalysis, DeliveryClip
from .detection.base import Detector
from .events.base import EventSegmenter
from .io.clip_loader import load_frames
from .pose.base import PoseEstimator
from .tracking.base import Tracker


class VideoEngine:
    """Runs one delivery clip through detection, tracking, pose, and event segmentation.

    Build order matters: tracking depends on detection, pose estimation is matched onto
    player tracks, and event segmentation reads both tracks and poses.
    """

    def __init__(
        self,
        detector: Detector,
        tracker: Tracker,
        pose_estimator: PoseEstimator,
        event_segmenter: EventSegmenter,
    ) -> None:
        self._detector = detector
        self._tracker = tracker
        self._pose_estimator = pose_estimator
        self._event_segmenter = event_segmenter

    def analyze(self, clip: DeliveryClip) -> DeliveryAnalysis:
        frames = load_frames(clip)
        detections = self._detector.detect(frames)
        tracks = self._tracker.track(detections)
        poses = self._pose_estimator.estimate(frames, tracks)
        event = self._event_segmenter.segment(tracks, poses)
        return DeliveryAnalysis(delivery=clip.delivery, tracks=tracks, poses=poses, event=event)
