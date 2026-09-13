from __future__ import annotations

from collections import defaultdict

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker as _ByteTrack

from ..contracts import BoundingBox, Detection, ObjectClass, Track
from .base import Tracker


class ByteTrackTracker(Tracker):
    """Kalman-filter multi-object tracking via Roboflow's `trackers` library (ByteTrack).

    Recommended over IouTracker for anything beyond a quick local test: it predicts through
    brief occlusions and survives the erratic, high-speed motion of a cricket ball far better
    than greedy IOU matching. Runs one tracker instance per object class, since track IDs
    should never be shared between e.g. a player and the ball.

    A track only gets a stable ID after `minimum_consecutive_frames` matches (2 by default),
    so the first frame or two of a new object's appearance is dropped rather than mis-tracked.

    `high_conf_det_threshold`/`track_activation_threshold` default to `_MIN_TRACKABLE_CONFIDENCE`
    (0.4) rather than the `trackers` library's own defaults (0.6/0.7): those were tuned for dense
    pedestrian-tracking benchmarks, not this pipeline, and silently drop any detection between
    0.4 and 0.6 forever -- such a detection can only extend an *existing* track (the low-confidence
    recovery stage), never spawn one of its own, so an object that's never above ~0.6 in any frame
    just disappears. That collapsed two clearly-separate, correctly-boxed people (confidences 0.82
    and 0.58 -- both well above `YoloDetector`'s own 0.4 admission floor) into a single track when
    tested against real footage. 0.4 matches `YoloDetector.confidence_threshold`'s default: anything
    the detector already decided was real gets a chance to become its own track. Pass either kwarg
    explicitly to override.
    """

    _MIN_TRACKABLE_CONFIDENCE = 0.4

    def __init__(self, **tracker_kwargs) -> None:
        tracker_kwargs.setdefault("high_conf_det_threshold", self._MIN_TRACKABLE_CONFIDENCE)
        tracker_kwargs.setdefault("track_activation_threshold", self._MIN_TRACKABLE_CONFIDENCE)
        self._tracker_kwargs = tracker_kwargs

    def track(self, detections: list[Detection]) -> list[Track]:
        by_class: dict[ObjectClass, list[Detection]] = defaultdict(list)
        for det in detections:
            by_class[det.obj_class].append(det)

        tracks: list[Track] = []
        id_offset = 0
        for obj_class, class_detections in by_class.items():
            class_tracks = self._track_one_class(obj_class, class_detections, id_offset)
            tracks.extend(class_tracks)
            if class_tracks:
                id_offset = max(t.track_id for t in class_tracks) + 1
        return tracks

    def _track_one_class(
        self, obj_class: ObjectClass, detections: list[Detection], id_offset: int
    ) -> list[Track]:
        tracker = _ByteTrack(**self._tracker_kwargs)
        by_frame: dict[int, list[Detection]] = defaultdict(list)
        for det in detections:
            by_frame[det.frame_index].append(det)

        accumulated: dict[int, list[Detection]] = defaultdict(list)
        for frame_index in sorted(by_frame):
            frame_dets = by_frame[frame_index]
            sv_detections = sv.Detections(
                xyxy=np.array(
                    [[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in frame_dets], dtype=np.float32
                ),
                confidence=np.array([d.confidence for d in frame_dets], dtype=np.float32),
                class_id=np.zeros(len(frame_dets), dtype=int),
            )
            tracked = tracker.update(sv_detections)

            for xyxy, confidence, track_id in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id):
                if track_id < 0:
                    continue
                x1, y1, x2, y2 = xyxy
                accumulated[int(track_id) + id_offset].append(
                    Detection(
                        frame_index=frame_index,
                        obj_class=obj_class,
                        box=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(confidence),
                    )
                )

        return [
            Track(track_id=track_id, obj_class=obj_class, detections=dets)
            for track_id, dets in accumulated.items()
        ]
