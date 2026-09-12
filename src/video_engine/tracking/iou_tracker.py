from __future__ import annotations

from collections import defaultdict

from ..contracts import Detection, ObjectClass, Track
from .base import Tracker


class IouTracker(Tracker):
    """Greedy frame-to-frame IOU matching, per object class.

    A placeholder for a real multi-object tracker (e.g. ByteTrack): good enough to link
    detections within one short delivery clip, but it has no re-identification model, so a
    track that leaves and re-enters frame is counted as a new track.
    """

    def __init__(self, iou_threshold: float = 0.3, max_frames_missing: int = 5) -> None:
        self._iou_threshold = iou_threshold
        self._max_frames_missing = max_frames_missing

    def track(self, detections: list[Detection]) -> list[Track]:
        by_class: dict[ObjectClass, list[Detection]] = defaultdict(list)
        for det in detections:
            by_class[det.obj_class].append(det)

        tracks: list[Track] = []
        next_track_id = 0
        for obj_class, class_detections in by_class.items():
            frames = sorted({d.frame_index for d in class_detections})
            open_tracks: dict[int, tuple[Detection, int]] = {}
            accumulated: dict[int, list[Detection]] = defaultdict(list)

            for frame_index in frames:
                frame_dets = [d for d in class_detections if d.frame_index == frame_index]
                unmatched = set(range(len(frame_dets)))

                for track_id, (last_det, last_frame) in list(open_tracks.items()):
                    if frame_index - last_frame > self._max_frames_missing:
                        del open_tracks[track_id]
                        continue
                    best_index, best_iou = None, self._iou_threshold
                    for i in unmatched:
                        iou = last_det.box.iou(frame_dets[i].box)
                        if iou > best_iou:
                            best_index, best_iou = i, iou
                    if best_index is not None:
                        matched = frame_dets[best_index]
                        accumulated[track_id].append(matched)
                        open_tracks[track_id] = (matched, frame_index)
                        unmatched.discard(best_index)

                for i in unmatched:
                    det = frame_dets[i]
                    track_id = next_track_id
                    next_track_id += 1
                    accumulated[track_id].append(det)
                    open_tracks[track_id] = (det, frame_index)

            for track_id, track_detections in accumulated.items():
                tracks.append(Track(track_id=track_id, obj_class=obj_class, detections=track_detections))

        return tracks
