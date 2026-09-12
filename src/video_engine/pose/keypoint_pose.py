from __future__ import annotations

import numpy as np
import torch
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import to_tensor

from ..contracts import BoundingBox, Keypoint, ObjectClass, PoseFrame, Track
from .base import PoseEstimator

_COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class KeypointRcnnPoseEstimator(PoseEstimator):
    """Body-keypoint estimation via a pretrained Keypoint R-CNN, matched onto player tracks by box overlap.

    Runs `batch_size` frames per forward pass instead of one at a time — torchvision's detection
    models accept a list of images per call, so batching cuts the number of model dispatches
    from one per frame to one per chunk.
    """

    def __init__(
        self,
        score_threshold: float = 0.6,
        match_iou_threshold: float = 0.3,
        device: str = "cpu",
        batch_size: int = 8,
    ) -> None:
        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        self._model = keypointrcnn_resnet50_fpn(weights=weights)
        self._model.eval()
        self._device = torch.device(device)
        self._model.to(self._device)
        self._score_threshold = score_threshold
        self._match_iou_threshold = match_iou_threshold
        self._batch_size = batch_size

    def estimate(self, frames: list[np.ndarray], tracks: list[Track]) -> list[PoseFrame]:
        player_tracks = [t for t in tracks if t.obj_class == ObjectClass.PLAYER]
        if not player_tracks or not frames:
            return []

        pose_frames: list[PoseFrame] = []
        for start in range(0, len(frames), self._batch_size):
            chunk = frames[start : start + self._batch_size]
            tensors = [to_tensor(frame).to(self._device) for frame in chunk]
            with torch.no_grad():
                outputs = self._model(tensors)

            for offset, output in enumerate(outputs):
                frame_index = start + offset
                candidates = [t for t in player_tracks if t.box_at(frame_index) is not None]
                if candidates:
                    pose_frames.extend(self._to_pose_frames(frame_index, output, candidates))
        return pose_frames

    def _to_pose_frames(self, frame_index: int, output: dict, candidates: list[Track]) -> list[PoseFrame]:
        result: list[PoseFrame] = []
        for box, score, keypoints in zip(
            output["boxes"].tolist(), output["scores"].tolist(), output["keypoints"].tolist()
        ):
            if score < self._score_threshold:
                continue
            track = _best_matching_track(
                BoundingBox(*box), candidates, frame_index, self._match_iou_threshold
            )
            if track is None:
                continue
            result.append(
                PoseFrame(
                    frame_index=frame_index,
                    track_id=track.track_id,
                    keypoints=[
                        Keypoint(name=name, x=x, y=y, confidence=v)
                        for name, (x, y, v) in zip(_COCO_KEYPOINT_NAMES, keypoints)
                    ],
                )
            )
        return result


def _best_matching_track(
    detected_box: BoundingBox, candidates: list[Track], frame_index: int, threshold: float
) -> Track | None:
    best_track, best_iou = None, threshold
    for track in candidates:
        track_box = track.box_at(frame_index)
        if track_box is None:
            continue
        iou = detected_box.iou(track_box)
        if iou > best_iou:
            best_track, best_iou = track, iou
    return best_track
