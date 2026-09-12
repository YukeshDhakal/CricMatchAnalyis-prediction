from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from ultralytics import YOLO

from ..contracts import BoundingBox, Detection, ObjectClass
from .base import Detector

logger = logging.getLogger(__name__)

_COCO_PERSON_CLASS_ID = 0
_BALL_STUMPS_CLASS_MAP = {0: ObjectClass.BALL, 1: ObjectClass.STUMPS}


class YoloDetector(Detector):
    """Player detection via a pretrained YOLO model; ball/stumps need a fine-tuned checkpoint.

    COCO-pretrained weights know "person" but were never trained on a cricket ball or stumps,
    so ball/stumps detection is a no-op until `ball_stumps_weights` points at a model fine-tuned
    on cricket footage — this still returns player detections in the meantime.

    Runs each model once over the whole clip (`batch_size` frames per forward pass) rather than
    once per frame — a delivery clip is only a few hundred frames, so batching it is the
    difference between one model dispatch and hundreds of them.
    """

    def __init__(
        self,
        player_weights: str = "yolov8n.pt",
        ball_stumps_weights: str | None = None,
        confidence_threshold: float = 0.4,
        batch_size: int = 16,
    ) -> None:
        self._player_model = YOLO(player_weights)
        self._ball_stumps_model = YOLO(ball_stumps_weights) if ball_stumps_weights else None
        if self._ball_stumps_model is None:
            logger.warning(
                "No ball_stumps_weights configured; YoloDetector will only emit PLAYER detections."
            )
        self._confidence_threshold = confidence_threshold
        self._batch_size = batch_size

    def detect(self, frames: list[np.ndarray]) -> list[Detection]:
        if not frames:
            return []
        detections = self._run(self._player_model, frames, self._to_player_detection)
        if self._ball_stumps_model is not None:
            detections += self._run(self._ball_stumps_model, frames, self._to_ball_stumps_detection)
        return detections

    def _run(
        self,
        model: YOLO,
        frames: list[np.ndarray],
        to_detection: Callable[[int, list[float], int, float], Detection | None],
    ) -> list[Detection]:
        results = model.predict(frames, batch=self._batch_size, verbose=False)
        out: list[Detection] = []
        for frame_index, result in enumerate(results):
            for box, cls, conf in zip(
                result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
            ):
                if conf < self._confidence_threshold:
                    continue
                detection = to_detection(frame_index, box, int(cls), conf)
                if detection is not None:
                    out.append(detection)
        return out

    @staticmethod
    def _to_player_detection(
        frame_index: int, box: list[float], cls: int, conf: float
    ) -> Detection | None:
        if cls != _COCO_PERSON_CLASS_ID:
            return None
        x1, y1, x2, y2 = box
        return Detection(frame_index, ObjectClass.PLAYER, BoundingBox(x1, y1, x2, y2), conf)

    @staticmethod
    def _to_ball_stumps_detection(
        frame_index: int, box: list[float], cls: int, conf: float
    ) -> Detection | None:
        obj_class = _BALL_STUMPS_CLASS_MAP.get(cls)
        if obj_class is None:
            return None
        x1, y1, x2, y2 = box
        return Detection(frame_index, obj_class, BoundingBox(x1, y1, x2, y2), conf)
