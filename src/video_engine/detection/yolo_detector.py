from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from ultralytics import YOLO

from ..contracts import BoundingBox, Detection, ObjectClass
from .base import Detector

logger = logging.getLogger(__name__)

_COCO_PERSON_CLASS_ID = 0

# Class-index -> `ObjectClass` maps for a fine-tuned cricket checkpoint. Which of these
# applies is a property of the *weights file*, not something that can be inferred from
# it, so it is a constructor argument rather than a constant.
#
# BALL_AND_STUMPS is the shape this pipeline was designed around and the one
# `video_engine.calibration` needs: stumps are the calibration reference that turns pixel
# coordinates into metres on the pitch.
BALL_AND_STUMPS_CLASSES = {0: ObjectClass.BALL, 1: ObjectClass.STUMPS}

# BALL_ONLY exists because the only genuinely fine-tuned cricket-ball detectors findable
# in public are single-class. Declaring it explicitly is the point: a one-class checkpoint
# loaded under `BALL_AND_STUMPS_CLASSES` would emit correct BALL detections and silently
# never emit STUMPS, which looks identical to "the stumps were not visible in this clip"
# and would leave every STUMPS-dependent feature quietly disabled with no error anywhere.
# Naming the map forces the caller to state which kind of model they have.
BALL_ONLY_CLASSES = {0: ObjectClass.BALL}


class YoloDetector(Detector):
    """Player detection via a pretrained YOLO model; ball/stumps need a fine-tuned checkpoint.

    COCO-pretrained weights know "person" but were never trained on a cricket ball or stumps,
    so ball/stumps detection is a no-op until `ball_stumps_weights` points at a model fine-tuned
    on cricket footage — this still returns player detections in the meantime.

    **No such checkpoint ships with this repo and none is downloaded.** `ball_stumps_weights`
    is a path the operator supplies. The README's "Ball and stumps detection" section records
    what was surveyed and why nothing found was usable as-is — in particular, the one real
    fine-tuned cricket-ball detector located in public has no licence file, which makes it
    all-rights-reserved and not redistributable or usable without the author's permission.
    Nothing here assumes otherwise, and no URL to it is baked in.

    `ball_stumps_classes` declares what the supplied checkpoint's class indices mean. It
    defaults to `BALL_AND_STUMPS_CLASSES`; pass `BALL_ONLY_CLASSES` for a single-class ball
    detector. `emits(ObjectClass.STUMPS)` lets downstream code (calibration, pitch geometry)
    check whether a capability is actually present rather than inferring it from an empty
    detection list.

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
        ball_stumps_classes: dict[int, ObjectClass] | None = None,
    ) -> None:
        self._player_model = YOLO(player_weights)
        self._ball_stumps_model = YOLO(ball_stumps_weights) if ball_stumps_weights else None
        self._ball_stumps_classes = dict(
            ball_stumps_classes if ball_stumps_classes is not None else BALL_AND_STUMPS_CLASSES
        )
        if self._ball_stumps_model is None:
            logger.warning(
                "No ball_stumps_weights configured; YoloDetector will only emit PLAYER detections."
            )
        elif ObjectClass.STUMPS not in self._ball_stumps_classes.values():
            # Loud, because this is the case that silently disables pitch calibration.
            logger.warning(
                "ball_stumps_weights loaded with a class map that has no STUMPS class (%s); "
                "pitch calibration and every pitch-geometry metric stay unavailable.",
                sorted(c.value for c in self._ball_stumps_classes.values()),
            )
        self._confidence_threshold = confidence_threshold
        self._batch_size = batch_size

    def emits(self, obj_class: ObjectClass) -> bool:
        """Whether this detector can ever produce `obj_class`.

        A capability question, not a result question. "This clip contained no stumps" and
        "this detector has no stumps class" both produce zero STUMPS detections, and only
        the second means a feature is unavailable rather than inapplicable.
        """
        if obj_class is ObjectClass.PLAYER:
            return True
        if self._ball_stumps_model is None:
            return False
        return obj_class in self._ball_stumps_classes.values()

    def detect(self, frames: list[np.ndarray]) -> list[Detection]:
        if not frames:
            return []
        detections = self._run(self._player_model, frames, self._to_player_detection)
        if self._ball_stumps_model is not None:
            detections += self._run(self._ball_stumps_model, frames, self._to_ball_stumps_detection)
        return detections

    def _to_ball_stumps_detection(
        self, frame_index: int, box: list[float], cls: int, conf: float
    ) -> Detection | None:
        obj_class = self._ball_stumps_classes.get(cls)
        if obj_class is None:
            return None
        x1, y1, x2, y2 = box
        return Detection(frame_index, obj_class, BoundingBox(x1, y1, x2, y2), conf)

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
