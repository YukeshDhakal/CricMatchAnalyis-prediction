from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

from ..contracts import DeliveryEvent, PoseFrame, Track


class EventSegmenter(ABC):
    """Finds the release and bat-contact frames within a delivery and classifies the shot."""

    @abstractmethod
    def segment(
        self,
        tracks: list[Track],
        poses: list[PoseFrame],
        frames: Optional[Sequence[np.ndarray]] = None,
    ) -> DeliveryEvent:
        """Segment one delivery.

        `frames` is optional, and its optionality is the contract rather than a
        convenience. A segmenter that has the pixels can ask whether a detection was
        *moving* -- the signal that separates the delivered ball from a ball resting on the
        outfield, which no per-frame detector can supply. A segmenter that does not have
        them must still return a defensible answer from tracks alone, so every caller that
        cannot cheaply supply frames (tests, replay over stored detections, the fusion
        layer working from persisted tracks) keeps working without being handed a worse
        answer silently. Implementations that use `frames` should say what degrades when
        it is absent.
        """
        ...
