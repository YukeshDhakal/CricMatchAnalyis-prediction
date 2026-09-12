from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..contracts import PoseFrame, Track


class PoseEstimator(ABC):
    """Estimates body keypoints for tracked players — feeds the technique-rating pillar."""

    @abstractmethod
    def estimate(self, frames: list[np.ndarray], tracks: list[Track]) -> list[PoseFrame]:
        """Return one PoseFrame per player track per frame it was detected in."""
