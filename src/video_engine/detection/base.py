from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..contracts import Detection


class Detector(ABC):
    """Finds players, ball, and stumps in each frame of a delivery clip."""

    @abstractmethod
    def detect(self, frames: list[np.ndarray]) -> list[Detection]:
        """Return every detection found across all frames, unordered."""
