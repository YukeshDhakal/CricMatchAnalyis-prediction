from __future__ import annotations

from abc import ABC, abstractmethod

from ..contracts import Detection, Track


class Tracker(ABC):
    """Links per-frame detections of the same object class into persistent tracks."""

    @abstractmethod
    def track(self, detections: list[Detection]) -> list[Track]:
        """Return one Track per distinct object followed across the clip."""
