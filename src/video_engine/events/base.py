from __future__ import annotations

from abc import ABC, abstractmethod

from ..contracts import DeliveryEvent, PoseFrame, Track


class EventSegmenter(ABC):
    """Finds the release and bat-contact frames within a delivery and classifies the shot."""

    @abstractmethod
    def segment(self, tracks: list[Track], poses: list[PoseFrame]) -> DeliveryEvent:
        ...
