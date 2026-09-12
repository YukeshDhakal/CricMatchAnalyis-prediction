from __future__ import annotations

from abc import ABC, abstractmethod

from ..contracts import DeliveryClip


class VideoAdapter(ABC):
    """Writes match video into the normalised video store as per-delivery clips (PRD 2.2, pipeline stage 1)."""

    @abstractmethod
    def ingest(self, *args, **kwargs) -> list[DeliveryClip]:
        ...
