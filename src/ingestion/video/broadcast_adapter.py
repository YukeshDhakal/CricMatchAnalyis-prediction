"""Broadcast feed adapter: intentionally not implemented.

PRD 2.4 (non-functional considerations) is explicit that broadcast footage use
requires a league/board licensing agreement before any video pipeline work
starts on that footage -- "a legal dependency, not an engineering one." This
class exists so the interface is settled and a real adapter can be dropped in
the moment rights are secured, but it refuses to pretend to work today.
"""
from __future__ import annotations

from ..contracts import DeliveryClip
from .base import VideoAdapter


class BroadcastFeedAdapter(VideoAdapter):
    def ingest(self, *args, **kwargs) -> list[DeliveryClip]:
        raise NotImplementedError(
            "Broadcast feed ingestion is blocked on league/board data-rights licensing "
            "(Third Umpire PRD 2.4). Use ManifestClipAdapter or SceneSplitAdapter for "
            "user-uploaded video until a licensing agreement is in place."
        )
