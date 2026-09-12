from .contracts import (
    BoundingBox,
    Detection,
    DeliveryAnalysis,
    DeliveryClip,
    DeliveryEvent,
    DeliveryRef,
    Keypoint,
    ObjectClass,
    PoseFrame,
    ShotType,
    Source,
    Track,
)

# VideoEngine (pipeline.py) pulls in the full detection/pose CV stack (ultralytics,
# torch, torchvision) transitively. Contracts is the handoff boundary with ingestion
# and other lightweight consumers, so it must stay importable without that stack --
# load VideoEngine lazily (PEP 562) rather than eagerly at package import time.


def __getattr__(name: str):
    if name == "VideoEngine":
        from .pipeline import VideoEngine

        return VideoEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BoundingBox",
    "Detection",
    "DeliveryAnalysis",
    "DeliveryClip",
    "DeliveryEvent",
    "DeliveryRef",
    "Keypoint",
    "ObjectClass",
    "PoseFrame",
    "ShotType",
    "Source",
    "Track",
    "VideoEngine",
]
