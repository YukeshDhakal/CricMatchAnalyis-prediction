"""Data contracts for the video engine.

`DeliveryClip` is the handoff boundary with ingestion: one already-segmented
clip per delivery, plus enough metadata to line it back up with the
ball-by-ball table later in the fusion stage. Everything downstream of that
is owned by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Source(str, Enum):
    BROADCAST = "broadcast"
    USER_UPLOAD = "user_upload"


@dataclass(frozen=True)
class DeliveryRef:
    """Identifies one delivery, matching the key ingestion/fusion use for the ball-by-ball table."""

    match_id: str
    innings: int
    over: int
    ball: int  # ball number within the over, as bowled (includes wides/no-balls)


@dataclass(frozen=True)
class DeliveryClip:
    """One delivery's video, as handed off by ingestion."""

    delivery: DeliveryRef
    video_path: str
    fps: float
    width: int
    height: int
    source: Source


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def iou(self, other: "BoundingBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0


class ObjectClass(str, Enum):
    PLAYER = "player"
    BALL = "ball"
    STUMPS = "stumps"


@dataclass(frozen=True)
class Detection:
    frame_index: int
    obj_class: ObjectClass
    box: BoundingBox
    confidence: float


@dataclass(frozen=True)
class Track:
    """One object followed across frames within a single delivery clip."""

    track_id: int
    obj_class: ObjectClass
    detections: list[Detection] = field(default_factory=list)

    def box_at(self, frame_index: int) -> BoundingBox | None:
        for det in self.detections:
            if det.frame_index == frame_index:
                return det.box
        return None


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class PoseFrame:
    frame_index: int
    track_id: int  # the player Track this pose belongs to
    keypoints: list[Keypoint]

    def get(self, name: str) -> Keypoint | None:
        return next((k for k in self.keypoints if k.name == name), None)


class ShotType(str, Enum):
    DRIVE = "drive"
    CUT = "cut"
    PULL = "pull"
    SWEEP = "sweep"
    DEFEND = "defend"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DeliveryEvent:
    release_frame: int | None
    contact_frame: int | None
    shot_type: ShotType
    notes: str = ""


@dataclass(frozen=True)
class DeliveryAnalysis:
    """The video engine's full output for one delivery — feeds Part 02's feature store."""

    delivery: DeliveryRef
    tracks: list[Track]
    poses: list[PoseFrame]
    event: DeliveryEvent
