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


class PitchLength(str, Enum):
    """Where a delivery pitched, as a coaching band rather than a raw distance.

    `str`-valued for the same reason `ShotType` is: it has to survive a round trip
    through JSON or a DataFrame column without a codec.

    The band boundaries in metres live in `calibration.LENGTH_BANDS`, not here, because
    they are a tunable convention and this enum is a vocabulary -- the same split
    `rating.contracts.Metric` keeps from `METRIC_SPECS`.

    `FULL_TOSS` is not a band on the length axis and is not derived from one: a full toss
    is a ball that reaches the batter without touching the ground, which is a fact about
    the trajectory. `UNKNOWN` is the honest answer whenever there is no calibration, no
    tracked bounce, or a position that isn't physically on the pitch -- and it is the
    only value anything in this repo produces today, because no ball or stumps detector
    exists to produce the others. See `calibration`'s module docstring.
    """

    FULL_TOSS = "full_toss"
    YORKER = "yorker"
    FULL = "full"
    GOOD = "good"
    BACK_OF_A_LENGTH = "back_of_a_length"
    SHORT = "short"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PitchPoint:
    """Where a ball made contact with the pitch, in metres on the pitch plane.

    Origin is the **striker's middle stump**; `length_m` runs down the pitch toward the
    bowler, `line_m` across it. Produced only by `calibration.PitchCalibration`, which
    needs stumps detections that no current model produces -- so nothing constructs one
    of these on real footage yet.

    `line_m` is signed but *unlabelled*: positive is one side of the middle stump, and
    which side is off or leg depends on the striker's handedness, which Cricsheet does
    not publish. `calibration.PitchCalibration` states that gap at length; it is
    repeated here because this is the type that leaves the video engine, and a consumer
    reading `line_m = -0.4` must not assume it means "outside off".
    """

    length_m: float
    line_m: float


class GeometryConfidence(str, Enum):
    """How much to trust a delivery's pitch geometry, as a band rather than a number.

    A band, not a float, because the thing being communicated is a *decision about
    trust* and the precision of a float would imply a calibration nothing here has. The
    bands mean:

    * `NONE` -- no geometry was produced at all. The only honest display is "insufficient
      data"; there is no number to show with a caveat attached.
    * `LOW` -- a bounce point exists but rests on a short track, a loose trajectory fit, or
      a stump-end assignment that was assumed rather than established. Show it only with
      the uncertainty visible beside it, never as a bare figure.
    * `MEDIUM` / `HIGH` -- progressively better supported, and still an estimate from one
      camera. Nothing in this repo can currently reach `HIGH` on real footage; the band
      exists so that the ceiling is a property of the evidence rather than of the enum.

    The surfacing rule that goes with this: anything below `MEDIUM` degrades to
    "insufficient data" in coaching output, matching how `ShotType.UNKNOWN` is already
    handled, rather than being rendered as a clean number a reader would take literally.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class DeliveryEvent:
    release_frame: int | None
    contact_frame: int | None
    shot_type: ShotType
    notes: str = ""

    # --- reserved for a ball detector + pitch calibration that don't exist yet ---
    # `bounce_frame` is the frame the ball made ground contact on; `pitch_point` is where
    # that contact was in real-world metres. Both need `ObjectClass.BALL` detections
    # (no fine-tuned checkpoint exists) and, for the second, `ObjectClass.STUMPS` ones to
    # calibrate against (likewise). They stay `None` rather than being estimated from the
    # player-only detections that do exist -- the same refusal, for the same reason, that
    # `fusion.contracts.FusedDelivery` makes for its biomechanics fields.
    bounce_frame: int | None = None
    pitch_point: PitchPoint | None = None
    pitch_length: PitchLength = PitchLength.UNKNOWN

    # --- how much the geometry above is worth, and why ---
    # These are not decoration on the fields above; they are the part a consumer is
    # required to read before displaying them. `pitch_confidence` is NONE whenever
    # `pitch_point` is None, but the reverse does not hold: a bounce point can exist and
    # still be LOW, and a LOW point must not reach a coaching note as a bare number.
    # `pitch_notes` carries the *reason* -- "only one stump set visible", "track too
    # short" -- because "insufficient data" with no explanation is what makes a user
    # assume the feature is broken rather than that the footage was unsuitable.
    pitch_confidence: GeometryConfidence = GeometryConfidence.NONE
    pitch_notes: str = ""
    # Support for the ball track itself, in [0, 1], from `trajectory.TrajectoryFit`. 0.0
    # when no trajectory was fitted at all.
    track_confidence: float = 0.0


@dataclass(frozen=True)
class DeliveryAnalysis:
    """The video engine's full output for one delivery — feeds Part 02's feature store."""

    delivery: DeliveryRef
    tracks: list[Track]
    poses: list[PoseFrame]
    event: DeliveryEvent
