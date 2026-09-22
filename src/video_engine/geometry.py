"""Turning stumps detections into a pitch calibration, and a bounce point into line/length.

This is the caller `calibration.calibrate_from_stumps` was written for and did not have.
Its docstring deliberately refuses to guess which detected stump set is the striker's end
and which is the bowler's, on the grounds that "the larger box is nearer" is *"right often
enough to be dangerous"*. That refusal is correct and is not overridden here. What this
module adds is the missing third option between guessing silently and refusing forever:
make the assumption **explicit, conditional, and visible in the output**.

Concretely, `resolve_stump_ends` applies the near/far size heuristic only when the two
clusters differ decisively in apparent height, refuses when they do not, and returns the
assumption it made as text that flows through `DeliveryEvent.pitch_notes` to the user. A
reader of a length figure therefore sees the assumption it rests on, which is the
difference between an estimate and a guess.

**What real footage says about how often this path will fire: rarely, for now.** All four
clips in `data/uploads/videos/` were probed with the trained checkpoint. Stumps are
detected well -- 103 frames at 0.40+ in the 4K nets clip, cleanly localised -- but only
ever *one* set, because the camera sits behind the bowler's arm and the far stumps are
occluded by the batter or out of frame. A homography needs four coplanar points, and one
stump set supplies two. So on every clip available here this module returns no calibration
and the pipeline reports `GeometryConfidence.NONE` with a reason. That is the honest
outcome, not a bug, and it is the specific thing a user can unblock by supplying footage
with both stump sets visible -- which is exactly the camera placement the fixed-tripod
setups this project targets already call for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .calibration import PitchCalibration, calibrate_from_stumps, classify_length
from .contracts import (
    BoundingBox,
    Detection,
    GeometryConfidence,
    ObjectClass,
    PitchLength,
    PitchPoint,
)

__all__ = [
    "MIN_END_SIZE_RATIO",
    "CLUSTER_RADIUS_MULTIPLE",
    "StumpEnds",
    "PitchGeometry",
    "resolve_stump_ends",
    "pitch_geometry",
]

# The two stump sets must differ in apparent height by at least this factor before the
# near/far assignment is made from size. The pitch is 20.12 m long, so under any normal
# camera placement the near stumps project far larger than the far ones -- a ratio near 1
# means the camera is side-on to the pitch, where the heuristic has no signal and would be
# a coin flip. Refusing there is the whole point.
MIN_END_SIZE_RATIO = 1.5

# A detection joins an existing stump cluster when its centre is within this multiple of
# the larger of the two box heights. Without a separation rule, two boxes on the *same*
# stumps across frames (ordinary detector jitter) would be handed to the homography as
# opposite ends of the pitch, producing a confident calibration in which 20 m maps onto a
# few pixels.
#
# Clustering is on the **2D centre**, and that detail is load-bearing. An earlier version
# separated clusters by horizontal position alone, which is wrong for the camera placement
# this project actually targets: filming down the pitch puts both stump sets at nearly the
# same x, differing in height in frame and in apparent size. Grouping by x merged them into
# one cluster and silently disabled calibration on precisely the footage it was written
# for. Caught by `tests/video_engine/test_geometry.py`, whose fixture is built from the
# real 4K clip's geometry rather than from a convenient side-on layout.
CLUSTER_RADIUS_MULTIPLE = 0.75


@dataclass(frozen=True)
class StumpEnds:
    """Two stump sets, assigned to ends, plus the assumption that assignment rests on."""

    strikers_end: BoundingBox
    bowlers_end: BoundingBox
    assumption: str


@dataclass(frozen=True)
class PitchGeometry:
    """Where a delivery pitched, how confident that is, and why.

    `point` is `None` whenever the geometry could not be established, and `notes` then
    says what was missing. `confidence` is `NONE` in exactly that case.
    """

    point: Optional[PitchPoint]
    length: PitchLength
    confidence: GeometryConfidence
    notes: str
    calibration: Optional[PitchCalibration] = None


def _mean_box(boxes: Sequence[BoundingBox]) -> BoundingBox:
    """The component-wise mean box of a cluster.

    Averaging across frames rather than taking the single most confident detection: the
    stumps do not move, so every frame is a repeated measurement of one fixed quantity,
    and the mean suppresses per-frame localisation jitter that would otherwise propagate
    straight into the homography. This is the one place in this pipeline where averaging
    over time is unambiguously correct, precisely because the target is static.
    """
    n = len(boxes)
    return BoundingBox(
        x1=sum(b.x1 for b in boxes) / n,
        y1=sum(b.y1 for b in boxes) / n,
        x2=sum(b.x2 for b in boxes) / n,
        y2=sum(b.y2 for b in boxes) / n,
    )


def _centre(box: BoundingBox) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)


def _cluster(detections: Sequence[Detection]) -> list[list[BoundingBox]]:
    """Group stumps detections into spatial clusters by 2D centre.

    Deliberately simple agglomeration rather than a general clustering algorithm: the
    thing being separated is two stump sets at opposite ends of a pitch, which are far
    apart compared with the per-frame jitter within either. Anything more elaborate would
    be unfalsifiable against the footage available here, where only one cluster ever
    exists.
    """
    if not detections:
        return []
    # Sorted by base height so the nearest (largest, lowest in frame) set is considered
    # last; the order only affects which cluster seeds first, never the grouping.
    boxes = sorted((d.box for d in detections), key=lambda b: b.y2)
    clusters: list[list[BoundingBox]] = []
    for box in boxes:
        cx, cy = _centre(box)
        height = max(box.y2 - box.y1, 1.0)
        for cluster in clusters:
            mean = _mean_box(cluster)
            mx, my = _centre(mean)
            mean_height = max(mean.y2 - mean.y1, 1.0)
            reach = CLUSTER_RADIUS_MULTIPLE * max(height, mean_height)
            if ((cx - mx) ** 2 + (cy - my) ** 2) ** 0.5 <= reach:
                cluster.append(box)
                break
        else:
            clusters.append([box])
    return clusters


def resolve_stump_ends(detections: Sequence[Detection]) -> Optional[StumpEnds]:
    """Assign detected stump sets to ends, or `None` when that cannot be done honestly.

    Returns `None` for: no stumps, one stump set only (the case every clip in this repo
    hits), more than two distinct sets, or two sets whose apparent sizes are too similar
    for the near/far assignment to mean anything.
    """
    stumps = [d for d in detections if d.obj_class is ObjectClass.STUMPS]
    clusters = _cluster(stumps)
    if len(clusters) != 2:
        return None

    first, second = (_mean_box(c) for c in clusters)
    first_h = first.y2 - first.y1
    second_h = second.y2 - second.y1
    if min(first_h, second_h) <= 0:
        return None

    ratio = max(first_h, second_h) / min(first_h, second_h)
    if ratio < MIN_END_SIZE_RATIO:
        return None

    near, far = (first, second) if first_h > second_h else (second, first)
    return StumpEnds(
        strikers_end=near,
        bowlers_end=far,
        assumption=(
            f"striker's end assigned to the larger stump set (apparent height ratio "
            f"{ratio:.1f}x); this assumes the camera is nearer the striker's end"
        ),
    )


def _confidence_band(track_confidence: float, calibration: PitchCalibration) -> GeometryConfidence:
    """Fold track support and fit quality into a trust band.

    Capped at `MEDIUM` on purpose. The remaining error is dominated by something neither
    input can see: the homography is anchored on a 0.2286 m x 20.12 m rectangle, an 88:1
    aspect ratio whose across-pitch conditioning is poor no matter how clean the
    reprojection residual looks (see `calibration`'s docstring). Reporting `HIGH` from a
    single camera would be claiming a precision the geometry does not support, so nothing
    here can produce it.
    """
    if track_confidence >= 0.6 and calibration.rms_error_px <= 1.0:
        return GeometryConfidence.MEDIUM
    return GeometryConfidence.LOW


def pitch_geometry(
    detections: Sequence[Detection],
    bounce_point_px: Optional[tuple[float, float]],
    track_confidence: float,
    bounced: bool = True,
) -> PitchGeometry:
    """Project a bounce point onto the pitch, or explain why that could not be done.

    `bounce_point_px` is the image-space ground-contact point from
    `trajectory.TrajectoryFit`. Only that single point is projected, and the reason is
    geometric rather than a matter of effort: the homography maps the *pitch plane*, a ball
    in flight is above it, and unprojecting an airborne detection through a ground-plane
    homography yields a confident number with no physical meaning. At ground contact the
    ball is on the plane by definition, so that one point is the only place the mapping is
    exact.
    """
    if bounce_point_px is None:
        # UNKNOWN, emphatically not FULL_TOSS. `classify_length(None, bounced=False)`
        # returns FULL_TOSS, and its docstring is explicit that this is for "a caller that
        # knows the ball never made ground contact". This caller knows no such thing: it
        # knows only that no bounce was *detected* within a track that, on real footage,
        # is routinely seven frames of a delivery that lasted fifty. Reporting a full toss
        # from a partial track would be inventing a cricketing fact out of a tracking gap
        # -- the same shape of error as the confident shot_type this project already
        # documents in MISTAKES.md. Verified against real_bowling_clip_full.mp4, where the
        # first version of this call reported `full_toss` for a ball visibly rolling along
        # the ground.
        return PitchGeometry(
            point=None,
            length=PitchLength.UNKNOWN,
            confidence=GeometryConfidence.NONE,
            notes=(
                "No ground contact was located in the tracked ball path, so there is no "
                "point to project onto the pitch. This is not evidence of a full toss -- "
                "the ball may simply have bounced outside the tracked frames."
            ),
        )

    ends = resolve_stump_ends(detections)
    if ends is None:
        stumps_seen = any(d.obj_class is ObjectClass.STUMPS for d in detections)
        return PitchGeometry(
            point=None,
            length=PitchLength.UNKNOWN,
            confidence=GeometryConfidence.NONE,
            notes=(
                "Stumps were detected at only one end (or were too similar in size to tell "
                "the ends apart), so the pitch could not be calibrated. Line and length need "
                "both stump sets visible in frame."
                if stumps_seen
                else "No stumps were detected, so the pitch could not be calibrated."
            ),
        )

    calibration = calibrate_from_stumps(ends.strikers_end, ends.bowlers_end)
    if calibration is None:
        return PitchGeometry(
            point=None,
            length=PitchLength.UNKNOWN,
            confidence=GeometryConfidence.NONE,
            notes=(
                "Both stump sets were found but the pitch-plane fit was rejected as "
                "unreliable, so no line or length is reported."
            ),
        )

    point = calibration.pitch_point(*bounce_point_px)
    if point is None:
        return PitchGeometry(
            point=None,
            length=PitchLength.UNKNOWN,
            confidence=GeometryConfidence.NONE,
            notes="The bounce point does not map onto the pitch plane (it lies at or past the horizon).",
            calibration=calibration,
        )

    length = classify_length(point.length_m, bounced=bounced)
    if length is PitchLength.UNKNOWN:
        return PitchGeometry(
            point=point,
            length=length,
            confidence=GeometryConfidence.NONE,
            notes=(
                f"The projected bounce point ({point.length_m:.2f} m) falls outside the pitch, "
                "which means the calibration or the tracked bounce is wrong. Reported as unknown "
                "rather than snapped to the nearest band."
            ),
            calibration=calibration,
        )

    return PitchGeometry(
        point=point,
        length=length,
        confidence=_confidence_band(track_confidence, calibration),
        notes=(
            f"Single-camera estimate, not a measurement. {ends.assumption}. Line is inherently "
            f"less reliable than length with this method (the calibration rectangle is ~88:1)."
        ),
        calibration=calibration,
    )
