"""Pitch-plane calibration: turning pixel coordinates into metres on the pitch.

**Nothing in this module runs on real footage today, and that is the point of reading
its docstring before using it.** It is the missing half of the pixel-space problem
`fusion.contracts.FusedDelivery` documents: the pose estimator produces real keypoints
and the tracker produces real tracks, but both are in raw pixels, and there is no
mapping from pixels to metres without knowing where the camera is. Calibration is that
mapping. With it, a ball's bounce point becomes a length and a line; without it, a
bounce point is a pair of pixel coordinates that means nothing outside the frame it
came from.

**What this module needed that didn't exist for most of its life: stumps detections.**
`ObjectClass.STUMPS` has been reserved in `contracts.py` since the start and
`YoloDetector.ball_stumps_weights` has been the hook for producing it. A fine-tuned
2-class (ball, stumps) checkpoint now exists at `weights/ball_stumps_n.pt` -- see the
README's "Ball and stumps detection" section for the dataset, licence and training run --
and `app/streamlit_app.py` loads it automatically when present. `calibrate_from_stumps`
still has no caller in the pipeline today, though: wiring it in wants a real-footage check
of the detector's stump localisation first, not just this module's synthetic-camera tests.
It is implemented, tested and correct anyway, because the calibration approach is the
thing that determines what the detector has to detect, and getting that backwards (train
a detector, then discover the geometry needs something else from it) is a more expensive
mistake than writing this early.

**The approach**: a planar homography anchored on the stumps' known real-world size, as
used by low-cost fixed-camera setups in the wild (Fulltrack AI's published setup
instructions specify a fixed elevated tripod above ~60 in with the stumps visible at
both ends of frame, for exactly this reason). The pitch surface is a plane; a camera
views it through a projective transform; four points whose real-world positions are
known determine that transform. The base of each stump set at each end supplies exactly
four such points.

**Why the result is honest but not precise, stated up front** because a number in
metres looks authoritative in a way a pixel coordinate does not:

* The four reference points form a 0.2286 m x 20.12 m rectangle -- an aspect ratio of
  about 88:1. A homography estimated from so thin a quadrilateral is well conditioned
  along the pitch and poorly conditioned across it, so **line is inherently less
  accurate than length** from this method, and gets worse with distance from the
  camera, where the far stumps occupy few pixels. `PitchCalibration.rms_error_px`
  exists so a caller can see when the fit is bad, and `MAX_REPROJECTION_ERROR_PX`
  refuses a fit that is worse than useless rather than returning it.
* It assumes the bounce point lies *on* the pitch plane. That is true for a bounce and
  false for a full toss, which is why `classify_length` reports `FULL_TOSS` from the
  ball track's own behaviour (no ground contact) rather than from a projected position.
* It assumes a fixed camera for the delivery. A broadcast cut mid-delivery invalidates
  the homography, and there is nothing here that detects one.

Nothing in this module estimates, infers or falls back to a default calibration. Every
entry point returns `None` when its inputs cannot support a real answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .contracts import BoundingBox, PitchLength, PitchPoint

__all__ = [
    "PITCH_LENGTH_M",
    "STUMP_HEIGHT_M",
    "STUMP_BLOCK_WIDTH_M",
    "STUMP_HALF_WIDTH_M",
    "POPPING_CREASE_FROM_STUMPS_M",
    "LENGTH_BANDS",
    "MAX_REPROJECTION_ERROR_PX",
    "PitchCalibration",
    "calibrate_from_stumps",
    "classify_length",
]

# --- Laws-of-Cricket reference geometry. Fixed, not tunable. -------------------------
#
# Distance between the two sets of stumps (equivalently, between the bowling creases):
# 22 yards. Worth stating precisely because it is easy to quote as "between the popping
# creases", which is a different and shorter measurement -- the popping creases are
# POPPING_CREASE_FROM_STUMPS_M in front of each set of stumps, so popping crease to
# popping crease is 17.68 m, not 20.12 m. This module measures from the *stumps*,
# because the stumps are what the calibration reference detects.
PITCH_LENGTH_M = 20.12

# Stump height above ground, and the width across all three stumps including the gaps.
STUMP_HEIGHT_M = 0.711  # 28 in
STUMP_BLOCK_WIDTH_M = 0.2286  # 9 in
STUMP_HALF_WIDTH_M = STUMP_BLOCK_WIDTH_M / 2  # outer stump offset from middle stump

POPPING_CREASE_FROM_STUMPS_M = 1.22  # 4 ft

# A fit worse than this is refused rather than returned. Four points determine a
# homography exactly, so a *perfect* algebraic fit is guaranteed and tells you nothing;
# this bound catches the cases that matter -- near-collinear or mis-ordered corners,
# where the solve succeeds numerically and the resulting map is nonsense. Picked as a
# few pixels of tolerance at broadcast resolution, not measured against annotated
# footage, because no annotated footage exists here to measure against.
MAX_REPROJECTION_ERROR_PX = 2.0

# Length bands in metres from the striker's stumps, as (name, lower_inclusive,
# upper_exclusive).
#
# **These are a documented convention, not fitted values**, in exactly the sense
# `rating.contracts.PHASE_IMPACT_WEIGHTS` is: published length bands differ between
# analytics providers by up to a metre at every boundary, and nothing in this repo has
# the labelled ball-tracking data to fit them. They are a single table so a later
# calibration pass changes one place, and so a test can state the mapping it depends on
# instead of hard-coding magic numbers. `PitchLength.FULL_TOSS` is deliberately absent:
# a full toss is defined by never bouncing, not by where it would have bounced, so it
# cannot be a band on this axis. See `classify_length`.
LENGTH_BANDS: tuple[tuple[PitchLength, float, float], ...] = (
    (PitchLength.YORKER, 0.0, 2.0),
    (PitchLength.FULL, 2.0, 4.0),
    (PitchLength.GOOD, 4.0, 7.0),
    (PitchLength.BACK_OF_A_LENGTH, 7.0, 9.0),
    (PitchLength.SHORT, 9.0, PITCH_LENGTH_M),
)


@dataclass(frozen=True)
class PitchCalibration:
    """A homography from image pixels to pitch-plane metres, plus how well it fits.

    The pitch frame this maps into: origin at the **striker's middle stump**, `x`
    positive across the pitch, `y` positive down the pitch toward the bowler's end. So
    a delivery pitching on a good length has `y` around 5 and `x` near 0.

    `x` is *not* labelled off side or leg side, and cannot be. Which side of the pitch
    is which depends on the striker's handedness, and Cricsheet's registry carries no
    batting-hand metadata -- the same gap `prediction.contracts.MatchupFeatures` and
    `rating.contracts.PlayerRole` document and decline to fake. A signed distance from
    the middle stump is the most that can honestly be said; converting it to "outside
    off" needs a handedness source this project does not have.

    `matrix` is stored as nested tuples rather than an ndarray so this contract is
    genuinely immutable and hashable like every other frozen contract here -- an ndarray
    field on a frozen dataclass is still writable in place, which would let a caller
    edit a calibration that a stored measurement already cited.

    `rms_error_px` is the root-mean-square reprojection error of the four reference
    points through the fitted matrix. With exactly four correspondences this is near
    zero by construction and is a numerical-sanity check, not an accuracy estimate --
    the real error is in where the stump boxes were detected, which this cannot see.
    """

    matrix: tuple[tuple[float, ...], ...]
    rms_error_px: float
    image_points: tuple[tuple[float, float], ...]

    def as_array(self) -> np.ndarray:
        return np.array(self.matrix, dtype=float)

    def pitch_point(self, x_px: float, y_px: float) -> Optional[PitchPoint]:
        """Map one image point onto the pitch plane.

        Returns `None` when the point maps to or past the horizon (a vanishing
        denominator), which is what happens to anything above the plane's horizon line
        in the image -- a real geometric answer of "this pixel does not correspond to a
        point on the pitch", not an error to be clamped away.
        """
        u, v, w = self.as_array() @ np.array([x_px, y_px, 1.0])
        if abs(w) < 1e-9:
            return None
        return PitchPoint(length_m=round(float(v / w), 3), line_m=round(float(u / w), 3))


def _homography(image_pts: np.ndarray, world_pts: np.ndarray) -> Optional[np.ndarray]:
    """Least-squares homography image -> world by the normalised DLT.

    Hartley normalisation (translate each point set to zero mean, scale to mean
    distance sqrt(2) from the origin) before the solve and un-normalisation after.
    Not optional decoration: the raw system mixes pixel coordinates in the hundreds
    with world coordinates spanning 0.23 m to 20.12 m, and solving that unconditioned
    is where a geometrically fine configuration turns into numerical mush.
    """
    if len(image_pts) < 4 or len(image_pts) != len(world_pts):
        return None

    t_img = _normalisation(image_pts)
    t_world = _normalisation(world_pts)
    if t_img is None or t_world is None:
        return None

    src = _apply(t_img, image_pts)
    dst = _apply(t_world, world_pts)

    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.array(rows, dtype=float))
    h = vt[-1].reshape(3, 3)

    matrix = np.linalg.inv(t_world) @ h @ t_img
    if abs(matrix[2, 2]) < 1e-12:
        return None
    return matrix / matrix[2, 2]


def _normalisation(points: np.ndarray) -> Optional[np.ndarray]:
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_distance = float(np.sqrt((shifted ** 2).sum(axis=1)).mean())
    if mean_distance < 1e-12:
        return None  # every point identical: no configuration to normalise
    scale = np.sqrt(2.0) / mean_distance
    return np.array(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]]
    )


def _apply(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    mapped = (transform @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def _stump_base_points(box: BoundingBox) -> tuple[tuple[float, float], tuple[float, float]]:
    """The two ground-level corners of a stumps box: (left base, right base).

    The *base* specifically. The top of the stumps is 0.711 m above the pitch plane, so
    a homography fitted to top corners would be fitting points that aren't on the plane
    it claims to map -- the single easiest way to get a calibration that solves cleanly
    and is wrong everywhere.
    """
    return (box.x1, box.y2), (box.x2, box.y2)


def calibrate_from_stumps(
    strikers_end: Optional[BoundingBox],
    bowlers_end: Optional[BoundingBox],
    max_error_px: float = MAX_REPROJECTION_ERROR_PX,
) -> Optional[PitchCalibration]:
    """A `PitchCalibration` from the two stump sets, or `None` if one can't be had.

    `strikers_end` and `bowlers_end` are the detected stumps bounding boxes at each end
    of the frame, which is why the camera placement matters: both sets have to be
    visible, which is the constraint the fixed elevated-tripod setups in the wild are
    built around.

    Which box is which end is the caller's to establish and is **not** inferred here.
    The obvious heuristic -- the larger box is nearer -- is right often enough to be
    dangerous and wrong whenever the camera is behind the bowler's arm at an angle,
    and getting it backwards produces a confident calibration with length measured from
    the wrong end. A wrong number that looks right is the failure mode this whole
    codebase is organised against, so the decision stays with the caller that has the
    camera geometry to make it.

    Returns `None` for: a missing box at either end, a degenerate (zero-width or
    zero-height) box, or a fit whose reprojection error exceeds `max_error_px`.
    """
    if strikers_end is None or bowlers_end is None:
        return None
    if min(strikers_end.area(), bowlers_end.area()) <= 0:
        return None

    near_left, near_right = _stump_base_points(strikers_end)
    far_left, far_right = _stump_base_points(bowlers_end)
    image_pts = np.array([near_left, near_right, far_left, far_right], dtype=float)

    # Four *distinct* image points are needed for a homography. Coincident ones (the two
    # ends detected at the same place, typically because one end wasn't really found)
    # leave the system rank-deficient, and solving it anyway produces a matrix full of
    # infinities that only fails later, in a division, with a warning. Rejecting the
    # configuration here says what is actually wrong.
    if len({(round(x, 6), round(y, 6)) for x, y in image_pts}) < 4:
        return None

    # Same order as `image_pts`. The near pair sits on y=0 (the striker's stumps define
    # the origin) and the far pair on y=PITCH_LENGTH_M; x is +/- the outer-stump offset.
    world_pts = np.array(
        [
            [-STUMP_HALF_WIDTH_M, 0.0],
            [STUMP_HALF_WIDTH_M, 0.0],
            [-STUMP_HALF_WIDTH_M, PITCH_LENGTH_M],
            [STUMP_HALF_WIDTH_M, PITCH_LENGTH_M],
        ],
        dtype=float,
    )

    matrix = _homography(image_pts, world_pts)
    if matrix is None or not np.all(np.isfinite(matrix)):
        return None

    reprojected = _apply(matrix, image_pts)
    # Residual is measured back in pixels, not metres: a metre of error at the far
    # stumps and a metre at the near stumps are wildly different amounts of evidence,
    # and the pixel figure is the one that says whether the *fit* is sound.
    back = _homography(world_pts, image_pts)
    if back is None:
        return None
    rms = float(np.sqrt((( _apply(back, reprojected) - image_pts) ** 2).sum(axis=1).mean()))
    if not np.isfinite(rms) or rms > max_error_px:
        return None

    return PitchCalibration(
        matrix=tuple(tuple(float(v) for v in row) for row in matrix),
        rms_error_px=round(rms, 4),
        image_points=tuple((float(x), float(y)) for x, y in image_pts),
    )


def classify_length(
    length_m: Optional[float], bounced: bool = True
) -> PitchLength:
    """The `PitchLength` band `length_m` falls in, measured from the striker's stumps.

    `bounced=False` reports `FULL_TOSS` regardless of `length_m`, because a full toss is
    defined by reaching the batter without touching the ground -- it is a property of
    the ball's trajectory, not a position on the pitch. A caller that knows the ball
    never made ground contact knows that more reliably than any projected coordinate
    would tell it.

    `UNKNOWN` for a missing length, and for one outside the pitch at either end (behind
    the striker's stumps, or past the bowler's). Those are physically impossible rather
    than merely extreme, so they indicate a bad calibration or a mis-tracked bounce, and
    the honest report is that the length isn't known -- not the nearest plausible band.
    """
    if not bounced:
        return PitchLength.FULL_TOSS
    if length_m is None:
        return PitchLength.UNKNOWN
    for band, low, high in LENGTH_BANDS:
        if low <= length_m < high:
            return band
    return PitchLength.UNKNOWN


def in_stump_corridor(line_m: Optional[float], tolerance_m: float = 0.0) -> Optional[bool]:
    """Whether a line is within the stumps' width, or `None` if the line isn't known.

    `tolerance_m` widens the corridor; a bowler aiming "at the stumps" is not expected
    to hit a 22.86 cm target every ball, so a caller measuring accuracy will usually
    want some. The default of zero is the literal stump block, so nothing is implied
    about what a reasonable tolerance is -- that is a coaching judgement and this
    function does not have an opinion on it.
    """
    if line_m is None:
        return None
    return abs(line_m) <= STUMP_HALF_WIDTH_M + tolerance_m
