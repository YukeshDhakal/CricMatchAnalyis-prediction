"""Classical temporal-motion scoring, used to tell a moving ball from a resting one.

**Why a classical signal earns a place next to a trained detector.** Real-footage probing
found the ball checkpoint detecting genuine cricket balls *lying still on the outfield* at
0.44-0.55 confidence for 83 consecutive frames, while the actually-delivered ball in a
different clip scored 0.38-0.65. The detector is not wrong in either case -- both are
cricket balls -- but only one of them is the delivery. A per-frame CNN cannot make that
distinction even in principle: it sees one frame, and "moving" is not a property of one
frame. Frame differencing is the cheapest signal that is *about* the thing the detector
structurally cannot see, which is the whole reason it is here rather than a second
detector or a bigger one.

**Three-frame differencing, not two.** A plain `|f[i] - f[i-1]|` lights up both where the
object now is and where it just was, so a fast ball produces two blobs and a box centred on
either scores high. Intersecting the difference against the *next* frame as well keeps only
what is moving at frame `i`, which is what a detection at frame `i` should be scored
against.

**What this deliberately does not do.** It does not propose candidate boxes. Probing the
native-4K nets footage showed the bowler's arm, the batter's bat and net shimmer dominating
the motion mask in exactly the corridor a delivery travels down, so a motion-first proposer
on this footage yields mostly limbs. Motion is used here as *evidence about a candidate the
detector already produced*, which is the role the measurements actually support. If footage
ever arrives where the ball is separable by motion alone, a proposer can be added -- but
claiming one works now would be the same mistake as trusting a held-out mAP.

**Camera motion is a stated limit, not a handled case.** Everything here assumes the camera
is effectively fixed for the delivery, which is what the tripod setups this project targets
provide. On a panning or zooming shot every static edge in the frame registers as motion
and these scores become meaningless -- `global_motion_ratio` exists so a caller can detect
that situation and refuse, rather than silently scoring noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..contracts import BoundingBox, Detection

__all__ = [
    "MOTION_DIFF_THRESHOLD",
    "STATIONARY_ENERGY_CEILING",
    "MotionEvidence",
    "motion_energy",
    "score_detections",
    "global_motion_ratio",
]

# Per-pixel intensity change counted as "this pixel changed". Low enough to catch a
# dim/blurred ball against grass, high enough to ignore sensor noise and compression
# mosquito noise on a flat background.
MOTION_DIFF_THRESHOLD = 12

# Below this fraction of changed pixels in its own box, a detection is treated as resting.
# Calibrated against the measured cases rather than guessed: the stationary balls in the
# 4K nets clip sit near zero, and the moving ball in `real_bowling_clip_full.mp4` is far
# above it. It is a wide margin on purpose -- it is separating "did not move at all" from
# "crossed the frame", not making a fine distinction.
STATIONARY_ENERGY_CEILING = 0.05


@dataclass(frozen=True)
class MotionEvidence:
    """How much the pixels under one detection were actually moving.

    `energy` is the fraction of pixels inside the detection's box that changed in both the
    backward and forward difference, in [0, 1]. `is_moving` applies
    `STATIONARY_ENERGY_CEILING` to it.

    `energy` is `0.0` and `is_moving` is `False` for a detection on the first or last
    frame, where no three-frame window exists. That is a deliberate refusal rather than a
    one-sided fallback: a two-frame difference at the boundary measures something
    different from what every other frame is measured with, and silently mixing the two
    would make the scores incomparable across the clip.
    """

    frame_index: int
    energy: float
    is_moving: bool


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.int16)
    return frame.mean(axis=2).astype(np.int16)


def _crop(array: np.ndarray, box: BoundingBox) -> Optional[np.ndarray]:
    height, width = array.shape[:2]
    x1 = max(0, int(box.x1))
    y1 = max(0, int(box.y1))
    x2 = min(width, int(round(box.x2)))
    y2 = min(height, int(round(box.y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return array[y1:y2, x1:x2]


def motion_energy(
    frames: Sequence[np.ndarray],
    frame_index: int,
    box: BoundingBox,
    threshold: int = MOTION_DIFF_THRESHOLD,
) -> float:
    """Fraction of pixels under `box` that are moving at `frame_index`, in [0, 1].

    Returns `0.0` at the clip boundaries, where the three-frame window does not exist, and
    for a box that falls entirely outside the frame.
    """
    if frame_index <= 0 or frame_index >= len(frames) - 1:
        return 0.0
    prev_c = _crop(_to_gray(frames[frame_index - 1]), box)
    curr_c = _crop(_to_gray(frames[frame_index]), box)
    next_c = _crop(_to_gray(frames[frame_index + 1]), box)
    if prev_c is None or curr_c is None or next_c is None:
        return 0.0
    if prev_c.shape != curr_c.shape or next_c.shape != curr_c.shape:
        return 0.0
    backward = np.abs(curr_c - prev_c) > threshold
    forward = np.abs(curr_c - next_c) > threshold
    moving = np.logical_and(backward, forward)
    return float(moving.mean()) if moving.size else 0.0


def score_detections(
    frames: Sequence[np.ndarray],
    detections: Sequence[Detection],
    threshold: int = MOTION_DIFF_THRESHOLD,
    ceiling: float = STATIONARY_ENERGY_CEILING,
) -> list[MotionEvidence]:
    """`MotionEvidence` for each detection, in the order given."""
    return [
        MotionEvidence(
            frame_index=det.frame_index,
            energy=(energy := motion_energy(frames, det.frame_index, det.box, threshold)),
            is_moving=energy > ceiling,
        )
        for det in detections
    ]


def global_motion_ratio(
    frames: Sequence[np.ndarray], frame_index: int, threshold: int = MOTION_DIFF_THRESHOLD
) -> float:
    """Fraction of the *whole frame* that is moving -- a camera-movement smell test.

    A fixed camera watching a delivery has a few percent of its pixels moving. A pan, a
    zoom, or a broadcast cut moves nearly all of them. A caller seeing a high value here
    should distrust every per-detection energy in that frame rather than reinterpret it,
    because the assumption those scores rest on has failed.
    """
    if frame_index <= 0 or frame_index >= len(frames) - 1:
        return 0.0
    prev_g = _to_gray(frames[frame_index - 1])
    curr_g = _to_gray(frames[frame_index])
    next_g = _to_gray(frames[frame_index + 1])
    if prev_g.shape != curr_g.shape or next_g.shape != curr_g.shape:
        return 0.0
    moving = np.logical_and(
        np.abs(curr_g - prev_g) > threshold, np.abs(curr_g - next_g) > threshold
    )
    return float(moving.mean())
