"""Physics-constrained trajectory fitting over noisy, sparse ball detections.

**What problem this actually solves, measured rather than assumed.** Real-footage probing
of `weights/ball_stumps_n.pt` over the four clips in `data/uploads/videos/` found that the
checkpoint's ball detections fall into three groups, and that *confidence does not
separate them*:

* genuine cricket balls lying **stationary on the ground** -- in the 4K nets clip, two of
  them, held at 0.44-0.55 confidence for 83 consecutive frames. These are correct
  detections of the wrong object.
* genuine false positives on light-coloured background clutter -- a bag by the fence in
  `real_bowling_clip_full.mp4`, peaking at 0.34.
* the **actual ball**, which in that same clip is detected across frames 80-86 at
  0.38-0.65 -- i.e. *lower* peak confidence than the stationary ball in the other clip.

No confidence threshold orders those three groups correctly, because "is this a cricket
ball" and "is this *the delivered* cricket ball" are different questions and a per-frame
detector only answers the first. What separates them is **motion across frames**: the
stationary balls drift under 2 px over 80 frames, the real ball moves roughly 50 px per
frame. That is a property of a *set* of detections, never of one, which is why it belongs
here and not in the detector.

**Why this replaces the pooled-spread gate rather than sitting beside it.** The previous
guard (`events.heuristic_segmenter._looks_like_real_motion_detections`) pooled every BALL
detection across track_ids and asked whether the pooled set's maximum spread cleared a
multiple of the mean box size. That question has a false-positive answer whenever a static
object and a real ball are *both* present: the spread between the two clusters is large,
so the gate passes, and the segmenter then takes `release_frame` from the first detection
in frame order -- which is the static object. Verified on real detections, not argued:
at a 0.15 confidence threshold on `real_bowling_clip_full.mp4` the gate passes and reports
`release_frame=1`, the bag. The shipped 0.40 default escapes this only because that bag
peaks at 0.34, a margin of 0.06 that nothing enforces. See MISTAKES.md.

A RANSAC fit asks a different and stricter question: *is there a subset of these detections
that lies on one physically plausible path?* A static cluster and a moving ball cannot both
be inliers to the same trajectory, so the answer selects one of them instead of blessing
the union.

**The hard-constraint ordering is the load-bearing detail.** A static cluster is perfectly
consistent with a degenerate zero-velocity "trajectory", and in the 4K clip it has 83
members against the real ball's 7. Scored on inlier count alone, the static cluster wins
every time. So the minimum-displacement requirement is applied as a **rejection filter on
each hypothesis before it is ever scored**, not as a tiebreak afterwards. Hypotheses that
do not move are not weak candidates; they are not candidates.

**Honesty about the model.** The fit is a quadratic in image `y` and a linear in image `x`
against frame index. That is *not* a claim that a projected trajectory is exactly
parabolic -- it is not, except for an affine (orthographic) camera. Under a real
perspective camera the image path is a rational function, and the approximation degrades
badly for a ball travelling toward or away from the lens. The quadratic is used as a
**smoothing and outlier-rejection prior**, whose job is to reject clutter, interpolate
frames the detector missed, and locate the bounce -- not to reconstruct 3D motion, which
single-camera footage cannot support at all (see the README). `TrajectoryFit.rms_px` is
reported so a caller can see when the approximation is failing, and confidence degrades
with it.

**The one place the geometry is exact** is the bounce. A ball in flight is above the pitch
plane, so unprojecting a mid-flight detection through the ground-plane homography is
meaningless. At ground contact it is *on* the plane by definition. That is why this module
locates a bounce point and hands only that single point to `calibration`, rather than
projecting the whole track.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence

from ..contracts import BoundingBox, Detection

__all__ = [
    "MIN_INLIERS",
    "MIN_DISPLACEMENT_BOX_MULTIPLE",
    "TOLERANCE_BOX_MULTIPLE",
    "MIN_TOLERANCE_PX",
    "TrajectoryFit",
    "fit_ball_trajectory",
]

# A trajectory needs enough support to be more than a coincidence. Three points define the
# quadratic exactly, so three inliers is not evidence of anything -- any three detections
# fit some parabola perfectly. Five is the smallest number that leaves real residual
# degrees of freedom while still being reachable on the sparse tracks this detector
# produces (the one real ball track in the available footage has seven points).
MIN_INLIERS = 5

# Carried over unchanged from the gate this module replaces, and for the same measured
# reason: a real ball crosses a large fraction of the frame during a delivery, a static
# object re-detected frame after frame does not move beyond inference jitter. Expressed as
# a multiple of the object's own box size so it is resolution-independent -- the same
# footage at 960x540 and at 4K must reach the same verdict.
MIN_DISPLACEMENT_BOX_MULTIPLE = 3.0

# Inlier tolerance, also scaled to the ball's own apparent size rather than fixed in
# pixels, for the same resolution-independence reason.
TOLERANCE_BOX_MULTIPLE = 1.5

# Two constraints that exist because the first version of this module failed without them,
# on real detections rather than in theory. Both encode the same physical fact -- a
# delivery is *one continuous flight, and the ball never stops* -- and each kills a
# distinct observed failure:
#
# * MAX_FRAME_GAP: without it, the fit happily linked a static cluster on frames 0-83 to a
#   falling object on frames 527-571 of the 4K clip, because a gentle enough quadratic
#   passes through both and 65 inliers beat the real ball's 7. Detections hundreds of
#   frames apart with nothing between them are not one flight, whatever curve joins them.
# * MIN_STEP_SPEED_BOX_MULTIPLE: total displacement is the wrong question, because a
#   hypothesis spanning two distant clusters has a large *total* displacement while being
#   motionless for almost all of it. The median frame-to-frame step is the right one: a
#   real ball moves every frame, a resting ball moves on none. This is what finally
#   separated the stationary outfield balls from the delivery, and it is checked on the
#   median rather than the minimum so one noisy localisation cannot veto a true track.
#
# The step-speed bound is set where it separates *moving from not moving*, and nowhere
# near where it would separate one moving object from another. Measured: resting balls
# drift about 0.001 of their box per frame, so 0.10 sits a hundredfold above the noise
# floor while still admitting a slow track. It is deliberately not tuned upward to make a
# particular clip come out right -- a threshold that happens to exclude the wrong answer
# by a hair is the same 0.06-of-luck trap documented in MISTAKES.md, and tuning one here
# would hide a limitation rather than fix it. Which moving object is *the delivery* is a
# question this module genuinely cannot answer; see `fit_ball_trajectory`.
MAX_FRAME_GAP = 8
MIN_STEP_SPEED_BOX_MULTIPLE = 0.10

# ...but with a floor, because a heavily motion-blurred ball can be detected with a box a
# few pixels across, and a tolerance of a few pixels would reject the true track on
# ordinary localisation noise.
MIN_TOLERANCE_PX = 4.0

# Enumerating every triple is exact and order-independent, which matters for tests and for
# reproducible output. It is only affordable while the candidate count is small; beyond
# this the sampler switches to a *seeded* RNG, so the result stays deterministic either
# way. Determinism is not a nicety here: a pipeline that returns a different line and
# length for the same clip on two runs cannot be audited.
_MAX_CANDIDATES_FOR_EXHAUSTIVE = 45
_RANDOM_ITERATIONS = 4000
_RANDOM_SEED = 0


@dataclass(frozen=True)
class TrajectoryFit:
    """One physically-plausible ball path selected from a noisy candidate set.

    `inlier_frames` are the frames whose detections the fit accepted; `coeffs_y` and
    `coeffs_x` describe the path in **image pixels against frame index**, not in metres
    and not in world coordinates. Read the module docstring before treating them as
    physics: they are a smoothing prior over a perspective projection, not a trajectory
    reconstruction.

    `bounce_frame` is `None` when no ground contact was detected within the tracked span.
    That is the common and honest case for a short or partial track, and callers must not
    substitute the lowest observed point for it -- the lowest *observed* point is an
    artefact of which frames the detector happened to fire on, while a bounce is a real
    event that the two-arc split either found or did not.

    `confidence` is in [0, 1] and is deliberately *not* a probability. It is a monotone
    summary of how much support the fit has (inlier count, how much of the span was
    covered, and residual relative to tolerance), meant for ordering and for the
    degrade-to-unknown thresholds callers apply -- not for arithmetic.
    """

    inlier_frames: tuple[int, ...]
    coeffs_x: tuple[float, float]  # x(t) = coeffs_x[0] * t + coeffs_x[1]
    coeffs_y: tuple[float, float, float]  # y(t) = c0*t^2 + c1*t + c2
    rms_px: float
    tolerance_px: float
    displacement_px: float
    median_box_px: float
    confidence: float
    bounce_frame: Optional[float] = None
    bounce_point_px: Optional[tuple[float, float]] = None

    @property
    def first_frame(self) -> int:
        return self.inlier_frames[0]

    @property
    def last_frame(self) -> int:
        return self.inlier_frames[-1]

    def position_at(self, frame_index: float) -> tuple[float, float]:
        """The fitted image position at `frame_index`, including frames with no detection.

        This is the interpolation that directly addresses the detector's recall gap: a
        frame the detector missed entirely still has a position, derived from the frames
        around it rather than from nothing. Extrapolating far beyond `last_frame` is the
        caller's risk -- the quadratic is only constrained where there were inliers.
        """
        t = float(frame_index)
        a, b, c = self.coeffs_y
        m, k = self.coeffs_x
        return (m * t + k, a * t * t + b * t + c)


def _center(box: BoundingBox) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def _box_size(box: BoundingBox) -> float:
    return max(box.x2 - box.x1, box.y2 - box.y1)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def _fit_quadratic(points: Sequence[tuple[float, float]]) -> Optional[tuple[float, float, float]]:
    """Least-squares `y = a t^2 + b t + c` via the normal equations.

    Written out rather than pulled from numpy.polyfit so that a degenerate configuration
    returns `None` instead of emitting a RankWarning and handing back coefficients that
    are numerically meaningless -- the same "refuse rather than guess" rule the rest of
    this package follows.
    """
    n = len(points)
    if n < 3:
        return None
    s0 = float(n)
    s1 = s2 = s3 = s4 = 0.0
    t0 = t1 = t2 = 0.0
    for t, y in points:
        t_2 = t * t
        s1 += t
        s2 += t_2
        s3 += t_2 * t
        s4 += t_2 * t_2
        t0 += y
        t1 += y * t
        t2 += y * t_2
    # Solve the 3x3 normal-equation system by Cramer's rule.
    m = [[s4, s3, s2], [s3, s2, s1], [s2, s1, s0]]
    rhs = [t2, t1, t0]
    det = _det3(m)
    if abs(det) < 1e-12:
        return None
    out = []
    for col in range(3):
        swapped = [row[:] for row in m]
        for row in range(3):
            swapped[row][col] = rhs[row]
        out.append(_det3(swapped) / det)
    return (out[0], out[1], out[2])


def _det3(m: Sequence[Sequence[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _fit_linear(points: Sequence[tuple[float, float]]) -> Optional[tuple[float, float]]:
    n = len(points)
    if n < 2:
        return None
    s_t = sum(t for t, _ in points)
    s_y = sum(y for _, y in points)
    s_tt = sum(t * t for t, _ in points)
    s_ty = sum(t * y for t, y in points)
    denom = n * s_tt - s_t * s_t
    if abs(denom) < 1e-12:
        return None
    slope = (n * s_ty - s_t * s_y) / denom
    return (slope, (s_y - slope * s_t) / n)


def _candidate_triples(count: int) -> list[tuple[int, int, int]]:
    if count <= _MAX_CANDIDATES_FOR_EXHAUSTIVE:
        return [
            (i, j, k)
            for i in range(count)
            for j in range(i + 1, count)
            for k in range(j + 1, count)
        ]
    rng = random.Random(_RANDOM_SEED)
    seen: set[tuple[int, int, int]] = set()
    for _ in range(_RANDOM_ITERATIONS):
        triple = tuple(sorted(rng.sample(range(count), 3)))
        seen.add(triple)  # type: ignore[arg-type]
    return sorted(seen)  # type: ignore[return-value]


def fit_ball_trajectory(
    detections: Sequence[Detection],
    min_inliers: int = MIN_INLIERS,
    min_displacement_box_multiple: float = MIN_DISPLACEMENT_BOX_MULTIPLE,
) -> Optional[TrajectoryFit]:
    """The best physically-plausible moving path through `detections`, or `None`.

    `detections` should be every BALL detection in the clip, pooled across track_ids --
    tracking fragments a fast ball into several short-lived ids, so a per-track fit would
    discard the very track it is looking for. Pooling is safe because the fit then
    *selects* a consistent subset rather than trusting the union, which is precisely the
    property the pooled-spread gate it replaces lacked.

    Returns `None` when no subset of at least `min_inliers` detections both fits one
    quadratic path and actually moves. `None` means "no trustworthy ball track in this
    clip", which for the currently available footage is frequently the correct answer and
    must be reported rather than papered over.

    **The limitation a caller must not forget**: this finds *a coherent moving object*, and
    that is not the same thing as *the delivered ball*. Measured on the 4K nets clip, the
    best-scoring track is a white object falling outside the net, fitted cleanly at
    ~2 px RMS over 18 frames -- a correct trajectory of the wrong object. Nothing in the
    detections themselves distinguishes it from a delivery. Making that distinction needs
    to know where the pitch is, i.e. a `calibration.PitchCalibration`, which in turn needs
    both stump sets visible. Callers that have no calibration should treat a fit as
    "something moved along a plausible arc", not as "this is the ball", and the
    pitch-geometry path in `events.heuristic_segmenter` does exactly that.
    """
    candidates = sorted(detections, key=lambda d: (d.frame_index, -d.confidence))
    if len({d.frame_index for d in candidates}) < min_inliers:
        return None

    points = [(float(d.frame_index),) + _center(d.box) for d in candidates]
    sizes = [_box_size(d.box) for d in candidates]
    median_box = _median(sizes)
    if median_box <= 0:
        return None
    tolerance = max(MIN_TOLERANCE_PX, TOLERANCE_BOX_MULTIPLE * median_box)

    best: Optional[TrajectoryFit] = None
    best_indices: tuple[int, ...] = ()
    best_key: tuple[int, float] = (0, -math.inf)

    for i, j, k in _candidate_triples(len(candidates)):
        t_i, t_j, t_k = points[i][0], points[j][0], points[k][0]
        if len({t_i, t_j, t_k}) < 3:
            continue  # same frame twice: no temporal baseline to fit against
        sample = [points[i], points[j], points[k]]
        coeffs_y = _fit_quadratic([(t, y) for t, _, y in sample])
        coeffs_x = _fit_linear([(t, x) for t, x, _ in sample])
        if coeffs_y is None or coeffs_x is None:
            continue

        inliers = _longest_contiguous_run(points, _select_inliers(points, coeffs_x, coeffs_y, tolerance))
        if len(inliers) < min_inliers:
            continue

        refined = _refit(points, inliers)
        if refined is None:
            continue
        coeffs_x, coeffs_y = refined
        # Re-derive the tolerance from *this hypothesis's own* inliers before the second
        # pass. The bootstrap tolerance has to use the global median box, but that median
        # is contaminated whenever the candidate set mixes objects of different apparent
        # size -- in `real_bowling_clip_full.mp4` the bag's 60 px box drags the median up
        # and widens the corridor enough to admit the bag into the real ball's track. A
        # ball-sized track should be judged against a ball-sized tolerance.
        local_box = _median([sizes[idx] for idx in inliers])
        local_tolerance = (
            max(MIN_TOLERANCE_PX, TOLERANCE_BOX_MULTIPLE * local_box) if local_box > 0 else tolerance
        )
        inliers = _longest_contiguous_run(
            points, _select_inliers(points, coeffs_x, coeffs_y, local_tolerance)
        )
        if len(inliers) < min_inliers:
            continue

        displacement = _displacement(points, inliers)
        # The hard constraints, applied before scoring -- see the module docstring. A
        # stationary cluster is not a weak hypothesis to be out-voted later; at this
        # point it stops being a hypothesis at all.
        scale_box = local_box if local_box > 0 else median_box
        if displacement < min_displacement_box_multiple * scale_box:
            continue
        if _median_step_speed(points, inliers) < MIN_STEP_SPEED_BOX_MULTIPLE * scale_box:
            continue

        rms = _rms(points, inliers, coeffs_x, coeffs_y)
        key = (len(inliers), -rms)
        if key <= best_key and best is not None:
            continue
        best_key = key
        best_indices = tuple(inliers)
        frames = tuple(int(points[idx][0]) for idx in inliers)
        best = TrajectoryFit(
            inlier_frames=frames,
            coeffs_x=coeffs_x,
            coeffs_y=coeffs_y,
            rms_px=round(rms, 3),
            tolerance_px=round(local_tolerance, 3),
            displacement_px=round(displacement, 2),
            median_box_px=round(scale_box, 2),
            confidence=_confidence(len(inliers), frames, rms, local_tolerance),
        )

    if best is None:
        return None
    bounce_frame, bounce_point = _find_bounce(points, best, best_indices)
    if bounce_frame is None:
        return best
    return TrajectoryFit(
        inlier_frames=best.inlier_frames,
        coeffs_x=best.coeffs_x,
        coeffs_y=best.coeffs_y,
        rms_px=best.rms_px,
        tolerance_px=best.tolerance_px,
        displacement_px=best.displacement_px,
        median_box_px=best.median_box_px,
        confidence=best.confidence,
        bounce_frame=bounce_frame,
        bounce_point_px=bounce_point,
    )


def _select_inliers(
    points: Sequence[tuple[float, float, float]],
    coeffs_x: tuple[float, float],
    coeffs_y: tuple[float, float, float],
    tolerance: float,
) -> tuple[int, ...]:
    """Indices of the points within `tolerance` of the path, at most one per frame.

    The one-per-frame rule matters: a ball occupies one position at a time, so admitting
    two detections from the same frame would let a static cluster and a real ball both be
    "on" the same trajectory and inflate the score of a fit that explains neither.
    """
    a, b, c = coeffs_y
    m, k = coeffs_x
    per_frame: dict[float, tuple[float, int]] = {}
    for idx, (t, x, y) in enumerate(points):
        dx = x - (m * t + k)
        dy = y - (a * t * t + b * t + c)
        error = math.hypot(dx, dy)
        if error > tolerance:
            continue
        current = per_frame.get(t)
        if current is None or error < current[0]:
            per_frame[t] = (error, idx)
    return tuple(idx for _, idx in sorted((points[i][0], i) for _, i in per_frame.values()))


def _refit(
    points: Sequence[tuple[float, float, float]], inliers: Sequence[int]
) -> Optional[tuple[tuple[float, float], tuple[float, float, float]]]:
    coeffs_y = _fit_quadratic([(points[i][0], points[i][2]) for i in inliers])
    coeffs_x = _fit_linear([(points[i][0], points[i][1]) for i in inliers])
    if coeffs_y is None or coeffs_x is None:
        return None
    return coeffs_x, coeffs_y


def _longest_contiguous_run(
    points: Sequence[tuple[float, float, float]],
    inliers: Sequence[int],
    max_gap: int = MAX_FRAME_GAP,
) -> tuple[int, ...]:
    """The longest stretch of `inliers` with no frame gap wider than `max_gap`.

    Gaps up to `max_gap` are kept deliberately -- interpolating across frames the detector
    missed is the point of fitting a trajectory at all. What is rejected is a gap so wide
    that the two sides cannot be the same flight, which is how a resting ball early in a
    clip was previously joined to an unrelated moving object hundreds of frames later.
    """
    if not inliers:
        return ()
    best: list[int] = []
    current: list[int] = [inliers[0]]
    for prev, idx in zip(inliers, inliers[1:]):
        if points[idx][0] - points[prev][0] <= max_gap:
            current.append(idx)
        else:
            if len(current) > len(best):
                best = current
            current = [idx]
    return tuple(current if len(current) > len(best) else best)


def _median_step_speed(
    points: Sequence[tuple[float, float, float]], inliers: Sequence[int]
) -> float:
    """Median pixels moved per frame between consecutive inliers.

    Normalised by the frame gap so a track with missed frames is not credited with extra
    speed for the frames it skipped.
    """
    steps = []
    for prev, idx in zip(inliers, inliers[1:]):
        dt = points[idx][0] - points[prev][0]
        if dt <= 0:
            continue
        steps.append(
            math.hypot(points[idx][1] - points[prev][1], points[idx][2] - points[prev][2]) / dt
        )
    return _median(steps) if steps else 0.0


def _displacement(points: Sequence[tuple[float, float, float]], inliers: Sequence[int]) -> float:
    xs = [points[i][1] for i in inliers]
    ys = [points[i][2] for i in inliers]
    return max(
        math.hypot(xs[a] - xs[b], ys[a] - ys[b])
        for a in range(len(xs))
        for b in range(a + 1, len(xs))
    )


def _rms(
    points: Sequence[tuple[float, float, float]],
    inliers: Sequence[int],
    coeffs_x: tuple[float, float],
    coeffs_y: tuple[float, float, float],
) -> float:
    a, b, c = coeffs_y
    m, k = coeffs_x
    total = 0.0
    for i in inliers:
        t, x, y = points[i]
        total += (x - (m * t + k)) ** 2 + (y - (a * t * t + b * t + c)) ** 2
    return math.sqrt(total / len(inliers))


def _confidence(inlier_count: int, frames: Sequence[int], rms: float, tolerance: float) -> float:
    """A bounded, monotone support summary -- explicitly not a probability.

    Three things make a track more trustworthy and each is capped so none can dominate:
    how many frames support it, how densely it covers the span it claims (a fit spanning
    60 frames on 5 detections is far weaker than one spanning 7 on 7), and how small the
    residual is relative to the tolerance that admitted it.
    """
    span = max(1, frames[-1] - frames[0] + 1)
    support = min(1.0, inlier_count / 10.0)
    density = min(1.0, inlier_count / span)
    tightness = max(0.0, 1.0 - (rms / tolerance)) if tolerance > 0 else 0.0
    # Tightness *multiplies* rather than contributing a weighted share. A contaminated fit
    # -- one that reached its inlier count by absorbing a nearby unrelated object -- scores
    # well on support and density precisely because it has many points, so an additive
    # blend let it report high confidence while fitting badly (measured: 0.83 on a track
    # that had swallowed the bag). Making residual a gate on the whole score means no
    # amount of support can compensate for a path the points do not actually lie on.
    return round(min(1.0, (0.6 * support + 0.4 * density) * tightness), 3)


def _find_bounce(
    points: Sequence[tuple[float, float, float]],
    fit: TrajectoryFit,
    inlier_indices: Sequence[int],
) -> tuple[Optional[float], Optional[tuple[float, float]]]:
    """Locate ground contact by splitting the track into two arcs, or report none.

    A bounce is a *cusp* in the image path -- vertical velocity reverses discontinuously --
    and a single quadratic cannot represent one. So each interior frame is tried as a
    split, two quadratics are fitted either side, and the split with the lowest combined
    residual wins, subject to the descent/ascent sign actually reversing.

    Returning `None` is a real answer, not a failure: a track that is still descending when
    the clip ends has not bounced within what was observed, and inventing a bounce point
    past the last inlier would be extrapolating exactly the quantity the caller most wants
    to trust. The requirement that both arcs have three points is what stops a two-frame
    tail from being promoted into an "ascent".
    """
    inliers = list(fit.inlier_frames)
    if len(inliers) < 6:
        return None, None
    # Keyed off the inlier *indices*, not off frame membership. Selecting by frame number
    # re-reads `points` and lets a rejected detection that happens to share a frame with an
    # inlier overwrite it -- which made the bounce frame flip between 82.5, 83.5 and None
    # across confidence thresholds that produced byte-identical inlier sets. An unstable
    # bounce point silently becomes an unstable length and line downstream.
    by_frame = {int(points[idx][0]): (points[idx][1], points[idx][2]) for idx in inlier_indices}
    best_split = None
    best_residual = math.inf
    for split in range(3, len(inliers) - 2):
        before = [(float(f), by_frame[f][1]) for f in inliers[:split]]
        after = [(float(f), by_frame[f][1]) for f in inliers[split:]]
        if len(before) < 3 or len(after) < 3:
            continue
        cb = _fit_quadratic(before)
        ca = _fit_quadratic(after)
        if cb is None or ca is None:
            continue
        # Image y grows downward: descent is dy/dt > 0, ascent after the bounce is < 0.
        # Requiring the reversal is what distinguishes a bounce from ordinary curvature.
        if _slope_at(cb, before[-1][0]) <= 0 or _slope_at(ca, after[0][0]) >= 0:
            continue
        residual = _arc_residual(before, cb) + _arc_residual(after, ca)
        if residual < best_residual:
            best_residual = residual
            best_split = (split, cb, ca)
    if best_split is None:
        return None, None
    split, cb, ca = best_split
    frame = (inliers[split - 1] + inliers[split]) / 2.0
    x, _ = fit.position_at(frame)
    a, b, c = cb
    y = a * frame * frame + b * frame + c
    return frame, (round(x, 2), round(y, 2))


def _slope_at(coeffs: tuple[float, float, float], t: float) -> float:
    a, b, _ = coeffs
    return 2 * a * t + b


def _arc_residual(points: Sequence[tuple[float, float]], coeffs: tuple[float, float, float]) -> float:
    a, b, c = coeffs
    return sum((y - (a * t * t + b * t + c)) ** 2 for t, y in points)
