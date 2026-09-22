from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from ..contracts import (
    BoundingBox,
    DeliveryEvent,
    Detection,
    GeometryConfidence,
    ObjectClass,
    PoseFrame,
    ShotType,
    Track,
)
from ..geometry import pitch_geometry
from ..motion import STATIONARY_ENERGY_CEILING, global_motion_ratio, motion_energy
from ..trajectory import fit_ball_trajectory
from ..trajectory.fit import MIN_INLIERS
from .base import EventSegmenter

_SWING_WINDOW = 4  # frames after contact over which wrist displacement is measured

# Above this fraction of the whole frame changing, the camera is moving rather than the
# subjects in it, and per-detection motion energy stops meaning anything -- every static
# edge registers. A fixed camera watching a delivery sits far below this even with a bowler
# running through frame. See `motion.global_motion_ratio`.
_MAX_GLOBAL_MOTION = 0.35


class HeuristicEventSegmenter(EventSegmenter):
    """First-pass event segmentation from ball position and batter wrist movement.

    This is a rule-based v0, not a trained model: release/contact frames come from ball-track
    proximity to a player, and shot_type from a coarse wrist-swing heuristic. Expect it to be
    replaced by a trained classifier once labelled delivery clips exist (Part 03.3's model layer).
    """

    def segment(
        self,
        tracks: list[Track],
        poses: list[PoseFrame],
        frames: Optional[Sequence[np.ndarray]] = None,
    ) -> DeliveryEvent:
        # IOU-based tracking (and ByteTrack under fast motion) fragments a genuinely fast
        # ball into several short-lived track_ids, since a ball's box barely overlaps
        # frame to frame -- so the trajectory fit below runs over every BALL detection
        # pooled across tracks, not per track_id. Pooling used to be risky: the previous
        # motion gate accepted the *union* of a static false positive and a real ball,
        # then took the release frame from whichever came first (the static one). The fit
        # instead *selects* a mutually consistent subset, so pooling is now safe -- see
        # `trajectory.fit`'s docstring and MISTAKES.md.
        ball_detections = sorted(
            (d for t in tracks if t.obj_class == ObjectClass.BALL for d in t.detections),
            key=lambda d: d.frame_index,
        )
        all_detections = [d for t in tracks for d in t.detections]
        ball_candidates = _drop_resting_detections(ball_detections, frames)
        fit = fit_ball_trajectory(ball_candidates) if ball_candidates else None

        if fit is None:
            return DeliveryEvent(
                release_frame=None,
                contact_frame=None,
                shot_type=ShotType.UNKNOWN,
                notes=_no_track_note(ball_detections),
                pitch_notes="No ball path, so no pitch geometry.",
            )

        # Only the fit's own inliers become the ball track. Handing the raw pooled
        # detections downstream would reintroduce exactly what the fit just rejected.
        inlier_frames = set(fit.inlier_frames)
        ball_track = Track(
            track_id=-1,
            obj_class=ObjectClass.BALL,
            detections=[d for d in ball_detections if d.frame_index in inlier_frames],
        )

        geometry = pitch_geometry(
            all_detections,
            fit.bounce_point_px,
            fit.confidence,
            bounced=fit.bounce_frame is not None,
        )
        bounce_frame = int(round(fit.bounce_frame)) if fit.bounce_frame is not None else None

        release_frame = fit.first_frame
        contact_frame, batter_track_id = self._find_contact(ball_track, tracks)

        if contact_frame is None or batter_track_id is None:
            return DeliveryEvent(
                release_frame=release_frame,
                contact_frame=None,
                shot_type=ShotType.UNKNOWN,
                notes="Ball never came within range of a tracked player.",
                bounce_frame=bounce_frame,
                pitch_point=geometry.point,
                pitch_length=geometry.length,
                pitch_confidence=geometry.confidence,
                pitch_notes=geometry.notes,
                track_confidence=fit.confidence,
            )

        shot_type = self._classify_shot(poses, batter_track_id, contact_frame)
        return DeliveryEvent(
            release_frame=release_frame,
            contact_frame=contact_frame,
            shot_type=shot_type,
            bounce_frame=bounce_frame,
            pitch_point=geometry.point,
            pitch_length=geometry.length,
            pitch_confidence=geometry.confidence,
            pitch_notes=geometry.notes,
            track_confidence=fit.confidence,
        )

    def _find_contact(self, ball_track: Track, tracks: list[Track]) -> tuple[int | None, int | None]:
        """The player nearest the ball at its closest approach is assumed to be the batter."""
        player_tracks = [t for t in tracks if t.obj_class == ObjectClass.PLAYER]
        best_frame, best_track_id, best_distance = None, None, math.inf

        for ball_det in ball_track.detections:
            ball_center = _center(ball_det.box)
            for player in player_tracks:
                player_box = player.box_at(ball_det.frame_index)
                if player_box is None:
                    continue
                distance = _distance(ball_center, _center(player_box))
                if distance < best_distance:
                    best_frame, best_track_id, best_distance = ball_det.frame_index, player.track_id, distance

        return best_frame, best_track_id

    def _classify_shot(self, poses: list[PoseFrame], batter_track_id: int, contact_frame: int) -> ShotType:
        window = sorted(
            (
                p for p in poses
                if p.track_id == batter_track_id and contact_frame <= p.frame_index <= contact_frame + _SWING_WINDOW
            ),
            key=lambda p: p.frame_index,
        )
        if len(window) < 2:
            return ShotType.UNKNOWN

        start_wrist = window[0].get("right_wrist") or window[0].get("left_wrist")
        end_wrist = window[-1].get("right_wrist") or window[-1].get("left_wrist")
        if start_wrist is None or end_wrist is None:
            return ShotType.UNKNOWN

        dx = end_wrist.x - start_wrist.x
        dy = end_wrist.y - start_wrist.y
        magnitude = math.hypot(dx, dy)

        if magnitude < 15:
            return ShotType.DEFEND
        if abs(dy) > abs(dx) * 1.5 and dy < 0:
            return ShotType.PULL
        if abs(dx) > abs(dy) * 1.5:
            return ShotType.DRIVE
        if dy > 0 and abs(dx) > abs(dy):
            return ShotType.SWEEP
        return ShotType.CUT


def _center(box: BoundingBox) -> tuple[float, float]:
    return (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _drop_resting_detections(
    detections: list[Detection], frames: Optional[Sequence[np.ndarray]]
) -> list[Detection]:
    """Discard ball detections whose own pixels were not moving, when frames are available.

    This is the motion signal doing work the trajectory fit cannot do alone. The fit
    rejects a resting cluster because it fails the displacement and step-speed filters --
    but it has to *consider* it first, and on the 4K nets clip that cluster is 83 of the
    candidates. Removing them up front shrinks the hypothesis space before the search and,
    more importantly, removes them as potential inliers to some *other* hypothesis, which
    is how a resting ball got absorbed into a real ball's track at low thresholds.

    **Degrades to a no-op without frames**, deliberately. The trajectory fit already
    rejects resting clusters on its own, so a caller replaying stored detections gets the
    same verdict by a slower route rather than a worse one -- this is an optimisation and a
    safety margin, never the only thing standing between a resting ball and a wrong answer.

    Refuses to filter at all when the camera is moving: `global_motion_ratio` above
    `_MAX_GLOBAL_MOTION` means most of the frame is changing, so per-detection energy no
    longer distinguishes a moving object from a stationary one in a panning shot, and
    filtering on it would discard real detections. Doing nothing is the correct response to
    an assumption having failed.
    """
    if not frames or not detections:
        return detections

    kept: list[Detection] = []
    for det in detections:
        if global_motion_ratio(frames, det.frame_index) > _MAX_GLOBAL_MOTION:
            return detections  # camera is moving: this signal means nothing here
        energy = motion_energy(frames, det.frame_index, det.box)
        # Boundary frames report 0.0 energy because no three-frame window exists there.
        # Keeping them rather than dropping them avoids trimming the ends off a genuine
        # track for a reason that has nothing to do with the ball.
        at_boundary = det.frame_index <= 0 or det.frame_index >= len(frames) - 1
        if at_boundary or energy > STATIONARY_ENERGY_CEILING:
            kept.append(det)

    # If filtering would leave too little to verify a trajectory with, hand back the
    # original set and let the fit decide. Motion energy is only meaningful when the frames
    # carry real texture and real movement; on degenerate input (a blank or synthetic clip,
    # a near-static locked-off shot, heavy compression flattening a small fast object) it
    # returns near-zero for everything, and a filter that trusts it then deletes the very
    # track it was meant to protect. The trajectory fit rejects resting clusters on its own
    # anyway -- this filter is an optimisation and a second line of defence, never the only
    # thing between a resting ball and a wrong answer, so falling back costs correctness
    # nothing. Caught by `tests/test_pipeline_smoke.py`, whose frames are all zeros.
    return kept if len(kept) >= MIN_INLIERS else detections


def _no_track_note(detections: list[Detection]) -> str:
    """Say which kind of "no ball" this is, because the three are not the same problem.

    "Nothing was detected" points at the detector's recall; "detections existed but never
    formed a moving path" points at a static false positive; "too few to verify" points at
    the clip being too short or the ball too briefly visible. A single generic message
    would collapse three different user actions -- supply better footage, ignore the
    background object, film the whole delivery -- into one shrug.
    """
    if not detections:
        return "No ball track available for this clip."
    distinct_frames = len({d.frame_index for d in detections})
    if distinct_frames < 5:
        return (
            f"Ball detections found on only {distinct_frames} frame(s) -- too few to verify a "
            "trajectory, so no ball path is reported."
        )
    return (
        "Ball detections found but rejected: no subset of them formed a physically "
        "plausible moving path (likely false positives on a static object, not the ball)."
    )
