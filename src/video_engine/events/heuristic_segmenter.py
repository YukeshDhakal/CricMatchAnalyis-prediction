from __future__ import annotations

import math

from ..contracts import BoundingBox, DeliveryEvent, Detection, ObjectClass, PoseFrame, ShotType, Track
from .base import EventSegmenter

_SWING_WINDOW = 4  # frames after contact over which wrist displacement is measured

# A real cricket ball crosses most of the frame during a delivery. Real-footage testing
# (see README's "Ball and stumps detection") found the fine-tuned ball checkpoint can lock
# onto a static background object (a bag, a sign) at plausible confidence for dozens of
# frames -- that detection never moves beyond ordinary frame-to-frame inference jitter
# around the same physical spot. Requiring a ball track's own displacement to clear a
# multiple of its own bounding-box size catches exactly that failure mode without needing
# a better checkpoint: real motion is a multiple of the object's own size; jitter isn't.
_MIN_DISPLACEMENT_BOX_MULTIPLE = 3.0


class HeuristicEventSegmenter(EventSegmenter):
    """First-pass event segmentation from ball position and batter wrist movement.

    This is a rule-based v0, not a trained model: release/contact frames come from ball-track
    proximity to a player, and shot_type from a coarse wrist-swing heuristic. Expect it to be
    replaced by a trained classifier once labelled delivery clips exist (Part 03.3's model layer).
    """

    def segment(self, tracks: list[Track], poses: list[PoseFrame]) -> DeliveryEvent:
        # IOU-based tracking (and ByteTrack under fast motion) fragments a genuinely fast
        # ball into several short-lived track_ids, since a ball's box barely overlaps
        # frame to frame -- so the motion check below runs over every BALL detection
        # pooled across tracks, not per track_id. That pooling is safe here (unlike for
        # PLAYER tracks) because a delivery clip has at most one real ball in flight.
        ball_detections = sorted(
            (d for t in tracks if t.obj_class == ObjectClass.BALL for d in t.detections),
            key=lambda d: d.frame_index,
        )
        ball_track = (
            Track(track_id=-1, obj_class=ObjectClass.BALL, detections=ball_detections)
            if ball_detections and _looks_like_real_motion_detections(ball_detections)
            else None
        )

        if ball_track is None:
            notes = (
                "No ball track available for this clip."
                if not ball_detections
                else "Ball detections found but rejected: none showed real motion across "
                "frames (likely a false positive on a static object, not the ball)."
            )
            return DeliveryEvent(
                release_frame=None,
                contact_frame=None,
                shot_type=ShotType.UNKNOWN,
                notes=notes,
            )

        release_frame = ball_track.detections[0].frame_index
        contact_frame, batter_track_id = self._find_contact(ball_track, tracks)

        if contact_frame is None or batter_track_id is None:
            return DeliveryEvent(
                release_frame=release_frame,
                contact_frame=None,
                shot_type=ShotType.UNKNOWN,
                notes="Ball never came within range of a tracked player.",
            )

        shot_type = self._classify_shot(poses, batter_track_id, contact_frame)
        return DeliveryEvent(release_frame=release_frame, contact_frame=contact_frame, shot_type=shot_type)

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


def _looks_like_real_motion_detections(detections: list[Detection]) -> bool:
    """True if `detections` (all the same object, pooled across however many track_ids
    tracking fragmented it into) move enough, relative to their own size, to plausibly be a
    ball in flight rather than a stationary false positive re-detected frame after frame."""
    if len(detections) < 2:
        return False

    centers = [_center(d.box) for d in detections]
    box_sizes = [max(d.box.x2 - d.box.x1, d.box.y2 - d.box.y1) for d in detections]
    avg_box_size = sum(box_sizes) / len(box_sizes)
    if avg_box_size <= 0:
        return False

    max_spread = max(
        _distance(centers[i], centers[j])
        for i in range(len(centers))
        for j in range(i + 1, len(centers))
    )
    return max_spread >= _MIN_DISPLACEMENT_BOX_MULTIPLE * avg_box_size
