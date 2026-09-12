from __future__ import annotations

import math

from ..contracts import BoundingBox, DeliveryEvent, ObjectClass, PoseFrame, ShotType, Track
from .base import EventSegmenter

_SWING_WINDOW = 4  # frames after contact over which wrist displacement is measured


class HeuristicEventSegmenter(EventSegmenter):
    """First-pass event segmentation from ball position and batter wrist movement.

    This is a rule-based v0, not a trained model: release/contact frames come from ball-track
    proximity to a player, and shot_type from a coarse wrist-swing heuristic. Expect it to be
    replaced by a trained classifier once labelled delivery clips exist (Part 03.3's model layer).
    """

    def segment(self, tracks: list[Track], poses: list[PoseFrame]) -> DeliveryEvent:
        ball_track = next((t for t in tracks if t.obj_class == ObjectClass.BALL), None)
        if ball_track is None or not ball_track.detections:
            return DeliveryEvent(
                release_frame=None,
                contact_frame=None,
                shot_type=ShotType.UNKNOWN,
                notes="No ball track available for this clip.",
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
