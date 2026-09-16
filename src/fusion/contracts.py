"""Data contracts for the fusion layer (PRD 2.2 stage 3 -- not built yet).

Fusion joins ingestion's per-delivery ball-by-ball truth (`Delivery`) with the
video engine's per-delivery CV output (`DeliveryAnalysis`) into one row per
bowled delivery (`FusedDelivery`), then rolls those rows up into per-match and
rolling-N-match player stats. Modeled on a reference post-game ETL design
(ingest telemetry -> aggregate single-match stats -> refresh a rolling
5-match summary) but reshaped around this repo's actual contracts and what
the CV pipeline currently produces -- see `FusedDelivery`'s docstring for
which fields have no producer yet.

Nothing in this module executes: it's the shape `fusion/pipeline.py` will
build once someone writes the join + aggregation logic. Contracts only, same
division of labor as `ingestion/contracts.py` and `video_engine/contracts.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ingestion.contracts import Delivery, DeliveryRef
from video_engine.contracts import PitchLength, ShotType

__all__ = [
    "FusedDelivery",
    "PlayerMatchStats",
    "PlayerRollingSummary",
]


@dataclass(frozen=True)
class FusedDelivery:
    """One delivery: ball-by-ball truth plus whatever video-derived signal exists for it.

    Flat and scalar-typed rather than embedding `Track`/`PoseFrame` lists -- this is
    the row shape stats/rating aggregation consumes, not a CV artifact. Raw tracks
    and poses stay inside `video_engine`, same boundary discipline `DeliveryClip`
    already keeps with ingestion.

    Ball-outcome fields mirror `ingestion.contracts.Delivery` by name rather than
    inventing new ones, so a fused row and a pure-stats row (`player_reports`) stay
    reconcilable against the same ingested truth.

    `video_available` is False whenever this delivery has no matching
    `DeliveryAnalysis` -- not every delivery will have footage (a match might only
    have some deliveries uploaded, and `BroadcastFeedAdapter` isn't implemented).
    Every field below it is `None` in that case, not zero, so aggregation can tell
    "no shot played" apart from "no video to tell."

    CV biomechanics fields (wrist speed, elbow extension, release/swing speed) are
    reserved but unpopulated: `HeuristicEventSegmenter` finds release/contact frames
    and a coarse `shot_type` from wrist *displacement* in raw pixel space -- it has
    no camera calibration, so it cannot honestly produce a real-world speed or joint
    angle. These fields exist so `fusion`/`player_reports` code can be written
    against a stable shape now, ahead of a calibrated biomechanics extractor landing
    in `video_engine/pose/`. Leave them `None` until that extractor exists --
    relabeling the heuristic's pixel-space displacement as "wrist_speed_ms" would
    make the field actively misleading, not just incomplete.

    Pitch-geometry fields (`pitch_length_m`, `pitch_line_m`, `pitch_length`) follow the
    same rule and are blocked on the same missing piece from the other direction. They
    need a *bounce point* in real-world metres, which needs `ObjectClass.BALL`
    detections to find the bounce and `ObjectClass.STUMPS` detections to calibrate the
    pixels-to-metres map (`video_engine.calibration`). Neither detector exists -- there
    is no fine-tuned 2-class checkpoint for this, and the README's "Ball and stumps
    detection" section records why the externally available options are not drop-ins.
    So every one of these is `None`/`UNKNOWN` on every row today.

    They are defined now rather than later because the consumer side is what makes them
    worth having: `rating.contracts.Metric` can carry pitch-length metrics and
    `rating.suggestions` can flag them only if there is an agreed row shape to compute
    them from, and agreeing that shape after a detector lands would mean rewriting both.
    `pitch_line_m` is signed distance from the middle stump and is deliberately *not*
    off/leg -- that needs batting-handedness metadata Cricsheet doesn't publish; see
    `video_engine.contracts.PitchPoint`.
    """

    delivery: DeliveryRef
    match_id: str
    innings: int
    over: int
    ball: int
    phase: str  # "powerplay" | "middle" | "death", from Delivery.phase

    striker: str
    non_striker: str
    bowler: str
    runs_batter: int
    runs_extras: int
    runs_total: int
    extras_type: Optional[str]
    wicket_player_out: Optional[str]
    wicket_kind: Optional[str]

    video_available: bool
    shot_type: Optional[ShotType] = None
    release_frame: Optional[int] = None
    contact_frame: Optional[int] = None

    # --- reserved for a not-yet-built calibrated biomechanics extractor ---
    wrist_speed_ms: Optional[float] = None
    bowling_elbow_extension_deg: Optional[float] = None
    ball_release_speed_kmh: Optional[float] = None
    bat_swing_speed_kmh: Optional[float] = None

    # --- reserved for a not-yet-built ball detector + pitch calibration ---
    # Metres from the striker's stumps down the pitch, and signed metres across it from
    # the middle stump. `pitch_length` is the coaching band the first of those falls in.
    pitch_length_m: Optional[float] = None
    pitch_line_m: Optional[float] = None
    pitch_length: PitchLength = PitchLength.UNKNOWN

    def has_pitch_geometry(self) -> bool:
        """Whether this row carries a real, calibrated bounce point.

        A single predicate rather than each consumer writing its own `is not None`
        check, because there are two fields plus an enum and "has geometry" has to mean
        the same thing to the baseline that pools them and to the flag that cites them.
        `FULL_TOSS` counts: a ball that never bounced has no `pitch_length_m` and is
        still a real, measured observation about length.
        """
        if self.pitch_length is PitchLength.FULL_TOSS:
            return True
        return self.pitch_length is not PitchLength.UNKNOWN and self.pitch_length_m is not None

    def ref(self) -> DeliveryRef:
        return self.delivery

    @staticmethod
    def from_delivery_only(delivery: Delivery) -> "FusedDelivery":
        """A delivery with no video coverage: ball-by-ball truth, everything CV-derived is None."""
        return FusedDelivery(
            delivery=delivery.ref(),
            match_id=delivery.match_id,
            innings=delivery.innings,
            over=delivery.over,
            ball=delivery.ball,
            phase=delivery.phase,
            striker=delivery.striker,
            non_striker=delivery.non_striker,
            bowler=delivery.bowler,
            runs_batter=delivery.runs_batter,
            runs_extras=delivery.runs_extras,
            runs_total=delivery.runs_total,
            extras_type=delivery.extras_type,
            wicket_player_out=delivery.wicket_player_out,
            wicket_kind=delivery.wicket_kind,
            video_available=False,
        )


@dataclass(frozen=True)
class PlayerMatchStats:
    """One player's batting + bowling + CV-biomechanics line for a single match.

    The successor to `player_reports.stats.batting_summary`/`bowling_summary` once
    video-derived signal exists to add -- not a replacement for them in the
    meantime; those functions keep working directly off `Delivery` rows for the
    "Execution"-only view.

    Keyed by `player` (name string), matching how `player_reports` already
    identifies players -- there's no separate player-id table in this repo, unlike
    the reference schema this was modeled on.
    """

    player: str
    match_id: str
    match_date: str

    runs_scored: int
    balls_faced: int
    fours: int
    sixes: int
    batting_strike_rate: float

    balls_bowled: int
    runs_conceded: int
    wickets_taken: int
    economy_rate: float

    # None whenever every delivery this player bowled had video_available=False
    avg_wrist_speed_ms: Optional[float] = None
    max_wrist_speed_ms: Optional[float] = None
    avg_bowling_elbow_extension_deg: Optional[float] = None
    # Only meaningful once avg_bowling_elbow_extension_deg is populated from a real
    # extractor -- see FusedDelivery's note. ICC's legal limit is 15 degrees.
    illegal_bowling_action_flag: Optional[bool] = None


@dataclass(frozen=True)
class PlayerRollingSummary:
    """Rolling last-N-match window for one player, refreshed after each new match.

    `matches_in_sample` can be less than `window_size` early in a player's covered
    history -- the same "not enough matches yet" case
    `player_reports.last_n_match_ids` already returns fewer than `n` for. Ties
    on match_date should break the same deterministic way that function documents
    (sort by `(date, match_id)`), not an unstable default sort.
    """

    player: str
    window_size: int
    matches_in_sample: int
    last_updated: str

    total_runs: int
    avg_runs: float
    strike_rate: float
    highest_score: int
    boundary_percent: float

    total_wickets: int
    bowling_average: Optional[float]
    economy_rate: float

    avg_wrist_speed_ms: Optional[float] = None
    max_wrist_speed_ms: Optional[float] = None
    avg_bowling_elbow_extension_deg: Optional[float] = None
