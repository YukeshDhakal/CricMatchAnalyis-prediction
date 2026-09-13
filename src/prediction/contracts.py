"""Data contracts for the live match-prediction feature-engineering layer.

This is Yukesh's separate 4-layer architecture for live win-probability / next-ball
prediction (data ingestion -> feature engineering -> phase-aware ML -> API/serving) --
distinct from `fusion/`'s post-match rating and coaching-suggestion work, though both
consume the same ingested ball-by-ball table. See memory note
"third_umpire_prediction_engine_architecture" for the full architecture; only layer 2
(feature engineering) is scaffolded here, per Yukesh's stated priority that clean,
context-rich features matter more than model choice. No ML model or FastAPI/Redis
serving layer exists yet, and none is implied by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "LiveTelemetryEvent",
    "MatchState",
    "ContextualFeatures",
    "MatchupFeatures",
    "PitchFatigueFeatures",
]


@dataclass(frozen=True)
class LiveTelemetryEvent:
    """One delivery's live CV-tracking telemetry, shaped the way a future live-feed
    REST endpoint would post it (layer 1 of the architecture -- not built; there is no
    producer of this yet, same "shape reserved, ahead of a producer" pattern as
    `fusion.contracts.FusedDelivery`'s CV fields). Field names/units match Yukesh's own
    example payload (`ball_speed`, `spin_rpm`, `pitch_map_x`).
    """

    match_id: str
    innings: int
    over: int
    ball: int
    ball_speed_kmh: Optional[float] = None
    spin_rpm: Optional[float] = None
    pitch_map_x: Optional[float] = None


@dataclass(frozen=True)
class MatchState:
    """Live match state needed for contextual features. Derivable from the ball-by-ball
    table alone (ingestion) -- no CV telemetry required, unlike pitch/fatigue features.
    """

    balls_bowled_this_innings: int
    total_balls_in_innings: int  # e.g. 120 for a T20 innings
    current_score: int
    wickets_down: int
    target: Optional[int] = None  # None in the first innings -- there's no chase yet


@dataclass(frozen=True)
class ContextualFeatures:
    current_run_rate: float
    required_run_rate: Optional[float]  # None whenever there's no target to chase
    balls_remaining: int
    wickets_in_hand: int


@dataclass(frozen=True)
class MatchupFeatures:
    """Batter-vs-this-specific-bowler rolling record.

    Yukesh's architecture calls for batter-vs-bowler-*type* matchups (e.g. left-arm
    orthodox vs right-hand batter, generalizing across every bowler of that type) --
    that needs a `bowling_style`/`batting_style` field ingestion doesn't capture yet
    (Cricsheet's player registry has no such metadata, unlike the reference ETL's
    `players` table). This contract is the concrete-bowler granularity that's actually
    derivable from what's ingested today; generalizing to bowler-type is future work
    once that metadata exists, not something to fake here.
    """

    batter: str
    bowler: str
    balls_faced: int
    runs_scored: int
    dismissals: int
    strike_rate: float


@dataclass(frozen=True)
class PitchFatigueFeatures:
    """Pitch-wear and fast-bowler-fatigue signal derived from a bowler's live telemetry
    across one innings. Both fields are `None` whenever there isn't at least two
    non-overlapping comparison windows of data yet (early in an innings, or -- today --
    always, since no `LiveTelemetryEvent` producer exists).
    """

    pitch_degradation_index: Optional[float]  # spin_rpm drift, late-innings minus early (positive = more turn)
    avg_speed_drop_kmh: Optional[float]  # early-innings avg ball speed minus late (positive = bowler slowing down)
