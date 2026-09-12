"""Data contracts for ingestion.

Re-exports the video engine's `DeliveryRef` / `DeliveryClip` / `Source` so ingestion's
video adapters produce exactly the shape the video engine consumes -- see
`src/video_engine/contracts.py` for the canonical definitions of those three.
Everything else here is ingestion-owned: the official ball-by-ball table
(PRD 2.1, "ICC / board ball-by-ball tables") that the fusion stage lines up
against the video engine's per-delivery output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from video_engine.contracts import DeliveryClip, DeliveryRef, Source

__all__ = [
    "DeliveryRef",
    "DeliveryClip",
    "Source",
    "MatchMeta",
    "Delivery",
]


@dataclass(frozen=True)
class MatchMeta:
    """One match's metadata -- venue, teams, result. Keyed by match_id, same key DeliveryRef joins on."""

    match_id: str
    competition: str
    match_type: str
    gender: str
    teams: tuple[str, str]
    venue: str
    city: Optional[str]
    dates: tuple[str, ...]
    toss_winner: Optional[str]
    toss_decision: Optional[str]
    outcome_winner: Optional[str]
    outcome_by: Optional[str]


@dataclass(frozen=True)
class Delivery:
    """One row of the official ball-by-ball table.

    Joins to the video engine's DeliveryRef by (match_id, innings, over, ball).
    `ball` is a 1-based count of every delivery bowled in the over, wides and
    no-balls included -- matching DeliveryRef.ball's documented semantics.
    """

    match_id: str
    innings: int
    over: int
    ball: int
    batting_team: str
    bowling_team: str
    striker: str
    non_striker: str
    bowler: str
    runs_batter: int
    runs_extras: int
    runs_total: int
    extras_type: Optional[str]
    wicket_player_out: Optional[str]
    wicket_kind: Optional[str]
    phase: str  # "powerplay" | "middle" | "death"

    def ref(self) -> DeliveryRef:
        return DeliveryRef(match_id=self.match_id, innings=self.innings, over=self.over, ball=self.ball)
