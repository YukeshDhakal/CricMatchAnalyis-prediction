"""Feature engineering for the live match-prediction layer: contextual, matchup, and
pitch/fatigue features -- layer 2 of Yukesh's 4-layer architecture, built first per his
stated priority ("even a strong model fails without clean, context-rich features").
Layer 1 (live telemetry intake) and layers 3/4 (phase-aware ML, API/serving) aren't
built; see `contracts.py`.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .contracts import (
    ContextualFeatures,
    LiveTelemetryEvent,
    MatchState,
    MatchupFeatures,
    PitchFatigueFeatures,
)

__all__ = ["contextual_features", "batter_vs_bowler_matchup", "pitch_fatigue_features"]


def contextual_features(state: MatchState) -> ContextualFeatures:
    """Current/required run rate, balls remaining, wickets in hand. Rates are per-6-ball
    over, matching how run rate is conventionally reported (not per-ball)."""
    balls_remaining = state.total_balls_in_innings - state.balls_bowled_this_innings
    current_run_rate = (
        round(6 * state.current_score / state.balls_bowled_this_innings, 2)
        if state.balls_bowled_this_innings
        else 0.0
    )

    required_run_rate: Optional[float] = None
    if state.target is not None and balls_remaining > 0:
        runs_needed = state.target - state.current_score
        required_run_rate = round(6 * runs_needed / balls_remaining, 2)

    return ContextualFeatures(
        current_run_rate=current_run_rate,
        required_run_rate=required_run_rate,
        balls_remaining=balls_remaining,
        wickets_in_hand=10 - state.wickets_down,
    )


def batter_vs_bowler_matchup(deliveries: pd.DataFrame, batter: str, bowler: str) -> MatchupFeatures:
    """Rolling record for this exact batter/bowler pair across all ingested deliveries.
    Same ball-faced convention as `player_reports.stats.batting_summary`: a wide isn't a
    ball faced.
    """
    faced = deliveries[(deliveries["striker"] == batter) & (deliveries["bowler"] == bowler)]
    balls_faced_rows = faced[faced["extras_type"] != "wides"]
    balls_faced = len(balls_faced_rows)
    runs_scored = int(faced["runs_batter"].sum())
    dismissals = int((faced["wicket_player_out"] == batter).sum())

    return MatchupFeatures(
        batter=batter,
        bowler=bowler,
        balls_faced=balls_faced,
        runs_scored=runs_scored,
        dismissals=dismissals,
        strike_rate=round(100 * runs_scored / balls_faced, 2) if balls_faced else 0.0,
    )


def pitch_fatigue_features(
    telemetry: list[LiveTelemetryEvent], recent_overs: int = 3
) -> PitchFatigueFeatures:
    """Spin-deviation trend and speed drop across one innings, comparing the first
    `recent_overs` overs against the most recent `recent_overs`. Returns both fields as
    `None` until at least `2 * recent_overs` overs of telemetry exist -- comparing
    overlapping or too-thin windows would be noise, not signal.
    """
    if not telemetry:
        return PitchFatigueFeatures(pitch_degradation_index=None, avg_speed_drop_kmh=None)

    ordered = sorted(telemetry, key=lambda e: (e.over, e.ball))
    max_over = ordered[-1].over
    if max_over < 2 * recent_overs - 1:
        return PitchFatigueFeatures(pitch_degradation_index=None, avg_speed_drop_kmh=None)

    early = [e for e in ordered if e.over < recent_overs]
    late = [e for e in ordered if e.over > max_over - recent_overs]

    pitch_degradation_index = _trend(early, late, lambda e: e.spin_rpm)
    avg_speed_drop_kmh = _trend(late, early, lambda e: e.ball_speed_kmh)  # early minus late: positive = slowing down

    return PitchFatigueFeatures(
        pitch_degradation_index=pitch_degradation_index,
        avg_speed_drop_kmh=avg_speed_drop_kmh,
    )


def _trend(baseline, comparison, field) -> Optional[float]:
    """comparison's average minus baseline's average, over whichever events have `field` set."""
    baseline_vals = [v for e in baseline if (v := field(e)) is not None]
    comparison_vals = [v for e in comparison if (v := field(e)) is not None]
    if not baseline_vals or not comparison_vals:
        return None
    return round(sum(comparison_vals) / len(comparison_vals) - sum(baseline_vals) / len(baseline_vals), 2)
