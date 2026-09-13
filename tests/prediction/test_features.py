import pandas as pd

from prediction.contracts import LiveTelemetryEvent, MatchState
from prediction.features import batter_vs_bowler_matchup, contextual_features, pitch_fatigue_features


def test_contextual_features_with_no_target_yet():
    state = MatchState(balls_bowled_this_innings=60, total_balls_in_innings=120, current_score=80, wickets_down=3)

    features = contextual_features(state)

    assert features.current_run_rate == 8.0
    assert features.required_run_rate is None
    assert features.balls_remaining == 60
    assert features.wickets_in_hand == 7


def test_contextual_features_computes_required_run_rate_during_a_chase():
    state = MatchState(
        balls_bowled_this_innings=60, total_balls_in_innings=120, current_score=80, wickets_down=3, target=180
    )

    features = contextual_features(state)

    assert features.required_run_rate == 10.0


def test_contextual_features_before_any_ball_bowled():
    state = MatchState(balls_bowled_this_innings=0, total_balls_in_innings=120, current_score=0, wickets_down=0)

    features = contextual_features(state)

    assert features.current_run_rate == 0.0
    assert features.balls_remaining == 120
    assert features.wickets_in_hand == 10


def test_contextual_features_required_run_rate_is_none_with_no_balls_left():
    state = MatchState(
        balls_bowled_this_innings=120, total_balls_in_innings=120, current_score=170, wickets_down=4, target=180
    )

    features = contextual_features(state)

    assert features.required_run_rate is None


def test_batter_vs_bowler_matchup_excludes_wides_from_balls_faced():
    deliveries = pd.DataFrame(
        [
            {"striker": "Bob", "bowler": "Carl", "runs_batter": 4, "extras_type": None, "wicket_player_out": None},
            {"striker": "Bob", "bowler": "Carl", "runs_batter": 0, "extras_type": "wides", "wicket_player_out": None},
            {"striker": "Bob", "bowler": "Carl", "runs_batter": 1, "extras_type": None, "wicket_player_out": None},
            {"striker": "Bob", "bowler": "Carl", "runs_batter": 0, "extras_type": None, "wicket_player_out": "Bob"},
            {"striker": "Dave", "bowler": "Carl", "runs_batter": 6, "extras_type": None, "wicket_player_out": None},
        ]
    )

    matchup = batter_vs_bowler_matchup(deliveries, "Bob", "Carl")

    assert matchup.balls_faced == 3  # wide excluded
    assert matchup.runs_scored == 5
    assert matchup.dismissals == 1
    assert matchup.strike_rate == round(100 * 5 / 3, 2)


def test_batter_vs_bowler_matchup_with_no_history_is_zeroed():
    deliveries = pd.DataFrame(
        [{"striker": "Dave", "bowler": "Carl", "runs_batter": 6, "extras_type": None, "wicket_player_out": None}]
    )

    matchup = batter_vs_bowler_matchup(deliveries, "Bob", "Carl")

    assert matchup.balls_faced == 0
    assert matchup.strike_rate == 0.0


def _telemetry(over: int, ball: int, speed: float | None, spin: float | None) -> LiveTelemetryEvent:
    return LiveTelemetryEvent(match_id="m1", innings=1, over=over, ball=ball, ball_speed_kmh=speed, spin_rpm=spin)


def test_pitch_fatigue_features_needs_two_full_windows_of_data():
    telemetry = [_telemetry(o, b, 140.0, 2000.0) for o in range(2) for b in range(1, 7)]

    features = pitch_fatigue_features(telemetry, recent_overs=3)

    assert features.pitch_degradation_index is None
    assert features.avg_speed_drop_kmh is None


def test_pitch_fatigue_features_detects_speed_drop_and_spin_increase():
    early = [_telemetry(o, b, 145.0, 1800.0) for o in range(3) for b in range(1, 7)]
    late = [_telemetry(o, b, 138.0, 2200.0) for o in range(3, 6) for b in range(1, 7)]

    features = pitch_fatigue_features(early + late, recent_overs=3)

    assert features.avg_speed_drop_kmh == 7.0  # bowler slowing down
    assert features.pitch_degradation_index == 400.0  # more spin deviation late


def test_pitch_fatigue_features_empty_telemetry():
    assert pitch_fatigue_features([]) == pitch_fatigue_features([])  # no crash, both None
    features = pitch_fatigue_features([])
    assert features.pitch_degradation_index is None
    assert features.avg_speed_drop_kmh is None
