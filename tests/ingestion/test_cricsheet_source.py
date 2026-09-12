import json
from pathlib import Path

from ingestion.sources.cricsheet import parse_match

FIXTURE = Path(__file__).parent / "fixtures" / "cricsheet_sample_match.json"


def _load():
    with FIXTURE.open(encoding="utf-8") as f:
        raw = json.load(f)
    return parse_match("1273712", raw)


def test_match_meta():
    meta, _ = _load()
    assert meta.match_id == "1273712"
    assert meta.teams == ("Papua New Guinea", "Oman")
    assert meta.match_type == "T20"
    assert meta.outcome_winner == "Oman"
    assert meta.outcome_by == "10 wickets"
    assert meta.venue.startswith("Al Amerat")


def test_delivery_count_and_keys_are_unique():
    _, deliveries = _load()
    # Innings 1: overs 0-2 = 6 + 6 + 6 = 18. Innings 2: overs 0-2 = 7 (1 wide) + 6 + 7 (1 no-ball) = 20.
    assert len(deliveries) == 18 + 20

    keys = [(d.match_id, d.innings, d.over, d.ball) for d in deliveries]
    assert len(keys) == len(set(keys)), "every (match_id, innings, over, ball) must be a unique join key"


def test_wicket_is_parsed():
    _, deliveries = _load()
    wicket_ball = next(d for d in deliveries if d.innings == 1 and d.over == 0 and d.ball == 5)
    assert wicket_ball.wicket_player_out == "TP Ura"
    assert wicket_ball.wicket_kind == "bowled"
    assert wicket_ball.striker == "TP Ura"
    assert wicket_ball.bowler == "Bilal Khan"


def test_wide_and_its_rebowled_ball_get_distinct_ball_numbers():
    # Cricsheet's own "actual_delivery" field labels both the wide (0.3) and its
    # redo (also 0.3) the same -- our ball numbering must not collide on that.
    _, deliveries = _load()
    over0_inn2 = [d for d in deliveries if d.innings == 2 and d.over == 0]
    wide = next(d for d in over0_inn2 if d.extras_type == "wides")
    redo = next(d for d in over0_inn2 if d.ball == wide.ball + 1)
    assert wide.ball != redo.ball
    assert redo.runs_batter == 4
    assert len(over0_inn2) == 7  # 6 legal + 1 wide


def test_no_ball_and_leg_bye_extras_types():
    _, deliveries = _load()
    over2_inn2 = [d for d in deliveries if d.innings == 2 and d.over == 2]
    noball = next(d for d in over2_inn2 if d.extras_type == "noballs")
    assert noball.runs_total == 2 and noball.runs_batter == 1

    legbye = next(d for d in over2_inn2 if d.extras_type == "legbyes")
    assert legbye.runs_total == 1 and legbye.runs_batter == 0


def test_batting_and_bowling_team_and_phase():
    _, deliveries = _load()
    first = deliveries[0]
    assert first.batting_team == "Papua New Guinea"
    assert first.bowling_team == "Oman"
    assert first.phase == "powerplay"  # over 0 is always powerplay


def test_ref_matches_delivery_ref_join_key():
    _, deliveries = _load()
    d = deliveries[0]
    ref = d.ref()
    assert (ref.match_id, ref.innings, ref.over, ref.ball) == (d.match_id, d.innings, d.over, d.ball)
