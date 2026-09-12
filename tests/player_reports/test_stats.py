import pandas as pd

from player_reports.stats import batting_summary, bowling_summary, last_n_match_ids


def _delivery(**overrides) -> dict:
    base = dict(
        match_id="m1",
        innings=1,
        over=0,
        ball=1,
        batting_team="A",
        bowling_team="B",
        striker="Nobody",
        non_striker="Nobody",
        bowler="Nobody",
        runs_batter=0,
        runs_extras=0,
        runs_total=0,
        extras_type=None,
        wicket_player_out=None,
        wicket_kind=None,
        phase="middle",
    )
    base.update(overrides)
    return base


def test_last_n_match_ids_orders_most_recent_first():
    matches = pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3"],
            "dates": [["2026-01-01"], ["2026-01-05"], ["2026-01-03"]],
        }
    )
    deliveries = pd.DataFrame(
        [
            _delivery(match_id="m1", striker="Bob"),
            _delivery(match_id="m2", striker="Bob"),
            _delivery(match_id="m3", striker="Bob"),
        ]
    )

    assert last_n_match_ids(deliveries, matches, "Bob", n=2) == ["m2", "m3"]


def test_last_n_match_ids_breaks_same_date_ties_deterministically():
    # Same-date ties are common (e.g. two World Cup matches on one day); pandas'
    # default sort is quicksort, which is not stable, so a naive date-only sort
    # could return a different order on repeated calls against identical data.
    matches = pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3"],
            "dates": [["2026-01-05"], ["2026-01-05"], ["2026-01-01"]],
        }
    )
    deliveries = pd.DataFrame(
        [
            _delivery(match_id="m1", striker="Bob"),
            _delivery(match_id="m2", striker="Bob"),
            _delivery(match_id="m3", striker="Bob"),
        ]
    )

    result = [last_n_match_ids(deliveries, matches, "Bob", n=2) for _ in range(20)]

    assert all(r == result[0] for r in result), "tie-breaking must be deterministic across repeated calls"
    assert result[0] == ["m2", "m1"]  # same date -> higher match_id (m2) sorts first


def test_batting_summary():
    deliveries = pd.DataFrame(
        [
            _delivery(striker="Bob", runs_batter=4),
            _delivery(striker="Bob", runs_batter=0, extras_type="wides"),  # not a ball faced
            _delivery(striker="Bob", runs_batter=1, extras_type="noballs"),  # a ball faced
            _delivery(striker="Bob", runs_batter=6),
            _delivery(striker="Bob", runs_batter=0, wicket_player_out="Bob", wicket_kind="bowled"),
        ]
    )

    summary = batting_summary(deliveries, "Bob")

    assert summary["innings"] == 1
    assert summary["runs"] == 11
    assert summary["balls_faced"] == 4
    assert summary["fours"] == 1
    assert summary["sixes"] == 1
    assert summary["dismissals"] == 1
    assert summary["highest_score"] == 11
    assert summary["strike_rate"] == 275.0


def test_bowling_summary_excludes_wides_from_balls_and_run_outs_from_wickets():
    deliveries = pd.DataFrame(
        [
            _delivery(bowler="Carl", runs_total=1),
            _delivery(bowler="Carl", runs_total=1, extras_type="wides"),  # illegal: not a ball bowled
            _delivery(bowler="Carl", runs_total=1, runs_extras=1, extras_type="legbyes"),  # legal, not charged to bowler
            _delivery(bowler="Carl", runs_total=0, wicket_player_out="X", wicket_kind="caught"),
            _delivery(bowler="Carl", runs_total=0, wicket_player_out="Y", wicket_kind="run out"),  # not credited
            _delivery(bowler="Carl", runs_total=4),
        ]
    )

    summary = bowling_summary(deliveries, "Carl")

    assert summary["overs"] == "0.5"  # 5 legal balls (wide excluded)
    assert summary["runs_conceded"] == 6  # 7 total runs - 1 legbye
    assert summary["wickets"] == 1  # run out excluded
    assert summary["economy"] == 7.2


def test_summaries_return_zeroed_stats_for_a_player_with_no_deliveries():
    deliveries = pd.DataFrame([_delivery()])

    assert batting_summary(deliveries, "Nobody Else") == {
        "innings": 0,
        "runs": 0,
        "balls_faced": 0,
        "fours": 0,
        "sixes": 0,
        "dismissals": 0,
        "highest_score": 0,
        "strike_rate": 0.0,
    }
    assert bowling_summary(deliveries, "Nobody Else") == {
        "overs": "0.0",
        "runs_conceded": 0,
        "wickets": 0,
        "economy": 0.0,
    }
