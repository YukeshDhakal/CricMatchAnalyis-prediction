from ingestion.contracts import Delivery, MatchMeta
from ingestion.storage import ParquetStore


def _sample_match():
    return MatchMeta(
        match_id="123",
        competition="Test Cup",
        match_type="T20",
        gender="male",
        teams=("A", "B"),
        venue="Some Ground",
        city="Some City",
        dates=("2026-01-01",),
        toss_winner="A",
        toss_decision="bat",
        outcome_winner="A",
        outcome_by="5 wickets",
    )


def _sample_delivery():
    return Delivery(
        match_id="123",
        innings=1,
        over=0,
        ball=1,
        batting_team="A",
        bowling_team="B",
        striker="P1",
        non_striker="P2",
        bowler="P3",
        runs_batter=4,
        runs_extras=0,
        runs_total=4,
        extras_type=None,
        wicket_player_out=None,
        wicket_kind=None,
        phase="powerplay",
    )


def test_write_and_read_roundtrip(tmp_path):
    store = ParquetStore(tmp_path)
    store.write_matches([_sample_match()])
    store.write_deliveries([_sample_delivery()])

    matches_df = store.read_matches()
    deliveries_df = store.read_deliveries()

    assert len(matches_df) == 1
    assert matches_df.iloc[0]["match_id"] == "123"
    assert list(matches_df.iloc[0]["teams"]) == ["A", "B"]

    assert len(deliveries_df) == 1
    assert deliveries_df.iloc[0]["runs_total"] == 4
    assert deliveries_df.iloc[0]["phase"] == "powerplay"
