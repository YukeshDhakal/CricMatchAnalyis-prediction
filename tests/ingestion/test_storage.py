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


def test_write_matches_accumulates_across_calls_instead_of_overwriting(tmp_path):
    """The bug this guards against: ingesting competition Y after competition X used
    to silently wipe X's matches, because the old write_matches was a plain
    `to_parquet` with no read of what was already on disk."""
    store = ParquetStore(tmp_path)
    other_match = MatchMeta(
        match_id="999",
        competition="Other Cup",
        match_type="T20",
        gender="male",
        teams=("C", "D"),
        venue="Another Ground",
        city=None,
        dates=("2026-02-02",),
        toss_winner="C",
        toss_decision="field",
        outcome_winner="D",
        outcome_by="3 runs",
    )

    store.write_matches([_sample_match()])
    store.write_matches([other_match])

    matches_df = store.read_matches()
    assert set(matches_df["match_id"]) == {"123", "999"}
    assert set(matches_df["competition"]) == {"Test Cup", "Other Cup"}


def test_write_matches_upserts_same_match_id(tmp_path):
    """Re-ingesting a match_id that's already stored replaces its row (last write
    wins) rather than duplicating it -- covers a genuine re-fetch after cricsheet.org
    corrects a match's data."""
    store = ParquetStore(tmp_path)
    store.write_matches([_sample_match()])

    corrected = MatchMeta(**{**_sample_match().__dict__, "outcome_winner": "B"})
    store.write_matches([corrected])

    matches_df = store.read_matches()
    assert len(matches_df) == 1
    assert matches_df.iloc[0]["outcome_winner"] == "B"


def test_write_deliveries_accumulates_and_upserts(tmp_path):
    store = ParquetStore(tmp_path)
    second_ball = Delivery(**{**_sample_delivery().__dict__, "ball": 2, "runs_batter": 1, "runs_total": 1})
    store.write_deliveries([_sample_delivery()])
    store.write_deliveries([second_ball])

    corrected_first_ball = Delivery(**{**_sample_delivery().__dict__, "runs_batter": 6, "runs_total": 6})
    store.write_deliveries([corrected_first_ball])

    deliveries_df = store.read_deliveries()
    assert len(deliveries_df) == 2  # accumulated, not overwritten
    first = deliveries_df[deliveries_df["ball"] == 1].iloc[0]
    assert first["runs_total"] == 6  # upserted, not duplicated


# --- scoped reads -------------------------------------------------------------------
#
# `read_matches`/`read_deliveries` grew optional filters that are pushed down to the
# Parquet reader so a question about a handful of matches doesn't materialise the whole
# warehouse (see the module docstring). These tests pin the *semantics* of those filters,
# especially the None-vs-empty distinction: getting that backwards would turn a cohort
# that legitimately selected nothing into a silent full-warehouse scan, which is the
# exact failure the filters exist to prevent.


def _match(match_id: str, competition: str = "Test Cup"):
    return MatchMeta(**{**_sample_match().__dict__, "match_id": match_id, "competition": competition})


def _ball(match_id: str, ball: int, striker="P1", non_striker="P2", bowler="P3"):
    return Delivery(
        **{
            **_sample_delivery().__dict__,
            "match_id": match_id, "ball": ball,
            "striker": striker, "non_striker": non_striker, "bowler": bowler,
        }
    )


def _seeded(tmp_path) -> ParquetStore:
    store = ParquetStore(tmp_path)
    store.write_matches([_match("m1"), _match("m2", "Other Cup"), _match("m3")])
    store.write_deliveries(
        [
            _ball("m1", 1, striker="Bob"),
            _ball("m1", 2, bowler="Bob"),
            _ball("m2", 1, non_striker="Bob"),
            _ball("m3", 1, striker="Ann", non_striker="Dee", bowler="Carl"),
        ]
    )
    return store


def test_reads_without_filters_return_everything(tmp_path):
    """Backward compatibility: the no-argument call is exactly the old behaviour."""
    store = _seeded(tmp_path)
    assert len(store.read_matches()) == 3
    assert len(store.read_deliveries()) == 4


def test_match_id_filter_narrows_both_tables(tmp_path):
    store = _seeded(tmp_path)
    assert set(store.read_matches(match_ids=["m1", "m3"])["match_id"]) == {"m1", "m3"}
    assert len(store.read_deliveries(match_ids=["m1"])) == 2


def test_an_empty_filter_selects_nothing_and_keeps_the_schema(tmp_path):
    """`[]` means "nothing was selected", not "no filter".

    The schema has to survive, because a caller that selected an empty cohort still
    operates on the returned frame's columns rather than special-casing emptiness.
    """
    store = _seeded(tmp_path)
    empty = store.read_deliveries(match_ids=[])
    assert len(empty) == 0
    assert list(empty.columns) == list(store.read_deliveries().columns)


def test_competition_filter_is_pushed_down(tmp_path):
    store = _seeded(tmp_path)
    assert set(store.read_matches(competitions=["Other Cup"])["match_id"]) == {"m2"}


def test_filters_work_on_columns_that_are_not_projected(tmp_path):
    """Filtering happens before projection, so a caller can narrow on `match_id` without
    paying to read it back."""
    store = _seeded(tmp_path)
    rows = store.read_deliveries(match_ids=["m1"], columns=["striker"])
    assert list(rows.columns) == ["striker"]
    assert len(rows) == 2


def test_involving_player_covers_all_three_ways_of_being_in_the_middle(tmp_path):
    """Striker, non-striker and bowler all count -- the same definition
    `player_reports.last_n_match_ids` and `rating.cohort.player_match_ids` use. A
    bowling-only appearance is still an appearance."""
    store = _seeded(tmp_path)
    rows = store.read_deliveries(involving_player="Bob")
    assert len(rows) == 3
    assert set(rows["match_id"]) == {"m1", "m2"}
    assert store.read_deliveries(involving_player="Nobody").empty


def test_player_and_match_filters_combine_as_a_conjunction(tmp_path):
    """The match restriction has to survive into every disjunct of the player filter.

    Dropping it from any one branch would silently widen that branch to the whole
    warehouse -- the bug this test exists to catch.
    """
    store = _seeded(tmp_path)
    rows = store.read_deliveries(match_ids=["m1"], involving_player="Bob")
    assert len(rows) == 2
    assert set(rows["match_id"]) == {"m1"}

    assert store.read_deliveries(match_ids=["m3"], involving_player="Bob").empty
