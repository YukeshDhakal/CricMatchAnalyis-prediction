from dataclasses import replace

import pandas as pd

from fusion.contracts import FusedDelivery
from fusion.pipeline import (
    aggregate_player_match_stats,
    fuse_match_deliveries,
    refresh_all_rolling_summaries,
    refresh_rolling_summary,
)
from video_engine.contracts import DeliveryAnalysis, DeliveryEvent, DeliveryRef, ShotType


def _delivery(**overrides) -> dict:
    base = dict(
        match_id="m1",
        innings=1,
        over=0,
        ball=1,
        batting_team="A",
        bowling_team="B",
        striker="Bob",
        non_striker="Nobody",
        bowler="Carl",
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


def _analysis(ref: DeliveryRef, shot_type: ShotType = ShotType.DRIVE) -> DeliveryAnalysis:
    return DeliveryAnalysis(
        delivery=ref,
        tracks=[],
        poses=[],
        event=DeliveryEvent(release_frame=0, contact_frame=3, shot_type=shot_type),
    )


def test_fuse_match_deliveries_marks_video_available_only_where_an_analysis_matches():
    deliveries = pd.DataFrame(
        [
            _delivery(ball=1, runs_batter=4),
            _delivery(ball=2, runs_batter=1),
        ]
    )
    analyses = [_analysis(DeliveryRef(match_id="m1", innings=1, over=0, ball=1))]

    fused = fuse_match_deliveries(deliveries, "m1", analyses)

    assert len(fused) == 2
    with_video = next(d for d in fused if d.ball == 1)
    without_video = next(d for d in fused if d.ball == 2)

    assert with_video.video_available is True
    assert with_video.shot_type == ShotType.DRIVE
    assert with_video.contact_frame == 3

    assert without_video.video_available is False
    assert without_video.shot_type is None
    assert without_video.wrist_speed_ms is None


def test_fuse_match_deliveries_only_returns_rows_for_the_requested_match():
    deliveries = pd.DataFrame([_delivery(match_id="m1"), _delivery(match_id="m2")])

    fused = fuse_match_deliveries(deliveries, "m1")

    assert len(fused) == 1
    assert fused[0].match_id == "m1"


def test_aggregate_player_match_stats_mirrors_player_reports_scoring_conventions():
    deliveries = pd.DataFrame(
        [
            _delivery(ball=1, striker="Bob", bowler="Carl", runs_batter=4, runs_total=4),
            _delivery(ball=2, striker="Bob", bowler="Carl", runs_batter=0, runs_total=1, extras_type="wides"),
            _delivery(ball=3, striker="Bob", bowler="Carl", runs_batter=1, runs_total=1, extras_type="noballs"),
            _delivery(ball=4, striker="Bob", bowler="Carl", runs_total=1, runs_extras=1, extras_type="legbyes"),
            _delivery(
                ball=5, striker="Dave", bowler="Carl", runs_batter=0, runs_total=0,
                wicket_player_out="Dave", wicket_kind="caught",
            ),
            _delivery(
                ball=6, striker="Dave", bowler="Carl", runs_batter=0, runs_total=0,
                wicket_player_out="Bob", wicket_kind="run out",
            ),
        ]
    )

    stats = aggregate_player_match_stats(fuse_match_deliveries(deliveries, "m1"), match_date="2026-01-01")
    bob = next(s for s in stats if s.player == "Bob")
    carl = next(s for s in stats if s.player == "Carl")

    assert bob.runs_scored == 5  # 4 + 0(wide, not faced) + 1(noball, faced) + 0
    assert bob.balls_faced == 3  # wide excluded, noball counts
    assert bob.batting_strike_rate == round(100 * 5 / 3, 2)

    assert carl.balls_bowled == 4  # wide and noball both excluded (illegal deliveries)
    assert carl.runs_conceded == 6  # 7 total - 1 legbye
    assert carl.wickets_taken == 1  # run out not credited
    assert carl.avg_wrist_speed_ms is None  # no CV producer populates this yet
    assert carl.illegal_bowling_action_flag is None


def test_aggregate_player_match_stats_averages_cv_biomechanics_only_over_video_rows():
    deliveries = pd.DataFrame(
        [
            _delivery(ball=1, bowler="Carl"),
            _delivery(ball=2, bowler="Carl"),
            _delivery(ball=3, bowler="Carl"),
        ]
    )
    fused = fuse_match_deliveries(deliveries, "m1")
    # Simulate a populated biomechanics extractor (none exists yet -- this stands in for
    # one) on two of the three deliveries; the third stays video-unavailable.
    fused = [
        replace(fused[0], video_available=True, wrist_speed_ms=30.0, bowling_elbow_extension_deg=10.0),
        replace(fused[1], video_available=True, wrist_speed_ms=34.0, bowling_elbow_extension_deg=18.0),
        fused[2],
    ]

    stats = aggregate_player_match_stats(fused, match_date="2026-01-01")
    carl = next(s for s in stats if s.player == "Carl")

    assert carl.avg_wrist_speed_ms == 32.0
    assert carl.max_wrist_speed_ms == 34.0
    assert carl.avg_bowling_elbow_extension_deg == 14.0
    assert carl.illegal_bowling_action_flag is True  # max 18deg > ICC's 15deg limit


def test_refresh_rolling_summary_breaks_same_date_ties_deterministically():
    history = [
        aggregate_player_match_stats(
            fuse_match_deliveries(pd.DataFrame([_delivery(match_id=mid, striker="Bob", runs_batter=runs)]), mid),
            match_date=date,
        )[0]
        for mid, date, runs in [("m1", "2026-01-05", 10), ("m2", "2026-01-05", 20), ("m3", "2026-01-01", 30)]
    ]

    results = [refresh_rolling_summary(history, "Bob", window_size=2) for _ in range(20)]

    assert all(r.total_runs == results[0].total_runs for r in results), "must be deterministic across repeated calls"
    assert results[0].total_runs == 30  # m2 (same date, higher match_id) + m1, not m3


def test_refresh_rolling_summary_returns_none_for_an_unknown_player():
    assert refresh_rolling_summary([], "Nobody", window_size=5) is None


def test_refresh_all_rolling_summaries_covers_every_player_with_history():
    deliveries = pd.DataFrame([_delivery(striker="Bob", bowler="Carl", runs_batter=6)])
    stats = aggregate_player_match_stats(fuse_match_deliveries(deliveries, "m1"), match_date="2026-01-01")

    summaries = refresh_all_rolling_summaries(stats, window_size=5)

    assert {s.player for s in summaries} == {"Bob", "Carl"}
