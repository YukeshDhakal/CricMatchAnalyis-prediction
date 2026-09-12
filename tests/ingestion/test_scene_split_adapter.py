from ingestion.contracts import Delivery
from ingestion.video.scene_split_adapter import SceneSplitAdapter, detect_scene_cuts
from .video_helpers import concat_clips, make_color_clip


def _delivery(ball: int) -> Delivery:
    return Delivery(
        match_id="match_001",
        innings=1,
        over=0,
        ball=ball,
        batting_team="A",
        bowling_team="B",
        striker="P1",
        non_striker="P2",
        bowler="P3",
        runs_batter=0,
        runs_extras=0,
        runs_total=0,
        extras_type=None,
        wicket_player_out=None,
        wicket_kind=None,
        phase="powerplay",
    )


def test_falls_back_to_uniform_split_when_no_cuts_detected(tmp_path):
    # A single solid-colour clip has no scene cuts, so this must fall back to a
    # uniform split rather than producing the wrong number of segments.
    video_path = tmp_path / "innings.mp4"
    make_color_clip(video_path, "green", duration=2.0, size="32x32", fps=10)

    deliveries = [_delivery(b) for b in range(1, 5)]
    clips = SceneSplitAdapter().ingest("match_001", video_path, deliveries, tmp_path / "out")

    assert len(clips) == 4
    assert [c.delivery.ball for c in clips] == [1, 2, 3, 4]
    for clip in clips:
        assert clip.width == 32 and clip.height == 32


def test_ingest_returns_empty_list_for_no_deliveries(tmp_path):
    video_path = tmp_path / "innings.mp4"
    make_color_clip(video_path, "green", duration=1.0, size="32x32", fps=10)

    assert SceneSplitAdapter().ingest("match_001", video_path, [], tmp_path / "out") == []


def test_detect_scene_cuts_finds_hard_cuts(tmp_path):
    segments = [
        make_color_clip(tmp_path / "red.mp4", "red", duration=1.0, size="32x32", fps=10),
        make_color_clip(tmp_path / "blue.mp4", "blue", duration=1.0, size="32x32", fps=10),
        make_color_clip(tmp_path / "green.mp4", "green", duration=1.0, size="32x32", fps=10),
    ]
    combined = concat_clips(segments, tmp_path / "combined.mp4")

    cuts = detect_scene_cuts(str(combined), threshold=0.3)

    assert len(cuts) >= 2  # red->blue and blue->green are both hard cuts
