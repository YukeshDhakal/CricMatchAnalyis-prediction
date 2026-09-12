import json

import pytest

from ingestion.contracts import Source
from ingestion.video.manifest_adapter import ManifestClipAdapter
from .video_helpers import make_color_clip


def test_ingest_wraps_pre_split_clips(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    make_color_clip(clips_dir / "ball1.mp4", "red", duration=1, size="32x32", fps=10)
    make_color_clip(clips_dir / "ball2.mp4", "blue", duration=1, size="32x32", fps=10)

    manifest = {
        "ball1.mp4": {"innings": 1, "over": 0, "ball": 1},
        "ball2.mp4": {"innings": 1, "over": 0, "ball": 2},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    clips = ManifestClipAdapter().ingest("match_001", clips_dir, manifest_path)

    assert len(clips) == 2
    assert {c.delivery.ball for c in clips} == {1, 2}
    for clip in clips:
        assert clip.delivery.match_id == "match_001"
        assert clip.width == 32 and clip.height == 32
        assert clip.fps == pytest.approx(10.0, rel=0.05)
        assert clip.source == Source.USER_UPLOAD


def test_ingest_raises_on_missing_clip(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"missing.mp4": {"innings": 1, "over": 0, "ball": 1}}), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        ManifestClipAdapter().ingest("match_001", clips_dir, manifest_path)
