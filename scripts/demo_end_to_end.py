"""End-to-end demo: ingestion -> video engine, on real ball-by-ball data plus one clip.

What's real:
  - The ball-by-ball table (fetched from cricsheet.org, or read from the local warehouse
    if already fetched).
  - ManifestClipAdapter probing an actual video file and producing a real DeliveryClip.
  - load_frames actually decoding that file with OpenCV.
  - IouTracker and HeuristicEventSegmenter -- the real tracking/event-segmentation logic.

What's simulated, and why:
  - Detection and pose estimation use fixed, hand-built detections/poses instead of the
    real YoloDetector/KeypointRcnnPoseEstimator. There's no cricket footage or a
    fine-tuned ball/stumps model to run them on yet (see README "Known gaps") -- this
    demo exists to show the pipeline's own logic (tracking, event segmentation, shot
    classification) wired up and running correctly end-to-end, not to claim real
    computer vision happened on real footage.

Run: python scripts/demo_end_to_end.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# tests/ first, src/ last-inserted so it ends up checked first -- tests/ingestion/ would
# otherwise shadow the real src/ingestion package (same collision src/ingestion's own
# tests hit; see tests/__init__.py).
sys.path.insert(0, str(REPO_ROOT / "tests"))  # for the shared FakeDetector/FakePoseEstimator
sys.path.insert(0, str(REPO_ROOT / "src"))

from ingestion.sources.cricsheet import CricsheetSource  # noqa: E402
from ingestion.storage import ParquetStore  # noqa: E402
from ingestion.video.manifest_adapter import ManifestClipAdapter  # noqa: E402
from video_engine.contracts import BoundingBox, Detection, Keypoint, ObjectClass, PoseFrame  # noqa: E402
from video_engine.events.heuristic_segmenter import HeuristicEventSegmenter  # noqa: E402
from video_engine.pipeline import VideoEngine  # noqa: E402
from video_engine.tracking.iou_tracker import IouTracker  # noqa: E402
from fakes import FakeDetector, FakePoseEstimator  # noqa: E402

COMPETITION = "icc_mens_t20_world_cup_male"
DEMO_DIR = REPO_ROOT / "data" / "demo"


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def step_1_real_ball_by_ball() -> None:
    section("1. Ingestion: real ball-by-ball table (cricsheet.org)")
    warehouse = ParquetStore()
    matches_path = warehouse.root / "matches.parquet"
    if not matches_path.exists():
        print(f"No local warehouse yet -- fetching '{COMPETITION}' from cricsheet.org...")
        from ingestion.pipeline import run_stats_ingestion

        match_count, delivery_count = run_stats_ingestion(CricsheetSource(COMPETITION), warehouse)
        print(f"Ingested {match_count} matches, {delivery_count} deliveries.")
    else:
        print(f"Using existing warehouse at {warehouse.root}/")

    deliveries = warehouse.read_deliveries()
    wickets = deliveries[deliveries["wicket_player_out"].notna()]
    sample = wickets.iloc[0]
    print(f"\n{len(deliveries):,} deliveries loaded. Example wicket ball, straight from the real table:")
    print(
        f"  match {sample.match_id}, innings {sample.innings}, over {sample.over}, ball {sample.ball}: "
        f"{sample.bowler} to {sample.striker} -- {sample.wicket_kind} ({sample.phase})"
    )


def step_2_real_clip_ingestion() -> Path:
    section("2. Ingestion: a real video file, wrapped into a DeliveryClip")
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    clip_path = DEMO_DIR / "delivery_0001.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=green:s=128x128:d=1:r=10",
            "-pix_fmt", "yuv420p", str(clip_path),
        ],
        capture_output=True, check=True,
    )
    manifest_path = DEMO_DIR / "manifest.json"
    manifest_path.write_text(json.dumps({clip_path.name: {"innings": 1, "over": 12, "ball": 3}}), encoding="utf-8")

    clips = ManifestClipAdapter().ingest(match_id="1273712", clips_dir=DEMO_DIR, manifest_path=manifest_path)
    clip = clips[0]
    print(f"Real file on disk: {clip.video_path} ({clip.width}x{clip.height} @ {clip.fps:.0f}fps)")
    print(f"DeliveryClip: {clip.delivery} source={clip.source.value}")
    return clip


def step_3_video_engine_analysis(clip) -> None:
    section("3. Video engine: tracking + event segmentation on that clip")
    print("Detection/pose are simulated below (see module docstring) -- everything else is real.")

    # A ball approaching a batter over 3 frames (each overlapping the last, so IouTracker
    # links them into one continuous track rather than fragmenting), then a fast
    # rightward wrist swing after contact: a drive.
    detections = [
        Detection(0, ObjectClass.PLAYER, BoundingBox(40, 30, 90, 120), 0.95),
        Detection(1, ObjectClass.PLAYER, BoundingBox(41, 30, 91, 120), 0.95),
        Detection(2, ObjectClass.PLAYER, BoundingBox(42, 30, 92, 120), 0.95),
        Detection(0, ObjectClass.BALL, BoundingBox(0, 60, 20, 80), 0.9),
        Detection(1, ObjectClass.BALL, BoundingBox(8, 60, 28, 80), 0.9),
        Detection(2, ObjectClass.BALL, BoundingBox(16, 60, 36, 80), 0.9),
    ]
    poses = [
        PoseFrame(2, track_id=0, keypoints=[Keypoint("right_wrist", 60.0, 70.0, 0.9)]),
        PoseFrame(3, track_id=0, keypoints=[Keypoint("right_wrist", 90.0, 68.0, 0.9)]),
    ]

    engine = VideoEngine(
        detector=FakeDetector(detections),
        tracker=IouTracker(iou_threshold=0.2),
        pose_estimator=FakePoseEstimator(poses),
        event_segmenter=HeuristicEventSegmenter(),
    )

    analysis = engine.analyze(clip)

    print(f"\nDeliveryAnalysis for {analysis.delivery}:")
    print(f"  tracks:        {[(t.track_id, t.obj_class.value, len(t.detections)) for t in analysis.tracks]}")
    print(f"  poses:         {len(analysis.poses)} frame(s)")
    print(f"  release_frame: {analysis.event.release_frame}")
    print(f"  contact_frame: {analysis.event.contact_frame}")
    print(f"  shot_type:     {analysis.event.shot_type.value}")


if __name__ == "__main__":
    step_1_real_ball_by_ball()
    clip = step_2_real_clip_ingestion()
    step_3_video_engine_analysis(clip)
    section("Done")
    print("Ingestion -> DeliveryClip -> VideoEngine.analyze() -> DeliveryAnalysis, wired end-to-end.")
