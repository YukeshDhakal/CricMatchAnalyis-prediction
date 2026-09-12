"""Live test console for Third Umpire.

Three tabs:
  1. Overview        -- warehouse stats, and a live refresh from cricsheet.org.
  2. Player Performance -- last-N-match batting/bowling summaries from the real
     ingested ball-by-ball table. Stats only: there's no trained CV model or
     rated-footage yet, so this covers the PRD's "Execution" pillar, not a full
     composite rating (that needs video-derived "Technique" signals too).
  3. Video Pipeline Test -- upload any video and run it through the *real*
     detection/tracking/pose pipeline (pretrained YOLO + Keypoint R-CNN --
     genuine inference, not simulated). There's no fine-tuned ball/stumps model
     yet, so shot classification stays "unknown" on real footage regardless of
     how good the detections are; that's an honest, documented gap, not a bug.

Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from ingestion.pipeline import run_stats_ingestion  # noqa: E402
from ingestion.sources.cricsheet import CricsheetSource  # noqa: E402
from ingestion.storage import ParquetStore  # noqa: E402
from ingestion.video.manifest_adapter import ManifestClipAdapter  # noqa: E402
from player_reports.stats import batting_summary, bowling_summary, last_n_match_ids  # noqa: E402

st.set_page_config(page_title="Third Umpire -- Live Test Console", layout="wide")
st.title("🏏 Third Umpire -- Live Test Console")


def _as_display_table(stats: dict) -> pd.DataFrame:
    """batting_summary/bowling_summary mix strings ("overs": "0.5") with numbers in
    one dict -- Arrow can't infer a single column type for that, so stringify
    everything before handing it to st.table."""
    return pd.DataFrame({"value": [str(v) for v in stats.values()]}, index=list(stats.keys()))


@st.cache_resource
def get_pipeline_components() -> dict:
    """Cached so model weights load once per server process, not once per click."""
    from video_engine.detection.yolo_detector import YoloDetector
    from video_engine.events.heuristic_segmenter import HeuristicEventSegmenter
    from video_engine.pose.keypoint_pose import KeypointRcnnPoseEstimator
    from video_engine.tracking.bytetrack_tracker import ByteTrackTracker

    return {
        "detector": YoloDetector(),
        "tracker": ByteTrackTracker(),
        "pose_estimator": KeypointRcnnPoseEstimator(),
        "event_segmenter": HeuristicEventSegmenter(),
    }


def run_pipeline_with_timing(clip):
    """Mirrors VideoEngine.analyze() stage-by-stage (rather than calling it as one
    black box) so each stage's wall-clock time is measurable -- that's the actual
    "model performance" signal, alongside the detections/frames themselves."""
    from video_engine.contracts import DeliveryAnalysis
    from video_engine.io.clip_loader import load_frames

    components = get_pipeline_components()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    frames = load_frames(clip)
    timings["load_frames"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    detections = components["detector"].detect(frames)
    timings["detection"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tracks = components["tracker"].track(detections)
    timings["tracking"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    poses = components["pose_estimator"].estimate(frames, tracks)
    timings["pose_estimation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    event = components["event_segmenter"].segment(tracks, poses)
    timings["event_segmentation"] = time.perf_counter() - t0

    analysis = DeliveryAnalysis(delivery=clip.delivery, tracks=tracks, poses=poses, event=event)
    return analysis, frames, detections, timings


def draw_overlay(frame: np.ndarray, detections_at_frame: list, poses_at_frame: list) -> np.ndarray:
    """Draws real detection boxes (with confidence) and pose keypoints onto a real
    frame -- the actual visual check of whether the model is seeing anything sensible,
    not just a count in a table."""
    img = frame.copy()
    for d in detections_at_frame:
        x1, y1, x2, y2 = (int(v) for v in (d.box.x1, d.box.y1, d.box.x2, d.box.y2))
        color = (255, 80, 80) if d.obj_class.value == "player" else (80, 220, 80)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img, f"{d.obj_class.value} {d.confidence:.2f}", (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
    for pose in poses_at_frame:
        for kp in pose.keypoints:
            if kp.confidence > 0.3:
                cv2.circle(img, (int(kp.x), int(kp.y)), 3, (255, 230, 0), -1)
    return img


@st.cache_data
def load_warehouse() -> tuple[pd.DataFrame, pd.DataFrame]:
    store = ParquetStore()
    if not (store.root / "matches.parquet").exists():
        return pd.DataFrame(), pd.DataFrame()
    return store.read_matches(), store.read_deliveries()


tab_overview, tab_player, tab_video = st.tabs(["Overview", "Player Performance", "Video Pipeline Test"])

with tab_overview:
    matches, deliveries = load_warehouse()

    if matches.empty:
        st.warning("No warehouse data yet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matches", f"{len(matches):,}")
        c2.metric("Deliveries", f"{len(deliveries):,}")
        c3.metric("Competitions", matches["competition"].nunique())
        c4.metric("Players seen", pd.concat([deliveries["striker"], deliveries["bowler"]]).nunique())
        st.dataframe(matches[["match_id", "competition", "teams", "venue", "outcome_winner"]].tail(10))

    st.divider()
    st.subheader("Refresh from cricsheet.org")
    st.caption(
        "Every fetch checks the live Last-Modified header against what's cached, "
        "and only re-downloads if cricsheet.org's archive has actually changed."
    )
    competition = st.text_input("Competition slug", "icc_mens_t20_world_cup_male")
    if st.button("Check for fresh data"):
        with st.spinner(f"Checking cricsheet.org for '{competition}'..."):
            store = ParquetStore()
            match_count, delivery_count = run_stats_ingestion(CricsheetSource(competition), store)
        st.success(f"Warehouse now has {match_count} matches, {delivery_count} deliveries.")
        st.cache_data.clear()

with tab_player:
    matches, deliveries = load_warehouse()
    if matches.empty:
        st.warning("No warehouse data yet -- fetch some in the Overview tab first.")
    else:
        players = sorted(pd.concat([deliveries["striker"], deliveries["bowler"]]).unique())
        player = st.selectbox("Player", players)
        n = st.slider("Last N matches", 1, 10, 5)

        match_ids = last_n_match_ids(deliveries, matches, player, n=n)
        st.caption(f"Matches considered (most recent first): {', '.join(match_ids) or 'none found'}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Batting")
            st.table(_as_display_table(batting_summary(deliveries, player, match_ids)))
        with col2:
            st.subheader("Bowling")
            st.table(_as_display_table(bowling_summary(deliveries, player, match_ids)))

        st.subheader("Runs per match")
        per_match_runs = (
            deliveries[(deliveries["striker"] == player) & (deliveries["match_id"].isin(match_ids))]
            .groupby("match_id")["runs_batter"]
            .sum()
            .reindex(match_ids)
        )
        st.bar_chart(per_match_runs)

with tab_video:
    st.caption(
        "Runs the *real* pipeline -- pretrained YOLO for player detection, ByteTrack "
        "for tracking, Keypoint R-CNN for pose. Genuine inference, not simulated. "
        "There's no fine-tuned ball/stumps model yet, so shot classification will "
        "read 'unknown' on real footage regardless of detection quality -- that's a "
        "documented gap (see README), not a failure of this test."
    )

    uploaded = st.file_uploader("Upload a video clip", type=["mp4", "mov", "avi", "mkv"])
    c1, c2, c3, c4 = st.columns(4)
    match_id = c1.text_input("Match ID", "manual-test")
    innings = c2.number_input("Innings", 1, 2, 1)
    over = c3.number_input("Over", 0, 19, 0)
    ball = c4.number_input("Ball", 1, 9, 1)

    if uploaded is not None:
        dest_dir = REPO_ROOT / "data" / "uploads" / "videos"
        dest_dir.mkdir(parents=True, exist_ok=True)
        video_path = dest_dir / uploaded.name
        video_path.write_bytes(uploaded.getbuffer())
        st.video(str(video_path))

        if st.button("Run real CV pipeline"):
            manifest_path = dest_dir / f".{video_path.stem}.manifest.json"
            manifest_path.write_text(
                json.dumps({video_path.name: {"innings": int(innings), "over": int(over), "ball": int(ball)}}),
                encoding="utf-8",
            )
            clip = ManifestClipAdapter().ingest(match_id, dest_dir, manifest_path)[0]
            st.write(f"DeliveryClip: {clip.width}x{clip.height} @ {clip.fps:.1f}fps, source={clip.source.value}")

            with st.spinner("Loading models (first run downloads pretrained weights) and running inference..."):
                analysis, frames, detections, timings = run_pipeline_with_timing(clip)

            st.subheader("Model performance")
            n_frames = len(frames)
            inference_time = timings["detection"] + timings["tracking"] + timings["pose_estimation"]
            perf_cols = st.columns(5)
            perf_cols[0].metric("Frames processed", n_frames)
            perf_cols[1].metric("Detection", f"{timings['detection']:.2f}s")
            perf_cols[2].metric("Tracking", f"{timings['tracking']:.3f}s")
            perf_cols[3].metric("Pose estimation", f"{timings['pose_estimation']:.2f}s")
            perf_cols[4].metric(
                "Effective FPS", f"{n_frames / inference_time:.1f}" if inference_time > 0 else "n/a"
            )
            st.bar_chart(pd.Series(timings, name="seconds"))

            st.subheader("Results")
            player_tracks = [t for t in analysis.tracks if t.obj_class.value == "player"]
            c1, c2, c3 = st.columns(3)
            c1.metric("People tracked", len(player_tracks))
            c2.metric("Pose frames", len(analysis.poses))
            c3.metric("Shot classification", analysis.event.shot_type.value)
            st.write("Event detail:", analysis.event)

            if detections:
                st.subheader("Detection confidence")
                det_df = pd.DataFrame(
                    [{"frame": d.frame_index, "class": d.obj_class.value, "confidence": d.confidence} for d in detections]
                )
                st.dataframe(det_df.groupby("class")["confidence"].describe())

                st.subheader("Sample frames (real detections drawn on real frames)")
                sample_frame_indices = sorted({d.frame_index for d in detections})
                step = max(1, len(sample_frame_indices) // 4)
                for frame_index in sample_frame_indices[::step][:4]:
                    dets_here = [d for d in detections if d.frame_index == frame_index]
                    poses_here = [p for p in analysis.poses if p.frame_index == frame_index]
                    overlay = draw_overlay(frames[frame_index], dets_here, poses_here)
                    st.image(overlay, caption=f"frame {frame_index}: {len(dets_here)} detection(s)")

                st.dataframe(
                    pd.DataFrame(
                        [{"track_id": t.track_id, "class": t.obj_class.value, "frames_seen": len(t.detections)} for t in analysis.tracks]
                    )
                )
            else:
                st.info(
                    "No people detected in this clip. Either there aren't visible players in frame, "
                    "or the footage/resolution doesn't suit the general-purpose pretrained model. "
                    f"Still processed {n_frames} frame(s) in {inference_time:.2f}s -- that's the honest result."
                )
