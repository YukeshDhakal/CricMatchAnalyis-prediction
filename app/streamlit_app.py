"""Live test console for Third Umpire.

Four tabs:
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
  4. Rating & Suggestions -- runs the real PRD stages 5-6 pipeline (fusion ->
     compute_baseline -> rate_player -> suggest_for_player) against whatever's
     in the warehouse. Technique always reads "not measured" (see rating/'s
     docstring -- no calibrated biomechanics extractor exists yet); notes come
     from the local Ollama model by default and fall back to the deterministic
     template writer visibly (note_source shown on every card), never silently.

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

from fusion.contracts import FusedDelivery, PlayerMatchStats  # noqa: E402
from fusion.pipeline import aggregate_player_match_stats, fuse_match_deliveries  # noqa: E402
from ingestion.pipeline import run_stats_ingestion  # noqa: E402
from ingestion.sources.cricsheet import CricsheetSource  # noqa: E402
from ingestion.storage import ParquetStore  # noqa: E402
from ingestion.video.manifest_adapter import ManifestClipAdapter  # noqa: E402
from player_reports.stats import batting_summary, bowling_summary, last_n_match_ids  # noqa: E402
from rating.contracts import Pillar, PlayerRole  # noqa: E402
from rating.engine import compute_baseline, rate_player  # noqa: E402
from rating.llm import (  # noqa: E402
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    OllamaClient,
    TemplateNoteWriter,
)
from rating.suggestions import flag_player, write_suggestions  # noqa: E402

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


@st.cache_data
def build_rating_inputs(matches: pd.DataFrame, deliveries: pd.DataFrame) -> tuple[list, list]:
    """Every match in the warehouse, fused (no video -- see `FusedDelivery.video_available`)
    and aggregated, once. This is the cohort `compute_baseline` pools over and the
    history `rate_player` windows -- cached so switching the player/role dropdowns
    below doesn't re-fuse the whole warehouse on every rerun.
    """
    match_dates = matches.set_index("match_id")["dates"].apply(lambda d: d[0] if len(d) else "")
    history: list[PlayerMatchStats] = []
    fused_all: list[FusedDelivery] = []
    for match_id, match_date in match_dates.items():
        fused = fuse_match_deliveries(deliveries, match_id)
        fused_all.extend(fused)
        history.extend(aggregate_player_match_stats(fused, match_date))
    return history, fused_all


def _pillar_rows(rating) -> pd.DataFrame:
    labels = {
        Pillar.TECHNIQUE: "Technique",
        Pillar.EXECUTION: "Execution",
        Pillar.GAME_IMPACT: "Game impact",
        Pillar.CONSISTENCY: "Consistency",
    }
    weights = rating.weights_used.as_dict()
    scores = {
        Pillar.TECHNIQUE: rating.pillars.technique,
        Pillar.EXECUTION: rating.pillars.execution,
        Pillar.GAME_IMPACT: rating.pillars.game_impact,
        Pillar.CONSISTENCY: rating.pillars.consistency,
    }
    rows = []
    for pillar in Pillar:
        measured = pillar not in rating.unmeasured_pillars
        rows.append(
            {
                "Pillar": labels[pillar],
                "Score (0-100, 50=baseline)": f"{scores[pillar]:.1f}" if measured else "not measured",
                "Weight used": f"{weights[pillar]:.0%}" if measured else "0% (redistributed)",
            }
        )
    return pd.DataFrame(rows).set_index("Pillar")


tab_overview, tab_player, tab_video, tab_rating = st.tabs(
    ["Overview", "Player Performance", "Video Pipeline Test", "Rating & Suggestions"]
)

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

with tab_rating:
    st.caption(
        "Runs the real stage 5-6 pipeline: fuse the warehouse's ball-by-ball rows "
        "(no video -- BroadcastFeedAdapter isn't implemented, so every FusedDelivery "
        "here has video_available=False), compute a cohort baseline, rate the "
        "selected player, then flag/rank/write their coaching suggestions. Technique "
        "always reads 'not measured' -- see rating/'s module docstring for why that's "
        "a deliberate weight-redistribution decision, not a bug."
    )

    matches, deliveries = load_warehouse()
    if matches.empty:
        st.warning("No warehouse data yet -- fetch some in the Overview tab first.")
    else:
        players = sorted(pd.concat([deliveries["striker"], deliveries["bowler"]]).unique())

        c1, c2, c3 = st.columns(3)
        player = c1.selectbox("Player", players, key="rating_player")
        role_label = c2.selectbox("Role", ["Top-order batter", "Death-overs bowler"])
        role = PlayerRole.TOP_ORDER_BATTER if role_label == "Top-order batter" else PlayerRole.DEATH_OVERS_BOWLER
        window_size = c3.slider("Rating window (last N matches)", 1, 10, 5, key="rating_window")

        st.subheader("Note writer")
        writer_choice = st.radio(
            "How coaching notes get phrased",
            ["Template only (deterministic, no network)", "Ollama (local LLM, falls back to template)"],
            horizontal=True,
        )
        if writer_choice.startswith("Ollama"):
            wc1, wc2 = st.columns(2)
            ollama_url = wc1.text_input("Ollama base URL", DEFAULT_OLLAMA_BASE_URL)
            ollama_model = wc2.text_input(
                "Model", DEFAULT_OLLAMA_MODEL,
                help="Default was chosen by measured citation/number fidelity against this repo's own "
                     "eval (scripts/eval_llm_notewriter.py), not by MMLU/GSM8K leaderboard rank -- see "
                     "README's 'Rating and coaching suggestions'.",
            )
            writer = OllamaClient(base_url=ollama_url, model=ollama_model)
        else:
            writer = TemplateNoteWriter()

        if st.button("Compute rating and suggestions"):
            with st.spinner("Fusing warehouse deliveries and computing the cohort baseline..."):
                history, fused_all = build_rating_inputs(matches, deliveries)
                baseline = compute_baseline(history, fused_all)

            if baseline is None:
                st.error("No matches in the warehouse to build a baseline from.")
            else:
                st.caption(f"Baseline: {baseline.source}")

                rating = rate_player(history, player, role, baseline, fused_all, window_size=window_size)
                if rating is None:
                    st.warning(f"No matches for {player} in the warehouse.")
                else:
                    st.subheader(f"{player} -- {role_label}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric(
                        "Composite rating",
                        f"{rating.composite:.1f} / 100" if rating.composite is not None else "insufficient data",
                    )
                    m2.metric("Matches in sample", rating.matches_in_sample)
                    m3.metric("Pillars measured", f"{4 - len(rating.unmeasured_pillars)} / 4")
                    if rating.unmeasured_pillars:
                        st.info(
                            f"Unmeasured: {', '.join(p.value for p in rating.unmeasured_pillars)} -- weight "
                            "redistributed across the rest. Not comparable to a future four-pillar rating "
                            "once Technique has a real producer."
                        )
                    st.table(_pillar_rows(rating))

                    # Same window rate_player used internally, for suggestions -- keeps the two
                    # views of "this player, recently" from silently disagreeing.
                    window_match_ids = set(last_n_match_ids(deliveries, matches, player, n=window_size))
                    window_fused = [d for d in fused_all if d.match_id in window_match_ids]

                    with st.spinner(f"Flagging deltas and writing notes ({writer_choice.split(' (')[0]})..."):
                        all_flags = flag_player(window_fused, player, baseline)
                        weakness_flags = [f for f in all_flags if f.is_concern]
                        strength_flags = [f for f in all_flags if not f.is_concern]
                        weaknesses = write_suggestions(weakness_flags, writer=writer, limit=5)
                        strengths = write_suggestions(strength_flags, writer=writer, limit=3)

                    if not all_flags:
                        st.info(
                            "No flags cleared the threshold for this player/window -- either performance is "
                            "close to baseline, or there weren't enough balls faced/bowled to flag "
                            "(see rating.suggestions.MIN_BALLS_FOR_FLAG)."
                        )

                    def _render_suggestion(s) -> None:
                        badge = "🤖 LLM" if s.note_source == "llm" else "📋 template"
                        with st.container(border=True):
                            st.markdown(f"**{s.rank}. {s.title}** &nbsp; `{badge}`")
                            st.write(s.body)
                            st.caption(
                                f"{s.flag.metric.value}"
                                + (f" ({s.flag.phase})" if s.flag.phase else "")
                                + f": {s.flag.actual:.2f} vs baseline {s.flag.baseline:.2f} "
                                f"({s.flag.relative_delta:+.0%}), {s.flag.sample_size} balls -- "
                                f"citations: {', '.join(s.citation_tokens()) or 'none'}"
                            )

                    st.subheader(f"Weaknesses -- coaching priorities ({len(weaknesses)})")
                    if not weaknesses:
                        st.caption("None cleared the threshold for this player/window.")
                    for s in weaknesses:
                        _render_suggestion(s)

                    st.subheader(f"Strengths -- keep doing ({len(strengths)})")
                    if not strengths:
                        st.caption("None cleared the threshold for this player/window.")
                    for s in strengths:
                        _render_suggestion(s)
