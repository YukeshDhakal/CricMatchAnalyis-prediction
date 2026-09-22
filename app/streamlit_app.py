"""Live test console for Third Umpire.

Four tabs:
  1. Overview        -- warehouse stats, and a live refresh from cricsheet.org.
  2. Player Performance -- last-N-match batting/bowling summaries from the real
     ingested ball-by-ball table. Stats only: there's no trained CV model or
     rated-footage yet, so this covers the PRD's "Execution" pillar, not a full
     composite rating (that needs video-derived "Technique" signals too).
  3. Video Pipeline Test -- upload any video and run it through the *real*
     detection/tracking/pose pipeline (pretrained YOLO + Keypoint R-CNN --
     genuine inference, not simulated). A fine-tuned ball+stumps YOLOv8n
     checkpoint now loads from `weights/ball_stumps_n.pt` when present (see
     README's "Ball and stumps detection"); without it, shot classification
     falls back to "unknown", which stays an honest, documented gap rather
     than a bug.
  4. Rating & Suggestions -- runs the real PRD stages 5-6 pipeline against the
     warehouse, scoped: `rating.pipeline` picks a comparable cohort and the
     player's last-N window first, reads only those rows, pools the baseline
     vectorised, and fuses only the window. It does *not* fuse the warehouse --
     that took 557s at 8,026 matches and is what `rating/pipeline.py`'s
     docstring explains the ordering of. Technique always reads "not measured"
     (see rating/'s docstring -- no calibrated biomechanics extractor exists
     yet); notes come from the local Ollama model by default and fall back to
     the deterministic template writer visibly (note_source shown on every
     card), never silently.

Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import contextlib
import html as _html
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from ingestion.fetch import download_url  # noqa: E402
from ingestion.pipeline import run_stats_ingestion  # noqa: E402
from ingestion.sources.cricsheet import CricsheetSource  # noqa: E402
from ingestion.storage import ParquetStore  # noqa: E402
from ingestion.video.manifest_adapter import ManifestClipAdapter  # noqa: E402
from player_reports.stats import batting_summary, bowling_summary, last_n_match_ids  # noqa: E402
from rating.cohort import CohortSelection  # noqa: E402
from rating.contracts import METRIC_SPECS, Pillar, PlayerRole  # noqa: E402
from rating.engine import rate_player  # noqa: E402
from rating.llm import (  # noqa: E402
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    OllamaClient,
    TemplateNoteWriter,
)
from rating.pipeline import cohort_baseline, prepare_rating, resolve_scope  # noqa: E402
from rating.suggestions import flag_player, write_suggestions  # noqa: E402

st.set_page_config(page_title="Third Umpire -- Live Test Console", layout="wide")

# Design pass covering all four tabs (see the design canvas linked from the project
# README/PR description). Tokens here are the *same* oklch values as that canvas, not
# a re-derivation -- .streamlit/config.toml carries hex approximations of the same
# palette for Streamlit's own native chrome (buttons, tabs, sliders, dataframes), which
# can't take oklch. Native widgets (st.dataframe, st.table, st.bar_chart, st.image,
# st.file_uploader) stay native -- fighting their internals with brittle CSS selectors
# isn't worth it -- but every section gets wrapped in the same `tu_card` frame via the
# open-div/close-div markdown pattern below, so the whole console reads as one system
# instead of "one designed tab + three bare ones."
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
  --tu-bg: oklch(0.16 0.012 262);
  --tu-surface: oklch(0.20 0.013 262);
  --tu-surface-2: oklch(0.24 0.014 262);
  --tu-border: oklch(0.28 0.011 262);
  --tu-border-soft: oklch(0.26 0.011 262);
  --tu-text: oklch(0.97 0.004 262);
  --tu-text-dim: oklch(0.85 0.006 262);
  --tu-text-muted: oklch(0.55 0.008 262);
  --tu-text-faint: oklch(0.48 0.008 262);
  --tu-accent: #c9a24b;
  --tu-concern: oklch(0.62 0.14 25);
  --tu-concern-text: oklch(0.68 0.13 25);
  --tu-strength: oklch(0.68 0.12 150);
  --tu-strength-text: oklch(0.72 0.11 150);
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; }
h1, h2, h3, .tu-display { font-family: 'Space Grotesk', 'Segoe UI', sans-serif !important; }
code, .tu-mono, [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', 'Courier New', monospace !important; }

/* Bespoke building blocks used by the Rating & Suggestions tab's card renderers below.
   Everything else in the app still uses plain Streamlit widgets + the native theme. */
.tu-card {
  background: var(--tu-surface);
  border: 1px solid var(--tu-border);
  border-radius: 12px;
  padding: 20px 24px;
  margin-bottom: 14px;
}
.tu-eyebrow {
  font-size: 11.5px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--tu-text-muted); display: flex; justify-content: space-between; align-items: baseline;
  margin-bottom: 12px;
}
.tu-pill {
  display: inline-block; font-size: 12.5px; font-weight: 500; color: var(--tu-text-dim);
  background: var(--tu-surface-2); border: 1px solid var(--tu-border); border-radius: 99px;
  padding: 3px 10px;
}
.tu-pillar-row { display: grid; grid-template-columns: 118px 1fr 42px; align-items: center; gap: 14px; margin-bottom: 12px; }
.tu-pillar-label { font-size: 13px; font-weight: 500; color: var(--tu-text-dim); }
.tu-pillar-label-dim { color: var(--tu-text-muted); }
.tu-pillar-value { font-size: 13px; font-weight: 600; color: var(--tu-text-dim); text-align: right; }
.tu-pillar-value-dim { color: var(--tu-text-muted); font-weight: 400; }
.tu-meter { position: relative; height: 9px; border-radius: 99px; background: var(--tu-border-soft); }
.tu-meter-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 99px; background: var(--tu-accent); }
.tu-meter-tick { position: absolute; left: 50%; top: -3px; bottom: -3px; width: 2px; background: color-mix(in oklab, var(--tu-text-dim) 50%, transparent); }
.tu-meter-hatch {
  background: repeating-linear-gradient(135deg, var(--tu-border) 0 6px, var(--tu-border-soft) 6px 12px);
}
.tu-suggestion-card { padding: 18px 20px; }
.tu-suggestion-title { font-size: 14.5px; font-weight: 600; color: var(--tu-text); line-height: 1.35; }
.tu-suggestion-body { font-size: 13px; line-height: 1.55; color: var(--tu-text-dim); margin-bottom: 12px; }
.tu-rank {
  font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600; color: var(--tu-bg);
  width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; margin-top: 1px;
}
.tu-badge-llm, .tu-badge-template {
  font-size: 10.5px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
  padding: 3px 8px; border-radius: 5px; white-space: nowrap; flex-shrink: 0;
}
.tu-badge-llm { color: oklch(0.7 0.1 85); background: oklch(0.26 0.03 85 / 0.5); border: 1px solid oklch(0.4 0.06 85 / 0.6); }
.tu-badge-template { color: var(--tu-text-muted); background: var(--tu-surface-2); border: 1px solid var(--tu-border); }
.tu-chip-row { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 6px; }
.tu-chip {
  font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; font-weight: 600; color: var(--tu-text-dim);
  border: 1px solid var(--tu-border); padding: 4px 9px; border-radius: 6px;
}
.tu-chip-novideo { font-weight: 500; color: var(--tu-text-muted); border: 1px dashed var(--tu-border); }
.tu-novideo-note { font-size: 11.5px; color: var(--tu-text-muted); margin-bottom: 4px; }

/* Overview / Player Performance / Video Pipeline Test: stat tiles + the card frame
   that wraps native widgets (dataframes, charts, images, uploaders). */
.tu-stat-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin-bottom: 14px; }
.tu-stat-tile { background: var(--tu-surface); border: 1px solid var(--tu-border); border-radius: 12px; padding: 16px 18px; }
.tu-stat-label { font-size: 11.5px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--tu-text-muted); margin-bottom: 8px; }
.tu-stat-value { font-size: 26px; font-weight: 600; color: var(--tu-text); line-height: 1; }

/* `tu_card()` wraps native widgets in a real st.container(border=True) rather than a
   string-HTML div (that was tried first and silently didn't nest -- see tu_card's
   docstring). Streamlit gives every vertical block the same data-testid
   ("stVerticalBlock") whether it's bordered or not, so the only thing that actually
   distinguishes a border=True container in this Streamlit build (1.63.0) is the
   emotion-cache class below, which IS version-pinned and will stop matching on a
   Streamlit upgrade. That's an accepted, graceful failure mode: if the selector goes
   stale, these containers just fall back to Streamlit's own native bordered-container
   look (still dark-themed via .streamlit/config.toml's [theme] block) rather than
   rendering visibly broken -- re-derive the class from a running app (inspect a
   `st.container(border=True)` element's className) if this drifts after an upgrade.
*/
div.st-emotion-cache-n66fta[data-testid="stVerticalBlock"] {
  background: var(--tu-surface) !important;
  border-color: var(--tu-border) !important;
  border-radius: 12px !important;
  padding: 18px 22px !important;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

st.title("🏏 Third Umpire -- Live Test Console")


@contextlib.contextmanager
def tu_card(eyebrow: str | None = None, right: str | None = None):
    """Wrap a section of native Streamlit widgets (dataframe, chart, image, uploader --
    anything that can't be authored as raw HTML) in the same visual frame the
    Rating & Suggestions tab's bespoke `.tu-card` divs use.

    Uses a real `st.container(border=True)` rather than an unclosed
    `<div class="tu-card">` opened in one st.markdown call and closed in a later one:
    that seemed like it should work (Streamlit widgets render as DOM siblings in call
    order) but doesn't -- each `unsafe_allow_html` string is parsed into its own
    isolated fragment before insertion, so the browser auto-closes the unclosed div at
    the end of *that* fragment instead of at the later close call, leaving an empty
    box followed by unstyled widgets. A real container nests correctly; THEME_CSS's
    emotion-class rule (see the comment above `.st-emotion-cache-n66fta` there) gives
    it the same look."""
    with st.container(border=True):
        if eyebrow:
            right_html = f"<span>{_html.escape(right)}</span>" if right else ""
            st.markdown(
                f'<div class="tu-eyebrow"><span>{_html.escape(eyebrow)}</span>{right_html}</div>',
                unsafe_allow_html=True,
            )
        yield


def _stat_tile_html(label: str, value: str) -> str:
    return (
        '<div class="tu-stat-tile">'
        f'<div class="tu-stat-label">{_html.escape(label)}</div>'
        f'<div class="tu-stat-value tu-mono">{_html.escape(str(value))}</div>'
        "</div>"
    )


def _stat_row(items: list[tuple[str, str]]) -> None:
    """A row of stat tiles in one HTML block -- used everywhere the old code used a
    row of `st.metric` columns (Overview's warehouse counts, Video Pipeline Test's
    model-performance and results rows)."""
    tiles = "".join(_stat_tile_html(label, value) for label, value in items)
    st.markdown(f'<div class="tu-stat-row">{tiles}</div>', unsafe_allow_html=True)


def _render_pitch_geometry(event) -> None:
    """Show line and length only when the evidence supports it, and say so when it doesn't.

    This tab is where a number is most likely to be believed, because it sits next to real
    frames with real boxes drawn on them. So the rule here is the strict one: anything
    below `GeometryConfidence.MEDIUM` renders as "Insufficient data" plus the reason, and
    never as a metre figure with a caveat underneath that a reader can skip. That matches
    how `ShotType.UNKNOWN` already behaves rather than inventing a second convention.

    Even at MEDIUM the figures are labelled an estimate and the line is labelled as the
    less reliable of the two, because the calibration rectangle is ~88:1 and is poorly
    conditioned across the pitch -- see `video_engine.calibration`.
    """
    from video_engine.contracts import GeometryConfidence

    confidence = getattr(event, "pitch_confidence", GeometryConfidence.NONE)
    point = getattr(event, "pitch_point", None)
    trustworthy = point is not None and confidence in (
        GeometryConfidence.MEDIUM,
        GeometryConfidence.HIGH,
    )

    if not trustworthy:
        reason = getattr(event, "pitch_notes", "") or "No pitch geometry was produced for this clip."
        st.markdown(
            f'<div class="tu-display" style="font-size:15px;font-weight:600;color:var(--tu-text);">'
            f"Insufficient data</div>"
            f'<div style="color:var(--tu-muted);font-size:13px;margin-top:4px;">'
            f"{_html.escape(reason)}</div>",
            unsafe_allow_html=True,
        )
        return

    _stat_row(
        [
            ("Length band", event.pitch_length.value.replace("_", " ")),
            ("Length (m from striker's stumps)", f"{point.length_m:.2f}"),
            ("Line (m from middle stump)", f"{point.line_m:+.2f}"),
            ("Confidence", confidence.value),
        ]
    )
    st.markdown(
        f'<div style="color:var(--tu-muted);font-size:13px;margin-top:6px;">'
        f"Single-camera estimate, not a measurement. Line is less reliable than length. "
        f"The sign of the line is distance from the middle stump and is <em>not</em> "
        f"labelled off or leg -- that needs the striker's handedness, which this project "
        f"has no source for. {_html.escape(getattr(event, 'pitch_notes', '') or '')}</div>",
        unsafe_allow_html=True,
    )


def _section_title_html(text: str) -> str:
    return (
        f'<div class="tu-display" style="font-size:16px;font-weight:600;'
        f'color:var(--tu-text);margin:6px 0 4px;">{_html.escape(text)}</div>'
    )


def _as_display_table(stats: dict) -> pd.DataFrame:
    """batting_summary/bowling_summary mix strings ("overs": "0.5") with numbers in
    one dict -- Arrow can't infer a single column type for that, so stringify
    everything before handing it to st.table."""
    return pd.DataFrame({"value": [str(v) for v in stats.values()]}, index=list(stats.keys()))


# `weights/` is gitignored -- this file is a local training artifact (Roboflow's CC BY 4.0
# "Cricket Dataset", ball+stump, trained via the Colab notebook in this project's history),
# not something the repo ships. THIRD_UMPIRE_BALL_STUMPS_WEIGHTS overrides the path; if
# neither exists, YoloDetector falls back to player-only detection (see its docstring).
_DEFAULT_BALL_STUMPS_WEIGHTS = REPO_ROOT / "weights" / "ball_stumps_n.pt"


def _ball_stumps_weights_path() -> str | None:
    override = os.environ.get("THIRD_UMPIRE_BALL_STUMPS_WEIGHTS")
    if override:
        return override
    return str(_DEFAULT_BALL_STUMPS_WEIGHTS) if _DEFAULT_BALL_STUMPS_WEIGHTS.exists() else None


@st.cache_resource
def get_pipeline_components() -> dict:
    """Cached so model weights load once per server process, not once per click."""
    from video_engine.detection.yolo_detector import YoloDetector
    from video_engine.events.heuristic_segmenter import HeuristicEventSegmenter
    from video_engine.pose.keypoint_pose import KeypointRcnnPoseEstimator
    from video_engine.tracking.bytetrack_tracker import ByteTrackTracker

    return {
        "detector": YoloDetector(ball_stumps_weights=_ball_stumps_weights_path()),
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


@st.cache_data(hash_funcs={CohortSelection: lambda c: c.match_ids})
def cached_cohort_baseline(cohort: CohortSelection):
    """The pooled baseline for one cohort, cached on the cohort's match ids.

    Two things are deliberate here.

    *It takes a `CohortSelection`, not a DataFrame.* The previous version of this cache
    took `matches` and `deliveries` frames, and Streamlit warned that its default
    hashing had failed and it was "falling back to pickling" them -- which meant
    pickling the entire ball-by-ball table to compute a cache key on every widget
    interaction. At 1.8M rows that is a substantial cost incurred *before* the cache can
    tell you it has the answer already. Keying on a tuple of match-id strings is cheap
    and is a complete description of what the baseline depends on.

    *It reads its own rows.* The cohort's deliveries are fetched inside, through the
    store's predicate pushdown, so no caller has to hold the whole warehouse in memory
    to ask for a baseline over part of it.

    The baseline depends only on the cohort, never on which player is selected, so
    switching players reuses it.
    """
    return cohort_baseline(ParquetStore(), cohort)


_PILLAR_LABELS = {
    Pillar.TECHNIQUE: "Technique",
    Pillar.EXECUTION: "Execution",
    Pillar.GAME_IMPACT: "Game Impact",
    Pillar.CONSISTENCY: "Consistency",
}


def _identity_card_html(player: str, role_label: str) -> str:
    """Name + role, styled like the design canvas's player-identity row."""
    return (
        '<div style="display:flex;align-items:baseline;gap:14px;margin:4px 0 18px;">'
        f'<div class="tu-display" style="font-size:26px;font-weight:700;color:var(--tu-text);">'
        f"{_html.escape(player)}</div>"
        f'<div class="tu-pill">{_html.escape(role_label)}</div>'
        "</div>"
    )


def _composite_card_html(rating) -> str:
    """The composite score, with the unmeasured-pillar disclosure surfaced right next
    to it -- never buried in a table row, per the PRD's redistribution caveat (see
    `PlayerRating`'s docstring: a composite without its unmeasured pillars visible
    is not something any caller should show or persist)."""
    if rating.composite is None or rating.insufficient_data:
        number_html = (
            '<div class="tu-mono" style="font-size:20px;font-weight:600;'
            'color:var(--tu-text-muted);margin-bottom:16px;">insufficient data</div>'
        )
    else:
        delta = rating.composite - 50
        sign = "+" if delta >= 0 else ""
        if delta > 0:
            delta_color = "var(--tu-strength-text)"
        elif delta < 0:
            delta_color = "var(--tu-concern-text)"
        else:
            delta_color = "var(--tu-text-muted)"
        number_html = (
            '<div style="display:flex;align-items:flex-end;gap:8px;margin-bottom:6px;">'
            f'<div class="tu-mono" style="font-size:52px;font-weight:600;line-height:1;'
            f'color:var(--tu-text);">{rating.composite:.0f}</div>'
            '<div class="tu-mono" style="font-size:16px;color:var(--tu-text-muted);'
            'padding-bottom:5px;">/100</div>'
            "</div>"
            f'<div class="tu-mono" style="font-size:12.5px;font-weight:600;'
            f'color:{delta_color};margin-bottom:16px;">{sign}{delta:.0f} vs baseline (50)</div>'
        )

    disclosure_html = ""
    if rating.unmeasured_pillars:
        names = ", ".join(
            _PILLAR_LABELS[p] for p in sorted(rating.unmeasured_pillars, key=lambda p: p.value)
        )
        disclosure_html = (
            '<div style="background:var(--tu-surface-2);border:1px solid var(--tu-border);'
            'border-radius:8px;padding:10px 12px;">'
            f'<div style="font-size:12.5px;font-weight:600;color:var(--tu-text-dim);">'
            f"{_html.escape(names)} &mdash; not measured</div>"
            '<div style="font-size:11.5px;color:var(--tu-text-muted);margin-top:2px;line-height:1.4;">'
            "Weight redistributed across the remaining pillars &mdash; not comparable to a "
            "future full-pillar composite.</div>"
            "</div>"
        )

    return (
        '<div class="tu-card">'
        f'<div class="tu-eyebrow"><span>Composite Rating</span>'
        f"<span>{rating.matches_in_sample} match(es)</span></div>"
        f"{number_html}{disclosure_html}"
        "</div>"
    )


def _pillar_bars_html(rating) -> str:
    weights = rating.weights_used.as_dict()
    scores = {
        Pillar.TECHNIQUE: rating.pillars.technique,
        Pillar.EXECUTION: rating.pillars.execution,
        Pillar.GAME_IMPACT: rating.pillars.game_impact,
        Pillar.CONSISTENCY: rating.pillars.consistency,
    }
    rows = []
    for pillar in Pillar:
        label = _PILLAR_LABELS[pillar]
        if pillar in rating.unmeasured_pillars:
            rows.append(
                '<div class="tu-pillar-row">'
                f'<div class="tu-pillar-label tu-pillar-label-dim">{label}</div>'
                '<div class="tu-meter tu-meter-hatch"></div>'
                '<div class="tu-pillar-value tu-pillar-value-dim tu-mono">&mdash;</div>'
                "</div>"
            )
        else:
            score = scores[pillar] or 0.0
            pct = max(0.0, min(100.0, score))
            rows.append(
                '<div class="tu-pillar-row">'
                f'<div class="tu-pillar-label">{label}</div>'
                f'<div class="tu-meter"><div class="tu-meter-fill" style="width:{pct:.1f}%;">'
                "</div><div class=\"tu-meter-tick\"></div></div>"
                f'<div class="tu-pillar-value tu-mono">{score:.0f}</div>'
                "</div>"
            )
    measured_count = 4 - len(rating.unmeasured_pillars)
    return (
        '<div class="tu-card">'
        '<div class="tu-eyebrow"><span>Pillars (0&ndash;100, 50=baseline)</span>'
        f"<span>{measured_count}/4 measured</span></div>"
        f"{''.join(rows)}"
        '<div style="font-size:11px;color:var(--tu-text-faint);margin-top:4px;">'
        f"Weight used: {', '.join(f'{_PILLAR_LABELS[p]} {weights[p]:.0%}' for p in Pillar if p not in rating.unmeasured_pillars)}"
        "</div>"
        "</div>"
    )


def _suggestion_card_html(s, kind: str) -> str:
    """`kind` is "concern" or "strength" -- purely a rank-badge/metric-readout color
    choice, driven by the same `flag.is_concern` split the weaknesses/strengths
    columns already use. Every dynamic string here is real pipeline output (an LLM
    note, a player-sourced label, a citation token), so it's escaped before going
    into markup -- unlike the illustrative design canvas, this renders live data."""
    accent_color = "var(--tu-concern)" if kind == "concern" else "var(--tu-strength)"
    text_color = "var(--tu-concern-text)" if kind == "concern" else "var(--tu-strength-text)"
    badge_class = "tu-badge-llm" if s.note_source == "llm" else "tu-badge-template"
    badge_label = "LLM-written" if s.note_source == "llm" else "Template"

    phase_label = s.flag.phase.replace("_", " ").title() if s.flag.phase else "Full innings"
    spec = METRIC_SPECS[s.flag.metric]
    metric_readout = (
        f"{s.flag.actual:.1f} vs {s.flag.baseline:.1f} {spec.unit} &middot; "
        f"{s.flag.relative_delta:+.0%}"
    )

    citations = s.citation_tokens()
    if citations:
        # `citations_have_video` is one bool for the whole flag (see PerformanceFlag's
        # docstring) -- there's no per-citation video status in the real schema, so
        # every chip in a card gets the same treatment, not a mix.
        chip_class = "tu-chip" if s.flag.citations_have_video else "tu-chip tu-chip-novideo"
        chips_html = "".join(
            f'<span class="{chip_class}">{_html.escape(tok)}</span>' for tok in citations
        )
        novideo_note = (
            '<div class="tu-novideo-note">Stat-only &mdash; no clip ingested for these '
            "deliveries.</div>"
            if not s.flag.citations_have_video
            else ""
        )
        citations_html = (
            '<div class="tu-eyebrow" style="margin-bottom:6px;">Cited deliveries</div>'
            f'<div class="tu-chip-row">{chips_html}</div>{novideo_note}'
        )
    else:
        citations_html = (
            '<div class="tu-eyebrow" style="margin-bottom:6px;">Cited deliveries</div>'
            '<div style="font-size:12px;color:var(--tu-text-faint);">none</div>'
        )

    return (
        '<div class="tu-card tu-suggestion-card">'
        '<div style="display:flex;justify-content:space-between;align-items:flex-start;'
        'gap:12px;margin-bottom:10px;">'
        '<div style="display:flex;align-items:flex-start;gap:10px;">'
        f'<div class="tu-rank" style="background:{accent_color};">{s.rank}</div>'
        f'<div class="tu-suggestion-title tu-display">{_html.escape(s.title)}</div>'
        "</div>"
        f'<div class="{badge_class}">{badge_label}</div>'
        "</div>"
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap;">'
        f'<div class="tu-pill">{_html.escape(phase_label)}</div>'
        f'<div class="tu-mono" style="font-size:12px;font-weight:600;color:{text_color};">'
        f"{metric_readout}</div>"
        "</div>"
        f'<div class="tu-suggestion-body">{_html.escape(s.body)}</div>'
        f"{citations_html}"
        "</div>"
    )


tab_overview, tab_player, tab_video, tab_rating = st.tabs(
    ["Overview", "Player Performance", "Video Pipeline Test", "Rating & Suggestions"]
)

with tab_overview:
    matches, deliveries = load_warehouse()

    if matches.empty:
        st.warning("No warehouse data yet.")
    else:
        _stat_row(
            [
                ("Matches", f"{len(matches):,}"),
                ("Deliveries", f"{len(deliveries):,}"),
                ("Competitions", str(matches["competition"].nunique())),
                (
                    "Players seen",
                    str(pd.concat([deliveries["striker"], deliveries["bowler"]]).nunique()),
                ),
            ]
        )
        with tu_card(eyebrow="Recent matches", right="last 10"):
            st.dataframe(matches[["match_id", "competition", "teams", "venue", "outcome_winner"]].tail(10))

    st.markdown(_section_title_html("Refresh from cricsheet.org"), unsafe_allow_html=True)
    with tu_card():
        st.caption(
            "Every fetch checks the live Last-Modified header against what's cached, "
            "and only re-downloads if cricsheet.org's archive has actually changed."
        )
        competition = st.text_input("Competition slug", "icc_mens_t20_world_cup_male")
        if st.button("Check for fresh data"):
            with st.spinner(f"Checking cricsheet.org for '{competition}'..."):
                store = ParquetStore()
                match_count, delivery_count = run_stats_ingestion(CricsheetSource(competition), store)
                # ParquetStore merges by match_id now instead of overwriting (see its
                # module docstring), so match_count/delivery_count above are just this
                # fetch's counts -- re-read for the real warehouse-wide totals rather
                # than mislabeling a per-run count as "the warehouse now has".
                total_matches = len(store.read_matches())
                total_deliveries = len(store.read_deliveries())
            st.success(
                f"Fetched {match_count} matches, {delivery_count} deliveries for "
                f"'{competition}'. Warehouse now has {total_matches} matches, "
                f"{total_deliveries} deliveries total."
            )
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
            with tu_card(eyebrow="Batting"):
                st.table(_as_display_table(batting_summary(deliveries, player, match_ids)))
        with col2:
            with tu_card(eyebrow="Bowling"):
                st.table(_as_display_table(bowling_summary(deliveries, player, match_ids)))

        per_match_runs = (
            deliveries[(deliveries["striker"] == player) & (deliveries["match_id"].isin(match_ids))]
            .groupby("match_id")["runs_batter"]
            .sum()
            .reindex(match_ids)
        )
        with tu_card(eyebrow="Runs per match"):
            st.bar_chart(per_match_runs, color="#c9a24b")

with tab_video:
    _weights_note = (
        "a fine-tuned ball+stumps checkpoint is loaded"
        if _ball_stumps_weights_path()
        else "no fine-tuned ball/stumps model is loaded, so shot classification will "
        "read 'unknown' regardless of detection quality -- see README"
    )
    st.caption(
        "Runs the *real* pipeline -- pretrained YOLO for player detection, ByteTrack "
        "for tracking, Keypoint R-CNN for pose. Genuine inference, not simulated. "
        f"Currently, {_weights_note}. Real-footage testing found this checkpoint can "
        "lock onto a static round/light-colored background object; a motion check now "
        "rejects any ball track that doesn't move enough to be real flight, falling back "
        "to 'unknown' instead of a wrong-but-confident shot call. That fixes false "
        "positives, not recall -- the checkpoint still often doesn't find the real ball "
        "at all, so 'unknown' remains the common, honest result. See README's \"Ball and "
        "stumps detection\" section for the full picture."
    )

    dest_dir = REPO_ROOT / "data" / "uploads" / "videos"
    dest_dir.mkdir(parents=True, exist_ok=True)

    source_mode = st.radio(
        "Video source", ["Upload a file", "Fetch from a URL"], horizontal=True,
    )
    video_path: Path | None = None
    if source_mode == "Upload a file":
        uploaded = st.file_uploader("Upload a video clip", type=["mp4", "mov", "avi", "mkv"])
        if uploaded is not None:
            video_path = dest_dir / uploaded.name
            video_path.write_bytes(uploaded.getbuffer())
    else:
        st.caption(
            "Needs a **direct** link to a video file (ends in .mp4/.mov/etc, e.g. a "
            "broadcast CDN or S3 URL) -- this fetches the bytes at that exact URL, it "
            "can't extract video from YouTube or any page that just embeds a player. "
            "Same `ingestion.fetch.download_url` the CLI's `fetch-url` command uses."
        )
        video_url = st.text_input("Video URL", placeholder="https://example.com/clip.mp4")
        if st.button("Fetch video") and video_url:
            with st.spinner(f"Downloading {video_url}..."):
                try:
                    video_path = download_url(video_url, dest_dir)
                except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
                    st.error(f"Couldn't fetch that URL: {exc}")
                    video_path = None
                else:
                    # A 200 response isn't proof of a video -- a page URL (e.g. a
                    # YouTube watch link, which is exactly what this caught while
                    # testing) downloads real bytes just fine, they're just an HTML
                    # page, not a video. download_url can't tell the difference by
                    # itself (it's a generic fetcher, also used for stats files), so
                    # this checks the one thing the caption above already promises:
                    # a *direct* link ends in a real video extension. Catching it
                    # here gives a clear message instead of the file silently
                    # reaching ffprobe and failing three layers deeper with a raw
                    # subprocess traceback.
                    if video_path.suffix.lower().lstrip(".") not in {"mp4", "mov", "avi", "mkv"}:
                        st.error(
                            f"That didn't fetch a video -- got a file named "
                            f"'{video_path.name}' with no recognised video extension "
                            "(expected .mp4/.mov/.avi/.mkv). This usually means the "
                            "URL points at a webpage (like a YouTube watch link), "
                            "not a direct file -- see the note above."
                        )
                        video_path.unlink(missing_ok=True)
                        video_path = None
            if video_path is not None:
                st.session_state["video_pipeline_fetched_path"] = str(video_path)
            else:
                st.session_state.pop("video_pipeline_fetched_path", None)
        elif "video_pipeline_fetched_path" in st.session_state:
            # Re-runs (e.g. clicking "Run real CV pipeline" below) shouldn't forget a
            # video that was already fetched -- st.button's True is only true on the
            # click itself, not on every subsequent rerun.
            video_path = Path(st.session_state["video_pipeline_fetched_path"])

    c1, c2, c3, c4 = st.columns(4)
    match_id = c1.text_input("Match ID", "manual-test")
    innings = c2.number_input("Innings", 1, 2, 1)
    over = c3.number_input("Over", 0, 19, 0)
    ball = c4.number_input("Ball", 1, 9, 1)

    if video_path is not None:
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

            st.markdown(_section_title_html("Model performance"), unsafe_allow_html=True)
            n_frames = len(frames)
            inference_time = timings["detection"] + timings["tracking"] + timings["pose_estimation"]
            _stat_row(
                [
                    ("Frames processed", str(n_frames)),
                    ("Detection", f"{timings['detection']:.2f}s"),
                    ("Tracking", f"{timings['tracking']:.3f}s"),
                    ("Pose estimation", f"{timings['pose_estimation']:.2f}s"),
                    (
                        "Effective FPS",
                        f"{n_frames / inference_time:.1f}" if inference_time > 0 else "n/a",
                    ),
                ]
            )
            with tu_card(eyebrow="Stage timings", right="seconds"):
                st.bar_chart(pd.Series(timings, name="seconds"), color="#c9a24b")

            st.markdown(_section_title_html("Results"), unsafe_allow_html=True)
            player_tracks = [t for t in analysis.tracks if t.obj_class.value == "player"]
            _stat_row(
                [
                    ("People tracked", str(len(player_tracks))),
                    ("Pose frames", str(len(analysis.poses))),
                    ("Shot classification", analysis.event.shot_type.value),
                ]
            )
            with tu_card(eyebrow="Line and length", right="single-camera estimate"):
                _render_pitch_geometry(analysis.event)

            with tu_card(eyebrow="Event detail"):
                st.write(analysis.event)

            if detections:
                det_df = pd.DataFrame(
                    [{"frame": d.frame_index, "class": d.obj_class.value, "confidence": d.confidence} for d in detections]
                )
                with tu_card(eyebrow="Detection confidence"):
                    st.dataframe(det_df.groupby("class")["confidence"].describe())

                with tu_card(eyebrow="Sample frames", right="real detections drawn on real frames"):
                    sample_frame_indices = sorted({d.frame_index for d in detections})
                    step = max(1, len(sample_frame_indices) // 4)
                    for frame_index in sample_frame_indices[::step][:4]:
                        dets_here = [d for d in detections if d.frame_index == frame_index]
                        poses_here = [p for p in analysis.poses if p.frame_index == frame_index]
                        overlay = draw_overlay(frames[frame_index], dets_here, poses_here)
                        st.image(overlay, caption=f"frame {frame_index}: {len(dets_here)} detection(s)")

                with tu_card(eyebrow="Tracks"):
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
        "Runs the real stage 5-6 pipeline. The baseline is pooled over a *comparable* "
        "cohort -- same competition(s), gender and format as the player's own recent "
        "matches, bounded to recent seasons -- not over the whole warehouse, which now "
        "mixes several competitions and both genders and would average into a player "
        "who resembles nobody. The cohort actually used is shown below. No video: "
        "BroadcastFeedAdapter isn't implemented, so every FusedDelivery here has "
        "video_available=False. Technique always reads 'not measured' -- see rating/'s "
        "module docstring for why that's a deliberate weight-redistribution decision, "
        "not a bug."
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
            # Select, read, then fuse -- see `rating.pipeline`'s module docstring. The
            # cohort is chosen from the (small) match table first, the baseline is pooled
            # over just that cohort's rows, and only the player's own window is fused into
            # FusedDelivery objects. The previous version fused every match in the
            # warehouse before it could answer anything, which measured 557s at 8,026
            # matches.
            store = ParquetStore()
            with st.spinner("Selecting a comparable cohort and pooling its baseline..."):
                # Scope first, so the baseline can be cached on the cohort rather than
                # recomputed for every player switch -- a baseline depends on the cohort
                # and the warehouse, never on who is selected.
                scope = resolve_scope(store, matches, player, window_size)
                baseline = cached_cohort_baseline(scope[0])
                inputs = prepare_rating(
                    store, matches, player, window_size=window_size,
                    baseline=baseline, scope=scope,
                )

            if baseline is None:
                st.error(
                    f"No comparable cohort for {player} -- either they have no matches in "
                    "the warehouse, or the matches they played in carry no others to "
                    "pool a baseline from."
                )
            else:
                cohort = inputs.cohort
                st.caption(f"Baseline: {baseline.source}")
                coverage = (
                    f"{len(cohort.match_ids):,} matches pooled"
                    + (f" (sampled from {cohort.matches_available:,} in scope)" if cohort.was_capped else "")
                    + f" · {inputs.deliveries_fused:,} deliveries fused for {player}'s window"
                )
                st.caption(coverage)

                history = list(inputs.window_history)
                fused_all = list(inputs.window_fused)
                rating = rate_player(history, player, role, baseline, fused_all, window_size=window_size)
                if rating is None:
                    st.warning(f"No matches for {player} in the warehouse.")
                else:
                    st.markdown(_identity_card_html(player, role_label), unsafe_allow_html=True)
                    hero_col1, hero_col2 = st.columns([2, 3])
                    with hero_col1:
                        st.markdown(_composite_card_html(rating), unsafe_allow_html=True)
                    with hero_col2:
                        st.markdown(_pillar_bars_html(rating), unsafe_allow_html=True)

                    # The exact same rows `rate_player` scored, not a separately re-derived
                    # window. `prepare_rating` resolves the window once and both halves
                    # read it, so the two views of "this player, recently" cannot disagree
                    # -- previously each derived its own and only matched by coincidence.
                    window_fused = fused_all

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

                    weak_col, strong_col = st.columns(2)
                    with weak_col:
                        st.markdown(
                            f'<div class="tu-display" style="font-size:15px;font-weight:600;'
                            f'color:var(--tu-text);margin-bottom:12px;">'
                            f"Weaknesses <span class=\"tu-mono\" style=\"font-size:11.5px;"
                            f'font-weight:500;color:var(--tu-text-muted);">'
                            f"({len(weaknesses)})</span></div>",
                            unsafe_allow_html=True,
                        )
                        if not weaknesses:
                            st.caption("None cleared the threshold for this player/window.")
                        for s in weaknesses:
                            st.markdown(_suggestion_card_html(s, "concern"), unsafe_allow_html=True)

                    with strong_col:
                        st.markdown(
                            f'<div class="tu-display" style="font-size:15px;font-weight:600;'
                            f'color:var(--tu-text);margin-bottom:12px;">'
                            f"Strengths <span class=\"tu-mono\" style=\"font-size:11.5px;"
                            f'font-weight:500;color:var(--tu-text-muted);">'
                            f"({len(strengths)})</span></div>",
                            unsafe_allow_html=True,
                        )
                        if not strengths:
                            st.caption("None cleared the threshold for this player/window.")
                        for s in strengths:
                            st.markdown(_suggestion_card_html(s, "strength"), unsafe_allow_html=True)
