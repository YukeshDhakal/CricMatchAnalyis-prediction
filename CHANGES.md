# Changes — for review

What changed in this build pass, grouped by area, for a human review pass. Each item names
the commit(s) it lives in. See DECISIONS.md for the reasoning behind the non-obvious calls,
and MISTAKES.md for what went wrong along the way and how it was caught.

**Test suite status at the end of this pass**: 247 passed, 8 skipped, 4 failed.
The 4 failures are all in `tests/ingestion/test_scene_split_adapter.py` and
`test_manifest_adapter.py` and are environmental — ffmpeg is not on this machine's PATH.
**Git status**: 7 commits ahead of `origin/main`, working tree clean, nothing pushed.

---

# Pass 2 (2026-09-22): ball tracking, trajectory fitting, and line/length

## Ball selection: trajectory fit replaces the pooled-spread motion gate
*(`4b8d3c5`)*

The headline change, and it starts from a correction to this repo's own record. Re-probing
the trained checkpoint over all four clips in `data/uploads/videos/` — and then opening the
frames rather than reading the coordinates — found that the detector's "static false
positives" are frequently **genuine cricket balls lying still on the ground**, detected at
0.44–0.55 across 83 frames of the 4K clip. The real delivered ball scores 0.38–0.65. No
confidence threshold orders those correctly, because a per-frame detector cannot see
motion at all.

It also found a **latent wrong-answer bug in the shipped motion gate**: it pools every ball
detection and measures the spread of the union, so it passes whenever a static object and a
real ball are both present, then reports the release frame of the static one. Measured on
`real_bowling_clip_full.mp4` at a 0.15 confidence threshold: `release_frame=1`, the bag by
the fence. The 0.40 default avoids it only because that bag peaks at 0.34.

- New `src/video_engine/trajectory/` — RANSAC fit selecting a mutually consistent moving
  subset, interpolating frames the detector missed, and locating a bounce by a two-arc
  split or honestly reporting none.
- New `src/video_engine/motion/` — three-frame differencing, scoring how much the pixels
  under a detection actually moved.
- New `src/video_engine/geometry.py` — gives `calibrate_from_stumps` its first caller.
- **Review this if**: you touch ball detection thresholds. The minimum-displacement and
  minimum-step-speed checks are *rejection filters applied before scoring*, not tiebreaks —
  a static cluster outnumbers the real ball 12:1 and wins any inlier-count ranking. That
  ordering is the load-bearing part; see `trajectory/fit.py`'s docstring before changing it.
- **New tests**: `tests/video_engine/test_trajectory.py` (13),
  `test_trajectory_real_footage.py` (4, carrying verbatim recorded detections from the real
  clip), `test_motion_energy.py` (6), `test_geometry.py` (12),
  `test_segmenter_motion_filter.py` (5).

## Motion energy wired in as a pre-filter
*(`0f54497`)*

`EventSegmenter.segment` gained an optional `frames` argument so the segmenter can ask
whether a detection's own pixels were moving. Measured effect on real detections:
`real_bowling_clip_full.mp4` at conf 0.25 goes from 9 candidates to exactly the 7 frames
the real ball occupies; at 0.10, from 59 to 12.

- **Review this if**: you are wondering why this isn't redundant with the trajectory fit.
  It stops a resting ball being absorbed as an *inlier* into a real ball's track, which is
  how contamination appeared at low thresholds — not just a speed optimisation.
- Two refusals are deliberate: the filter disables itself on a panning camera
  (`global_motion_ratio`), and falls back to the unfiltered set when fewer than
  `MIN_INLIERS` survive, because motion energy is meaningless on degenerate input. The
  second was found by `test_pipeline_smoke.py`'s all-zero frames, where a trusting filter
  deleted the entire candidate set.

## Pitch geometry surfaced with its uncertainty attached
*(`459775b`)*

- `DeliveryEvent` gained `pitch_confidence`, `pitch_notes` and `track_confidence`.
- API responses carry a nested `pitch_geometry` block whose `available` flag must be read
  before the numbers; `insufficient_data_reason` is always populated when it is false.
- The Streamlit Video Pipeline Test tab gained a "Line and length" card that renders
  "Insufficient data" and the reason below MEDIUM confidence, rather than a metre figure
  with a caveat under it.
- Also fixes a wrong-answer bug found by running the real pipeline rather than a unit test:
  an undetected bounce was passed through as `bounced=False`, which `classify_length` turns
  into `FULL_TOSS` — reporting a full toss for a ball visibly rolling along the ground. A
  tracking gap is not a cricketing fact; it now reports `UNKNOWN` and says why.
- **Review this if**: you add another consumer of pitch geometry. The rule is that the
  confidence band travels with the number and anything below MEDIUM degrades to
  "insufficient data", matching the existing `ShotType.UNKNOWN` pattern.

## What this pass verified against real footage, and what it did not

Stated separately because the distinction is the point:

- **Verified on real footage**: ball selection. On `real_bowling_clip_full.mp4` the
  pipeline now reports `release_frame=80` (the real ball) where the old gate reported
  frame 1 (the bag), and on `real_bowling_clip3.mp4` / `real_bowling_clip.mp4` it reports
  no track at all rather than a confident wrong one. End-to-end through the real detector,
  tracker and segmenter, not through fixtures.
- **Implemented and unit-tested only**: line, length and the bounce projection. The
  geometry is exercised by synthetic-camera tests and has **never produced a number on real
  footage**, because no available clip shows both stump sets — every clip is filmed from
  behind the bowler's arm, and a homography needs four coplanar points where one stump set
  supplies two. Treat the line/length path as unproven until it runs on suitable footage.
- **Not solved**: identifying *which* moving object is the delivery. On the 4K clip the
  best-scoring track is a white object falling outside the net, fitted cleanly at ~2 px
  RMS — a correct trajectory of the wrong object. Disambiguating it needs the pitch
  corridor, which needs the calibration above, which needs the footage above.

---

# Pass 1: console reskin, warehouse fixes, detector training

## UI: Live Test Console reskin
*(`bfb7d96`)*

- Full dark-theme visual pass across all four tabs (Overview, Player Performance, Video
  Pipeline Test, Rating & Suggestions) — custom CSS, Google Fonts, card/tile components.
- `.streamlit/config.toml` added so Streamlit's own native chrome matches.
- **Review this if**: you want to sanity-check the visual design choices, or you're about to
  touch `app/streamlit_app.py`'s CSS/HTML helper functions (`tu_card`, `_stat_tile_html`,
  etc.) and want to understand the existing pattern first.

## Data: warehouse overwrite bug fixed
*(`bfb7d96`)*

- `ParquetStore.write_matches`/`write_deliveries` now merge/upsert instead of overwriting —
  ingesting a second competition no longer wipes the first.
- `read_matches`/`read_deliveries` gained optional filters (`match_ids`, `competitions`,
  `involving_player`) pushed down via pyarrow, so a scoped read doesn't materialize the
  whole warehouse.
- **Review this if**: you're adding a new ingestion source or another scoped-read caller —
  check `tests/ingestion/test_storage.py`'s "scoped reads" section for the filter semantics
  (the None-vs-empty-list distinction is load-bearing).

## Rating: pipeline performance fix
*(`bfb7d96`)*

- New `src/rating/cohort.py`, `src/rating/pipeline.py`: cohort selection happens before any
  delivery-level read, cutting rating-a-player from 550s+ to ~8.5s against the real 8,026-
  match warehouse.
- **Review this if**: you're changing what counts as a "comparable cohort" — that logic is
  now centralized in `cohort.py` rather than scattered through the rating engine.

## Video engine: pitch calibration groundwork
*(`bfb7d96`)*

- New `src/video_engine/calibration.py`: stump-anchored homography, implemented and tested
  against a synthetic camera, but **has no caller in the pipeline yet** — see its own
  docstring and the README's "Pitch calibration" section for why.
- **Review this if**: you're about to wire calibration into the pipeline — read the
  docstring's honesty section on precision limits (length is more reliable than line) first.

## Video pipeline: URL-based video fetching
*(`6f18713`, carrying a fix from earlier in the session)*

- Video Pipeline Test tab gained a "Fetch from a URL" option alongside file upload.
- `src/ingestion/fetch.py`: `download_url` now sends a real browser User-Agent — several
  real CDNs (GCS buckets, sample-video hosts) 403 the bare `urlretrieve` default.
- **Review this if**: you're adding another network-fetch path — the User-Agent fix applies
  only to this one function; check whether a new caller needs the same treatment.

## Ball/stumps detector: trained and wired, then found unreliable, then partially mitigated
*(`6f18713`, `f3cae57`, `9339429` — read as one arc, not three independent changes)*

1. **Trained** a YOLOv8n on a real, correctly-licensed (CC BY 4.0) 2-class dataset; wired the
   resulting checkpoint (`weights/ball_stumps_n.pt`, gitignored) into `YoloDetector` via
   `app/streamlit_app.py`, with a `THIRD_UMPIRE_BALL_STUMPS_WEIGHTS` env override.
2. **Found**, via real-footage testing requested by the user, that the checkpoint can lock
   onto a static background object at plausible confidence and drive a wrong-but-confident
   shot classification — documented in the README and MISTAKES.md rather than left
   undiscovered.
3. **Mitigated** with a temporal-motion check in `src/video_engine/events/
   heuristic_segmenter.py` that rejects a ball track that doesn't move enough to be real
   flight, pooling detections across track_ids to survive tracking fragmentation. Verified
   against the exact real clip that produced the original wrong answer — same detection
   still fires, now correctly rejected.
- **Review this if**: you're evaluating whether this feature is ready for real use — **it is
  not**. The motion check fixes precision (no more confidently wrong answers) but not recall
  (the real ball is still frequently missed). Read the README's "Ball and stumps detection"
  section in full before deciding whether to build anything on top of this.
- **New tests**: `tests/video_engine/test_heuristic_segmenter.py` (4 tests: no-track,
  rejected-static-track, accepted-moving-track, prefers-moving-over-static).

---

## What's explicitly still open (after pass 2)

- **Footage that shows both stump sets.** The single biggest blocker, and the one only the
  user can clear. Every clip here is filmed from behind the bowler's arm, so only one stump
  set is ever visible and the pitch cannot be calibrated. Line and length are implemented,
  tested and inert until suitable footage exists — a fixed elevated camera with the stumps
  at *both* ends in frame, which is the setup the README's "Pitch calibration" section
  already describes.
- **Footage that shows the ball in flight.** Separate from the above and equally blocking.
  Probing found the delivered ball is recovered by neither the checkpoint nor frame
  differencing on these clips: the deliveries are filmed down the pitch axis, so the ball's
  image displacement is small and heavily blurred, and the bowler's arm dominates the motion
  mask in exactly the corridor the ball travels down. Native-4K differencing did not rescue
  it. Side-on footage would be a far better test of everything built this pass.
- **Telling the delivery from any other moving object.** Currently unsolved and not solvable
  without the calibration above — see the note in the pass-2 section.
- **Ball detection recall.** Unchanged as a model problem: still needs real, non-Roboflow
  training data with motion-blurred balls in flight. What changed is that the pipeline no
  longer converts a recall gap into a confident wrong answer, and the trajectory fit now
  interpolates across frames the detector missed *within* a track it has verified.
- **A retrain.** Not attempted: no labelled data and no GPU training infrastructure in this
  environment. Noted as a deliberate exclusion, not an oversight.
- **`git push`.** 6 commits sit ahead of `origin/main`, unpublished as of this file — left
  for review rather than pushed.
