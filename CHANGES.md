# Changes — for review

What changed in this build pass, grouped by area, for a human review pass. Each item names
the commit(s) it lives in. See DECISIONS.md for the reasoning behind the non-obvious calls,
and MISTAKES.md for what went wrong along the way and how it was caught.

**Test suite status at the end of this pass**: 206 passed, 4 skipped, 0 failures.
**Git status**: 4 commits ahead of `origin/main`, working tree clean.

---

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

## What's explicitly still open (not done in this pass)

- **Ball detection recall.** The checkpoint needs either more real (non-Roboflow) training
  data with motion-blurred balls in flight, or a different detection approach entirely
  (classical background-subtraction/Kalman tracking is what small-fast-object trackers
  typically use instead of a frame-by-frame CNN). Neither was attempted this pass.
- **Pitch calibration wiring.** `calibrate_from_stumps` is implemented and tested but has no
  caller — deliberately blocked on the ball detector being trustworthy first.
- **`git push`.** 4 commits sit ahead of `origin/main`, unpublished as of this file.
