# Third Umpire

An AI cricket analyst: it reads match video and official T20 stat tables, and returns a
per-player rating with clip-referenced coaching suggestions. Product and architecture
context lives in the team's PRD ("Third Umpire" Artifact); this repo is the implementation.

## Modules

- **`ingestion/`** (this section) — pipeline stage 1 (PRD 2.2): fetches the official
  ball-by-ball table and produces the `DeliveryClip`s the video engine consumes.
- **`video_engine/`** — turns a per-delivery video clip into tracked objects, player
  poses, and a segmented event (shot type, release/contact frames). See its own section
  below for the CV pipeline.
- **`fusion/`** — joins `ingestion`'s ball-by-ball table with `video_engine`'s
  `DeliveryAnalysis` into `FusedDelivery` rows, then aggregates those into per-match and
  rolling-N-match player stats (`PlayerMatchStats`/`PlayerRollingSummary`). CV
  biomechanics fields (wrist speed, bowling elbow extension) are reserved in the
  contracts but always `None` today -- no calibrated extractor produces them yet, see
  `fusion/contracts.py`.
- **`rating/`** — pipeline stages 5-6 (PRD 3.1-3.3): the composite four-pillar
  `PlayerRating` and the flag → rank → note-write coaching-suggestion engine, including
  a local-LLM note writer with a deterministic template fallback. `cohort.py` chooses
  which matches a baseline pools over (competition/gender/format-scoped, recency-bounded,
  capped) and `pipeline.py` wires selection → scoped read → vectorised baseline → fuse
  only the player's window, so rating one player doesn't cost a pass over the warehouse.
  See "Rating and coaching suggestions" below for what's implemented and what's
  deliberately left open.
- **`player_reports/`** — stats-only player performance summaries (batting/bowling over
  a player's last N matches) computed directly from the ingested ball-by-ball table.
  Covers the PRD's "Execution" pillar only, not a full composite rating -- that needs
  video-derived "Technique" signals the video engine doesn't produce yet.
- **`app/`** — a Streamlit live test console tying ingestion, player_reports, and the
  video engine together in one running UI. See "Live test console" below.
- **`prediction/`** — feature engineering (layer 2) for a *separate* live win-probability
  / next-ball-prediction subsystem: contextual (run rate, wickets in hand), batter-vs-bowler
  matchup, and pitch/fatigue features. Not to be confused with `fusion/` above -- this is
  in-match live prediction, `fusion/` is post-match rating. No live telemetry intake, ML
  model, or serving layer exists yet; see `prediction/contracts.py`.

## Live test console

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`. Three tabs:

1. **Overview** -- warehouse stats, and a button to check cricsheet.org for fresh data
   (same live-freshness check as the CLI).
2. **Player Performance** -- pick any player who appears in the ingested data, see their
   batting/bowling summary over their last N matches. Real numbers straight from the
   parquet warehouse, e.g. V Kohli's last 5 T20 World Cup innings: 122 runs off 102
   balls, SR 119.61, 5 dismissals.
3. **Video Pipeline Test** -- upload any video file and run it through the *real*
   detection/tracking/pose pipeline (pretrained YOLO + ByteTrack + Keypoint R-CNN --
   genuine inference, downloads its weights on first use, not simulated). There's no
   fine-tuned ball/stumps model yet, so shot classification will read "unknown" on real
   footage regardless of how good the person-detection is -- that's the documented gap
   above, not a bug in this test. Shows real per-stage timing, detection confidence
   stats, and draws actual bounding boxes/keypoints on real frames -- not just an
   aggregate count -- so you can see what the model actually saw.

**Measured performance, not a claim:** ran the real pipeline against a 20-frame/2s clip
made from a real photo with two people (`ultralytics/assets/zidane.jpg`) on CPU-only
inference:

| Stage | Time |
|---|---|
| Detection (YOLO) | 0.94s |
| Tracking (ByteTrack) | 0.013s |
| **Pose estimation (Keypoint R-CNN)** | **78.7s** |
| Event segmentation | 0.0015s |

Pose estimation is ~99% of total runtime -- roughly 4s/frame on CPU. That's the real
bottleneck for anything beyond a quick local test; a production system processing full
matches (thousands of deliveries) needs GPU inference for this stage, matching what the
broader architecture pitch (see "Prototypes" above) already assumed. Also observed on
that same test, now root-caused and fixed: `ByteTrackTracker` collapsed both
clearly-visible, separately-boxed people into a single track rather than two. Cause was
a threshold mismatch, not a tracking bug -- `YoloDetector` admits any detection >=0.4
confidence, but the `trackers` library's own defaults
(`high_conf_det_threshold=0.6`/`track_activation_threshold=0.7`) only let a detection
*extend* an existing track below 0.6, never *spawn* a new one; the second person never
cleared 0.6 in any frame (confidence ~0.58) and so never got a track of his own. Fixed
by defaulting both thresholds to 0.4 in `ByteTrackTracker` -- see its docstring and
`tests/test_bytetrack_tracker.py::test_two_separately_boxed_players_get_two_tracks_even_at_borderline_confidence`.

## Feeding in your own data

Two intake folders under `data/uploads/` (auto-created, gitignored -- these are your
local test inputs, not repo content):

- **`data/uploads/videos/`** -- drop a delivery/match video file here, or fetch one from
  a direct link: `python -m ingestion.cli fetch-url --url <link> --dest videos`. Then
  wrap a single file into a `DeliveryClip`:
  `python -m ingestion.cli ingest-video --match-id <id> --video data/uploads/videos/<file> --innings 1 --over 0 --ball 1`
  (a whole continuous recording instead of one delivery? use `SceneSplitAdapter`
  directly -- see `scripts/demo_end_to_end.py` for the pattern). Note: `fetch-url` needs
  a direct file link (ends in `.mp4` etc.); it can't pull from YouTube or a page that
  just embeds a player.
- **`data/uploads/stats/`** -- drop a stats file here (or `fetch-url ... --dest stats`).
  Tell me what's in it (ball-by-ball data vs. a tournament standings/points table) --
  ingestion currently only has a `StatsSource` for Cricsheet's ball-by-ball format;
  anything else needs a small parser added first, same shape as `sources/cricsheet.py`.

## Ingestion

Two independent jobs, matching PRD 2.1's data sources:

### 1. Ball-by-ball table (`ingestion/sources/`)

`CricsheetSource` fetches and normalises data from [Cricsheet](https://cricsheet.org)
(free, open, structured ball-by-ball data covering T20Is and most T20 leagues) into the
`MatchMeta` / `Delivery` shapes in `ingestion/contracts.py`. This stands in for the PRD's
licensed "ICC / board ball-by-ball tables" source until a data-partner deal is signed;
swap in a licensed-feed `StatsSource` later without touching anything downstream, since
both implement the same interface.

**Always live, not a frozen snapshot:** Cricsheet isn't a streaming API -- it's a zip
archive updated periodically as new matches are played -- so "live" here means every
call checks the archive's current `Last-Modified` header against what was last
downloaded, and re-fetches only when it's actually changed. That's a HEAD request, not
a full download, so calling `ingest-stats` repeatedly (a cron job, a re-run before a
demo) is cheap and never serves stale data past whatever cricsheet.org itself has
published.

Run it:

```bash
python -m ingestion.cli ingest-stats --competition icc_mens_t20_world_cup_male
```

`--competition` is any Cricsheet JSON download slug (`t20s_male`, `ipl_male`, etc. --
see cricsheet.org/downloads). Output lands in `data/warehouse/{matches,deliveries}.parquet`.

**Runs accumulate across competitions.** `ParquetStore` upserts (keyed by `match_id`
for matches, `(match_id, innings, over, ball)` for deliveries) instead of overwriting,
so running `ingest-stats` for a second competition adds to the warehouse rather than
replacing it -- re-ingesting the *same* competition still refreshes its rows
(last-write-wins) rather than duplicating them. `RatingBaseline`'s cohort/season
baseline (see `rating/contracts.py`) still pools over whatever's in the warehouse at
baseline-computation time, so a multi-competition warehouse gives you a
multi-competition baseline -- ingest only what you want pooled together if that's not
what you want.

**Ball numbering note:** `ball` is the 1-based position of each delivery within its over,
wides and no-balls included -- *not* Cricsheet's own `actual_delivery` field, which
reuses the same ball number for an illegal delivery and its re-bowled replacement (both
come back as e.g. `"17.5"`). Using list order instead keeps `(match_id, innings, over,
ball)` a genuinely unique join key, and it's covered by
`tests/ingestion/test_cricsheet_source.py::test_wide_and_its_rebowled_ball_get_distinct_ball_numbers`.

### 2. Video (`ingestion/video/`)

Three adapters behind one `VideoAdapter` interface, matching PRD 2.1's three video
sources:

| Adapter | Source | Status |
|---|---|---|
| `ManifestClipAdapter` | User-uploaded video, already split one file per delivery (the common nets/club-footage case) | Working -- just probes each file and attaches the delivery key |
| `SceneSplitAdapter` | User-uploaded video as one continuous recording | Working, heuristic v0 -- splits on detected scene cuts, falls back to a uniform time split when cut count doesn't match the known delivery count |
| `BroadcastFeedAdapter` | Broadcast feed | **Not implemented.** PRD 2.4 is explicit that broadcast footage needs a league/board licensing agreement before any pipeline work starts on it -- a legal, not engineering, blocker. Raises `NotImplementedError` rather than pretending to work. |

All three produce `video_engine.contracts.DeliveryClip` directly (imported, not
mirrored) -- ingestion hands off a decoded video file per delivery plus `fps`/`width`/
`height`/`source`, exactly what `video_engine.io.clip_loader.load_frames` expects.

## Contract with video engine

`DeliveryClip` (`src/video_engine/contracts.py`) is the handoff boundary:

| Field | Type | Notes |
|---|---|---|
| `delivery.match_id` | `str` | Joins back to the official ball-by-ball table |
| `delivery.innings` / `.over` / `.ball` | `int` | Ball number as bowled (wides/no-balls included) |
| `video_path` | `str` | Path to the delivery's video file |
| `fps`, `width`, `height` | `float` / `int` | Source video properties |
| `source` | `"broadcast"` \| `"user_upload"` | Broadcast footage vs. a phone/user upload |

If a video adapter's actual output looks different, update `DeliveryClip` and
`load_frames` in `src/video_engine/io/clip_loader.py` rather than adapting around the
mismatch elsewhere -- that's the one place this boundary is meant to live.

**Not yet built:** the fusion step that lines up `ingestion`'s ball-by-ball `Delivery`
rows with `video_engine`'s `DeliveryAnalysis` output (PRD 2.2, stage 3) -- that's next.

## Video engine pipeline

Four stages, in dependency order -- each one built to unblock the next:

1. **Detection** (`detection/`) -- player, ball, and stumps bounding boxes per frame.
   `YoloDetector` uses a pretrained YOLO model for players; ball/stumps detection needs a
   checkpoint fine-tuned on cricket footage (`ball_stumps_weights`), which doesn't exist
   yet -- pass `None` and it will emit player detections only, with a logged warning. Runs
   each model once over the whole clip (`batch_size` frames per forward pass, default 16)
   rather than once per frame.
2. **Tracking** (`tracking/`) -- links per-frame detections into persistent tracks.
   `ByteTrackTracker` (recommended default) wraps a real Kalman-filter multi-object tracker
   -- it predicts through brief occlusions and handles the ball's erratic, fast motion far
   better than IOU matching alone. `IouTracker` is kept as a zero-dependency fallback for
   quick local tests.
3. **Pose estimation** (`pose/`) -- `KeypointRcnnPoseEstimator` runs a pretrained
   Keypoint R-CNN and matches each detected pose onto the nearest player track by box
   overlap. Batches `batch_size` frames per forward pass (default 8) instead of one at a time.
4. **Event segmentation** (`events/`) -- `HeuristicEventSegmenter` finds the release frame
   (first ball detection) and contact frame (closest ball/player approach), then classifies
   the shot from the batter's wrist swing. This is a rule-based v0; replace it with a
   trained classifier once labelled clips exist.

`pipeline.VideoEngine` wires all four into `analyze(clip) -> DeliveryAnalysis`.

### Ball and stumps detection: resolved via a CC BY 4.0 dataset, trained checkpoint

`ObjectClass.BALL` and `ObjectClass.STUMPS` have been reserved since the start, and
`YoloDetector.ball_stumps_weights` is the hook for a fine-tuned checkpoint that produces
them. **A checkpoint now exists**: a YOLOv8n model fine-tuned on Roboflow's "Cricket
Dataset" (workspace `yukeee`, project `cricket-dataset-z2wkt-wcfjg`, version 1 — 7,452
images, classes `ball`/`stump`, licensed CC BY 4.0), trained for 30 epochs on a Colab T4
GPU (0.893 mAP50 overall: 0.809 ball, 0.976 stump). It lives at `weights/ball_stumps_n.pt`,
which is **gitignored** (same as every `*.pt` in this repo) because it's a local training
artifact, not something to vendor in git — `app/streamlit_app.py`'s
`get_pipeline_components()` loads it automatically when present (override the path with
`THIRD_UMPIRE_BALL_STUMPS_WEIGHTS`), and falls back to player-only detection when it isn't,
exactly as `YoloDetector`'s docstring describes. `data.yaml`'s class order (`0: ball, 1:
stump`) matches `BALL_AND_STUMPS_CLASSES` exactly, so no `ball_stumps_classes` override is
needed for this checkpoint.

**Real-footage validation (three clips, none from the training dataset) found a real gap,
not a clean pass.** The training/validation mAP above is measured against Roboflow's own
held-out split; three independent clips (stock nets footage, two different angles, plus a
user-supplied 4K clip) tell a different story:

- Clip 1 (nets footage, ball not yet released in the trimmed window): correctly reported no
  ball track — an honest negative, not a detection.
- Clip 2 (same footage, full 9s including the actual delivery): the "ball" track locked onto
  a **static white sack sitting on the ground by the fence** for 43/90 frames at a
  borderline-confident 0.40-0.68 — a real false positive, not the ball. `shot_type=DRIVE`
  was produced from that false track, which is worse than reporting `unknown`: a wrong
  answer that looks confident.
- Clip 3 (a different, higher-quality nets clip): no ball false positive this time, but also
  no true positive — the model never found the real ball. It did correctly detect
  **stumps** in 8/100 frames, which is a genuine, useful partial result: the stumps half of
  this checkpoint generalizes better than the ball half.

**Conclusion: this checkpoint alone is not reliable enough to drive shot classification or
calibration on arbitrary footage.** It clearly overfit to Roboflow's own image distribution
(likely camera angle, lighting, and ball motion blur characteristics) rather than learning a
robust "small fast-moving round object" detector — a stationary round/light-colored
background object was mistaken for the ball with just enough confidence to pass the
pipeline's detection threshold.

**Mitigation shipped: a temporal-motion check, not a better checkpoint.**
`video_engine.events.heuristic_segmenter._looks_like_real_motion_detections` now rejects any
candidate ball track whose detections don't move at least `3x` their own bounding-box size
across the clip — a real ball crosses most of the frame during a delivery; a static false
positive re-detected frame after frame doesn't move beyond ordinary inference jitter. The
check pools ball detections across every track_id before measuring motion (not per track_id)
because IOU-based tracking fragments a genuinely fast ball into several short-lived tracks
whose individual overlap is often zero — checking motion per-track would reject real balls
for the same reason it correctly rejects static ones. This closes the specific failure mode
found above (a wrong-but-confident `shot_type` from a static object) by falling back to the
honest `unknown` with a note explaining why, instead of a silently wrong answer.

**What this does not fix**: recall. The checkpoint still doesn't reliably detect a real ball
in flight (clip 3 above found stumps but never found the ball at all) — the motion check only
prevents false positives from being trusted, it can't manufacture a true positive. Improving
recall still needs one of: more real (non-Roboflow) training data with motion-blurred balls
in flight, or a different detection approach entirely (e.g. classical background-subtraction
tracking, which is what small-fast-object trackers typically use instead of a frame-by-frame
CNN). Pitch calibration and the pitch length/line metrics below remain unblocked in code but
are **not safe to trust** until real ball recall improves — the motion check makes a bad
detection safe, not a missing one available.

The rest of this section records the original survey and the licensing reasoning behind
*not* using the two public alternatives below, since a training-from-scratch decision that
looked expensive at the time is worth keeping the reasoning for.

What a survey of publicly available models turned up:

- **`sanjusabu/Cricket-Ball-and-Stumps-Detection`** — despite the name, not usable in any
  form. The repository is ~25 KB total: one Jupyter notebook and a README. No dataset, no
  weights, no licence file. A dead end.
- **`kushagra3204/Cricket-Ball-Trajectory-Prediction`** — a real, genuinely fine-tuned
  YOLOv8 ball detector. Verified from the repository's own files rather than its README:
  `data.yaml` declares exactly one class (`0: cricketBall`), and
  `runs/detect/train4/args.yaml` records a real Ultralytics training run (`yolov8l.pt`
  base, 80 epochs, real dataset path). Five training runs exist at different model sizes;
  `modelSave.py` exports `train3`'s `best.pt` (~6.25 MB, yolov8n-sized) to ONNX. The
  associated dataset is published separately on Kaggle (~1,778 annotated images). Two
  problems: it is **ball-only**, so it supplies neither half of the stumps pair that
  calibration needs, and **the repository has no LICENSE file**, which under default
  copyright means all rights reserved.

**The decision: do not use those weights in this pass, and do not assume permission.**
No licence file means not redistributable and not reusable without the author's explicit
consent. Vendoring them, downloading them at runtime, or writing code that assumes their
availability would all bake in a permission nobody has been given. So this pass spent its
effort on the two things that are genuinely blocked on *design* rather than on someone
else's licence:

1. **Calibration** (`video_engine/calibration.py`) — the stump-anchored homography that
   turns pixels into metres, implemented and tested against a synthetic camera. See below.
2. **Schema plumbing** — `FusedDelivery.pitch_length_m` / `pitch_line_m` /
   `pitch_length`, `Metric`'s four pitch metrics, and the flagging logic that consumes
   them, all in place and tested, all inert until a detector exists.

Getting the calibration approach settled *first* is deliberate: it determines what the
detector has to detect. Training a detector and then discovering the geometry needed
something else from it is the more expensive order.

**If you want to supply a different checkpoint** (or retrain this one — see
`data/datasets/ball_stump_v1/data.yaml` for the exact dataset reference), `YoloDetector`
takes `ball_stumps_weights` (a path you provide, and whose licence is yours to establish)
plus `ball_stumps_classes`, which declares what the class indices mean:

```python
from video_engine.detection.yolo_detector import YoloDetector, BALL_ONLY_CLASSES

detector = YoloDetector(
    ball_stumps_weights="/path/to/your/ball.pt",
    ball_stumps_classes=BALL_ONLY_CLASSES,   # a one-class ball detector
)
detector.emits(ObjectClass.STUMPS)  # False — pitch calibration stays unavailable
```

The class map is an explicit argument rather than an assumption precisely because a
one-class ball model loaded under the default two-class map would emit correct `BALL`
detections and silently never emit `STUMPS` — indistinguishable from "no stumps were
visible in this clip", and it would leave every stumps-dependent feature quietly
disabled with no error anywhere. `emits()` answers the capability question so callers can
tell "this detector can't" from "this clip didn't".

Of the two realistic routes considered at the time — (a) obtain the author's permission
for the ball-only weights and dataset, then source or annotate a stumps dataset to
complete it, or (b) train a 2-class model from scratch — this project took (b), using the
CC BY 4.0 Roboflow dataset described above rather than the two unusable/unlicensed options.

### Pitch calibration: implemented, unblocked, pending real-footage validation

`video_engine/calibration.py` maps image pixels onto the pitch plane via a homography
anchored on the stumps' known real-world size — the same approach low-cost fixed-camera
setups use in the wild (Fulltrack AI's published setup instructions specify a fixed
elevated tripod with stumps visible at both ends of frame, for exactly this reason).

Reference geometry is from the Laws: 20.12 m (22 yards) **stumps to stumps**, 0.711 m
stump height, 0.2286 m across the stump block. Worth being precise about the first: it is
easy to quote 20.12 m as the distance between the *popping* creases, which is a different
and shorter measurement — the popping creases are 1.22 m in front of each set of stumps,
so popping crease to popping crease is 17.68 m. This module measures from the stumps,
because the stumps are what the calibration reference detects.

The four ground-level stump-base corners (two ends × two outer stumps) are exactly the
four coplanar correspondences a homography needs. `calibrate_from_stumps` solves it by
normalised DLT and returns `None` — never a best guess — when the stumps are missing,
degenerate, coincident, or the fit exceeds `MAX_REPROJECTION_ERROR_PX`.

**Stated limits, because a number in metres looks authoritative in a way a pixel
coordinate does not:** the four anchors form a 0.2286 m × 20.12 m rectangle, an aspect
ratio of about 88:1, so the fit is well conditioned along the pitch and poorly
conditioned across it — **line is inherently less accurate than length**, and worse at
the far end where the stumps occupy few pixels. It assumes the bounce lies on the pitch
plane (true for a bounce, false for a full toss, which is why a full toss is reported
from the ball's trajectory rather than from a projected position) and a fixed camera for
the delivery (a broadcast cut mid-delivery invalidates it, and nothing detects one).

`tests/video_engine/test_calibration.py` verifies the actual mathematical property —
points are projected through a synthetic camera, a calibration is recovered from the
resulting "detections", and other points are mapped back and checked against the metres
they started from — rather than pinning whatever the code returned the day it was
written.

## Rating and coaching suggestions

`rating/` implements PRD stages 5-6 on top of `fusion`'s output. Three layers, in
strict order, and the order is what makes the output auditable:

1. **Rule/flagging layer** (`suggestions.flag_player`) -- deterministic arithmetic over
   real `FusedDelivery` rows against a `RatingBaseline`, producing `PerformanceFlag`s
   that carry the metric, the measured value, the baseline, the delta, and real
   `DeliveryRef` citations. No LLM touches this layer and none can.
2. **Ranking** (`suggestions.rank_flags`) -- a documented deterministic formula.
3. **Note-writing** (`rating/llm.py`) -- phrasing only, from an already-computed flag.

That split is how PRD 2.4's "no black-box verdicts -- every AI-derived suggestion must
cite its underlying clip and stat so an analyst can verify it" gets enforced rather than
merely intended: a model that cannot originate a number or a citation cannot produce an
unverifiable claim.

### What's implemented

- **Composite rating** (`rating/engine.py`) -- PRD 3.1's four weighted pillars on a
  0-100 scale where **50 means "exactly at the cohort baseline"**, not "half marks".
  Weights are tunable per role in one table (`contracts.ROLE_PROFILES`); the two roles
  PRD 3.1 tabulates ("top-order batter", "death-overs bowler") are there, and adding a
  role is a one-entry change with no code change in the engine.
- **Execution, Game impact and Consistency** are computed from real data today. Game
  impact weights each delivery by phase (`PHASE_IMPACT_WEIGHTS`) *and* compares it
  against that phase's own cohort rate, so a batter is neither penalised for facing a
  hard phase nor credited merely for batting at the death.
- **Flagging** covers strike rate, economy, dot-ball %, boundary % and per-phase strike
  rate/economy, with citations chosen deterministically and **preferring deliveries that
  actually have footage** (falling back to ball-by-ball-only refs, and saying so via
  `PerformanceFlag.citations_have_video`).
- **LLM note-writer** (`rating/llm.py`) -- `OllamaClient` against a local Ollama server,
  with `TemplateNoteWriter` (the PRD's curated template library) as a hard fallback.
  Both implement the same small `NoteWriter` interface and are swappable at the call
  site. Suggestion **titles are never model-written**; only bodies are.
- **Human-in-the-loop** is scaffolded only: `CoachingSuggestion.status` is
  `"pending" | "accepted" | "edited" | "rejected"`, defaulting to `"pending"`. No UI
  consumes it yet.

### Technique pillar: weight-redistributed, not measured

PRD 3.1 weights Technique at 20-30%. **It is never scored, for any player.** Its only
possible inputs are `FusedDelivery`'s biomechanics fields, every one of which is `None`
because no calibrated extractor exists -- what `video_engine` produces is wrist
displacement in raw pixel space with no camera calibration, which cannot become a real
joint angle or speed (see `fusion/contracts.py`).

The deliberate choice was **(a) proportional weight redistribution**: Technique's weight
is removed and the other three rescaled to sum to 1.0. Option (b) (returning `None`) was
rejected because Technique is missing for *every* player, so (b) would make the entire
engine return `None` for every input -- untestable dead code. Option (b) is still
reachable and not dead: when *nothing* is measurable, `composite` is `None` and
`insufficient_data` is `True`, rather than a number invented from an empty weight set.

The cost is real and is not hidden: a redistributed composite is **not** comparable to a
future four-pillar composite. `PlayerRating.unmeasured_pillars` and `weights_used` are
not debug fields -- any surface showing `composite` must show them too.
`engine._technique_score` deliberately does not read the biomechanics fields even if a
caller populates them, and there's a regression test pinning that.

### Baseline: cohort/season, not PRD 3.1's matchup baseline

PRD 3.1 asks for a matchup baseline (this batter against this class of bowling). That
needs `bowling_style`/`batting_style` metadata Cricsheet's registry doesn't carry -- the
same gap `prediction/contracts.py` documents and declines to fake. So
`engine.compute_baseline` pools **every player in the supplied sample**, over *totals*
rather than a mean of per-player rates, so a four-ball cameo at a strike rate of 300
can't drag the cohort. `RatingBaseline.source` describes the cohort in words and is
carried onto every flag and suggestion -- "15% below baseline" is itself a black-box
verdict unless the analyst can see which baseline.

**Which matches go into that sample is now an explicit choice (`rating/cohort.py`).**
It used to be whatever was on disk, and that was only ever safe by accident: `ParquetStore`
overwrote on every ingest, so a warehouse was implicitly one competition. Once writes
started merging, the same call pooled the IPL, the BBL, the PSL, the NPL, several World
Cups and the full men's *and* women's T20I record into a single "average player" who
resembles nobody — and every flag scored against it inherited that. A multi-competition
warehouse made the old default a correctness problem, not just a slow one.

`cohort_for_player` reads the competition, gender and match type off the player's **own
recent matches** and scopes the cohort to those, so a player rated on their last five IPL
matches is compared against IPL cricket. Three deliberate biases, all reported rather than
hidden:

- **Like-for-like scoping**, inferred from the player's own window — never guessed from
  anything the data doesn't carry.
- **A recency bound** (`DEFAULT_SEASONS_BACK = 3`), anchored to the most recent match *in
  the data*, not to today's clock — so the same warehouse and spec select the same cohort
  next year, and a stored suggestion stays checkable against the baseline it cites.
- **A cap** (`DEFAULT_MAX_COHORT_MATCHES = 600`), applied most-recent-first, bounding the
  work one rating can trigger however large the warehouse grows. This is a real sampling
  bias toward recent matches, so `CohortSelection.was_capped` and `matches_available`
  expose it and `source` says "most recent 600 of 4,120" rather than just "600 matches".

`CohortSpec` overrides any dimension explicitly, and `WHOLE_WAREHOUSE` restores the old
unbounded pooling for a genuinely single-competition warehouse — reachable and named,
rather than reachable by omission.

### Rating at warehouse scale: select, read, then fuse

The rating path used to fuse **every match in the warehouse** into `FusedDelivery`
objects and then filter the objects down to the handful it needed. That is
O(matches × deliveries): selecting one match's rows costs a scan of the whole frame, and
it did that once per match. Measured on a warehouse of 8,026 matches / 1.83M deliveries,
one rating cost **557.7 s** before producing anything. The same loop on the old
single-competition warehouse (230 matches) took 1.3 s — the per-match cost had grown from
~6 ms to ~69 ms purely because the *frame* grew.

`rating/pipeline.py` inverts the order:

1. `cohort.cohort_and_window` picks the match ids — cohort plus the player's window —
   from the small match table, touching no delivery rows.
2. `ParquetStore.read_deliveries(match_ids=...)` pushes that selection down to the
   Parquet reader (`pyarrow` `filters=`), so rows outside it are never decoded. "Which
   matches has this player played in" likewise pushes down, as a disjunction over
   striker/non-striker/bowler.
3. The cohort baseline is pooled **vectorised** (`engine.baseline_from_deliveries`) —
   every field on `RatingBaseline` is a pooled total, so the per-delivery objects cancel
   out of the sums entirely.
4. Only the player's window — typically five matches, ~1,000 deliveries — is fused into
   objects, because that is the only place per-delivery identity is used (phase weighting,
   and the `DeliveryRef` citations a flag must carry).

Measured on the same 8,026-match warehouse: **557.7 s → 0.50 s** cold, and ~0.27 s per
player once the cohort baseline is cached (it depends on the cohort and the warehouse,
never on who is selected). `tests/app/` went from 10 m 52 s to ~3 s.

`engine.baseline_from_deliveries` is pinned to produce **field-for-field identical**
output to `compute_baseline` over the same rows, rounding included — the fast path is the
same answer, not a close-enough reimplementation. One asymmetry is documented rather than
glossed: pitch geometry is video-derived and has no column in the ball-by-ball table, so
the vectorised path always returns `pitch=None`, and the equality holds only while no
fused row carries geometry. The test asserts that precondition rather than assuming it,
so it fails loudly when a detector lands.

The scale tests (`tests/rating/test_pipeline_scale.py`) assert the **property** — same
work per rating against warehouses an order of magnitude apart — rather than a stopwatch
reading, which would be flaky on shared CI while saying less about why it regressed. One
end-to-end case at ~8,000 matches is marked `slow` and still runs by default; a
regression here would be a 500x one, so it shouldn't be opt-in.

### Pitch length and line: contracts ahead of the producer

`Metric` now carries `GOOD_LENGTH_PERCENT`, `SHORT_BALL_PERCENT`, `FULL_BALL_PERCENT` and
`STUMP_LINE_PERCENT`, `RatingBaseline` carries an optional `PitchBaseline`, and
`suggestions.flag_player` computes and cites them. **None of it fires today**, because no
delivery carries a bounce point — see "Ball and stumps detection" above.

This is a deliberate narrowing of a rule `Metric`'s docstring used to state absolutely
("nothing is reserved-but-unpopulated in this enum"). The distinction that actually
matters is not "does this fire today" but "is whether it fires decided by the data or by
a branch unreachable in principle". `flag_player` already emitted `DOT_BALL_PERCENT` and
`BOUNDARY_PERCENT` only when the baseline carries the corresponding rate — conditional
emission driven by the input, which is exactly the shape these use. The revised rule is
written out in `rating/contracts.py`.

They are expressed as **percentages of deliveries in a band**, not as a mean length in
metres: a bowler alternating yorkers and bouncers averages a good length and has bowled
neither, and a mean has no defensible direction for `higher_is_better`. Band boundaries
live in `calibration.LENGTH_BANDS` and are a documented convention, not fitted values —
published bands differ between providers by up to a metre at every boundary, and nothing
here has the labelled ball-tracking data to fit them.

Two guards are load-bearing and tested. The denominator is **deliveries that carry
geometry**, not all deliveries — otherwise a bowler's good-length share reads near zero
whenever most of their deliveries simply weren't filmed, turning a coverage artefact into
a bowling problem. And a flag needs *both* a cohort with geometry and a player window with
geometry: a covered cohort plus an uncovered player would otherwise produce a damning,
citation-backed "0% good length" finding out of missing footage.

### Ranking: deterministic stand-in for the PRD's "learned ranker"

`rank_flags` orders by `abs(relative_delta) * min(1, sample_size / 30)` -- relative, so
metrics in different units compete fairly, and sample-discounted, so a six-ball blip
can't outrank a 40-ball pattern. Every tie-break is total (concerns before strengths,
then metric, phase, player), so input order can't change the output.

It is **not** a learned ranker, deliberately. Training one needs labelled data about
which suggestions analysts found useful, which is exactly what
`CoachingSuggestion.status` exists to accumulate. Fitting one now would mean inventing
the labels. Swapping in a real ranker later means replacing this one function and
nothing else.

### The local LLM, and how citations are kept honest

`OllamaClient` posts to `/api/generate` on `http://localhost:11434`. Base URL and model
are configurable via `THIRD_UMPIRE_OLLAMA_URL` / `THIRD_UMPIRE_OLLAMA_MODEL` or
constructor arguments -- nothing is hardcoded to one model or host.

Every generated note must (1) reproduce each citation token character for character and
(2) contain no number the flag didn't supply. A failure at any point -- unreachable
server, timeout, malformed JSON, dropped citation, invented number, `requests` not
installed -- retries once and then returns the `TemplateNoteWriter` note. **The failure
mode is always "blunter phrasing", never "wrong number" or "no suggestion"**, and
`CoachingSuggestion.note_source` records which one the analyst is reading.

**Model choice: `llama3.2:3b` is the default -- chosen by measurement, across two
rounds of it, the second of which overturned the first.**

*Round one.* General benchmarks rank `phi3.5` (Microsoft's 3.8B instruct model,
MIT-licensed) well ahead of `llama3.2:1b` on breadth:

| Model | Params | Disk | MMLU | GSM8K | Context | Licence |
|---|---|---|---|---|---|---|
| Llama 3.2 1B | 1B | ~1.3GB | ~49% | much weaker | 128K | Llama 3.2 Community |
| Gemma 2 2B | 2B | ~1.6GB | ~52% | ~40% | 8K | Gemma |
| Llama 3.2 3B | 3B | ~2.5GB | ~63% | ~77% | 128K | Llama 3.2 Community |
| Qwen2.5 3B | 3B | ~1.9GB | ~65% | ~79% | 32K | Qwen (not Apache at this size) |
| Phi-3.5-mini | 3.8B | ~2.2GB | ~69% | ~86% | 128K | MIT |

`phi3.5` was set as the initial default on that basis. But **no public benchmark
measures this task** (coaching-note generation with mandatory verbatim citation echo,
zero fabricated numbers) -- so it was measured against the actual job with
`scripts/eval_llm_notewriter.py`. `phi3.5`'s greater fluency worked against it: it
"helpfully" restates and elaborates the given facts, which reads well but reliably
introduces a number that wasn't in the input -- exactly the failure PRD 2.4 forbids.
`llama3.2:1b` won that round and became the default.

*Round two.* The eval's fixture only tested flags with one or two citations. Running
`suggest_for_player` against **real warehouse data** (not the fixture) showed why that
mattered: real flags carry up to three citations (`suggestions.MAX_CITATIONS_PER_FLAG`),
and `llama3.2:1b` echoed **0 of 24** real suggestions' citations correctly -- every
single one fell back to the template, despite measuring ~60% "accepted" on the fixture
that never tested the harder case. The fixture was corrected to include a
three-citation case, and `llama3.2:3b` was measured against both the corrected fixture
and the same real data:

| Model | Citation fidelity (eval) | Number fidelity | Accepted (both) | Real data (24 flags) | Latency |
|---|---|---|---|---|---|
| `llama3.2:1b` | 59.5% | 100% | 59.5% | **0 / 24 (0%)** | 2.8s |
| `phi3.5` | 74.4%* | 41.0%* | 41.0%* | not retested | 9.8s |
| **`llama3.2:3b`** | **88.1%** | **100%** | **88.1%** | **23 / 24 (95.8%)** | 4.4s |

\* measured against the pre-fix, two-citation-max fixture -- not re-run against the
corrected one, since it already lost round one and the fix only makes three-citation
cases harder, not easier.

`llama3.2:3b` isn't a compromise: it beats `llama3.2:1b` on citation fidelity *and*
matches it at 100% number fidelity, with no fluency-vs-fabrication trade-off the way
`phi3.5` had one, and it's still faster than `phi3.5`.

```bash
python scripts/eval_llm_notewriter.py                       # the default model
python scripts/eval_llm_notewriter.py --model llama3.2:1b   # any pulled model
THIRD_UMPIRE_LLM_EVAL=1 pytest tests/rating/test_llm_citation_fidelity.py -v
```

**Lesson for future model swaps:** general capability benchmarks do not predict
performance on this task, in either direction -- and neither does a synthetic eval
fixture whose citation-count distribution doesn't match production. Re-run
`scripts/eval_llm_notewriter.py` *and* spot-check real `suggest_for_player` output
against real data before trusting a comparison or changing `DEFAULT_OLLAMA_MODEL`.
`llama3.2:1b` and `phi3.5` are both kept as documented, measured-and-rejected
alternatives in `rating.llm.ALTERNATIVE_OLLAMA_MODELS`; `qwen2.5:3b` remains untested.
The fidelity test's floor (`MIN_CITATION_FIDELITY = 0.80` in
`tests/rating/test_llm_citation_fidelity.py`) is calibrated to `llama3.2:3b`'s
measured rate with margin, not to an aspirational number.

### Tests

`tests/rating/` covers the rating math (including the Technique-missing case and a
regression test that populated pixel-space biomechanics still produce no score), flag
generation and citation selection, ranking determinism, and note-writing against a
`FakeNoteWriter`/`FakeOllamaTransport` (`tests/rating/fakes.py`). **No test in the
default suite needs a live Ollama server**; the live check skips unless
`THIRD_UMPIRE_LLM_EVAL=1` is set and the server answers.

Added with the scale and pitch-geometry work:

- `tests/rating/test_cohort.py` — which matches get into a baseline, and whether the
  resulting `source` string tells an analyst what happened (including that the recency
  window is anchored to the data, not the clock, and that `player_match_ids` still agrees
  with `player_reports.last_n_match_ids` on the same-date tie-break).
- `tests/rating/test_pipeline_scale.py` — builds synthetic warehouses up to ~8,000
  matches and asserts the scaling *property*: the same work per rating at 20x the
  warehouse, the cohort pooled but never fused, an existing rating unchanged by matches
  outside its cohort, and the vectorised baseline field-for-field equal to the old one.
- `tests/rating/test_pitch_metrics.py` — the pitch metrics' arithmetic when geometry
  exists, and (equally load-bearing) their silence when it doesn't.
- `tests/video_engine/test_calibration.py` — round-trips points through a synthetic
  camera to verify the homography is genuinely the inverse of a known projection, plus
  every documented refusal path.

Known environment gaps in this checkout, not test failures: `tests/scoreboard_ocr`,
`tests/test_bytetrack_tracker.py`, `tests/test_iou_tracker.py` and
`tests/test_pipeline_smoke.py` need `ultralytics` / `supervision`, and some
`tests/ingestion` video-adapter tests need the `ffmpeg` binary. Install those to run
them; `tests/video_engine/test_detector_class_maps.py` skips itself cleanly without
`ultralytics` rather than failing to collect.

## Prototypes (beyond the current MVP scope)

Speculative pieces from a broader architecture pitch, built standalone (not wired into
`ingestion` or `video_engine`) to test whether the approach holds up before deciding
where they'd plug in.

- **`scoreboard_ocr/`** -- reads runs/wickets/overs off a broadcast scoreboard graphic
  with PyTesseract (PSM 7 + regex), prototyping the pitch's "OCR Extraction Framework"
  (intended as a gating signal ahead of highlight detection). `parse_scoreboard_text` is
  pure regex, tested independently of OCR; `read_scoreboard` runs the real Tesseract
  binary against a rendered image. Needs the Tesseract binary installed separately (not
  a pip package) -- `winget install --id UB-Mannheim.TesseractOCR` on Windows.
  **Accuracy caveat:** measured 100% on 60 randomized *clean, synthetic* scoreboard
  renders -- that's a sanity check on the parsing logic, not a claim about real
  broadcast footage, which has motion blur, compression artifacts, and far more varied
  graphic styles. The pitch's "99%+ accuracy" figure is unverified against real footage;
  don't repeat it as fact until it's been measured against actual broadcast crops.

## Security notes

- **`torch>=2.6` is a pinned floor, not just a version bump.** That release switched
  `torch.load`'s default to `weights_only=True`, which stops a malicious or corrupted
  `.pt` checkpoint from executing arbitrary code via pickle on load. Don't downgrade
  below it, and don't override `weights_only` when loading a checkpoint from outside
  the team.
- **Only load model weights from a source you trust.** Treat an unfamiliar `.pt` file
  the same as you'd treat unfamiliar executable code.
- **All dependencies are version-range or exact pinned** (`pyproject.toml` /
  `requirements.txt`) rather than left unbounded, so a future release can't silently
  change behavior or pull in a compromised transitive dependency without the pin being
  revisited.
- **`video_path` is trusted input for now**, on both sides of the ingestion/video-engine
  boundary. That's fine while everything runs in the same trusted pipeline; if a future
  API ever accepts a path (or an uploaded file) from an external caller, validate and
  sandbox it before it reaches `ManifestClipAdapter`, `SceneSplitAdapter`, or
  `load_frames`.
- **`CricsheetSource.fetch` downloads a zip over HTTPS from a fixed, hardcoded host**
  (`cricsheet.org`) and extracts it with `zipfile.extractall` -- fine for a single trusted
  public source; don't point this at an arbitrary/untrusted URL without adding path
  validation against zip-slip first.

## Known gaps (tracked, not blocking)

- No fine-tuned ball/stumps detector yet -- needs a labelled cricket dataset. Surveyed,
  with the findings and the licensing decision written up under "Ball and stumps
  detection" above: the one real public cricket-ball detector is ball-only and carries no
  licence file (all rights reserved), so it is **not** used here and no permission to use
  it is assumed. `YoloDetector` takes an operator-supplied checkpoint plus an explicit
  `ball_stumps_classes` map so a ball-only model can't silently pass as a ball+stumps one.
- `video_engine/calibration.py` (stump-anchored homography, pixels to pitch metres) is
  implemented and tested but **has no caller in the pipeline**, because it needs the
  stumps detections above. Its accuracy limits -- line much less reliable than length,
  given the 88:1 anchor rectangle -- are documented in the module and in the README
  section above.
- `rating/`'s pitch-length/line metrics (`GOOD_LENGTH_PERCENT`, `SHORT_BALL_PERCENT`,
  `FULL_BALL_PERCENT`, `STUMP_LINE_PERCENT`) and `FusedDelivery`'s `pitch_length_m` /
  `pitch_line_m` / `pitch_length` exist, are tested against hand-built rows, and are
  inert on real data for the same reason. Contracts ahead of the producer, same pattern
  and same justification as the biomechanics fields.
- Pitch **line** cannot be reported as off side / leg side at all, at any point, without
  batting-handedness metadata Cricsheet doesn't publish. `pitch_line_m` is signed
  distance from the middle stump and is deliberately unlabelled -- the same gap
  `prediction/contracts.py` records for matchups.
- The "batter" in a clip is inferred as whichever player track is nearest the ball at
  contact; there's no explicit role labelling (batter vs. bowler vs. fielder) yet.
- Shot classification is a hand-set heuristic on wrist displacement, not learned.
- `SceneSplitAdapter`'s scene-cut detection is untuned against real broadcast footage --
  the `threshold` default (0.3) and the uniform-split fallback are both starting points.
- Ingestion <-> video-engine fusion (PRD 2.2 stage 3): contracts *and* join/aggregation
  logic both exist now (`src/fusion/` -- `fuse_match_deliveries`, `fuse_matches`,
  `fuse_and_aggregate`, `aggregate_player_match_stats`, the rolling-summary refreshers).
  This entry previously claimed the logic didn't; it was stale. What is still missing is
  anything to fuse *with*: `BroadcastFeedAdapter` is unimplemented, so every
  `FusedDelivery` in practice has `video_available=False` and the join runs against an
  empty analysis set.
- `ingestion.sources.cricsheet` currently only handles Cricsheet's men's/women's
  international and league JSON schema (`data_version` 1.x); a licensed-feed source will
  need its own parser behind the same `StatsSource` interface.
- `player_reports` covers only stats already in the ball-by-ball table -- no fielding,
  no phase-by-phase breakdown, no matchup-vs-baseline (PRD 3.1's "Execution" pillar
  needs that comparison, this just reports raw totals). `rating/` now does the
  phase-by-phase and vs-baseline work on top of `fusion`'s rows.
- `rating/`'s Technique pillar is never scored and its weight is redistributed across
  the other three -- blocked on the same missing calibrated biomechanics extractor as
  `fusion`'s `None` fields, not on rating code. A redistributed composite is not
  comparable to a future four-pillar one; see "Rating and coaching suggestions" above.
- `rating/`'s baseline is a cohort/season mean, not PRD 3.1's matchup baseline -- that
  needs `bowling_style`/`batting_style` metadata Cricsheet doesn't publish, the same gap
  `prediction/contracts.py` records. The *cohort* is now chosen deliberately rather than
  being "whatever is on disk" (see "Baseline" above), but a like-for-like competition
  cohort is still a weaker comparison than a real matchup baseline.
- The cohort cap (`DEFAULT_MAX_COHORT_MATCHES = 600`) and the recency bound
  (`DEFAULT_SEASONS_BACK = 3`) are picked, not fitted -- same status as
  `PHASE_IMPACT_WEIGHTS` and `RATIO_SCORE_SPREAD`. Fitting them would mean measuring when
  cohort rates stop predicting the next season's, which needs the multi-season warehouse
  this work was done to make usable in the first place.
- The warehouse is still two Parquet files read whole-file with predicate pushdown. That
  is enough at ~1.8M deliveries (a filtered cohort read is well under a second), but it
  is not partitioned by competition or season, so a filtered read still opens one large
  file. Partitioning, or the "real database" this is heading toward, is the next step if
  the warehouse grows another order of magnitude; `ParquetStore`'s reader-side filters are
  the seam for it, since a SQL store satisfies the same interface with a `WHERE` clause.
- `rating/`'s ranker is a documented deterministic formula standing in for PRD 3.2's
  "learned ranker". Training a real one needs the analyst accept/edit/reject labels that
  `CoachingSuggestion.status` exists to collect and that no UI produces yet.
- PRD 3.3's human-in-the-loop review is a contract field only (`status`), with no UI
  behind it.
- Found via the live LLM eval, now fixed: the generated-note number check counted the
  metric's own unit ("runs per 100 balls") as a fabricated statistic, rejecting
  otherwise-correct notes. Digits arriving as prompt *text* -- the unit, the baseline
  description -- are now treated as input; see
  `test_the_metrics_own_unit_is_not_mistaken_for_an_invented_statistic`.
- Fixed via a two-round model swap: `llama3.2:1b` (an earlier default) echoed a single
  citation almost perfectly but reliably dropped multi-citation flags, and scored
  **0/24 real suggestions correctly cited** against real warehouse data once flags hit
  the real maximum of three citations -- a failure the synthetic eval's fixture didn't
  surface because it only tested up to two. `phi3.5` was tried as the fix and made
  things worse (fabricated numbers instead). `llama3.2:3b` is the actual fix: 88.1%
  accepted on the corrected fixture, 23/24 (95.8%) on the same real data, still faster
  than `phi3.5` -- see "Rating and coaching suggestions" above. `qwen2.5:3b` remains an
  untested candidate if further improvement is wanted.
- Found via testing, now fixed: `last_n_match_ids` used to break same-date ties with
  pandas' default (non-stable) sort, so "last N matches" could silently return a
  different match on repeated calls against identical data whenever two matches shared
  a date (common -- 169 of 230 ingested matches share a date with another). Now sorts
  by `(date, match_id)` for a deterministic order; see
  `test_last_n_match_ids_breaks_same_date_ties_deterministically`.
- Found via testing, now fixed: `ByteTrackTracker` collapsed two real, separately-boxed
  people into one track because the `trackers` library's default confidence thresholds
  (0.6/0.7) let a sub-0.6-confidence detection extend an existing track but never spawn
  its own -- see the Video Pipeline Test section above and
  `test_two_separately_boxed_players_get_two_tracks_even_at_borderline_confidence`.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Video-engine tests run against fakes/stubs for the model-backed stages (`Detector`,
`PoseEstimator`), so they don't need model downloads or a GPU. Ingestion's video-adapter
tests generate tiny synthetic clips with `ffmpeg` (must be on `PATH`) rather than using
real footage, and the stats-source tests run against a trimmed real Cricsheet match
fixture (`tests/ingestion/fixtures/`).
