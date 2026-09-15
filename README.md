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
  a local-LLM note writer with a deterministic template fallback. See "Rating and
  coaching suggestions" below for what's implemented and what's deliberately left open.
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
`engine.compute_baseline` pools **every player in the supplied `PlayerMatchStats`
sample**: feed it one competition's matches and it is that competition's season average.
It pools *totals*, not a mean of per-player rates, so a four-ball cameo at a strike rate
of 300 can't drag the cohort. `RatingBaseline.source` describes the cohort in words and
is carried onto every flag and suggestion -- "15% below baseline" is itself a black-box
verdict unless the analyst can see which baseline.

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

**Model choice: `phi3.5` (3.8B instruct) is the default**, not the `llama3.2:1b` that
happened to be pulled on the dev machine. For a constrained note-writer, instruction
adherence matters far more than breadth:

| Model | Params | Disk | MMLU | GSM8K | Context | Licence |
|---|---|---|---|---|---|---|
| Llama 3.2 1B | 1B | ~1.3GB | ~49% | much weaker | 128K | Llama 3.2 Community |
| Gemma 2 2B | 2B | ~1.6GB | ~52% | ~40% | 8K | Gemma |
| Llama 3.2 3B | 3B | ~2.5GB | ~63% | ~77% | 128K | Llama 3.2 Community |
| Qwen2.5 3B | 3B | ~1.9GB | ~65% | ~79% | 32K | Qwen (not Apache at this size) |
| **Phi-3.5-mini** | 3.8B | ~2.2GB | **~69%** | **~86%** | 128K | **MIT** |

Phi-3.5-mini leads its size class on the closest available proxies for "follow the
constraints exactly, don't embellish", fits a 4GB-VRAM GPU (2.2GB of weights plus a
short KV cache), and is MIT-licensed. `ollama pull phi3.5`. **Llama 3.2 3B is the
documented alternative** for a Meta-ecosystem preference; `llama3.2:1b` stays a
supported swap-in for low-resource machines. See `rating.llm.ALTERNATIVE_OLLAMA_MODELS`.

Those are borrowed leaderboard numbers, and **no public benchmark measures this task**
(coaching-note generation with mandatory verbatim citation echo). The evidence that
matters for this deployment is measured here:

```bash
python scripts/eval_llm_notewriter.py                        # the default model
python scripts/eval_llm_notewriter.py --model llama3.2:1b    # any pulled model
THIRD_UMPIRE_LLM_EVAL=1 pytest tests/rating/test_llm_citation_fidelity.py -v
```

Measured on this machine with **`llama3.2:1b`** (13 synthetic flags x 3 runs = 39
generations, `max_attempts=1`): **citation fidelity 62-69%** across repeated samples,
**number fidelity 100%**, mean latency 2.70s.

The failure pattern is sharp and worth knowing, because the headline number hides it:

| Flag carries | Generations | Every citation echoed |
|---|---|---|
| one citation | 27 | 26 (96%) |
| two citations | 12 | 1 (8%) |

The 1B model reproduces a single token almost perfectly and reliably drops one of a
pair. That is the empirical case for a larger default, and the reason the fidelity
test's floor is 85%. Note what was *not* done in response: capping
`MAX_CITATIONS_PER_FLAG` at one would make the number look fine by giving the analyst
less evidence, which is tuning the evidence to suit the model.

`phi3.5` is not pulled on this machine, so its own pass rate is unmeasured here --
`ollama pull phi3.5` and re-run the script above to fill that in.

### Tests

`tests/rating/` covers the rating math (including the Technique-missing case and a
regression test that populated pixel-space biomechanics still produce no score), flag
generation and citation selection, ranking determinism, and note-writing against a
`FakeNoteWriter`/`FakeOllamaTransport` (`tests/rating/fakes.py`). **No test in the
default suite needs a live Ollama server**; the live check skips unless
`THIRD_UMPIRE_LLM_EVAL=1` is set and the server answers.

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

- No fine-tuned ball/stumps detector yet -- needs a labelled cricket dataset.
- The "batter" in a clip is inferred as whichever player track is nearest the ball at
  contact; there's no explicit role labelling (batter vs. bowler vs. fielder) yet.
- Shot classification is a hand-set heuristic on wrist displacement, not learned.
- `SceneSplitAdapter`'s scene-cut detection is untuned against real broadcast footage --
  the `threshold` default (0.3) and the uniform-split fallback are both starting points.
- Ingestion <-> video-engine fusion (PRD 2.2 stage 3): data contracts exist
  (`src/fusion/contracts.py` -- `FusedDelivery`, `PlayerMatchStats`, `PlayerRollingSummary`),
  join/aggregation logic doesn't yet.
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
  `prediction/contracts.py` records.
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
- Measured, not fixed: `llama3.2:1b` echoes a single citation almost perfectly (26/27)
  but drops one of a pair when a flag carries two (1/12), so multi-citation notes fall
  back to templates on it. This is the empirical reason `phi3.5` is the default;
  `phi3.5` itself is unmeasured here because it isn't pulled on this machine.
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
