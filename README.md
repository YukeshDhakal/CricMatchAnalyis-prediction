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
- **Fusion / rating / suggestion engine** — not started yet; consumes `DeliveryAnalysis`
  from `video_engine` alongside `ingestion`'s ball-by-ball table.

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
- Ingestion <-> video-engine fusion (PRD 2.2 stage 3) isn't built yet.
- `ingestion.sources.cricsheet` currently only handles Cricsheet's men's/women's
  international and league JSON schema (`data_version` 1.x); a licensed-feed source will
  need its own parser behind the same `StatsSource` interface.

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
