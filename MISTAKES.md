# Mistakes

A running log of things that went wrong, what was actually happening, and what fixed it or
what to do differently next time. This is not a blame log — it exists so the same mistake
doesn't get repeated three sessions from now. Newest first.

---

## Took a memory metric at 57% as proof that memory was not the problem (2026-09-23)

**What happened**: a real 13.7 MB 4K/60fps clip was submitted to the live Railway service.
Partway through the job a poll returned a Railway edge `502 Application failed to respond`
and the logs showed the container restarting mid-analysis. `railway metrics` reported a
memory peak of 13.5 GB against a 24 GB limit — 57% — and a CPU peak of 3.5 of 24 vCPU. Both
were read as ruling out a resource kill, which left the GIL as the suspect: the pipeline
ran as a *thread* inside the uvicorn process, so the theory was that CPU-bound inference
starved the asyncio event loop until Railway's edge gave up.

**What was actually happening**: 13.5 GB *was* the bug, measured. The clip is 598 frames at
3840×2160, and `load_frames` decoded every one of them into a full-resolution uint8 RGB
array and held the whole list for the length of the job. 598 × 23.73 MiB is **13.86 GiB** —
within rounding of the reported "peak", and reached about 14 seconds into the job. The
metric was not showing headroom; it was showing the frame list, sampled at its plateau,
with detection and pose estimation still to pile their own allocations on top. A single
pose batch of eight 4K frames was measured at a further 3.74 GiB transient.

**Why the GIL theory was wrong, and how that was settled**: by measuring instead of
reasoning. A pipeline-shaped CPU-bound workload — 4K `cv2.cvtColor`, a torch conv forward,
`to_tensor` on 4K frames, and a plain-Python IoU loop — was pinned in a `ThreadPoolExecutor`
while asyncio's tick overshoot was sampled. Worst case: **51 ms**. CPython preempts a
pure-Python thread every 5 ms and OpenCV and torch release the GIL around native calls.
Event-loop latency in the tens of milliseconds does not make an edge proxy give up after
tens of seconds. (Through the full HTTP path the degradation is larger than that — a median
of ~540 ms under sustained pure-Python load — which is bad and worth fixing, and still two
orders of magnitude short of a timeout.)

**The part that decided it**: no `healthcheckPath` is configured on this service, so Railway
has no mechanism to restart a container for answering slowly. Slow responses produce 502s.
They do not produce `Started server process [1]`. Something *killed the process*, and on a
container the thing that kills a process for resource reasons is the OOM killer. The
metric's own number said which resource.

**Why it was nearly a wrong fix**: the proposed remedy — move the pipeline to a
`ProcessPoolExecutor` — is a good change for other reasons and would have been **strictly
worse on its own**. It does not reduce memory by a byte. It would have moved the OOM kill
from the server to the worker, and `ProcessPoolExecutor` is permanently broken after a
worker dies uncleanly: every subsequent job would have failed with `BrokenProcessPool`
until someone restarted the container. That trades a loud crash that self-heals for a
quiet one that does not, and every test would still have passed, because no test ever
decoded a real 4K clip.

**Why no test caught it**: every test that exercised `load_frames` built its frames itself,
or used small fixtures. The bug lives entirely in the step between a file on disk and a
list of arrays, and a test that starts from arrays cannot see it. The upload limit, the
only guard in front of this, is expressed in bytes — and the ratio between an H.264 file
and its decoded frames is a property of the codec, about **1083x** for this clip. Nothing
about "13.7 MB" hints at 13.86 GiB.

**Lessons**:

1. A resource metric at 57% is not an alibi. Ask what the number *is* before concluding
   what it rules out — 13.5 GB for a ten-second video was the answer sitting in plain
   sight. And a sampled metric cannot see a spike shorter than its interval; the peak it
   reports is a lower bound on the real peak, never an upper one.
2. A limit has to be expressed in the unit that actually runs out. Bytes on the wire bound
   nothing about bytes in memory when a decoder sits between them.
3. This is the same shape as this file's first entry and as the ffmpeg entries before it: a
   change that passes every test and was never run against something real. The tests now
   write real video files with OpenCV and assert on `nbytes`, and the 4K clip from the
   incident is a runnable (opt-in) test rather than a story.

---

## Read a dev-server hydration failure as a bug in the code under test (2026-09-22)

**What happened**: while verifying the new sign-in flow in a headless browser against
`next dev`, submitting the login form navigated to `/login?` and re-rendered the empty
form. That is the exact signature of a React `onSubmit` handler never running — a native
GET submission with no field names — so the first conclusion was that the form was not
wired up. Two rounds of "wait longer for hydration" changed nothing.

**What was actually happening**: the page never hydrated at all, in *any* route. The only
console message was an HMR WebSocket handshake failure (`ERR_INVALID_HTTP_RESPONSE`), which
looked like harmless dev noise. Running the same checks against `next build && next start`
showed the app hydrating and behaving correctly, including the form. Nothing was wrong with
the code.

**Why it was nearly a wrong fix**: the obvious next step from "onSubmit isn't firing" is to
start changing the form — adding `name` attributes, replacing the submit button with a
click handler, restructuring the Suspense boundary. Every one of those would have been a
change made to work around a broken harness, landed in the repo, and left behind.

**Lesson**: when the symptom is "no client-side JavaScript appears to run", check whether
*any* client-side JavaScript runs before attributing it to the component being tested. And
verify against the build that actually ships — a dev server's live-reload machinery is not
part of the artifact and can fail on its own.

---

## Nearly shipped an image encoder with red and blue swapped (2026-09-22)

**What happened**: `io.clip_loader.load_frames` converts every frame to RGB, and the
overlay colours were written as RGB triples to match. `cv2.imencode`, however, interprets
its input as BGR. Encoding the overlay directly would have produced a valid JPEG of the
right size, with every red detection box rendered blue and the footage's colours inverted.

**Why it would have survived a normal test pass**: it raises nothing. The field is present,
the base64 decodes, the dimensions are right, the JSON validates, and an assertion like
`assert "image_base64" in result` passes. The Streamlit path has never encoded anything —
it hands the raw array to `st.image` — so no existing code or test exercised this boundary
at all. It would have been found by a person looking at the web app and wondering why the
cricket pitch was purple.

**How it was caught**: by writing the tests to decode the base64 back into pixels and
assert on channel values (`tests/video_engine/test_overlay.py` encodes a pure-red frame and
checks the red channel survives), and then by running the real pipeline on real footage and
opening the returned frames.

**Lesson**: for anything that produces an image, "the field is present" is not a test. The
assertion has to be on the pixels, because the entire class of image bugs — wrong colour
space, wrong scale, blank frame, placeholder — produces output that is structurally valid.
This is the same rule pass 2 arrived at for detections ("open the frame and look at the
box"), reached independently from the encoding side.

---

## Described the ball detector's failures as "false positives" without looking at the frames (2026-09-22)

**What happened**: the README, DECISIONS.md and this file all recorded the ball
checkpoint as locking onto "a static white sack sitting on the ground by the fence" — a
false positive. That is true of one clip. Re-probing all four clips in
`data/uploads/videos/` and then *actually opening the frames* found the more common case
is different: in the 4K nets clip the detections at (408, 371) and (550, 372), held at
0.44–0.55 confidence for 83 consecutive frames, are **genuine cricket balls lying still on
the ground**. The detector is correct about them. They are simply not the delivered ball.

**Why it matters more than a wording correction**: it changes what the problem is. "The
detector hallucinates balls" suggests better weights or a higher threshold. "The detector
correctly finds every ball in frame, including the three that aren't in play" cannot be
fixed by either, because a per-frame detector is being asked a question — *which* of these
balls is the delivery — that a single frame does not contain. Measured confirmation that
thresholds cannot work: the resting ball scores 0.55 while the real moving ball in
`real_bowling_clip_full.mp4` scores 0.38–0.65. No cut orders those correctly.

**Fix applied**: motion became a first-class signal (`video_engine/motion/`) and ball
selection moved to a trajectory fit over a *set* of detections
(`video_engine/trajectory/`), because "is moving" is a property of a set and is exactly
what the per-frame detector structurally cannot see.

**Lesson**: "I know what this failure is" is itself a claim that needs checking against
the footage. Three documents carried the same unverified characterisation for days because
each was written from the previous one rather than from the frames. When a component's
behaviour is being described in a doc, open the frame and look at the box.

---

## Shipped a motion gate whose correctness depended on a 0.06 confidence margin (2026-09-22)

**What happened**: the temporal-motion gate added in `9339429` pools every BALL detection
across track_ids and asks whether the *pooled* set's maximum spread clears a multiple of
the mean box size. With a static object and a real ball both present, the spread being
measured is the distance *between the two objects*, which is large — so the gate passes
the union, and the segmenter then takes `release_frame` from the first detection in frame
order, which belongs to the static object.

Measured on real detections rather than argued: at a 0.15 confidence threshold on
`real_bowling_clip_full.mp4` the gate passes and reports `release_frame=1` — the bag by
the fence — instead of frame 80, where the ball actually appears. The shipped 0.40 default
never hit this only because that bag's confidence happens to peak at 0.34. Nothing
enforced that margin, nothing tested it, and it would have been crossed by any retrain, a
different clip, or the very threshold reduction that the same README recommends for
improving recall.

**Root cause**: the gate answered "do these detections, taken together, move?" when the
question that needed answering was "is there a subset of these detections that lies on one
plausible path?". The first question has a yes answer for a set containing two unrelated
objects; the second does not.

**Fix applied**: `video_engine/trajectory/fit.py` — a RANSAC fit that *selects* a mutually
consistent subset instead of blessing the union, with minimum displacement and minimum
median step speed as rejection filters applied **before** scoring. The ordering is the
load-bearing part: a static cluster is perfectly consistent with a zero-velocity path and
had 83 members against the real ball's 7, so on inlier count alone it wins every time.
Pinned by `tests/video_engine/test_trajectory_real_footage.py`, which carries the actual
recorded detections from that clip.

**Lesson**: a guard that passes on the available test data is not the same as a guard that
is correct, and "the threshold that happens to be configured keeps us on the right side of
it" is a latent bug with a timer on it. When a check's correctness depends on a numeric
margin, measure the margin and write it down — if it turns out to be 0.06, that is the
finding.

---

## Clustered two stump sets by horizontal position, for a camera angle this project doesn't use (2026-09-22)

**What happened**: `geometry.resolve_stump_ends` first separated the two stump sets by
their horizontal centres, with a generous gap threshold. That is correct for a side-on
camera and wrong for the placement this project actually targets and that all its footage
uses — filming *down the pitch*, where both sets sit at nearly the same `x` and differ in
height in frame and in apparent size. The two sets merged into one cluster, so
`resolve_stump_ends` returned `None` and calibration stayed silently disabled on precisely
the footage it was written for.

**How it was caught**: by a test fixture built from the real 4K clip's measured geometry
(near set ~60x160 px low in frame, far set ~20x55 px high in frame) rather than from a
convenient side-on layout. A fixture drawn to make the code look right would have passed.

**Fix applied**: cluster on the 2D centre with a radius proportional to box height.

**Lesson**: when writing the fixture for a geometric component, take the numbers off the
real footage. A fixture invented alongside the code shares the code's assumptions, and a
test built on it verifies that the code is self-consistent rather than that it is right.

---

## Trusted a trained checkpoint's dataset-split mAP as if it were real-world accuracy (2026-09-17)

**What happened**: after training the ball+stumps YOLOv8n checkpoint (0.893 mAP50 on
Roboflow's own held-out split), the first instinct was to call the detector "done" and wire
it into the app. The user pushed back and asked for a real bowling clip test. Three real
clips later: one false positive (a static bag detected as "ball" at plausible confidence,
driving a wrong `shot_type=DRIVE`), and two clips where the ball was never detected at all.

**Root cause**: training/validation mAP measures fit to the training distribution, not
generalization to a different one. The dataset's images likely share a camera angle,
lighting, and ball-clarity profile that real nets/phone footage doesn't. This is a standard
domain-generalization gap, not a training bug — no amount of additional epochs on the same
data would have caught it.

**Fix applied**: a temporal-motion check that rejects any ball track that doesn't move
enough to be real flight (see DECISIONS.md). This fixes precision, not recall.

**Lesson**: a model's own held-out metric is not evidence it works on the actual target
input distribution. Test on real, independently-sourced footage before calling a CV
component "working" — and when a user asks to see it on a real clip, that's not a formality
to rush past.

---

## Lost a fully-trained model to a Colab runtime disconnect (2026-09-16)

**What happened**: the first 30-epoch training run on Colab's free T4 tier completed
successfully (0.893 mAP50). Progress was checked on intermittently rather than grabbing the
weights the moment training finished. By the time the download cell ran, the runtime had
already disconnected (free-tier idle/session-length limit) and been replaced with a fresh
VM — `/content/` was wiped, taking the trained weights with it, even though the notebook's
own *output* (the training logs) persisted in the saved `.ipynb`.

**Fix applied**: combined training, `shutil.make_archive`, and `files.download` into a
single cell on the retrain, so there's no window between "training finishes" and
"weights leave the ephemeral VM" for a disconnect to land in.

**Lesson**: on any ephemeral compute (Colab free tier, spot instances, etc.), the save/export
step is not a separate follow-up task — it has to be atomic with the work that produced the
artifact, or scheduled to run immediately on completion. "I'll grab it once it's done"
assumes a window of availability that free-tier ephemeral compute does not guarantee.

---

## A stray keystroke corrupted a notebook cell's source without being noticed immediately (2026-09-16/17)

**What happened**: `!pip install -q ultralytics roboflow` in the first Colab cell was found
later to read `!pip install -q ultralytics rchfcoboflow` — extra characters inserted mid-word
at some point during browser-automation interaction with the notebook (exact trigger
unclear; plausibly a click that landed in edit mode on the wrong cell, or a keystroke that
fired before the mouse click that should have preceded it, registered). It went unnoticed
until a routine content check well after the fact.

**Consequence**: harmless this time — the cell had already executed successfully with
correct text before the corruption happened, so the already-installed packages were
unaffected. Would not have been harmless if a later runtime disconnect had required
re-running that cell.

**Lesson (reinforcing one already learned earlier this project)**: when automating a
browser-based code editor, verify a click actually focused the intended element
(`document.activeElement`) before typing, every time — not just after an incident. Spot-check
critical cells' actual content after a sequence of automated interactions, especially before
relying on them to still be correct after a page reload or reconnect.

---

## Assumed a coordinate-scale mismatch pattern from one Streamlit tab applied elsewhere (earlier in session)

**What happened**: screenshot pixel coordinates didn't match the page's actual DOM/CSS pixel
coordinates in browser automation, with the mismatch ratio varying by tab (~1.36x in one
Streamlit tab, ~1.63x in a Colab tab) rather than being a fixed constant. Early attempts to
click Streamlit tab labels using screenshot-derived coordinates repeatedly landed on the
wrong element.

**Fix applied**: stopped using raw pixel coordinates for element interaction entirely;
switched to `find`-tool element refs or direct JS `element.click()` (via a
shadow-DOM-piercing `deepQueryAll` helper for Colab's custom elements). This is now the
default approach, not a special case.

**Lesson**: don't assume a scale-mismatch ratio measured once carries over to a different
tab, page, or zoom state. When coordinate clicks misbehave, check
`window.innerWidth`/`devicePixelRatio` against the screenshot's reported dimensions rather
than guessing a correction factor — and prefer ref-based or JS-direct interaction over
coordinate math wherever the tooling allows it.
