# Mistakes

A running log of things that went wrong, what was actually happening, and what fixed it or
what to do differently next time. This is not a blame log — it exists so the same mistake
doesn't get repeated three sessions from now. Newest first.

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
