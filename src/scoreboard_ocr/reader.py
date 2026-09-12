"""Reads a live match state (runs/wickets/overs) off a broadcast scoreboard graphic.

Prototype for the "OCR Extraction Framework" piece of the expanded architecture
report: PyTesseract with PSM 7 (treat the image as a single text line) plus regex
over the result. This is a new capability -- it doesn't fit the current PRD's MVP
scope (ingestion's ball-by-ball table, video_engine's per-delivery analysis) and
isn't wired into either yet; it exists standalone so its own accuracy can be
measured against real broadcast crops before deciding where it plugs in (most
likely: a gating signal ahead of highlight detection, per the report's Part 5).

Accuracy note: the report's "99%+ accuracy" claim for this approach is not
something this prototype has verified -- it depends heavily on the actual
broadcast's font, resolution, and graphic style. Treat that number as unverified
until measured against real broadcast crops.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytesseract

# pytesseract shells out to the tesseract binary; on Windows it isn't always on
# PATH for a session started before the installer ran. Point at the standard
# install location if it's there and PATH doesn't already resolve it.
_WINDOWS_DEFAULT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if shutil.which("tesseract") is None and _WINDOWS_DEFAULT.exists():
    pytesseract.pytesseract.tesseract_cmd = str(_WINDOWS_DEFAULT)

# e.g. "142/3" or "142-3"
_SCORE_RE = re.compile(r"(\d{1,3})\s*[/-]\s*(\d{1,2})")
# e.g. "15.2", "(15.2 Ov)", "15.2 overs"
_OVERS_RE = re.compile(r"(\d{1,2})\.(\d)")


@dataclass(frozen=True)
class ScoreboardReading:
    runs: int
    wickets: int
    over: int
    ball: int
    raw_text: str


def parse_scoreboard_text(text: str) -> Optional[ScoreboardReading]:
    """Pure regex parse, independent of OCR -- kept separate so the parsing logic
    itself can be tested fast and deterministically."""
    score_match = _SCORE_RE.search(text)
    overs_match = _OVERS_RE.search(text)
    if not score_match or not overs_match:
        return None
    return ScoreboardReading(
        runs=int(score_match.group(1)),
        wickets=int(score_match.group(2)),
        over=int(overs_match.group(1)),
        ball=int(overs_match.group(2)),
        raw_text=text,
    )


def read_scoreboard(image, psm: int = 7) -> Optional[ScoreboardReading]:
    """`image` is anything pytesseract accepts: a PIL Image or a path. `psm=7`
    treats the crop as a single line of text, matching a typical scoreboard
    graphic once it's been cropped down to just the score/overs region."""
    text = pytesseract.image_to_string(image, config=f"--psm {psm}")
    return parse_scoreboard_text(text)
