"""T20 phase boundaries, shared by stats parsing and the feature store."""
from __future__ import annotations

POWERPLAY_END = 6  # overs 1-6 (0-based 0-5)
DEATH_START = 15  # overs 16-20 (0-based 15-19)


def phase_for_over(over_zero_based: int) -> str:
    if over_zero_based < POWERPLAY_END:
        return "powerplay"
    if over_zero_based < DEATH_START:
        return "middle"
    return "death"
