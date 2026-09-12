from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from ..contracts import Delivery, MatchMeta


class StatsSource(ABC):
    """A source of the official ball-by-ball table (PRD 2.1: 'ICC / board ball-by-ball tables')."""

    @abstractmethod
    def iter_matches(self) -> Iterator[tuple[MatchMeta, list[Delivery]]]:
        """Yields one (match metadata, ball-by-ball deliveries) pair per match."""
