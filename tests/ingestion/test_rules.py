import pytest

from ingestion.rules import phase_for_over


@pytest.mark.parametrize(
    "over,expected",
    [
        (0, "powerplay"),
        (5, "powerplay"),
        (6, "middle"),
        (14, "middle"),
        (15, "death"),
        (19, "death"),
    ],
)
def test_phase_boundaries(over, expected):
    assert phase_for_over(over) == expected
