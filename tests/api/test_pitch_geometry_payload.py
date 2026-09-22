"""The API must never hand a client a line/length number without its uncertainty.

These pin the shape of the `pitch_geometry` block rather than the values in it. The
failure being guarded against is a client reading `pitch_length_band` and rendering it as
fact, which is only prevented if `available` is false and a reason is present whenever the
evidence is thin.
"""
from api.pipeline_runner import _pitch_geometry_payload
from video_engine.contracts import (
    DeliveryEvent,
    GeometryConfidence,
    PitchLength,
    PitchPoint,
    ShotType,
)


def _event(**kwargs) -> DeliveryEvent:
    base = dict(release_frame=1, contact_frame=5, shot_type=ShotType.UNKNOWN)
    base.update(kwargs)
    return DeliveryEvent(**base)


def test_no_geometry_is_unavailable_and_carries_a_reason():
    payload = _pitch_geometry_payload(
        _event(pitch_notes="Stumps were detected at only one end.")
    )

    assert payload["available"] is False
    assert payload["confidence"] == "none"
    assert "one end" in payload["insufficient_data_reason"]
    assert "pitch_length_m" not in payload


def test_a_low_confidence_point_is_still_not_available():
    """A bounce point exists, but LOW must not read as a usable answer."""
    payload = _pitch_geometry_payload(
        _event(
            pitch_point=PitchPoint(length_m=5.4, line_m=-0.12),
            pitch_length=PitchLength.GOOD,
            pitch_confidence=GeometryConfidence.LOW,
            track_confidence=0.4,
        )
    )

    assert payload["available"] is False
    assert "insufficient_data_reason" in payload
    # The numbers are still present for debugging, but gated by `available`.
    assert payload["pitch_length_m"] == 5.4


def test_a_medium_confidence_point_is_available_and_line_stays_unlabelled():
    payload = _pitch_geometry_payload(
        _event(
            pitch_point=PitchPoint(length_m=6.1, line_m=0.08),
            pitch_length=PitchLength.GOOD,
            pitch_confidence=GeometryConfidence.MEDIUM,
            track_confidence=0.72,
        )
    )

    assert payload["available"] is True
    assert payload["pitch_length_band"] == "good"
    assert payload["pitch_line_m"] == 0.08
    assert payload["line_side_labelled"] is False
    assert "insufficient_data_reason" not in payload


def test_track_confidence_is_always_reported():
    payload = _pitch_geometry_payload(_event(track_confidence=0.615))

    assert payload["track_confidence"] == 0.615
