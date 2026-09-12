import pytest
from PIL import Image, ImageDraw, ImageFont

from scoreboard_ocr.reader import ScoreboardReading, parse_scoreboard_text, read_scoreboard

FONT_PATH = r"C:\Windows\Fonts\arial.ttf"


def _render_scoreboard_image(text: str) -> Image.Image:
    """Renders `text` the way a broadcast scoreboard graphic would look once
    cropped to just that line: dark background, light bold-ish text."""
    font = ImageFont.truetype(FONT_PATH, 40)
    image = Image.new("RGB", (400, 60), color=(20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.text((10, 5), text, font=font, fill=(255, 255, 255))
    return image


@pytest.mark.parametrize(
    "text,expected",
    [
        ("142/3 (15.2 Ov)", ScoreboardReading(142, 3, 15, 2, "142/3 (15.2 Ov)")),
        ("142-3 (15.2)", ScoreboardReading(142, 3, 15, 2, "142-3 (15.2)")),
        ("9/0 (0.1 Ov)", ScoreboardReading(9, 0, 0, 1, "9/0 (0.1 Ov)")),
    ],
)
def test_parse_scoreboard_text_handles_common_formats(text, expected):
    result = parse_scoreboard_text(text)
    assert result.runs == expected.runs
    assert result.wickets == expected.wickets
    assert result.over == expected.over
    assert result.ball == expected.ball


def test_parse_scoreboard_text_returns_none_for_unrelated_text():
    assert parse_scoreboard_text("SIXER!! crowd goes wild") is None


def test_parse_scoreboard_text_requires_both_score_and_overs():
    assert parse_scoreboard_text("142/3") is None  # no overs
    assert parse_scoreboard_text("(15.2 Ov)") is None  # no score


def test_read_scoreboard_ocrs_a_rendered_broadcast_style_graphic():
    image = _render_scoreboard_image("142/3 (15.2 Ov)")

    reading = read_scoreboard(image)

    assert reading is not None
    assert reading.runs == 142
    assert reading.wickets == 3
    assert reading.over == 15
    assert reading.ball == 2


def test_read_scoreboard_returns_none_on_a_blank_image():
    image = Image.new("RGB", (400, 60), color=(20, 20, 20))
    assert read_scoreboard(image) is None
