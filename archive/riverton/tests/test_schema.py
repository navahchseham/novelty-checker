from pathlib import Path

import pytest
from pydantic import ValidationError

from novelty.schema import (
    FIXED_CONTENT_MAX_WORDS,
    FixedContent,
    Stance,
    Submission,
    load_fixed_content,
    word_count,
)

DATA = Path(__file__).resolve().parent.parent / "data"

VALID_BODY = "Four ten-hour shifts could hurt parents who rely on fixed daycare pickup times every day."


def test_fixed_content_file_is_valid_and_within_limit():
    fc = load_fixed_content(DATA / "fixed_content.json")
    assert word_count(fc.body) <= FIXED_CONTENT_MAX_WORDS


def test_fixed_content_over_limit_rejected():
    with pytest.raises(ValidationError):
        FixedContent(id="x", title="t", body="word " * (FIXED_CONTENT_MAX_WORDS + 1))


def test_valid_submission():
    s = Submission(headline="  Daycare clash  ", body=VALID_BODY, stance="oppose")
    assert s.stance is Stance.OPPOSE
    assert s.headline == "Daycare clash"


@pytest.mark.parametrize(
    "field,value",
    [
        ("headline", ""),
        ("headline", "word " * 21),
        ("body", "too short to count"),
        ("body", "word " * 251),
        ("stance", "maybe"),
    ],
)
def test_invalid_submission_rejected(field, value):
    data = {"headline": "Daycare clash", "body": VALID_BODY, "stance": "oppose", field: value}
    with pytest.raises(ValidationError):
        Submission(**data)


def test_to_text_excludes_stance():
    s = Submission(headline="Daycare clash", body=VALID_BODY, stance="oppose")
    assert "oppose" not in s.to_text().lower()
    assert s.headline in s.to_text() and s.body in s.to_text()
