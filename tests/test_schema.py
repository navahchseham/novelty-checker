from pathlib import Path

import pytest
from pydantic import ValidationError

from novelty.schema import FIXED_CONTENT_MAX_WORDS, FixedContent, Stance, Submission, load_fixed_content

DATA = Path(__file__).resolve().parent.parent / "data"
BODY = "Drivers will just park on the residential streets right outside the zone boundary and walk in."


def test_fixed_content_file_within_limit():
    fc = load_fixed_content(DATA / "fixed_content.json")
    assert len(fc.body.split()) <= FIXED_CONTENT_MAX_WORDS


def test_fixed_content_over_limit_rejected():
    with pytest.raises(ValidationError):
        FixedContent(id="x", title="t", body="word " * (FIXED_CONTENT_MAX_WORDS + 1))


def test_valid_submission_is_trimmed():
    s = Submission(headline="  Boundary parking  ", body=BODY, stance="support_with_changes")
    assert s.headline == "Boundary parking" and s.stance is Stance.SUPPORT_WITH_CHANGES


@pytest.mark.parametrize("field,value", [
    ("headline", "Hi"),                # < 5 chars
    ("headline", "x" * 101),
    ("body", "Too short to count."),   # < 80 chars
    ("body", "x" * 801),
    ("stance", "mixed"),               # not an allowed value
])
def test_invalid_submission_rejected(field, value):
    data = {"headline": "Boundary parking", "body": BODY, "stance": "oppose", field: value}
    with pytest.raises(ValidationError):
        Submission(**data)


def test_to_text_excludes_stance():
    a = Submission(headline="Boundary parking", body=BODY, stance="oppose")
    b = a.model_copy(update={"stance": Stance.SUPPORT})
    assert a.to_text() == b.to_text()
