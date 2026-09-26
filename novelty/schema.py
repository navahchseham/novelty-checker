"""Data shapes (design spec: "Content shape" and "1. Validate").

FixedContent is the article (<= 100 words) that users respond to. Submission is the
three-field form: headline, body, stance.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, field_validator

FIXED_CONTENT_MAX_WORDS = 100
HEADLINE_CHARS = (5, 100)
BODY_CHARS = (80, 800)


class Stance(str, Enum):
    SUPPORT = "support"
    SUPPORT_WITH_CHANGES = "support_with_changes"
    OPPOSE = "oppose"
    UNDECIDED = "undecided"


class FixedContent(BaseModel):
    id: str
    title: str
    body: str

    @field_validator("body")
    @classmethod
    def _max_words(cls, v: str) -> str:
        n = len(v.split())
        if n > FIXED_CONTENT_MAX_WORDS:
            raise ValueError(f"fixed content has {n} words (max {FIXED_CONTENT_MAX_WORDS})")
        return v.strip()

    def to_text(self) -> str:
        return f"{self.title}\n\n{self.body}"


def _check_chars(v: str, limits: tuple[int, int], name: str) -> str:
    v = v.strip()
    lo, hi = limits
    if not lo <= len(v) <= hi:
        raise ValueError(f"{name} must be {lo}-{hi} characters (got {len(v)})")
    return v


class Submission(BaseModel):
    id: str | None = None
    headline: str
    body: str
    stance: Stance

    @field_validator("headline")
    @classmethod
    def _headline(cls, v: str) -> str:
        return _check_chars(v, HEADLINE_CHARS, "headline")

    @field_validator("body")
    @classmethod
    def _body(cls, v: str) -> str:
        return _check_chars(v, BODY_CHARS, "body")

    def to_text(self) -> str:
        """Text used for embeddings. Stance is excluded, so flipping it changes nothing."""
        return f"{self.headline}\n\n{self.body}"


def load_fixed_content(path: str | Path) -> FixedContent:
    return FixedContent.model_validate(json.loads(Path(path).read_text()))
