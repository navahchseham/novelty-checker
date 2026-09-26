"""Data shapes for the novelty scorer.

FixedContent is the short piece (<= 100 words) that users respond to.
Submission is what a user sends back: three discrete, user-provided properties
(headline, body, stance).
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

FIXED_CONTENT_MAX_WORDS = 100
HEADLINE_MAX_WORDS = 20
BODY_MIN_WORDS = 10
BODY_MAX_WORDS = 250


def word_count(text: str) -> int:
    return len(text.split())


class Stance(str, Enum):
    """Multi-choice property: the user's position on the fixed content."""

    SUPPORT = "support"
    OPPOSE = "oppose"
    MIXED = "mixed"
    QUESTION = "question"


class FixedContent(BaseModel):
    id: str
    title: str
    body: str

    @field_validator("body")
    @classmethod
    def _max_words(cls, v: str) -> str:
        n = word_count(v)
        if n > FIXED_CONTENT_MAX_WORDS:
            raise ValueError(f"fixed content body has {n} words (max {FIXED_CONTENT_MAX_WORDS})")
        return v.strip()

    def to_text(self) -> str:
        return f"{self.title}\n\n{self.body}"


class Submission(BaseModel):
    """A user's response to a FixedContent item."""

    id: str | None = None
    headline: str = Field(min_length=1)
    body: str = Field(min_length=1)
    stance: Stance

    @field_validator("headline")
    @classmethod
    def _headline_words(cls, v: str) -> str:
        v = v.strip()
        n = word_count(v)
        if n == 0 or n > HEADLINE_MAX_WORDS:
            raise ValueError(f"headline must be 1-{HEADLINE_MAX_WORDS} words (got {n})")
        return v

    @field_validator("body")
    @classmethod
    def _body_words(cls, v: str) -> str:
        v = v.strip()
        n = word_count(v)
        if not BODY_MIN_WORDS <= n <= BODY_MAX_WORDS:
            raise ValueError(f"body must be {BODY_MIN_WORDS}-{BODY_MAX_WORDS} words (got {n})")
        return v

    def to_text(self) -> str:
        """Text used for embeddings. Stance is deliberately excluded (see DESIGN.md §1)."""
        return f"{self.headline}\n\n{self.body}"


def load_fixed_content(path: str | Path) -> FixedContent:
    return FixedContent.model_validate(json.loads(Path(path).read_text()))


def load_submissions(path: str | Path) -> list[Submission]:
    return [Submission.model_validate(s) for s in json.loads(Path(path).read_text())]
