"""Offline test harness: deterministic stand-ins for the LLM steps (design spec: "Determinism").

The stub extractor maps keywords to canonical claims, so two differently worded texts that
use the same keyword produce the same point, just as a paraphrase would. The stub judge
matches a new point to an existing point with identical text.
"""

import re
from pathlib import Path

import pytest

from novelty.config import Params
from novelty.embeddings import FakeEmbedder
from novelty.extract import Extraction
from novelty.judge import Judgement, PointMatch
from novelty.pipeline import NoveltyScorer
from novelty.pool import Pool
from novelty.schema import Submission, load_fixed_content

DATA = Path(__file__).resolve().parent.parent / "data"

CLAIMS = {  # keyword -> canonical claim
    "traffic": "charge will reduce downtown traffic",
    "low-income": "flat charge is unfair to low-income drivers",
    "buses": "fix public transport before charging",
    "shops": "charge will hurt downtown shops",
    "air": "less traffic means cleaner air",
    "park": "drivers will park just outside the zone",
    "cameras": "plate cameras create a privacy risk",
    "consultation": "consultation is a formality",
}
ARTICLE_CLAIMS = ["charge will reduce downtown traffic", "revenue funds buses and bike lanes"]
OFFTOPIC_MARKERS = ("biryani", "basmati", "broadband")
GIBBERISH_MARKER = "lavender"


def stub_extractor(sub: Submission) -> Extraction:
    text = f"{sub.headline} {sub.body}".lower()
    if any(m in text for m in OFFTOPIC_MARKERS):
        return Extraction(about_article=False, points=[])
    return Extraction(about_article=True,
                      points=[c for k, c in CLAIMS.items() if re.search(rf"\b{re.escape(k)}\b", text)])


def stub_judge(sub, points, candidates) -> Judgement:
    by_text = {c.text: c.id for c in candidates}
    quality = 0 if GIBBERISH_MARKER in sub.body.lower() else 3
    return Judgement(matches=[PointMatch(index=i, match_ids=[by_text[p]] if p in by_text else [])
                              for i, p in enumerate(points, 1)],
                     quality=quality, reason="stub")


def sub(headline: str, body: str, stance: str = "support") -> Submission:
    return Submission(headline=headline, body=body, stance=stance)


@pytest.fixture
def fixed():
    return load_fixed_content(DATA / "fixed_content.json")


@pytest.fixture
def params():
    # Embedding floors off by default (the FakeEmbedder measures word overlap, not meaning);
    # tests that exercise the floor set them explicitly.
    return Params(r_min=-1.0, r_body_min=-1.0, t=0.0, dup_max=0.95)


@pytest.fixture
def make_scorer(fixed, params):
    def _make(pool: Pool | None = None, **overrides) -> NoveltyScorer:
        if pool is None:
            pool = Pool()
            pool.add_article_points(ARTICLE_CLAIMS, params.article_prior_mentions)
        p = Params(**{**params.__dict__, **overrides})
        return NoveltyScorer(fixed, pool, FakeEmbedder(), stub_extractor, stub_judge, p)
    return _make


@pytest.fixture
def seeded_scorer(make_scorer):
    """A scorer whose pool already holds a realistic cluster of common takes."""
    scorer = make_scorer()
    bodies = [
        ("Less gridlock at last", "This is going to cut the traffic jams on Main Street every single weekday morning."),
        ("Traffic will ease", "Fewer cars downtown means the traffic finally moves and my commute gets shorter."),
        ("Unfair on workers", "Twelve dollars a day is brutal for low-income families who have to drive to work."),
        ("Hits the poorest", "The low-income commuters with no bus route will pay the most for this scheme every month."),
        ("Transit first please", "Run more buses and make them reliable before you start charging people to drive."),
        ("Shops will suffer", "Customers will simply avoid the shops downtown and drive to the out-of-town mall instead."),
    ]
    for i, (h, b) in enumerate(bodies):
        scorer.score(sub(h, b), sub_id=f"seed{i}")
    return scorer
