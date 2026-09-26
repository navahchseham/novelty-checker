"""Integrity checks for the generated dataset (DESIGN.md §3). Runs offline."""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from novelty.schema import Submission

DATA = Path(__file__).resolve().parent.parent / "data"
PLAN = json.loads((DATA / "angles.json").read_text())
CORPUS = json.loads((DATA / "corpus.json").read_text())
GOLDEN = json.loads((DATA / "golden.json").read_text())

# Distinctive terms for each held-back angle. None may appear in the corpus, otherwise the
# golden "novel" item built from that angle would not really be novel.
HELD_OUT_PROBES = {
    "H1": r"police|firefight|911|dispatch|two-tier|first responder",
    "H2": r"control group|randomi[sz]|causation|staggered rollout",
    "H3": r"disabilit|opt[- ]out|arthritis",
    "H4": r"season|winter|summer|january",
    "H5": r"online|digital|24/7|portal",
    "H6": r"neighbou?ring (town|municipal)|bidding war|poach",
}


def test_corpus_counts_match_plan():
    planned = {a["id"]: a["count"] for a in PLAN["corpus_angles"]}
    assert Counter(x["angle_id"] for x in CORPUS) == planned
    assert len(CORPUS) == 50


def test_golden_counts_match_plan():
    counts = Counter(x["category"] for x in GOLDEN)
    gp = PLAN["golden_plan"]
    assert counts["novel_relevant"] == len(gp["novel_relevant"]["from"])
    assert counts["partly_novel"] == len(gp["partly_novel"]["items"])
    assert counts["paraphrase_common"] == len(gp["paraphrase_common"]["from"])
    assert counts["paraphrase_rare"] == len(gp["paraphrase_rare"]["from"])
    assert counts["offtopic_novel"] == len(gp["offtopic_novel"]["topics"])
    assert counts["low_effort"] + counts["gibberish"] == len(PLAN["golden_handwritten"])


@pytest.mark.parametrize("item", CORPUS + GOLDEN, ids=lambda x: x["id"])
def test_every_item_is_a_valid_submission(item):
    Submission.model_validate(item)


def test_ids_unique():
    ids = [x["id"] for x in CORPUS + GOLDEN]
    assert len(ids) == len(set(ids))


def test_golden_expected_ranges_valid():
    for x in GOLDEN:
        assert 0.0 <= x["expected_min"] <= x["expected_max"] <= 1.0


def test_no_held_out_angle_leaks_into_corpus():
    for angle, pattern in HELD_OUT_PROBES.items():
        leaks = [x["id"] for x in CORPUS if re.search(pattern, f"{x['headline']} {x['body']}".lower())]
        assert not leaks, f"held-out angle {angle} appears in corpus items {leaks}"


def test_single_generation_model():
    """Mixing models would let embeddings pick up writing style instead of ideas."""
    models = {x["model"] for x in CORPUS + GOLDEN if "model" in x}
    assert len(models) == 1
