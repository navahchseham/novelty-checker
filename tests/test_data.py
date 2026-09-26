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
    "H1": r"residential street|outside the zone|just outside|boundary|perimeter|side street",
    "H2": r"real-time|dynamic pric|vary with|surge pric",
    "H3": r"plate camera|traffic camera|cameras (will|log|record)|plate-recognition|surveillance|privacy|location data",
    "H4": r"loophole|register .* as (a )?van|(pretend|claim|disguise|reclassif).{0,30}van",
    "H5": r"7:01|taper|cut-off|cutoff",
    "H6": r"revenue (will )?(fall|drop|decline|shrink)|depends on the charge failing",
}


def test_corpus_counts_match_plan():
    assert Counter(x["angle_id"] for x in CORPUS) == {a["id"]: a["count"] for a in PLAN["corpus_angles"]}
    assert len(CORPUS) == 50


def test_golden_counts_match_plan():
    counts = Counter(x["category"] for x in GOLDEN)
    gp = PLAN["golden_plan"]
    for cat, spec in gp.items():
        expected = len(spec.get("from") or spec.get("items") or spec.get("topics"))
        assert counts[cat] == expected, cat
    for h in PLAN["golden_handwritten"]:
        assert counts[h["category"]] >= 1


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
    assert len({x["model"] for x in CORPUS + GOLDEN if "model" in x}) == 1


HOLDOUT_PATH = DATA / "golden_holdout.json"
HOLDOUT_PROBES = {
    "H7": r"carpool|car-pool|three or more (people|passengers)|high.occupancy",
    "H8": r"plumber|electrician|tradesperson|tradespeople|contractor",
    "H9": r"uber|lyft|ride.?hail|rideshare|ride-share",
}


@pytest.mark.skipif(not HOLDOUT_PATH.exists(), reason="holdout not generated")
def test_holdout_is_valid_leak_free_and_independent():
    from novelty.config import JUDGE_MODEL
    holdout = json.loads(HOLDOUT_PATH.read_text())
    for x in holdout:
        Submission.model_validate(x)
    assert len({x["id"] for x in holdout} | {x["id"] for x in GOLDEN}) == len(holdout) + len(GOLDEN)
    for angle, pattern in HOLDOUT_PROBES.items():
        leaks = [x["id"] for x in CORPUS + GOLDEN if re.search(pattern, f"{x['headline']} {x['body']}".lower())]
        assert not leaks, f"holdout angle {angle} appears in {leaks}"
    writers = {x["model"] for x in holdout if "model" in x}
    corpus_writers = {x["model"] for x in CORPUS}
    assert len(writers) == 1 and JUDGE_MODEL not in writers and not writers & corpus_writers
