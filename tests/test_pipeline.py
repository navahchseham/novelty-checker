"""Behaviour tests from the design spec ("Behaviour tests"), run offline with stubbed LLM steps."""

import math

import pytest

from tests.conftest import sub

NOVEL = sub("Parking chaos next door",
            "Commuters will park on the residential streets right outside the zone and walk in, clogging them.")
NOVEL_PARAPHRASE = sub("Boundary streets will fill up",
                       "People will just park their cars one block past the boundary, so those roads get jammed.")


def test_novel_and_relevant_scores_high(seeded_scorer):
    r = seeded_scorer.score(NOVEL, add_to_pool=False)
    assert r.status == "scored" and r.score > 0.6
    assert [p["prior_mentions"] for p in r.points] == [0]


def test_first_mover_then_paraphrase_drops(seeded_scorer):
    first = seeded_scorer.score(NOVEL)
    second = seeded_scorer.score(NOVEL_PARAPHRASE, add_to_pool=False)
    assert first.score > 0.6 and first.n_point == 1.0
    # The paraphrase finds the first mover's point in the pool: m = 1, so N_point = w(1) = exp(-1/lambda).
    assert [p["prior_mentions"] for p in second.points] == [1]
    assert second.n_point == pytest.approx(math.exp(-1 / seeded_scorer.p.rarity_lambda), abs=1e-3)
    assert second.score < first.score
    # The absolute threshold (< 0.2) depends on real embeddings for N_sem, so it is a live test.


def test_repeated_point_decays_below_point_two(seeded_scorer):
    """Each repetition lowers the point's worth; after 4 prior mentions N_point is below 0.2."""
    scores = []
    for i in range(4):
        scores.append(seeded_scorer.score(sub(f"Park outside {i}", f"Drivers will park just past the edge of the "
                                              f"zone and walk the rest of the way in, version {i}.")).n_point)
    r = seeded_scorer.score(NOVEL_PARAPHRASE, add_to_pool=False)
    assert scores == sorted(scores, reverse=True) and scores[0] == 1.0
    assert r.points[0]["prior_mentions"] == 4 and r.n_point < 0.2


def test_rarity_one_mention_beats_five(make_scorer):
    scorer = make_scorer()
    scorer.score(sub("Consultation is fake", "The consultation is just a formality, because the council clearly decided everything already."))
    for i in range(5):
        scorer.score(sub(f"Clean air {i}", f"With fewer cars on the road the air downtown will be much cleaner for everyone, point {i}."))
    rare = scorer.score(sub("Sham consultation", "Nobody I know believes this consultation will change anything at all about the plan."),
                        add_to_pool=False)
    common = scorer.score(sub("Breathing easier", "Our kids will finally breathe cleaner air once most of the cars are gone from here."),
                          add_to_pool=False)
    assert rare.points[0]["prior_mentions"] == 1 and common.points[0]["prior_mentions"] == 5
    assert rare.score > common.score


def test_partial_novelty_sits_between(seeded_scorer):
    new = seeded_scorer.score(NOVEL, add_to_pool=False)
    old = seeded_scorer.score(sub("Old takes", "It will cut traffic, it will hurt low-income drivers, and it will hurt the shops too."),
                              add_to_pool=False)
    partial = seeded_scorer.score(
        sub("Mostly old", "It will cut traffic, hurt low-income drivers, and hurt shops; and people will park outside."),
        add_to_pool=False)
    assert old.score < partial.score < new.score


def test_not_novel_common_point_scores_low(seeded_scorer):
    r = seeded_scorer.score(sub("Great idea, less traffic",
                                "Great idea, this will mean less traffic downtown. About time the council did it."),
                            add_to_pool=False)
    assert r.points[0]["restates_article"] and r.n_point == 0.0
    assert r.score < 0.2  # its only point restates the article, so it contributes nothing of its own


def test_restating_the_article_does_not_dilute_a_new_point(seeded_scorer):
    padded = seeded_scorer.score(sub("Parking chaos next door", "Great that it will cut traffic. But commuters will "
                                     "park on residential streets right outside the zone and walk in."), add_to_pool=False)
    assert [p["restates_article"] for p in padded.points] == [True, False]
    assert padded.n_point == 1.0


def test_exact_copy_is_duplicate(seeded_scorer):
    e = seeded_scorer.pool.entries[0]
    r = seeded_scorer.score(sub(e.headline, e.body, e.stance), add_to_pool=False)
    assert r.status == "duplicate" and r.score == 0.0


def test_stance_flip_cannot_buy_a_score(seeded_scorer):
    e = seeded_scorer.pool.entries[0]
    flipped = "oppose" if e.stance != "oppose" else "support"
    r = seeded_scorer.score(sub(e.headline, e.body, flipped), add_to_pool=False)
    assert r.status == "duplicate" and r.score == 0.0


def test_irrelevant_is_zero_and_was_blocked_by_relevance(seeded_scorer):
    r = seeded_scorer.score(sub("My biryani secret", "Layer the basmati over the marinated chicken, seal the pot and "
                                "steam it slowly with saffron milk and fried onions."), add_to_pool=False)
    assert r.score == 0.0 and r.reason.startswith("irrelevant")
    assert r.n_sem is not None  # reported even when gated; the ">= 90th percentile" check is a live test


def test_copied_headline_offtopic_body_fails_body_check(seeded_scorer, make_scorer):
    scorer = make_scorer(pool=seeded_scorer.pool, r_body_min=0.2)
    r = scorer.score(sub("Harborview approves downtown congestion charge",
                         "Soak the rice for thirty minutes, parboil with cardamom, then steam gently on a low heat."),
                     add_to_pool=False)
    assert r.status == "irrelevant" and "floor" in r.reason
    assert r.relevance > r.body_relevance  # the copied headline alone would have lifted it


def test_quality_floor_zeroes_fluent_gibberish(seeded_scorer):
    r = seeded_scorer.score(sub("Velvet tolls", "When the harbour hums in lavender arithmetic every bicycle negotiates "
                                "its cardboard pension with the moon and the traffic."), add_to_pool=False)
    assert r.status == "scored" and r.quality_level == 0
    assert r.score == 0.0 and r.quality_factor == 0.0


def test_quality_floor_rejects_are_not_pooled(seeded_scorer):
    n = len(seeded_scorer.pool)
    seeded_scorer.score(sub("Velvet tolls", "In lavender arithmetic the traffic negotiates with the moon on Thursdays and sings to the tolls."))
    assert len(seeded_scorer.pool) == n


def test_invalid_submission_scores_zero(seeded_scorer):
    r = seeded_scorer.score({"headline": "Hi", "body": "too short", "stance": "support"})
    assert r.status == "invalid" and r.score == 0.0


def test_scored_submission_joins_pool_and_counts_once(make_scorer):
    scorer = make_scorer()
    r = scorer.score(sub("Twice the traffic point", "Traffic will drop a lot. Honestly, less traffic downtown is the whole point of doing this."))
    assert r.added_to_pool and len(scorer.pool) == 1
    tid = next(p.id for p in scorer.pool.points if p.text == "charge will reduce downtown traffic")
    assert scorer.pool.mentions([tid]) == 11  # article prior (10) + this submission once


@pytest.mark.parametrize("seed", range(5))
def test_scores_always_in_unit_interval(seeded_scorer, seed):
    import random
    rng = random.Random(seed)
    words = ["traffic", "shops", "park", "cameras", "buses", "air", "zone", "drivers", "lavender", "streets"]
    for _ in range(10):
        body = " ".join(rng.choice(words) for _ in range(20))
        r = seeded_scorer.score(sub("Random words here", body + " and more words to pass the limit"))
        assert 0.0 <= r.score <= 1.0


def test_small_registry_shows_judge_every_point(seeded_scorer, make_scorer):
    """With no neighbour above t, a small registry still exposes every point (no retrieval misses)."""
    seen = []
    scorer = make_scorer(pool=seeded_scorer.pool, t=0.999)
    inner = scorer.judge_fn
    scorer.judge_fn = lambda s, pts, cands: seen.append({c.id for c in cands}) or inner(s, pts, cands)
    scorer.score(NOVEL, add_to_pool=False)
    assert seen[0] == {p.id for p in scorer.pool.points}


def test_large_registry_falls_back_to_retrieval(seeded_scorer, make_scorer):
    seen = []
    scorer = make_scorer(pool=seeded_scorer.pool, t=0.999, show_all_below=0, fallback_common_points=1)
    inner = scorer.judge_fn
    scorer.judge_fn = lambda s, pts, cands: seen.append({c.id for c in cands}) or inner(s, pts, cands)
    scorer.score(NOVEL, add_to_pool=False)
    assert seen[0] < {p.id for p in scorer.pool.points}  # article points + most common only
