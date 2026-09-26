import itertools

import numpy as np
import pytest

from novelty.scoring import (
    RUBRIC_MAX,
    ScoringConfig,
    combine,
    empirical_cdf,
    novelty_distance,
    rubric_gate,
)


def test_rubric_gate_ramp():
    assert [rubric_gate(l) for l in range(RUBRIC_MAX + 1)] == [0.0, 0.0, 0.5, 1.0, 1.0]


def test_rubric_gate_rejects_out_of_range():
    with pytest.raises(ValueError):
        rubric_gate(5)


def test_near_duplicate_has_low_distance():
    duplicate = novelty_distance([0.99, 0.5, 0.4, 0.3, 0.2, 0.1])
    distinct = novelty_distance([0.45, 0.4, 0.35, 0.3, 0.2, 0.1])
    assert duplicate < distinct


def test_crowded_region_has_lower_distance_than_single_match():
    # Same closest neighbour, but many close neighbours = a well-covered point.
    crowded = novelty_distance([0.8, 0.8, 0.8, 0.8, 0.8])
    lone = novelty_distance([0.8, 0.3, 0.3, 0.3, 0.3])
    assert crowded < lone


def test_empirical_cdf_bounds_and_ties():
    ref = np.array([0.1, 0.2, 0.3, 0.4])
    assert empirical_cdf(0.0, ref) == 0.0
    assert empirical_cdf(1.0, ref) == 1.0
    assert empirical_cdf(0.2, ref) == pytest.approx(0.375)  # 1 below + half of the tie


def test_off_topic_or_junk_scores_zero_even_if_maximally_novel():
    assert combine(1.0, relevance_level=1, quality_level=4, novelty_llm_level=4).score == 0.0
    assert combine(1.0, relevance_level=4, quality_level=0, novelty_llm_level=4).score == 0.0


def test_embedding_only_mode():
    b = combine(0.7, relevance_level=4, quality_level=4)
    assert b.novelty_llm is None and b.score == pytest.approx(0.7)


def test_blend_weights():
    cfg = ScoringConfig(emb_novelty_weight=0.4)
    b = combine(0.5, relevance_level=4, quality_level=4, novelty_llm_level=4, cfg=cfg)
    assert b.score == pytest.approx(0.4 * 0.5 + 0.6 * 1.0)


def test_score_always_in_unit_interval():
    for n, r, q, l in itertools.product([0.0, 0.3, 1.0], range(5), range(5), [None, *range(5)]):
        assert 0.0 <= combine(n, r, q, l).score <= 1.0
