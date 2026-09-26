import numpy as np
import pytest

from novelty.scoring import (empirical_cdf, final_score, loo_reference, point_novelty, quality_factor,
                             rarity_weight, semantic_distance)


def test_rarity_weights_match_spec():
    # Spec: lambda = 2 gives m=0 -> 1.0, m=1 -> 0.61, m=3 -> 0.22, m=10 -> almost nothing
    assert rarity_weight(0, 2) == 1.0
    assert rarity_weight(1, 2) == pytest.approx(0.61, abs=0.005)
    assert rarity_weight(3, 2) == pytest.approx(0.22, abs=0.005)
    assert rarity_weight(10, 2) < 0.01


def test_four_old_plus_one_new_is_about_point_two():
    assert point_novelty([10, 10, 10, 10, 0], 2) == pytest.approx(0.2, abs=0.01)


def test_no_points_scores_zero():
    assert point_novelty([], 2) == 0.0


@pytest.mark.parametrize("level,expected", [(0, 0.0), (1, 0.0), (2, 0.5 / 0.7), (3, 1.0), (4, 1.0)])
def test_quality_factor_is_a_floor_not_a_reward(level, expected):
    assert quality_factor(level, 0.4, 0.7) == pytest.approx(expected)


def test_empirical_cdf():
    ref = np.array([0.1, 0.2, 0.3, 0.4])
    assert empirical_cdf(0.0, ref) == 0.0 and empirical_cdf(1.0, ref) == 1.0
    assert empirical_cdf(0.2, ref) == pytest.approx(0.375)
    assert empirical_cdf(0.5, np.array([])) == 1.0


def test_semantic_distance_mixes_closest_and_top_k_mean():
    assert semantic_distance(np.array([0.9, 0.8, 0.1]), top_k=2, max_weight=0) == pytest.approx(1 - 0.85)
    assert semantic_distance(np.array([0.9, 0.8, 0.1]), top_k=2) == pytest.approx(1 - (0.45 + 0.425))


def test_one_near_paraphrase_lowers_semantic_distance():
    lone = np.array([0.92, 0.70, 0.70, 0.70, 0.70])     # one close match among distant ones
    far = np.array([0.72, 0.70, 0.70, 0.70, 0.70])
    assert semantic_distance(lone, 5) < semantic_distance(far, 5) - 0.05


def test_loo_reference_excludes_self():
    v = np.eye(3, dtype=np.float32)          # orthogonal vectors: each is distance 1 from the others
    np.testing.assert_allclose(loo_reference(v, 2), [1.0, 1.0, 1.0])


def test_failed_gate_is_exactly_zero():
    assert final_score(1.0, 1.0, 1.0, 0.85, gates_passed=False) == 0.0


def test_bounds_property():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        ms = list(rng.integers(0, 30, size=rng.integers(0, 9)))
        s = final_score(point_novelty(ms, rng.uniform(0.1, 5)), rng.uniform(0, 1),
                        quality_factor(int(rng.integers(0, 5)), 0.4, 0.7), rng.uniform(0, 1))
        assert 0.0 <= s <= 1.0
