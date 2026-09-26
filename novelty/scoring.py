"""Pure scoring math (design spec: "Scoring and normalization"). No model calls here.

S = G * Q * (alpha * N_point + (1 - alpha) * N_sem), where every factor is in [0, 1].
"""

from __future__ import annotations

import math

import numpy as np

QUALITY_LEVELS = 4  # the judge rates quality on anchored integer levels 0..4


def rarity_weight(m: int, lam: float) -> float:
    """w(m) = exp(-m / lambda). A first-mover point (m = 0) is worth 1.0."""
    if m < 0:
        raise ValueError(f"prior mentions must be >= 0, got {m}")
    return math.exp(-m / lam)


def point_novelty(prior_mentions: list[int], lam: float) -> float:
    """Mean rarity weight over a submission's points. No points gives 0 (nothing to reward)."""
    if not prior_mentions:
        return 0.0
    return float(np.mean([rarity_weight(m, lam) for m in prior_mentions]))


def quality_factor(level: int, q_min: float, q_full: float) -> float:
    """Q: 0 below q_min, otherwise min(1, q / q_full), where q = level / 4. It can only reduce S."""
    if not 0 <= level <= QUALITY_LEVELS:
        raise ValueError(f"quality level must be 0..{QUALITY_LEVELS}, got {level}")
    q = level / QUALITY_LEVELS
    return 0.0 if q < q_min else min(1.0, q / q_full)


def empirical_cdf(value: float, reference: np.ndarray) -> float:
    """Percentile rank of value within reference (ties count half). Empty reference gives 1.0."""
    ref = np.sort(np.asarray(reference, dtype=float))
    if ref.size == 0:
        return 1.0
    below = np.searchsorted(ref, value, side="left")
    at_or_below = np.searchsorted(ref, value, side="right")
    return float((below + at_or_below) / (2 * ref.size))


def semantic_distance(sims: np.ndarray, top_k: int, max_weight: float = 0.5) -> float:
    """1 - (max_weight * closest + (1 - max_weight) * mean of top_k) cosine to the pool.

    The spec uses the top-k mean alone, but then one near-paraphrase among otherwise distant
    neighbours barely moves the score (DESIGN.md §3, change 10). The closest-match term catches it."""
    sims = np.asarray(sims, dtype=float)
    if sims.size == 0:
        return 1.0
    top = np.sort(sims)[::-1][:top_k]
    return float(1.0 - (max_weight * top[0] + (1 - max_weight) * top.mean()))


def loo_reference(vectors: np.ndarray, top_k: int, max_weight: float = 0.5) -> np.ndarray:
    """Leave-one-out semantic distance of every pooled submission against the rest."""
    n = len(vectors)
    if n < 2:
        return np.array([])
    S = vectors @ vectors.T
    return np.array([semantic_distance(np.delete(S[i], i), top_k, max_weight) for i in range(n)])


def final_score(n_point: float, n_sem: float, q: float, alpha: float, gates_passed: bool = True) -> float:
    if not gates_passed:
        return 0.0
    s = q * (alpha * n_point + (1 - alpha) * n_sem)
    return round(float(np.clip(s, 0.0, 1.0)), 4)
