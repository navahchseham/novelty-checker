"""Pure scoring math: gates, novelty normalization and the final combination.

No model calls happen here. The embedding and LLM layers produce the inputs
(cosine similarities, rubric levels) and this module turns them into a score in
[0, 1]. See DESIGN.md §2 for the rationale behind each formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RUBRIC_MAX = 4  # LLM rubrics are scored on integer levels 0..4


@dataclass(frozen=True)
class ScoringConfig:
    top_k: int = 5                  # neighbours used for novelty (embedding and LLM judge)
    max_sim_weight: float = 0.5     # weight of the single closest neighbour vs. the top-k mean
    emb_novelty_weight: float = 0.4  # blend: w * N_emb + (1 - w) * N_llm
    gate_floor: int = 1             # rubric level at or below which a gate is fully closed
    gate_full: int = 3              # rubric level at or above which a gate is fully open


@dataclass(frozen=True)
class ScoreBreakdown:
    novelty_emb: float
    novelty_llm: float | None
    novelty: float
    relevance_gate: float
    quality_gate: float
    score: float


def rubric_gate(level: int, cfg: ScoringConfig = ScoringConfig()) -> float:
    """Map a 0..4 rubric level to a gate in [0, 1] that ramps linearly from floor to full."""
    if not 0 <= level <= RUBRIC_MAX:
        raise ValueError(f"rubric level must be 0..{RUBRIC_MAX}, got {level}")
    span = cfg.gate_full - cfg.gate_floor
    return float(np.clip((level - cfg.gate_floor) / span, 0.0, 1.0))


def novelty_distance(sims: np.ndarray, cfg: ScoringConfig = ScoringConfig()) -> float:
    """Raw novelty from cosine similarities to the corpus (self excluded).

    Mixes the closest neighbour (catches a near-duplicate of one item) with the top-k
    mean (catches a crowded region where many people made the same point).
    """
    sims = np.asarray(sims, dtype=float)
    if sims.size == 0:
        raise ValueError("need at least one corpus similarity")
    top = np.sort(sims)[::-1][: cfg.top_k]
    closeness = cfg.max_sim_weight * top[0] + (1 - cfg.max_sim_weight) * top.mean()
    return float(1.0 - closeness)


def empirical_cdf(value: float, reference: np.ndarray) -> float:
    """Fraction of the reference distribution at or below value, with ties counted half.

    Used to normalize raw novelty distance against the corpus's own leave-one-out
    distances, so the result does not depend on the embedding model's similarity scale.
    """
    ref = np.sort(np.asarray(reference, dtype=float))
    if ref.size == 0:
        raise ValueError("reference distribution is empty")
    below = np.searchsorted(ref, value, side="left")
    at_or_below = np.searchsorted(ref, value, side="right")
    return float((below + at_or_below) / (2 * ref.size))


def combine(
    novelty_emb: float,
    relevance_level: int,
    quality_level: int,
    novelty_llm_level: int | None = None,
    cfg: ScoringConfig = ScoringConfig(),
) -> ScoreBreakdown:
    """Final score = novelty x relevance_gate x quality_gate, always in [0, 1].

    If novelty_llm_level is None (embedding-only mode) the novelty is N_emb alone.
    """
    if not 0.0 <= novelty_emb <= 1.0:
        raise ValueError(f"novelty_emb must be in [0, 1], got {novelty_emb}")
    novelty_llm = None if novelty_llm_level is None else novelty_llm_level / RUBRIC_MAX
    if novelty_llm is None:
        novelty = novelty_emb
    else:
        w = cfg.emb_novelty_weight
        novelty = w * novelty_emb + (1 - w) * novelty_llm
    rel = rubric_gate(relevance_level, cfg)
    qual = rubric_gate(quality_level, cfg)
    return ScoreBreakdown(
        novelty_emb=novelty_emb,
        novelty_llm=novelty_llm,
        novelty=novelty,
        relevance_gate=rel,
        quality_gate=qual,
        score=round(novelty * rel * qual, 4),
    )
