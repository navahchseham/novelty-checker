"""End-to-end scoring pipeline (design spec: "Pipeline overview").

validate -> relevance gate -> duplicate gate -> extract points (+ LLM relevance)
-> retrieve neighbours -> judge -> score -> add to pool
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from pydantic import ValidationError

from novelty.config import Params
from novelty.embeddings import Embedder
from novelty.extract import Extraction, extract_points
from novelty.judge import Judgement, judge
from novelty.llm import LLM
from novelty.pool import CanonicalPoint, Pool, PoolEntry
from novelty.schema import FixedContent, Submission
from novelty.scoring import (empirical_cdf, final_score, loo_reference, point_novelty, quality_factor,
                             rarity_weight, semantic_distance)

Extractor = Callable[[Submission], Extraction]
JudgeFn = Callable[[Submission, list[str], list[CanonicalPoint]], Judgement]


@dataclass
class ScoreResult:
    score: float
    status: str                        # scored | invalid | irrelevant | duplicate | no_points
    reason: str
    relevance: float | None = None
    body_relevance: float | None = None
    max_pool_similarity: float | None = None
    n_sem: float | None = None
    n_point: float | None = None
    quality_level: int | None = None
    quality_factor: float | None = None
    points: list[dict] = field(default_factory=list)
    neighbours: list[str] = field(default_factory=list)
    added_to_pool: bool = False


class NoveltyScorer:
    def __init__(self, fixed: FixedContent, pool: Pool, embedder: Embedder, extractor: Extractor,
                 judge_fn: JudgeFn, params: Params | None = None):
        self.fixed, self.pool, self.embedder = fixed, pool, embedder
        self.extractor, self.judge_fn = extractor, judge_fn
        self.p = params or Params.load()
        self.article_vec = embedder.embed([fixed.to_text()])[0]
        self._loo: np.ndarray | None = None

    @classmethod
    def from_llm(cls, fixed: FixedContent, pool: Pool, embedder: Embedder, llm: LLM,
                 params: Params | None = None) -> "NoveltyScorer":
        return cls(fixed, pool, embedder,
                   extractor=lambda sub: extract_points(llm, fixed, sub),
                   judge_fn=lambda sub, pts, cands: judge(llm, fixed, sub, pts, cands),
                   params=params)

    def _reference(self) -> np.ndarray:
        if self._loo is None:
            self._loo = loo_reference(self.pool.vectors, self.p.sem_top_k, self.p.sem_max_weight)
        return self._loo

    def relevance(self, headline_vec: np.ndarray, body_vec: np.ndarray) -> tuple[float, float]:
        body_rel = float(body_vec @ self.article_vec)
        w = self.p.headline_weight
        return w * float(headline_vec @ self.article_vec) + (1 - w) * body_rel, body_rel

    def score(self, raw: Submission | dict[str, Any], add_to_pool: bool = True,
              sub_id: str | None = None) -> ScoreResult:
        p = self.p
        # 1. Validate
        try:
            sub = raw if isinstance(raw, Submission) else Submission.model_validate(raw)
        except ValidationError as e:
            return ScoreResult(0.0, "invalid", f"invalid submission: {e.errors()[0]['msg']}")

        # 2. Relevance gate (embedding floor). N_sem is computed first so rejected items still report it.
        h_vec, b_vec, v = self.embedder.embed([sub.headline, sub.body, sub.to_text()])
        rel, body_rel = self.relevance(h_vec, b_vec)
        sims = self.pool.vectors @ v if len(self.pool) else np.array([])
        n_sem = empirical_cdf(semantic_distance(sims, p.sem_top_k, p.sem_max_weight), self._reference())
        max_sim = float(sims.max()) if sims.size else 0.0
        base = dict(relevance=rel, body_relevance=body_rel, max_pool_similarity=max_sim, n_sem=n_sem)
        if body_rel < p.r_body_min or rel < p.r_min:
            return ScoreResult(0.0, "irrelevant", "irrelevant: below the embedding relevance floor", **base)

        # 3. Duplicate gate (stance excluded from the text, so flipping it does not escape)
        if max_sim >= p.dup_max:
            return ScoreResult(0.0, "duplicate", f"near-copy of a pooled submission (cos {max_sim:.3f})", **base)

        # 4. Extract points; the same call checks relevance with the LLM
        ext = self.extractor(sub)
        if not ext.about_article:
            return ScoreResult(0.0, "irrelevant", "irrelevant: the LLM judged it off-topic", **base)
        if not ext.points:
            return ScoreResult(0.0, "no_points", "no claims found", **base)

        # 5. Retrieve neighbours: similarity >= t, capped at K; article points are always candidates
        order = np.argsort(-sims) if sims.size else np.array([], dtype=int)
        neighbours = [int(i) for i in order if sims[i] >= p.t][: p.k_max]
        cand_ids = set(self.pool.article_point_ids())
        for i in neighbours:
            cand_ids.update(pid for ids in self.pool.entries[i].point_ids for pid in ids)
        if not neighbours:
            cand_ids.update(self.pool.most_common(p.fallback_common_points))
        if len(self.pool.points) <= p.show_all_below:
            # Small registry: show every point, so a retrieval miss cannot make an old point look new.
            cand_ids = {pt.id for pt in self.pool.points}
        candidates = [self.pool.points[i] for i in sorted(cand_ids)]

        # 6. Judge: which existing point each new point repeats, plus quality
        j = self.judge_fn(sub, ext.points, candidates)
        matches = [m.match_ids for m in j.matches]

        # 7. Score. Prior mentions come from the pool registry, not the LLM. Points that only restate
        #    the article are neutral context: excluded from N_point, so they neither add nor dilute.
        #    A submission that only restates the article has no claims of its own, so N_point = 0.
        article_ids = set(self.pool.article_point_ids())
        prior = [self.pool.mentions(ids) if ids else 0 for ids in matches]
        restates = [bool(set(ids) & article_ids) for ids in matches]
        own = [m for m, r in zip(prior, restates) if not r]
        n_point = point_novelty(own, p.rarity_lambda)
        q = quality_factor(j.quality, p.q_min, p.q_full)
        s = final_score(n_point, n_sem, q, p.alpha)
        detail = [{"point": pt, "match_ids": ids,
                   "match": " | ".join(self.pool.points[i].text for i in ids) or None,
                   "restates_article": r, "prior_mentions": m,
                   "weight": round(rarity_weight(m, p.rarity_lambda), 4)}
                  for pt, ids, m, r in zip(ext.points, matches, prior, restates)]
        result = ScoreResult(s, "scored", j.reason, **base, n_point=n_point, quality_level=j.quality,
                             quality_factor=q, points=detail,
                             neighbours=[self.pool.entries[i].id for i in neighbours])

        # 8. Add to pool. Submissions zeroed by the quality floor are not added, so noise cannot inflate counts.
        if add_to_pool and q > 0:
            entry = PoolEntry(sub_id or sub.id or f"s{len(self.pool) + 1:04d}", sub.headline, sub.body,
                              sub.stance.value, list(ext.points), [], s)
            self.pool.add(entry, v, matches)
            self._loo = None
            result.added_to_pool = True
        return result
