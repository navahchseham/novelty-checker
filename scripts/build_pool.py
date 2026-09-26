"""Calibrate thresholds on the corpus, then build the pool by scoring the corpus in order.

Usage:  ../myvenv/bin/python -m scripts.build_pool

1. Calibration (embeddings only, corpus only; the golden set is never used):
     r_min, r_body_min  = lowest corpus value - margin   (the floor only catches obvious junk)
     t                  = 80th percentile of pairwise corpus similarity
     dup_max            = max(0.9, highest corpus pair + margin): the corpus has no copies by
                          construction, so the near-copy cut-off must sit above every corpus pair
2. Pool: extract the article's points, then score every corpus item through the full pipeline
   (shuffled with a fixed seed) and add it to the pool, exactly as live submissions would be.

Writes data/params.json, data/calibration.json, data/pool.json, data/pool_vectors.npy and
data/corpus_scores.json.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from novelty.config import JUDGE_MODEL, JUDGE_PROVIDER, Params
from novelty.embeddings import get_embedder
from novelty.extract import extract_article_points
from novelty.llm import get_llm
from novelty.pipeline import NoveltyScorer
from novelty.pool import Pool
from novelty.schema import Submission, load_fixed_content

DATA = Path(__file__).resolve().parent.parent / "data"
MARGIN = 0.05
DUP_MARGIN = 0.02
NEIGHBOUR_PERCENTILE = 80
SHUFFLE_SEED = 7


def calibrate(fixed, corpus: list[Submission], embedder, base: Params) -> tuple[Params, dict]:
    a = embedder.embed([fixed.to_text()])[0]
    H = embedder.embed([s.headline for s in corpus])
    B = embedder.embed([s.body for s in corpus])
    V = embedder.embed([s.to_text() for s in corpus])
    body_rel = B @ a
    rel = base.headline_weight * (H @ a) + (1 - base.headline_weight) * body_rel
    S = V @ V.T
    pairwise = S[~np.eye(len(S), dtype=bool)]
    params = replace(base, r_min=round(float(rel.min()) - MARGIN, 4),
                     r_body_min=round(float(body_rel.min()) - MARGIN, 4),
                     t=round(float(np.percentile(pairwise, NEIGHBOUR_PERCENTILE)), 4),
                     dup_max=round(max(base.dup_max, float(pairwise.max()) + DUP_MARGIN), 4))
    stats = {
        "relevance": {"min": float(rel.min()), "mean": float(rel.mean()), "max": float(rel.max())},
        "body_relevance": {"min": float(body_rel.min()), "mean": float(body_rel.mean())},
        "pairwise_similarity": {"min": float(pairwise.min()), "median": float(np.median(pairwise)),
                                f"p{NEIGHBOUR_PERCENTILE}": float(np.percentile(pairwise, NEIGHBOUR_PERCENTILE)),
                                "max": float(pairwise.max())},
        "mean_neighbours_at_t": float(((S >= params.t).sum(axis=1) - 1).mean()),
    }
    return params, stats


def main() -> None:
    fixed = load_fixed_content(DATA / "fixed_content.json")
    rows = json.loads((DATA / "corpus.json").read_text())
    corpus = [Submission.model_validate(r) for r in rows]
    embedder = get_embedder()
    llm = get_llm(model=JUDGE_MODEL, fallbacks="", provider=JUDGE_PROVIDER)

    params, stats = calibrate(fixed, corpus, embedder, Params())
    params.save(DATA / "params.json")
    (DATA / "calibration.json").write_text(json.dumps(stats, indent=2) + "\n")
    print("calibrated:", {k: v for k, v in asdict(params).items() if k in ("r_min", "r_body_min", "t", "dup_max")})
    print("stats:", json.dumps(stats))

    pool = Pool()
    article_points = extract_article_points(llm, fixed)
    pool.add_article_points(article_points, params.article_prior_mentions)
    print(f"article points ({len(article_points)}):", article_points)

    scorer = NoveltyScorer.from_llm(fixed, pool, embedder, llm, params)
    order = list(range(len(rows)))
    random.Random(SHUFFLE_SEED).shuffle(order)
    results = []
    for n, i in enumerate(order, 1):
        r = scorer.score(corpus[i], sub_id=rows[i]["id"])
        results.append({"id": rows[i]["id"], "angle_id": rows[i]["angle_id"], "order": n, "score": r.score,
                        "status": r.status, "n_point": r.n_point, "n_sem": r.n_sem, "quality": r.quality_level,
                        "added": r.added_to_pool, "points": r.points, "reason": r.reason})
        print(f"{n:>2} {rows[i]['id']} {rows[i]['angle_id']:<3} {r.status:<10} S={r.score:.3f} "
              f"points={len(r.points)} new={sum(p['prior_mentions'] == 0 for p in r.points)}")

    pool.save(DATA / "pool.json", DATA / "pool_vectors.npy")
    (DATA / "corpus_scores.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"pool: {len(pool)} submissions, {len(pool.points)} canonical points")


if __name__ == "__main__":
    main()
