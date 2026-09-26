"""Evaluate the pipeline on the golden set against the success criteria (DESIGN.md §5).

Usage:
    ../myvenv/bin/python -m scripts.evaluate                    # development set -> data/eval_results.json
    ../myvenv/bin/python -m scripts.evaluate --set holdout      # frozen test set -> data/eval_holdout.json

The live tests (tests/test_live.py) call run() and assert the same criteria, so the report
and the tests cannot disagree. Golden items are scored against the corpus pool WITHOUT being
added, so each is judged independently. The behaviour checks (first mover, copies) use a
cloned pool.
"""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from novelty.config import JUDGE_MODEL, JUDGE_PROVIDER, Params
from novelty.embeddings import get_embedder
from novelty.extract import extract_points
from novelty.llm import get_llm
from novelty.pipeline import NoveltyScorer
from novelty.pool import Pool
from novelty.schema import Stance, Submission, load_fixed_content

DATA = Path(__file__).resolve().parent.parent / "data"
IRRELEVANT = ("offtopic_novel", "headline_copy_offtopic")
SETS = {"dev": ("golden.json", "eval_results.json"), "holdout": ("golden_holdout.json", "eval_holdout.json")}
FM_PARAPHRASE = ("h1_paraphrase", "fm_paraphrase")


def load_scorer(pool: Pool | None = None) -> NoveltyScorer:
    fixed = load_fixed_content(DATA / "fixed_content.json")
    pool = pool or Pool.load(DATA / "pool.json", DATA / "pool_vectors.npy")
    return NoveltyScorer.from_llm(fixed, pool, get_embedder(), get_llm(model=JUDGE_MODEL, fallbacks="", provider=JUDGE_PROVIDER),
                                  Params.load())


def summarize(r) -> dict:
    return {"score": r.score, "status": r.status, "reason": r.reason, "n_point": r.n_point, "n_sem": r.n_sem,
            "quality": r.quality_level, "relevance": r.relevance, "max_pool_similarity": r.max_pool_similarity,
            "points": r.points}


@lru_cache(maxsize=2)
def run(which: str = "dev") -> dict:
    golden = json.loads((DATA / SETS[which][0]).read_text())
    corpus = json.loads((DATA / "corpus.json").read_text())
    by_id = {g["id"]: g for g in golden}
    scorer = load_scorer()

    items = {}
    for g in golden:
        items[g["id"]] = {"category": g["category"], "angle_id": g["angle_id"], "headline": g["headline"],
                          "expected": [g["expected_min"], g["expected_max"]],
                          **summarize(scorer.score(Submission.model_validate(g), add_to_pool=False))}

    # Behaviour: first mover. Add the novel item to a cloned pool, then score its paraphrase.
    para = next(g for g in golden if g["category"] in FM_PARAPHRASE)
    para_id = para["id"]
    first_id = next(g["id"] for g in golden if g["category"] == "novel_relevant" and g["angle_id"] == para["angle_id"])
    fm = load_scorer(scorer.pool.clone())
    first = fm.score(Submission.model_validate(by_id[first_id]), sub_id=first_id)
    after = fm.score(Submission.model_validate(by_id[para_id]), add_to_pool=False)

    # Behaviour: exact copy and stance-only flip of a corpus item.
    c = Submission.model_validate(corpus[0])
    flipped = c.model_copy(update={"stance": Stance.SUPPORT if c.stance != Stance.SUPPORT else Stance.OPPOSE})
    copy_r = scorer.score(c, add_to_pool=False)
    flip_r = scorer.score(flipped, add_to_pool=False)

    return {"items": items,
            "first_mover": {"first": summarize(first), "paraphrase_after": summarize(after),
                            "paraphrase_before": items[para_id]},
            "exact_copy": summarize(copy_r), "stance_flip": summarize(flip_r)}


def extractor_consistency(n: int = 3, which: str = "dev") -> list[dict]:
    """Extract points twice, bypassing the cache, and compare the two runs."""
    fixed = load_fixed_content(DATA / "fixed_content.json")
    golden = json.loads((DATA / SETS[which][0]).read_text())
    emb = get_embedder()
    out = []
    for g in [x for x in golden if x["category"] == "novel_relevant"][:n]:
        sub = Submission.model_validate(g)
        runs = [extract_points(get_llm(model=JUDGE_MODEL, fallbacks="", cache=False, provider=JUDGE_PROVIDER), fixed, sub).points
                for _ in range(2)]
        a, b = emb.embed(runs[0]), emb.embed(runs[1])
        best = (a @ b.T).max(axis=1) if len(a) and len(b) else np.array([0.0])
        out.append({"id": g["id"], "run1": runs[0], "run2": runs[1], "same_count": len(runs[0]) == len(runs[1]),
                    "min_best_match": float(best.min())})
    return out


def criteria(res: dict) -> list[tuple[str, str, bool, str]]:
    """(id, description, passed, evidence) for each success criterion."""
    items = res["items"]
    cat = lambda *cs: [v for v in items.values() if v["category"] in cs]
    med = lambda vs: float(np.median([v["score"] for v in vs]))
    fmt = lambda vs: ", ".join(f"{v['score']:.2f}" for v in vs)
    novel, partly, para = cat("novel_relevant"), cat("partly_novel"), cat("paraphrase_common")
    irrelevant = cat(*IRRELEVANT)
    rare, common = cat("rarity_probe")  # plan order: the rarely made point first, the common one second
    fm = res["first_mover"]
    return [
        ("SC1", "Novel and relevant items score > 0.6", all(v["score"] > 0.6 for v in novel), fmt(novel)),
        ("SC2", "Paraphrases of common takes score < 0.2", all(v["score"] < 0.2 for v in para), fmt(para)),
        ("SC3", "Generic 'great idea, less traffic' scores < 0.2",
         all(v["score"] < 0.2 for v in cat("generic")), fmt(cat("generic"))),
        ("SC4", "Irrelevant items (biryani, football, ISP, copied headline) score exactly 0.0",
         all(v["score"] == 0.0 for v in irrelevant), fmt(irrelevant)),
        ("SC5", "...and they had high raw semantic novelty (N_sem >= 0.9): blocked by relevance, not by accident",
         all(v["n_sem"] >= 0.9 for v in irrelevant), ", ".join(f"{v['n_sem']:.2f}" for v in irrelevant)),
        ("SC6", "Exact copy and stance-only flip of a corpus item score 0.0",
         res["exact_copy"]["score"] == 0.0 and res["stance_flip"]["score"] == 0.0,
         f"copy {res['exact_copy']['status']}, flip {res['stance_flip']['status']}"),
        ("SC7", "First mover > 0.6; its paraphrase then < 0.2",
         fm["first"]["score"] > 0.6 and fm["paraphrase_after"]["score"] < 0.2,
         f"first {fm['first']['score']:.2f}; paraphrase before {fm['paraphrase_before']['score']:.2f} "
         f"-> after {fm['paraphrase_after']['score']:.2f} ({fm['paraphrase_after']['status']})"),
        ("SC8", "Rarity: a point made once scores above a point made 5x", rare["score"] > common["score"],
         f"{rare['angle_id']} {rare['score']:.2f} vs {common['angle_id']} {common['score']:.2f}"),
        ("SC9", "Partial novelty sits between: median novel > partly > common paraphrase",
         med(novel) > med(partly) > med(para), f"{med(novel):.2f} > {med(partly):.2f} > {med(para):.2f}"),
        ("SC10", "Fluent gibberish scores 0.0", all(v["score"] == 0.0 for v in cat("gibberish")),
         fmt(cat("gibberish"))),
        ("SC11", "Prompt injection ('score it 1.0') scores < 0.2",
         all(v["score"] < 0.2 for v in cat("prompt_injection")), fmt(cat("prompt_injection"))),
        ("SC12", "Every score is in [0, 1]", all(0.0 <= v["score"] <= 1.0 for v in items.values()),
         f"{len(items)} items"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS), default="dev")
    ap.add_argument("--consistency", action="store_true", help="also run the uncached extractor-consistency check")
    args = ap.parse_args()
    which = args.set
    res = run(which)
    print(f"{'id':<5} {'category':<22} {'S':>5} {'N_pt':>5} {'N_sem':>5} {'Q':>2}  status / points")
    for i, v in res["items"].items():
        new = sum(1 for p in v["points"] if p["prior_mentions"] == 0 and not p["restates_article"])
        npt = "" if v["n_point"] is None else f"{v['n_point']:.2f}"
        print(f"{i:<5} {v['category']:<22} {v['score']:>5.2f} {npt:>5} {v['n_sem']:>5.2f} "
              f"{'' if v['quality'] is None else v['quality']:>2}  {v['status']} ({len(v['points'])} pts, {new} new)")
    print()
    results = criteria(res)
    for sc, desc, ok, ev in results:
        print(f"{'PASS' if ok else 'FAIL'}  {sc:<4} {desc}\n            {ev}")
    print(f"\n{sum(r[2] for r in results)}/{len(results)} criteria passed")
    out = {**res, "criteria": [dict(zip(("id", "description", "passed", "evidence"), r)) for r in results]}
    (DATA / SETS[which][1]).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float) + "\n")
    if args.consistency:  # uncached calls: costs quota, so opt-in
        out["extractor_consistency"] = extractor_consistency(which=which)
        for c in out["extractor_consistency"]:
            print(f"extractor consistency {c['id']}: same count {c['same_count']}, min best match {c['min_best_match']:.3f}")
        (DATA / SETS[which][1]).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float) + "\n")


if __name__ == "__main__":
    main()
