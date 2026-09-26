"""Offline what-if: recompute golden scores for other rarity decays (lambda), no API calls.

Uses the prior mentions, article flags, N_sem and quality stored in data/eval_results.json,
so it shows exactly how the lambda choice moves each criterion (DESIGN.md §5).

Usage:  ../myvenv/bin/python -m scripts.lambda_sweep 2 1 0.5
"""

from __future__ import annotations

import json
import sys

from novelty.config import Params
from novelty.scoring import final_score, point_novelty, quality_factor
from scripts.evaluate import DATA


def rescore(v: dict, lam: float, p: Params) -> float:
    if v["status"] != "scored":
        return v["score"]
    own = [x["prior_mentions"] for x in v["points"] if not x["restates_article"]]
    return final_score(point_novelty(own, lam), v["n_sem"], quality_factor(v["quality"], p.q_min, p.q_full), p.alpha)


def main() -> None:
    lams = [float(x) for x in sys.argv[1:]] or [2.0, 1.0, 0.5]
    res, p = json.loads((DATA / "eval_results.json").read_text()), Params.load()  # development set only
    rows = list(res["items"].items()) + [("FM-after", res["first_mover"]["paraphrase_after"] | {"category": "first_mover_after"})]
    print(f"{'item':<9} {'category':<20}" + "".join(f"  λ={l:<4}" for l in lams))
    for i, v in rows:
        print(f"{i:<9} {v['category']:<20}" + "".join(f"  {rescore(v, l, p):>6.2f}" for l in lams))


if __name__ == "__main__":
    main()
