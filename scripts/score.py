"""Score one submission against the pool and explain the result.

Usage:
    ../myvenv/bin/python -m scripts.score --headline "..." --body "..." --stance oppose
    ../myvenv/bin/python -m scripts.score --json submission.json
    add --add to save the submission into the pool (data/pool.json) after scoring
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate import DATA, load_scorer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headline")
    ap.add_argument("--body")
    ap.add_argument("--stance", default="undecided",
                    choices=["support", "support_with_changes", "oppose", "undecided"])
    ap.add_argument("--json", type=Path, help="file with headline, body and stance fields")
    ap.add_argument("--add", action="store_true", help="add the submission to the saved pool")
    args = ap.parse_args()

    raw = json.loads(args.json.read_text()) if args.json else \
        {"headline": args.headline or "", "body": args.body or "", "stance": args.stance}
    scorer = load_scorer()
    r = scorer.score(raw, add_to_pool=args.add)

    print(f"\nSCORE  {r.score:.3f}   ({r.status})")
    print(f"reason {r.reason}")
    if r.relevance is not None:
        print(f"relevance {r.relevance:.3f} (body {r.body_relevance:.3f}) | closest pooled {r.max_pool_similarity:.3f}"
              f" | N_sem {r.n_sem:.2f}")
    if r.n_point is not None:
        print(f"N_point {r.n_point:.2f} | quality {r.quality_level}/4 -> Q {r.quality_factor:.2f} | "
              f"neighbours {len(r.neighbours)}")
        for p in r.points:
            tag = "article" if p["restates_article"] else ("NEW" if p["prior_mentions"] == 0 else f"m={p['prior_mentions']}")
            print(f"  [{tag:>7}] {p['point']}" + (f"   ~ {p['match']}" if p["match"] else ""))
    if args.add and r.added_to_pool:
        scorer.pool.save(DATA / "pool.json", DATA / "pool_vectors.npy")
        print("added to pool")


if __name__ == "__main__":
    main()
