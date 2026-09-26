"""Diagnostic: do the embeddings separate ideas? (Step 4, DESIGN.md §4.1)

Embeds the fixed content, corpus and golden set (cached after the first run), then reports:
  1. within-angle vs. between-angle similarity in the corpus
  2. nearest-neighbour angle accuracy (does an item's closest neighbour share its angle?)
  3. per golden category: similarity to the fixed content and to the closest corpus item
  4. a PREVIEW of normalized embedding novelty N_emb per golden category

Usage:  ../myvenv/bin/python -m scripts.inspect_embeddings
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from novelty.embeddings import get_embedder
from novelty.schema import Submission, load_fixed_content
from novelty.scoring import empirical_cdf, novelty_distance

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    fixed = load_fixed_content(DATA / "fixed_content.json")
    corpus = json.loads((DATA / "corpus.json").read_text())
    golden = json.loads((DATA / "golden.json").read_text())
    text = lambda x: Submission.model_validate(x).to_text()

    emb = get_embedder()
    print(f"embedder: {emb.cache_id}")
    f = emb.embed([fixed.to_text()])[0]
    C = emb.embed([text(x) for x in corpus])
    G = emb.embed([text(x) for x in golden])

    # 1. within vs between angle similarity (corpus, excluding self-pairs)
    S = C @ C.T
    angles = np.array([x["angle_id"] for x in corpus])
    same = angles[:, None] == angles[None, :]
    off_diag = ~np.eye(len(C), dtype=bool)
    within, between = S[same & off_diag].mean(), S[~same].mean()
    print(f"\n1. corpus similarity   within-angle {within:.3f} | between-angle {between:.3f} | gap {within - between:.3f}")

    # 2. nearest-neighbour angle accuracy, for items whose angle has at least 2 members
    S_masked = np.where(off_diag, S, -np.inf)
    nn = S_masked.argmax(axis=1)
    multi = np.array([(angles == a).sum() > 1 for a in angles])
    acc = (angles[nn] == angles)[multi].mean()
    print(f"2. nearest-neighbour angle accuracy (angles with >1 item): {acc:.0%}")

    # 3 and 4. golden categories
    loo = np.array([novelty_distance(np.delete(S[i], i)) for i in range(len(C))])
    rows = defaultdict(list)
    for x, g in zip(golden, G):
        sims = C @ g
        n_emb = empirical_cdf(novelty_distance(sims), loo)
        rows[x["category"]].append((float(g @ f), float(sims.max()), n_emb))

    print(f"3/4. golden categories   (corpus items' mean sim to fixed content: {(C @ f).mean():.3f})")
    print(f"   {'category':<18} {'n':>2}  {'sim→fixed':>9}  {'max sim→corpus':>14}  {'N_emb preview':>13}")
    for cat in ["novel_relevant", "partly_novel", "paraphrase_common", "paraphrase_rare",
                "offtopic_novel", "low_effort", "gibberish"]:
        r = np.array(rows[cat])
        print(f"   {cat:<18} {len(r):>2}  {r[:, 0].mean():>9.3f}  {r[:, 1].mean():>14.3f}  "
              f"{r[:, 2].mean():>6.2f} [{r[:, 2].min():.2f}–{r[:, 2].max():.2f}]")


if __name__ == "__main__":
    main()
