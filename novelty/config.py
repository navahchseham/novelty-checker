"""Pipeline parameters (design spec: parameter table).

Values marked "calibrated" are tuned once on the corpus by scripts/calibrate.py. They are
stored in data/params.json and then fixed, so tests check behaviour rather than values
fitted to themselves.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

PARAMS_PATH = Path(__file__).resolve().parent.parent / "data" / "params.json"

# Models used for extraction and judging (one model for the whole pool). Overridable by env.
# Gemini's free daily quota ran out during development, so judging moved to Groq (DESIGN.md §3).
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "groq")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")


@dataclass(frozen=True)
class Params:
    # Relevance gate (embedding floor). It only rejects obvious junk; the LLM decides the rest.
    r_min: float = 0.0            # calibrated: 0.3*cos(headline) + 0.7*cos(body) floor
    r_body_min: float = 0.0       # calibrated: cos(body, article) floor
    headline_weight: float = 0.3
    # Duplicate gate
    dup_max: float = 0.9          # near-copy cutoff on full-text cosine
    # Neighbour retrieval
    t: float = 0.0                # calibrated: similarity threshold for neighbours
    k_max: int = 15               # cap that protects the judge's context
    show_all_below: int = 150     # while the registry is this small, the judge sees every point (change 12)
    fallback_common_points: int = 10  # shown to the judge when no neighbour passes t
    # Rarity and quality
    rarity_lambda: float = 0.4    # w(m) = exp(-m / lambda); spec said 2, see DESIGN.md §3 change 13
    article_prior_mentions: int = 10  # points already made by the article itself
    q_min: float = 0.4
    q_full: float = 0.7
    # Blend: alpha * N_point + (1 - alpha) * N_sem
    alpha: float = 0.85
    sem_top_k: int = 5            # neighbours averaged for N_sem
    sem_max_weight: float = 0.5   # weight of the single closest neighbour in N_sem (change 10)

    def save(self, path: Path = PARAMS_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path = PARAMS_PATH) -> "Params":
        if not path.exists():
            return cls()
        return replace(cls(), **json.loads(path.read_text()))
