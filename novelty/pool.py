"""The pool of scored submissions and its canonical point registry (design spec: step 7).

Prior-mention counts live here, in code. The judge only says which existing points a new
point matches (possibly several near-synonyms), and the pool supplies how many distinct
submissions made any of them (DESIGN.md §3, changes 2 and 8).
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class CanonicalPoint:
    id: int
    text: str
    count: int                  # number of submissions making this point (article points start high)
    source: str = "submission"  # "article" or "submission"
    prior: int = 0              # mentions assumed before any submission (article points only)


@dataclass
class PoolEntry:
    id: str
    headline: str
    body: str
    stance: str
    points: list[str]
    point_ids: list[list[int]]  # per point: the canonical ids it was matched to (or its new id)
    score: float | None = None


@dataclass
class Pool:
    points: list[CanonicalPoint] = field(default_factory=list)
    entries: list[PoolEntry] = field(default_factory=list)
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))

    # --- registry ---------------------------------------------------------------------------
    def add_article_points(self, texts: list[str], prior_mentions: int) -> None:
        for t in texts:
            self.points.append(CanonicalPoint(len(self.points), t, prior_mentions, "article", prior_mentions))

    def mentions(self, point_ids: list[int]) -> int:
        """Prior mentions m of a claim matched to these canonical ids: the number of distinct
        pooled submissions containing any of them, plus the article prior if one is an article point.
        Taking the union makes m robust to one idea being split across several canonical points."""
        ids = set(point_ids)
        subs = sum(1 for e in self.entries if any(ids & set(p) for p in e.point_ids))
        prior = max((self.points[i].prior for i in ids), default=0)
        return subs + prior

    def most_common(self, n: int) -> list[int]:
        return [p.id for p in sorted(self.points, key=lambda p: -p.count)[:n]]

    def article_point_ids(self) -> list[int]:
        return [p.id for p in self.points if p.source == "article"]

    # --- submissions ------------------------------------------------------------------------
    def add(self, entry: PoolEntry, vector: np.ndarray, matches: list[list[int]]) -> None:
        """Add a scored submission. Matched canonical points gain one mention (once per
        submission); an unmatched point becomes a new canonical point."""
        point_ids, counted = [], set()
        for text, ids in zip(entry.points, matches):
            if not ids:
                ids = [len(self.points)]
                self.points.append(CanonicalPoint(ids[0], text, 0))
            for i in set(ids) - counted:
                self.points[i].count += 1
                counted.add(i)
            point_ids.append(list(ids))
        entry.point_ids = point_ids
        self.entries.append(entry)
        v = np.asarray(vector, dtype=np.float32)[None, :]
        self.vectors = v if self.vectors.size == 0 else np.vstack([self.vectors, v])

    def __len__(self) -> int:
        return len(self.entries)

    def clone(self) -> "Pool":
        return copy.deepcopy(self)

    # --- persistence ------------------------------------------------------------------------
    def save(self, json_path: Path, vectors_path: Path) -> None:
        data = {"points": [asdict(p) for p in self.points], "entries": [asdict(e) for e in self.entries]}
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        np.save(vectors_path, self.vectors)

    @classmethod
    def load(cls, json_path: Path, vectors_path: Path) -> "Pool":
        data = json.loads(json_path.read_text())
        return cls(
            points=[CanonicalPoint(**p) for p in data["points"]],
            entries=[PoolEntry(**e) for e in data["entries"]],
            vectors=np.load(vectors_path),
        )
