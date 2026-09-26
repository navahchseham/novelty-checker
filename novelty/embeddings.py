"""Provider-agnostic text embeddings (see DESIGN.md §4).

Every Embedder returns L2-normalized float32 vectors, so cosine similarity is a dot product.
Select the provider with EMBED_PROVIDER ("gemini" by default, "fake" for offline use).
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
from dotenv import load_dotenv

from novelty.llm import rate_limit_wait

load_dotenv()

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "emb"
BATCH_SIZE = 50
MAX_RETRIES = 6


class Embedder(Protocol):
    cache_id: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) array of L2-normalized vectors, one row per text."""
        ...


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(norms == 0, 1, norms)


class GeminiEmbedder:
    """Gemini embeddings with task_type SEMANTIC_SIMILARITY.

    Each text is wrapped in its own Content. gemini-embedding-2 is multimodal and would
    otherwise merge a list of strings into ONE embedding.
    """

    def __init__(self, model: str | None = None):
        from google import genai

        self._client = genai.Client()
        self.model = model or os.getenv("EMBED_MODEL", "gemini-embedding-2")
        self.cache_id = f"gemini:{self.model}:similarity"

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        from google.genai import errors, types

        contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
        config = types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.models.embed_content(model=self.model, contents=contents, config=config)
                if len(resp.embeddings) != len(texts):
                    raise RuntimeError(f"expected {len(texts)} embeddings, got {len(resp.embeddings)}")
                return np.array([e.values for e in resp.embeddings], dtype=np.float32)
            except errors.APIError as e:
                if e.code == 429 and "PerDay" not in str(e):
                    wait = rate_limit_wait(e)
                elif e.code >= 500:
                    wait = 2 ** attempt * 5
                else:
                    raise
                print(f"  {self.model}: {e.code} {e.status}, retrying in {wait}s")
                time.sleep(wait)
            except httpx.TransportError as e:
                wait = 2 ** attempt * 5
                print(f"  {self.model}: network error ({type(e).__name__}), retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"{self.model} unavailable after {MAX_RETRIES} attempts")

    def embed(self, texts: list[str]) -> np.ndarray:
        batches = [self._embed_batch(texts[i : i + BATCH_SIZE]) for i in range(0, len(texts), BATCH_SIZE)]
        return normalize(np.vstack(batches))


class FakeEmbedder:
    """Deterministic, offline bag-of-words embedder (feature hashing of words and bigrams).

    Similarity reflects word overlap only, not meaning. It exists so the pipeline and its
    tests run without an API key. It is not a substitute for real embeddings in evaluation.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.cache_id = f"fake:{dim}"

    def _features(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9']+", text.lower())
        return words + [f"{a} {b}" for a, b in zip(words, words[1:])]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for f in self._features(text):
                h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return normalize(out)


class CachedEmbedder:
    """Per-text on-disk cache, keyed by (embedder id, text). Only uncached texts are sent."""

    def __init__(self, inner: Embedder, cache_dir: str | Path = CACHE_DIR):
        self.inner = inner
        self.cache_id = inner.cache_id
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, text: str) -> Path:
        return self.dir / f"{hashlib.sha256(f'{self.cache_id}|{text}'.encode()).hexdigest()}.npy"

    def embed(self, texts: list[str]) -> np.ndarray:
        paths = [self._path(t) for t in texts]
        missing = sorted({t for t, p in zip(texts, paths) if not p.exists()})
        if missing:
            for t, v in zip(missing, self.inner.embed(missing)):
                np.save(self._path(t), v)
        return np.vstack([np.load(p) for p in paths])


def get_embedder(cache: bool = True) -> Embedder:
    provider = os.getenv("EMBED_PROVIDER", "gemini").lower()
    if provider == "gemini":
        emb: Embedder = GeminiEmbedder()
    elif provider == "fake":
        return FakeEmbedder()  # cheap and deterministic, nothing to cache
    else:
        raise ValueError(f"unknown EMBED_PROVIDER {provider!r}")
    return CachedEmbedder(emb) if cache else emb
