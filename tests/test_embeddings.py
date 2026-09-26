import numpy as np

from novelty.embeddings import CachedEmbedder, FakeEmbedder, normalize


class CountingEmbedder:
    """Wraps FakeEmbedder and records which texts actually reached the 'provider'."""

    cache_id = "counting"

    def __init__(self):
        self.inner = FakeEmbedder()
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return self.inner.embed(texts)


def test_fake_embedder_is_normalized_and_deterministic():
    texts = ["four day week reduces burnout", "sourdough starter on parchment"]
    a, b = FakeEmbedder().embed(texts), FakeEmbedder().embed(texts)
    assert a.shape == (2, 512)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(a, b)


def test_fake_embedder_similarity_tracks_word_overlap():
    v = FakeEmbedder().embed([
        "the four day week will reduce burnout for city staff",
        "a four day week will reduce burnout for staff",
        "dry the sourdough starter on parchment paper",
    ])
    assert v[0] @ v[1] > v[0] @ v[2]


def test_normalize_handles_zero_vector():
    out = normalize(np.zeros((1, 4)))
    assert np.all(np.isfinite(out))


def test_cache_only_sends_uncached_texts(tmp_path):
    inner = CountingEmbedder()
    emb = CachedEmbedder(inner, cache_dir=tmp_path)

    first = emb.embed(["alpha beta", "gamma delta"])
    second = emb.embed(["gamma delta", "epsilon zeta", "alpha beta"])

    assert inner.calls == [["alpha beta", "gamma delta"], ["epsilon zeta"]]
    np.testing.assert_array_equal(first[0], second[2])  # order preserved, value reused
    np.testing.assert_array_equal(first[1], second[0])


def test_cache_dedupes_repeated_texts_in_one_call(tmp_path):
    inner = CountingEmbedder()
    out = CachedEmbedder(inner, cache_dir=tmp_path).embed(["same", "same"])
    assert inner.calls == [["same"]] and out.shape[0] == 2
