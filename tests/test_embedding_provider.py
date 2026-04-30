#!/usr/bin/env python3
"""
Embedding Provider 单元测试
测试 Hash / ST / Cached embedding provider 功能
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.embedding_provider import (
    HashEmbeddingProvider,
    CachedEmbeddingProvider,
    cosine_similarity,
)


class TestHashEmbeddingProvider:
    """Hash fallback provider 测试"""

    def test_encode_returns_valid_dim(self):
        provider = HashEmbeddingProvider(dim=384)
        vec = provider.encode("test text")
        assert len(vec) == 384

    def test_encode_normalized(self):
        provider = HashEmbeddingProvider(dim=384)
        vec = provider.encode("hello world")
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    def test_deterministic(self):
        provider = HashEmbeddingProvider(dim=128)
        v1 = provider.encode("deterministic test")
        v2 = provider.encode("deterministic test")
        assert v1 == v2

    def test_different_texts_different_vectors(self):
        provider = HashEmbeddingProvider(dim=128)
        v1 = provider.encode("text one")
        v2 = provider.encode("text two")
        assert v1 != v2

    def test_encode_batch(self):
        provider = HashEmbeddingProvider(dim=64)
        texts = ["a", "b", "c"]
        vecs = provider.encode_batch(texts)
        assert len(vecs) == 3
        assert all(len(v) == 64 for v in vecs)

    def test_dim_property(self):
        provider = HashEmbeddingProvider(dim=256)
        assert provider.dim == 256

    def test_name(self):
        provider = HashEmbeddingProvider()
        assert "hash" in provider.name.lower()


class TestCachedEmbeddingProvider:
    """Cached provider 测试"""

    def test_cache_hit(self):
        inner = HashEmbeddingProvider(dim=64)
        cached = CachedEmbeddingProvider(inner, max_size=100, ttl=3600)
        v1 = cached.encode("cache test")
        v2 = cached.encode("cache test")
        assert v1 == v2  # same result

    def test_cache_dim_delegation(self):
        inner = HashEmbeddingProvider(dim=128)
        cached = CachedEmbeddingProvider(inner)
        assert cached.dim == 128

    def test_cache_name_delegation(self):
        inner = HashEmbeddingProvider()
        cached = CachedEmbeddingProvider(inner)
        assert cached.name == inner.name

    def test_cache_batch(self):
        inner = HashEmbeddingProvider(dim=32)
        cached = CachedEmbeddingProvider(inner)
        texts = ["x", "y", "z"]
        vecs = cached.encode_batch(texts)
        assert len(vecs) == 3
        assert all(len(v) == 32 for v in vecs)


class TestCosineSimilarity:
    """余弦相似度测试"""

    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b) - 0.0) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_similar_vectors(self):
        a = [1.0, 1.0, 1.0]
        b = [1.1, 0.9, 1.0]
        sim = cosine_similarity(a, b)
        assert sim > 0.9  # very similar
