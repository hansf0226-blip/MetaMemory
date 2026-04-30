"""
Embedding Provider - 语义嵌入抽象层

支持多后端切换:
- openai: OpenAI/DeepSeek 兼容 API (text-embedding-3-small 等)
- sentence_transformer: 本地模型 (all-MiniLM-L6-v2 等)
- hash: 确定性哈希 fallback (无语义，仅兜底)

架构: Provider → 生成嵌入 → 缓存 → 返回 768-dim 向量
"""
import hashlib
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ============================================================
# 配置
# ============================================================

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 / BGE-small 默认维度
CACHE_MAX_SIZE = 5000
CACHE_TTL = 3600  # 1 小时

# ============================================================
# 抽象基类
# ============================================================


class EmbeddingProvider(ABC):
    """嵌入提供者抽象基类"""

    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """编码单条文本为嵌入向量"""

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """批量编码"""

    @property
    @abstractmethod
    def dim(self) -> int:
        """嵌入维度"""

    @property
    @abstractmethod
    def name(self) -> str:
        """提供者名称"""


# ============================================================
# 余弦相似度
# ============================================================


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算余弦相似度"""
    a_np = np.array(a)
    b_np = np.array(b)
    dot = np.dot(a_np, b_np)
    norm = np.linalg.norm(a_np) * np.linalg.norm(b_np)
    if norm == 0:
        return 0.0
    return float(dot / norm)


# ============================================================
# Hash Fallback Provider
# ============================================================


class HashEmbeddingProvider(EmbeddingProvider):
    """基于 SHA256 哈希的确定性嵌入（无语义，仅兜底）"""

    def __init__(self, dim: int = EMBEDDING_DIM):
        self._dim = dim

    def encode(self, text: str) -> List[float]:
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        for i in range(self._dim):
            b = hash_bytes[i % len(hash_bytes)]
            vec.append((b - 128) / 128.0)
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.encode(t) for t in texts]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "hash(fallback)"


# ============================================================
# OpenAI 兼容 API Provider
# ============================================================


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """使用 OpenAI 兼容 API 生成语义嵌入"""

    # text-embedding-3-small 支持 512/1536, ada-002 固定 1536
    VALID_DIMS = {512, 1536}

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
    ):
        import os

        self._model = model
        self._dim = dim if dim in self.VALID_DIMS else 1536

        # 延迟导入避免强依赖
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai 库未安装，pip install openai")

        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "sk-"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def encode(self, text: str) -> List[float]:
        try:
            resp = self._client.embeddings.create(
                model=self._model,
                input=text[:8191],
                dimensions=self._dim,
            )
            vec = list(resp.data[0].embedding)
            if len(vec) != self._dim:
                self._dim = len(vec)
            return vec
        except Exception as e:
            print(f"  ⚠️ OpenAI encode failed: {e}")
            return HashEmbeddingProvider(self._dim).encode(text)

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            resp = self._client.embeddings.create(
                model=self._model,
                input=[t[:8191] for t in texts],
                dimensions=self._dim,
            )
            result = [list(d.embedding) for d in resp.data]
            if result and len(result[0]) != self._dim:
                self._dim = len(result[0])
            return result
        except Exception as e:
            print(f"  ⚠️ OpenAI encode_batch failed: {e}")
            hp = HashEmbeddingProvider(self._dim)
            return [hp.encode(t) for t in texts]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"openai({self._model})"


# ============================================================
# Sentence Transformer Provider (本地模型)
# ============================================================


class STEmbeddingProvider(EmbeddingProvider):
    """使用本地 sentence-transformers / transformers 模型"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        import torch
        self._model_name = model_name

        # Try sentence-transformers first (simpler API)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, local_files_only=True)
            try:
                self._dim = self._model.get_embedding_dimension()
            except AttributeError:
                self._dim = self._model.get_sentence_embedding_dimension()
            self._use_st = True
            return
        except Exception:
            pass

        # Fallback: transformers + mean pooling (works with PyTorch 2.2)
        from transformers import AutoTokenizer, AutoModel
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self._hf_model = AutoModel.from_pretrained(model_name, local_files_only=True)
        self._dim = self._hf_model.config.hidden_size
        self._use_st = False

    def _mean_pooling(self, model_output, attention_mask):
        import torch
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, text: str) -> List[float]:
        if self._use_st:
            return self._model.encode(text, normalize_embeddings=True).tolist()

        import torch
        encoded = self._tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output = self._hf_model(**encoded)
        embedding = self._mean_pooling(output, encoded['attention_mask'])
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding[0].tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        if self._use_st:
            return self._model.encode(texts, normalize_embeddings=True).tolist()

        import torch
        encoded = self._tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output = self._hf_model(**encoded)
        embedding = self._mean_pooling(output, encoded['attention_mask'])
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.tolist()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"st({self._model_name})"


# ============================================================
# 缓存层
# ============================================================


class CachedEmbeddingProvider(EmbeddingProvider):
    """带缓存的嵌入提供者包装器"""

    def __init__(self, provider: EmbeddingProvider, max_size: int = CACHE_MAX_SIZE, ttl: int = CACHE_TTL):
        self._provider = provider
        self._cache: Dict[str, tuple] = {}  # key -> (vector, expiry)
        self._max_size = max_size
        self._ttl = ttl
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def encode(self, text: str) -> List[float]:
        key = self._make_key(text)
        with self._lock:
            if key in self._cache:
                vec, expiry = self._cache[key]
                if time.time() < expiry:
                    self._hits += 1
                    return vec
                del self._cache[key]

        self._misses += 1
        vec = self._provider.encode(text)
        with self._lock:
            if len(self._cache) >= self._max_size:
                # 淘汰最旧的 20%
                to_remove = sorted(self._cache.keys())[: len(self._cache) // 5]
                for k in to_remove:
                    del self._cache[k]
            self._cache[key] = (vec, time.time() + self._ttl)
        return vec

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        # 批量中不缓存的直接透传，缓存命中则用缓存
        results = []
        uncached_texts = []
        uncached_indices = []
        for i, text in enumerate(texts):
            key = self._make_key(text)
            with self._lock:
                if key in self._cache:
                    vec, expiry = self._cache[key]
                    if time.time() < expiry:
                        self._hits += 1
                        results.append((i, vec))
                        continue
            uncached_texts.append(text)
            uncached_indices.append(i)

        if uncached_texts:
            self._misses += len(uncached_texts)
            new_vecs = self._provider.encode_batch(uncached_texts)
            now = time.time()
            with self._lock:
                for text, vec in zip(uncached_texts, new_vecs):
                    key = self._make_key(text)
                    self._cache[key] = (vec, now + self._ttl)
                results.extend(zip(uncached_indices, new_vecs))

        # 按原始顺序排列
        results.sort(key=lambda x: x[0])
        return [v for _, v in results]

    @property
    def dim(self) -> int:
        return self._provider.dim

    @property
    def name(self) -> str:
        return f"cached({self._provider.name})"

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "cache_size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
        }


# ============================================================
# 工厂函数 - 自动选择最佳提供者
# ============================================================


def create_embedding_provider(
    prefer: str = "auto",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> EmbeddingProvider:
    """
    创建 embedding provider。

    prefer:
        - "auto": 自动选择 siliconflow > openai > sentence_transformer > hash
        - "siliconflow": 硅基流动 BGE 模型（免费，推荐）
        - "openai": 强制 OpenAI 兼容 API
        - "sentence_transformer": 强制本地模型
        - "hash": 强制 hash fallback
    """
    provider: EmbeddingProvider

    if prefer == "hash":
        provider = HashEmbeddingProvider()
    elif prefer == "sentence_transformer":
        try:
            provider = STEmbeddingProvider()
        except ImportError:
            provider = HashEmbeddingProvider()
    elif prefer == "siliconflow":
        import os
        from core.config import get_config
        sf_key = api_key or os.environ.get("SILICONFLOW_API_KEY") or get_config("embedding.siliconflow.api_key")
        sf_url = base_url or get_config("embedding.siliconflow.base_url", "https://api.siliconflow.cn/v1")
        sf_model = get_config("embedding.siliconflow.model", "BAAI/bge-large-zh-v1.5")
        sf_dim = int(get_config("embedding.siliconflow.dim", 1024))
        try:
            provider = OpenAIEmbeddingProvider(api_key=sf_key, base_url=sf_url, model=sf_model, dim=sf_dim)
        except ImportError:
            provider = _try_local_or_hash()
    elif prefer == "openai":
        try:
            provider = OpenAIEmbeddingProvider(api_key=api_key, base_url=base_url)
        except ImportError:
            provider = HashEmbeddingProvider()
    else:  # auto
        # 优先 SiliconFlow（免费 BGE 模型）
        import os
        from core.config import get_config
        sf_key = os.environ.get("SILICONFLOW_API_KEY") or get_config("embedding.siliconflow.api_key")
        if sf_key and sf_key.startswith("sk-"):
            try:
                sf_url = get_config("embedding.siliconflow.base_url", "https://api.siliconflow.cn/v1")
                sf_model = get_config("embedding.siliconflow.model", "BAAI/bge-large-zh-v1.5")
                sf_dim = int(get_config("embedding.siliconflow.dim", 1024))
                provider = OpenAIEmbeddingProvider(api_key=sf_key, base_url=sf_url, model=sf_model, dim=sf_dim)
            except Exception:
                provider = _try_local_or_hash()
        elif api_key or os.environ.get("OPENAI_API_KEY"):
            try:
                provider = OpenAIEmbeddingProvider(api_key=api_key, base_url=base_url)
            except ImportError:
                provider = _try_local_or_hash()
        else:
            provider = _try_local_or_hash()

    return CachedEmbeddingProvider(provider)


def _try_local_or_hash() -> EmbeddingProvider:
    try:
        return STEmbeddingProvider()
    except Exception:
        return HashEmbeddingProvider()


# ============================================================
# 全局单例
# ============================================================

_global_provider: Optional[EmbeddingProvider] = None
_lock = threading.RLock()


def get_embedding_provider() -> EmbeddingProvider:
    """获取全局 embedding 提供者"""
    global _global_provider
    with _lock:
        if _global_provider is None:
            _global_provider = create_embedding_provider()
        return _global_provider


def set_embedding_provider(provider: EmbeddingProvider):
    """设置全局 embedding 提供者"""
    global _global_provider
    with _lock:
        _global_provider = provider
