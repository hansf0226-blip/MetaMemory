"""
嵌入生成器 - 使用 LLM 生成文本嵌入向量
支持多种嵌入策略
"""

import hashlib
import json
from typing import Dict, List

try:
    from api.llm_manager import LLMManager
except ImportError:
    LLMManager = None


class EmbeddingGenerator:
    """嵌入生成器 - 使用 LLM 生成向量嵌入"""

    def __init__(self, llm: LLMManager = None, embedding_dim: int = 768):
        self.llm = llm or LLMManager()
        self.embedding_dim = embedding_dim

        # 嵌入缓存
        self.cache: Dict[str, List[float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        print("✅ 嵌入生成器已初始化")
        print(f"   向量维度：{embedding_dim}")
        print(f"   LLM 提供商：{self.llm.active_provider if self.llm else 'None'}")

    def generate_embedding(self, text: str, method: str = "llm_projection", use_cache: bool = True) -> List[float]:
        """
        生成文本嵌入

        Args:
            text: 输入文本
            method: 嵌入生成方法
                - "llm_projection": 使用 LLM 生成投影向量（推荐）
                - "hash_embedding": 基于哈希的快速嵌入（降级方案）
                - "keyword_vector": 关键词向量（简单方案）
            use_cache: 是否使用缓存

        Returns:
            嵌入向量（固定维度）
        """
        # 检查缓存
        cache_key = f"{method}:{text}"
        if use_cache and cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        self.cache_misses += 1

        # 选择生成方法
        if method == "llm_projection":
            embedding = self._llm_projection(text)
        elif method == "hash_embedding":
            embedding = self._hash_embedding(text)
        elif method == "keyword_vector":
            embedding = self._keyword_vector(text)
        else:
            embedding = self._hash_embedding(text)  # 降级

        # 缓存结果
        if use_cache:
            self.cache[cache_key] = embedding

        return embedding

    def _llm_projection(self, text: str) -> List[float]:
        """使用 LLM 生成投影向量"""
        if not self.llm or not self.llm.providers:
            return self._hash_embedding(text)

        # 构建提示词，让 LLM 生成结构化向量
        prompt = """请将以下文本转换为一个数学向量表示。

文本：{text[:500]}

要求：
1. 生成一个长度为 {self.embedding_dim} 的浮点数数组
2. 数值范围在 -1 到 1 之间
3. 向量应该捕捉文本的语义特征
4. 直接返回 JSON 数组格式，不要其他解释

返回格式：[0.1, -0.5, 0.3, ...]"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat(messages, temperature=0.1, max_tokens=1000)

            if response:
                # 解析 JSON 数组
                response = response.strip()
                if response.startswith("[") and response.endswith("]"):
                    embedding = json.loads(response)
                    if len(embedding) == self.embedding_dim:
                        return embedding

                # 尝试提取数组
                import re

                match = re.search(r"\[[\d\.\-\s,]+\]", response)
                if match:
                    embedding = json.loads(match.group())
                    # 填充或截断到目标维度
                    return self._normalize_dimension(embedding)

            # 降级方案
            return self._hash_embedding(text)

        except Exception as e:
            print(f"⚠️ LLM 嵌入生成失败：{e}")
            return self._hash_embedding(text)

    def _hash_embedding(self, text: str) -> List[float]:
        """基于哈希的快速嵌入（降级方案）"""
        # 使用文本哈希生成确定性向量
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # 扩展哈希到目标维度
        embedding = []
        for i in range(self.embedding_dim):
            byte_idx = i % len(hash_bytes)
            # 将字节转换为 -1 到 1 的浮点数
            value = (hash_bytes[byte_idx] - 128) / 128.0
            embedding.append(value)

        return embedding

    def _keyword_vector(self, text: str) -> List[float]:
        """关键词向量（简单方案）"""
        # 提取关键词
        words = text.lower().split()
        word_freq = {}
        for word in words:
            word = "".join(c for c in word if c.isalnum())
            if len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1

        # 创建词频向量
        embedding = [0.0] * self.embedding_dim

        # 将词频映射到向量空间
        for i, (word, freq) in enumerate(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:100]):
            idx = i % self.embedding_dim
            embedding[idx] = min(1.0, freq / 10.0)

        # 添加一些噪声使向量更丰富
        import random

        random.seed(hash(text) % 2**32)
        for i in range(self.embedding_dim):
            if embedding[i] == 0:
                embedding[i] = random.uniform(-0.1, 0.1)

        return embedding

    def _normalize_dimension(self, embedding: List[float]) -> List[float]:
        """标准化向量维度"""
        if len(embedding) >= self.embedding_dim:
            return embedding[: self.embedding_dim]
        else:
            # 填充零
            return embedding + [0.0] * (self.embedding_dim - len(embedding))

    def generate_embeddings_batch(
        self, texts: List[str], method: str = "llm_projection", use_cache: bool = True
    ) -> List[List[float]]:
        """批量生成嵌入"""
        embeddings = []
        for i, text in enumerate(texts):
            embedding = self.generate_embedding(text, method, use_cache)
            embeddings.append(embedding)

            if (i + 1) % 10 == 0:
                print(f"   已生成 {i + 1}/{len(texts)} 个嵌入")

        return embeddings

    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0

        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
        }

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        print("🗑️ 嵌入缓存已清空")

    def print_status(self):
        """打印状态"""
        stats = self.get_cache_stats()

        print("\n【嵌入生成器状态】")
        print(f"缓存大小：{stats['cache_size']}")
        print(f"缓存命中：{stats['cache_hits']}")
        print(f"缓存未命中：{stats['cache_misses']}")
        print(f"命中率：{stats['hit_rate']*100:.1f}%")
