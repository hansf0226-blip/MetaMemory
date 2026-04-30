#!/usr/bin/env python3
"""
🚀 向量检索引擎 - 优化版
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("⚠️ FAISS 库未安装")


@dataclass
class VectorSearchResult:
    """向量搜索结果"""

    id: str
    score: float
    distance: float
    metadata: Dict[str, Any]


class FAISSVectorIndex:
    """FAISS 向量索引"""

    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "ivfflat",
        nlist: int = 100,
        m: int = 16,
    ):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS 库未安装")

        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.m = m

        self._index: Optional[Any] = None
        self._id_map: Dict[int, str] = {}
        self._id_reverse: Dict[str, int] = {}
        self._metadata: Dict[str, Dict] = {}
        self._next_idx: int = 0
        self._lock = threading.RLock()

        self._stats = {
            "total_adds": 0,
            "total_searches": 0,
            "total_deletes": 0,
        }

        self._build_index()
        logger.info(f"✅ FAISS 向量索引已初始化: {index_type}, 维度: {dimension}")

    def _build_index(self):
        """构建 FAISS 索引"""
        with self._lock:
            if self.index_type == "flat":
                base_index = faiss.IndexFlatIP(self.dimension)
                self._index = faiss.IndexIDMap2(base_index)

            elif self.index_type == "ivfflat":
                quantizer = faiss.IndexFlatIP(self.dimension)
                base_index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
                self._index = faiss.IndexIDMap2(base_index)

            elif self.index_type == "hnsw":
                base_index = faiss.IndexHNSWFlat(self.dimension, self.m)
                base_index.hnsw.efConstruction = 200
                base_index.hnsw.efSearch = 128
                self._index = faiss.IndexIDMap2(base_index)

            else:
                raise ValueError(f"不支持的索引类型: {self.index_type}")

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 归一化向量"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-10)

    def add(self, vector_id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """添加向量"""
        return self.add_batch([(vector_id, vector, metadata)]) == 1

    def add_batch(self, vectors: List[Tuple[str, np.ndarray, Optional[Dict]]]) -> int:
        """批量添加向量"""
        if not vectors:
            return 0

        with self._lock:
            vec_array = np.array([v for _, v, _ in vectors], dtype=np.float32)
            vec_array = self._normalize(vec_array)

            ids = []
            for vec_id, _, meta in vectors:
                if vec_id in self._id_reverse:
                    old_idx = self._id_reverse[vec_id]
                    del self._id_map[old_idx]
                    self._id_reverse.pop(vec_id, None)
                    self._metadata.pop(vec_id, None)

                faiss_id = self._next_idx
                self._next_idx += 1
                ids.append(faiss_id)
                self._id_map[faiss_id] = vec_id
                self._id_reverse[vec_id] = faiss_id
                self._metadata[vec_id] = meta or {}

            ids_array = np.array(ids, dtype=np.int64)

            if self.index_type.startswith("ivf") and not self._index.index.is_trained:
                if len(vectors) >= self.nlist:
                    self._index.index.train(vec_array)
                    logger.info(f"✅ IVF 索引训练完成，样本数: {len(vectors)}")

            self._index.add_with_ids(vec_array, ids_array)
            count = len(vectors)
            self._stats["total_adds"] += count
            return count

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[VectorSearchResult]:
        """搜索相似向量"""
        start_time = time.time()

        with self._lock:
            if self._index.ntotal == 0:
                return []

            query = query_vector.astype(np.float32).reshape(1, -1)
            query = self._normalize(query)

            scores, indices = self._index.search(query, min(top_k, self._index.ntotal))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx not in self._id_map:
                    continue

                vec_id = self._id_map[idx]
                normalized_score = (score + 1) / 2

                results.append(
                    VectorSearchResult(
                        id=vec_id,
                        score=float(normalized_score),
                        distance=float(1 - normalized_score),
                        metadata=self._metadata.get(vec_id, {}),
                    )
                )

            self._stats["total_searches"] += 1
            return results

    def delete(self, vector_id: str) -> bool:
        """删除向量"""
        with self._lock:
            if vector_id not in self._id_reverse:
                return False

            faiss_id = self._id_reverse[vector_id]
            if hasattr(self._index, "remove_ids"):
                ids_array = np.array([faiss_id], dtype=np.int64)
                self._index.remove_ids(ids_array)

            del self._id_map[faiss_id]
            del self._id_reverse[vector_id]
            self._metadata.pop(vector_id, None)

            self._stats["total_deletes"] += 1
            return True

    def save(self, path: str) -> bool:
        """保存索引到磁盘"""
        with self._lock:
            try:
                save_path = Path(path)
                save_path.parent.mkdir(exist_ok=True)
                faiss.write_index(self._index, str(save_path))

                meta_path = save_path.with_suffix(".meta.npz")
                np.savez(
                    meta_path,
                    id_map=np.array(list(self._id_map.items()), dtype=object),
                    metadata=np.array(list(self._metadata.items()), dtype=object),
                    next_idx=np.array([self._next_idx]),
                )

                logger.info(f"✅ 索引已保存: {path}")
                return True
            except Exception as e:
                logger.error(f"❌ 索引保存失败: {e}")
                return False

    def load(self, path: str) -> bool:
        """从磁盘加载索引"""
        with self._lock:
            try:
                save_path = Path(path)
                if not save_path.exists():
                    return False

                self._index = faiss.read_index(str(save_path))

                meta_path = save_path.with_suffix(".meta.npz")
                if meta_path.exists():
                    data = np.load(meta_path, allow_pickle=True)
                    self._id_map = dict(data["id_map"])
                    self._id_reverse = {v: k for k, v in self._id_map.items()}
                    self._metadata = dict(data["metadata"])
                    self._next_idx = int(data["next_idx"][0])

                logger.info(f"✅ 索引已加载: {path}, 向量数: {self._index.ntotal}")
                return True
            except Exception as e:
                logger.error(f"❌ 索引加载失败: {e}")
                return False

    def stats(self) -> Dict[str, Any]:
        """获取索引统计"""
        with self._lock:
            return {
                **self._stats,
                "index_type": self.index_type,
                "dimension": self.dimension,
                "total_vectors": self._index.ntotal if self._index else 0,
                "is_trained": self._index.is_trained if self._index else False,
            }
