#!/usr/bin/env python3
"""
🔢 向量数据库集成 - 生产级
支持 FAISS (本地) 和 Milvus (分布式)
"""

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("⚠️ FAISS 库未安装")

try:
    from pymilvus import (
        connections,
        utility,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
    )

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logger.warning("⚠️ Milvus 库未安装")


@dataclass
class VectorSearchResult:
    """向量搜索结果"""

    id: str
    score: float
    distance: float
    metadata: Dict[str, Any]


@dataclass
class VectorDBConfig:
    """向量数据库配置"""

    backend: str = "faiss"  # faiss, milvus
    dimension: int = 768
    index_type: str = "HNSW"

    # FAISS 配置
    faiss_index_path: str = "data/vectors.index"
    faiss_nlist: int = 100

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "memories"
    milvus_metric_type: str = "COSINE"


class VectorDatabaseBackend(ABC):
    """向量数据库后端抽象基类"""

    @abstractmethod
    def insert(self, vector_id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """插入向量"""
        pass

    @abstractmethod
    def insert_batch(self, vectors: List[tuple[str, np.ndarray, Optional[Dict]]]) -> int:
        """批量插入向量"""
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 10, filter_expr: Optional[str] = None) -> List[VectorSearchResult]:
        """搜索相似向量"""
        pass

    @abstractmethod
    def delete(self, vector_id: str) -> bool:
        """删除向量"""
        pass

    @abstractmethod
    def get(self, vector_id: str) -> Optional[tuple[np.ndarray, Dict]]:
        """获取向量"""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        pass


class FAISSVectorBackend(VectorDatabaseBackend):
    """FAISS 本地向量数据库"""

    def __init__(self, config: VectorDBConfig):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS 库未安装，请安装: pip install faiss-cpu")

        self.config = config
        self._index: Optional[Any] = None
        self._id_map: Dict[int, str] = {}
        self._id_reverse: Dict[str, int] = {}
        self._metadata: Dict[str, Dict] = {}
        self._next_idx: int = 0
        self._lock = threading.RLock()
        self._stats = {"inserts": 0, "searches": 0, "deletes": 0}

        Path(config.faiss_index_path).parent.mkdir(exist_ok=True)

        self._build_index()
        logger.info(f"✅ FAISS 向量数据库初始化完成: {config.index_type}")

    def _build_index(self):
        """构建 FAISS 索引"""
        with self._lock:
            dim = self.config.dimension

            if self.config.index_type == "IVFFlat":
                quantizer = faiss.IndexFlatIP(dim)
                self._index = faiss.IndexIVFFlat(quantizer, dim, self.config.faiss_nlist)
            elif self.config.index_type == "HNSW":
                self._index = faiss.IndexHNSWFlat(dim, 32)
                self._index.hnsw.efConstruction = 200
                self._index.hnsw.efSearch = 128
            else:  # Flat
                self._index = faiss.IndexFlatIP(dim)

            # 使用 ID 映射
            self._index = faiss.IndexIDMap2(self._index)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 归一化"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-10)

    def insert(self, vector_id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        return self.insert_batch([(vector_id, vector, metadata)]) == 1

    def insert_batch(self, vectors: List[tuple[str, np.ndarray, Optional[Dict]]]) -> int:
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

            if hasattr(self._index, "is_trained") and not self._index.is_trained:
                if len(vectors) >= self.config.faiss_nlist:
                    self._index.train(vec_array)

            self._index.add_with_ids(vec_array, ids_array)
            count = len(vectors)
            self._stats["inserts"] += count
            return count

    def search(self, query_vector: np.ndarray, top_k: int = 10, filter_expr: Optional[str] = None) -> List[VectorSearchResult]:
        if filter_expr:
            logger.warning("⚠️ FAISS 后端不支持过滤表达式")

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

            self._stats["searches"] += 1
            return results

    def delete(self, vector_id: str) -> bool:
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

            self._stats["deletes"] += 1
            return True

    def get(self, vector_id: str) -> Optional[tuple[np.ndarray, Dict]]:
        with self._lock:
            if vector_id not in self._id_reverse:
                return None
            return None, self._metadata.get(vector_id, {})

    def save(self, path: Optional[str] = None) -> bool:
        """保存索引到磁盘"""
        save_path = path or self.config.faiss_index_path
        with self._lock:
            try:
                faiss.write_index(self._index, save_path)

                meta_path = Path(save_path).with_suffix(".meta.npz")
                np.savez(
                    meta_path,
                    id_map=np.array(list(self._id_map.items()), dtype=object),
                    metadata=np.array(list(self._metadata.items()), dtype=object),
                    next_idx=np.array([self._next_idx]),
                )

                logger.info(f"✅ 向量索引已保存: {save_path}")
                return True
            except Exception as e:
                logger.error(f"❌ 保存向量索引失败: {e}")
                return False

    def load(self, path: Optional[str] = None) -> bool:
        """从磁盘加载索引"""
        load_path = Path(path or self.config.faiss_index_path)
        with self._lock:
            try:
                if not load_path.exists():
                    return False

                self._index = faiss.read_index(str(load_path))

                meta_path = load_path.with_suffix(".meta.npz")
                if meta_path.exists():
                    data = np.load(meta_path, allow_pickle=True)
                    self._id_map = dict(data["id_map"])
                    self._id_reverse = {v: k for k, v in self._id_map.items()}
                    self._metadata = dict(data["metadata"])
                    self._next_idx = int(data["next_idx"][0])

                logger.info(f"✅ 向量索引已加载: {load_path}, 向量数: {self._index.ntotal}")
                return True
            except Exception as e:
                logger.error(f"❌ 加载向量索引失败: {e}")
                return False

    def health_check(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": "healthy",
                "backend": "faiss",
                "total_vectors": self._index.ntotal if self._index else 0,
                "index_type": self.config.index_type,
                "dimension": self.config.dimension,
                "stats": self._stats.copy(),
            }


class MilvusVectorBackend(VectorDatabaseBackend):
    """Milvus 分布式向量数据库"""

    def __init__(self, config: VectorDBConfig):
        if not MILVUS_AVAILABLE:
            raise ImportError("Milvus 库未安装，请安装: pip install pymilvus")

        self.config = config
        self._collection: Optional[Collection] = None
        self._connected = False
        self._lock = threading.RLock()
        self._stats = {"inserts": 0, "searches": 0, "deletes": 0}

        self._connect()
        self._create_collection()
        logger.info(f"✅ Milvus 向量数据库初始化完成: {config.milvus_host}:{config.milvus_port}")

    def _connect(self):
        """连接 Milvus"""
        with self._lock:
            try:
                connections.connect(
                    alias="default",
                    host=self.config.milvus_host,
                    port=self.config.milvus_port,
                )
                self._connected = True
            except Exception as e:
                logger.error(f"❌ Milvus 连接失败: {e}")
                raise

    def _create_collection(self):
        """创建集合"""
        with self._lock:
            collection_name = self.config.milvus_collection

            if utility.has_collection(collection_name):
                self._collection = Collection(collection_name)
                logger.info(f"✅ 使用已有集合: {collection_name}")
                return

            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.config.dimension),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="importance", dtype=DataType.FLOAT),
            ]

            schema = CollectionSchema(fields=fields, description="记忆向量集合")

            self._collection = Collection(name=collection_name, schema=schema)

            # 创建索引
            index_params = {
                "metric_type": self.config.milvus_metric_type,
                "index_type": self.config.index_type,
                "params": {"M": 16, "efConstruction": 200},
            }
            self._collection.create_index(field_name="vector", index_params=index_params)
            self._collection.create_index(field_name="user_id", index_name="idx_user_id")

            # 加载集合
            self._collection.load()

            logger.info(f"✅ Milvus 集合创建完成: {collection_name}")

    def insert(self, vector_id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        return self.insert_batch([(vector_id, vector, metadata)]) == 1

    def insert_batch(self, vectors: List[tuple[str, np.ndarray, Optional[Dict]]]) -> int:
        if not vectors:
            return 0

        with self._lock:
            try:
                data = {
                    "id": [],
                    "vector": [],
                    "user_id": [],
                    "memory_type": [],
                    "importance": [],
                }

                for vec_id, vec, meta in vectors:
                    data["id"].append(vec_id)
                    data["vector"].append(vec.astype(np.float32).tolist())

                    meta = meta or {}
                    data["user_id"].append(meta.get("user_id", ""))
                    data["memory_type"].append(meta.get("memory_type", "memory"))
                    data["importance"].append(meta.get("importance", 0.5))

                self._collection.insert(data)
                self._collection.flush()

                count = len(vectors)
                self._stats["inserts"] += count
                return count

            except Exception as e:
                logger.error(f"❌ Milvus 插入失败: {e}")
                return 0

    def search(self, query_vector: np.ndarray, top_k: int = 10, filter_expr: Optional[str] = None) -> List[VectorSearchResult]:
        with self._lock:
            try:
                search_params = {
                    "metric_type": self.config.milvus_metric_type,
                    "params": {"ef": 128},
                }

                results = self._collection.search(
                    data=[query_vector.astype(np.float32).tolist()],
                    anns_field="vector",
                    param=search_params,
                    limit=top_k,
                    expr=filter_expr,
                    output_fields=["id", "user_id", "memory_type", "importance"],
                )

                output = []
                for hits in results:
                    for hit in hits:
                        normalized_score = (1 + hit.score) / 2
                        output.append(
                            VectorSearchResult(
                                id=hit.id,
                                score=float(normalized_score),
                                distance=float(hit.distance),
                                metadata={
                                    "user_id": hit.entity.get("user_id", ""),
                                    "memory_type": hit.entity.get("memory_type", ""),
                                    "importance": hit.entity.get("importance", 0.5),
                                },
                            )
                        )

                self._stats["searches"] += 1
                return output

            except Exception as e:
                logger.error(f"❌ Milvus 搜索失败: {e}")
                return []

    def delete(self, vector_id: str) -> bool:
        with self._lock:
            try:
                expr = f'id == "{vector_id}"'
                self._collection.delete(expr)
                self._stats["deletes"] += 1
                return True
            except Exception as e:
                logger.error(f"❌ Milvus 删除失败: {e}")
                return False

    def get(self, vector_id: str) -> Optional[tuple[np.ndarray, Dict]]:
        # Milvus 不支持直接获取向量
        return None, {}

    def health_check(self) -> Dict[str, Any]:
        with self._lock:
            try:
                utility.list_collections()
                return {
                    "status": "healthy",
                    "backend": "milvus",
                    "host": self.config.milvus_host,
                    "collection": self.config.milvus_collection,
                    "connected": self._connected,
                    "stats": self._stats.copy(),
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e),
                }


class VectorDatabase:
    """向量数据库管理器"""

    _backends = {
        "faiss": FAISSVectorBackend,
        "milvus": MilvusVectorBackend,
    }

    def __init__(self, config: Optional[VectorDBConfig] = None):
        self.config = config or VectorDBConfig()
        self._backend: Optional[VectorDatabaseBackend] = None
        self._lock = threading.RLock()
        self._initialize()

    def _initialize(self):
        """初始化后端"""
        with self._lock:
            backend_class = self._backends.get(self.config.backend)
            if not backend_class:
                raise ValueError(f"不支持的向量数据库后端: {self.config.backend}")

            self._backend = backend_class(self.config)

    @property
    def backend(self) -> VectorDatabaseBackend:
        return self._backend

    def insert(self, vector_id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        return self._backend.insert(vector_id, vector, metadata)

    def insert_batch(self, vectors: List[tuple[str, np.ndarray, Optional[Dict]]]) -> int:
        return self._backend.insert_batch(vectors)

    def search(self, query_vector: np.ndarray, top_k: int = 10, filter_expr: Optional[str] = None) -> List[VectorSearchResult]:
        return self._backend.search(query_vector, top_k, filter_expr)

    def delete(self, vector_id: str) -> bool:
        return self._backend.delete(vector_id)

    def health_check(self) -> Dict[str, Any]:
        return self._backend.health_check()

    def save(self, path: Optional[str] = None) -> bool:
        """保存索引（仅 FAISS 支持）"""
        if isinstance(self._backend, FAISSVectorBackend):
            return self._backend.save(path)
        return False

    def load(self, path: Optional[str] = None) -> bool:
        """加载索引（仅 FAISS 支持）"""
        if isinstance(self._backend, FAISSVectorBackend):
            return self._backend.load(path)
        return False
