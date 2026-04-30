"""
向量数据库模块 - 支持语义检索的向量存储
使用 SQLite 存储向量嵌入，实现余弦相似度搜索

功能：
- 支持向量的存储、检索和管理
- 支持知识库的存储、检索和管理
- 支持相似度搜索
- 线程安全的数据库操作
- 性能优化
- 安全验证
"""

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from core.utils import (
        DatabaseError,
        ValidationError,
        ensure_directory,
        generate_id,
        get_logger,
        validate_dict,
        validate_float,
        validate_integer,
        validate_list,
        validate_string,
    )
except ImportError:
    import logging
    import uuid

    def get_logger(name: str):
        return logging.getLogger(name)

    class DatabaseError(Exception):
        pass

    class ValidationError(ValueError):
        pass

    def ensure_directory(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def generate_id():
        return str(uuid.uuid4())[:12]

    def validate_dict(data, name="data"):
        if not isinstance(data, dict):
            raise ValidationError(f"{name} must be a dict")

    def validate_float(value, name="value", min_value=None, max_value=None):
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be a number")
        if min_value is not None and value < min_value:
            raise ValidationError(f"{name} must be >= {min_value}")
        if max_value is not None and value > max_value:
            raise ValidationError(f"{name} must be <= {max_value}")

    def validate_integer(value, name="value", min_value=None, max_value=None):
        if not isinstance(value, int):
            raise ValidationError(f"{name} must be an integer")
        if min_value is not None and value < min_value:
            raise ValidationError(f"{name} must be >= {min_value}")
        if max_value is not None and value > max_value:
            raise ValidationError(f"{name} must be <= {max_value}")

    def validate_list(data, name="data", min_length=0):
        if not isinstance(data, list):
            raise ValidationError(f"{name} must be a list")
        if len(data) < min_length:
            raise ValidationError(f"{name} must have at least {min_length} items")

    def validate_string(value, name="value", min_length=0):
        if not isinstance(value, str):
            raise ValidationError(f"{name} must be a string")
        if len(value) < min_length:
            raise ValidationError(f"{name} must be at least {min_length} chars")

# 尝试导入 FAISS 库
try:
    import faiss
except ImportError:
    faiss = None
    get_logger(__name__).warning("FAISS 库未安装，将使用默认的向量检索方法")


# 异常定义
class VectorDatabaseError(DatabaseError):
    """向量数据库异常"""


class VectorDatabase:
    """向量数据库 - 支持语义检索"""

    def __init__(self, db_path: str = None, embedding_dim: int = 768, index_type: str = "ivfflat"):
        """
        初始化向量数据库

        Args:
            db_path: 数据库文件路径，默认在 data/vector_store.db
            embedding_dim: 嵌入维度，默认 768
            index_type: FAISS 索引类型，可选值：
                - "flat": 简单暴力搜索，适合小数据集
                - "ivfflat": 倒排索引，适合中等数据集
                - "ivfpq": 乘积量化，适合大数据集（内存效率高）
                - "hnsw": 分层导航小世界图，适合高维向量
        """
        self.logger = get_logger(__name__)

        if db_path is None:
            # 默认路径：项目根目录/data/vector_store.db
            project_root = Path(__file__).parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "vector_store.db")

        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self._local = threading.local()  # 线程局部存储
        self._faiss_index = None
        self._vector_ids = []  # 存储向量 ID，与 FAISS 索引的索引对应
        self._lock = threading.RLock()  # 用于 FAISS 索引的线程安全

        # 确保目录存在
        ensure_directory(Path(db_path).parent)

        # 初始化数据库
        self._init_db()

        # 初始化 FAISS 索引
        self._init_faiss_index()

        self.logger.info(f"✅ 向量数据库已初始化：{db_path}")
        self.logger.info(f"   向量维度：{embedding_dim}")
        self.logger.info(f"   FAISS 索引类型：{index_type}")
        if faiss:
            self.logger.info(f"   FAISS 索引已初始化：{type(self._faiss_index).__name__}")

    def initialize(
        self, collection_name: str = "test", embedding_dim: int = 768, storage_dir: str = None, index_type: str = None
    ):
        """
        初始化向量数据库

        Args:
            collection_name: 集合名称
            embedding_dim: 嵌入维度
            storage_dir: 存储目录
            index_type: FAISS 索引类型，可选值：
                - "flat": 简单暴力搜索，适合小数据集
                - "ivfflat": 倒排索引，适合中等数据集
                - "ivfpq": 乘积量化，适合大数据集（内存效率高）
                - "hnsw": 分层导航小世界图，适合高维向量
        """
        if storage_dir:
            db_path = str(Path(storage_dir) / f"{collection_name}_vector_store.db")
            self.db_path = db_path
            # 确保目录存在
            ensure_directory(Path(db_path).parent)

        if embedding_dim:
            self.embedding_dim = embedding_dim

        if index_type:
            self.index_type = index_type

        # 重新初始化数据库
        self._init_db()

        # 重新初始化 FAISS 索引
        self._init_faiss_index()

        self.logger.info(f"✅ 向量数据库已初始化：{self.db_path}")
        self.logger.info(f"   向量维度：{self.embedding_dim}")
        self.logger.info(f"   FAISS 索引类型：{self.index_type}")
        if faiss:
            self.logger.info(f"   FAISS 索引已初始化：{type(self._faiss_index).__name__}")

    def _get_connection(self):
        """
        获取数据库连接

        Returns:
            sqlite3.Connection: 数据库连接
        """
        if not hasattr(self._local, "conn"):
            try:
                self._local.conn = sqlite3.connect(self.db_path)
                self.logger.debug("创建向量数据库连接")
            except Exception as e:
                self.logger.error(f"创建向量数据库连接失败: {e}")
                raise VectorDatabaseError(f"创建向量数据库连接失败: {e}")
        return self._local.conn

    def _init_db(self):
        """初始化数据库表结构"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # 创建向量表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    tags TEXT
                )
            """)

            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags ON vectors(tags)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created ON vectors(created_at)")

            # 创建知识库表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT,
                    embedding TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    source_url TEXT,
                    importance REAL DEFAULT 0.5
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON knowledge_base(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_importance ON knowledge_base(importance)")

            conn.commit()
            self.logger.info("   📊 数据库表结构已创建")
        except Exception as e:
            self.logger.error(f"初始化数据库表结构失败: {e}")
            raise VectorDatabaseError(f"初始化数据库表结构失败: {e}")

    def _init_faiss_index(self):
        """初始化 FAISS 索引"""
        if not faiss:
            return

        try:
            with self._lock:
                # 根据索引类型创建 FAISS 索引
                if self.index_type == "flat":
                    # 简单暴力搜索，适合小数据集
                    self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                elif self.index_type == "ivfflat":
                    # 倒排索引，适合中等数据集
                    nlist = 100  # 聚类中心数量
                    self._faiss_index = faiss.IndexIVFFlat(
                        faiss.IndexFlatIP(self.embedding_dim), self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT
                    )
                elif self.index_type == "ivfpq":
                    # 乘积量化，适合大数据集（内存效率高）
                    nlist = 100  # 聚类中心数量
                    m = 8  # 子量化器数量
                    self._faiss_index = faiss.IndexIVFPQ(
                        faiss.IndexFlatIP(self.embedding_dim), self.embedding_dim, nlist, m, 8  # 每个子量化器的 bits
                    )
                elif self.index_type == "hnsw":
                    # 分层导航小世界图，适合高维向量
                    M = 16  # 每个节点的最大邻居数
                    self._faiss_index = faiss.IndexHNSWFlat(self.embedding_dim, M)
                    # 设置搜索参数
                    self._faiss_index.hnsw.efConstruction = 40
                    self._faiss_index.hnsw.efSearch = 16
                else:
                    # 默认使用 IVF 索引
                    nlist = 100
                    self._faiss_index = faiss.IndexIVFFlat(
                        faiss.IndexFlatIP(self.embedding_dim), self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT
                    )

                # 加载现有向量到 FAISS 索引
                self._load_vectors_to_faiss()

                self.logger.info(f"   📊 FAISS 索引已初始化: {type(self._faiss_index).__name__}")
        except Exception as e:
            self.logger.error(f"初始化 FAISS 索引失败: {e}")
            self._faiss_index = None

    def _load_vectors_to_faiss(self):
        """加载现有向量到 FAISS 索引"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # 获取所有向量
            cursor.execute("SELECT id, embedding FROM vectors")
            rows = cursor.fetchall()

            if not rows:
                return

            # 准备向量数据
            vectors = []
            vector_ids = []

            for row in rows:
                vector_id, embedding_str = row
                embedding = json.loads(embedding_str)
                vectors.append(embedding)
                vector_ids.append(vector_id)

            # 转换为 numpy 数组
            vectors_np = np.array(vectors, dtype=np.float32)

            # 训练 FAISS 索引
            self._faiss_index.train(vectors_np)

            # 添加向量到索引
            self._faiss_index.add(vectors_np)
            self._vector_ids = vector_ids

            self.logger.info(f"   📊 加载了 {len(vector_ids)} 个向量到 FAISS 索引")
        except Exception as e:
            self.logger.error(f"加载向量到 FAISS 索引失败: {e}")

    # ========== 向量操作 ==========

    def add_vector(self, content: str, embedding: List[float], metadata: Dict = None, tags: List[str] = None) -> str:
        """
        添加向量

        Args:
            content: 向量内容
            embedding: 向量嵌入
            metadata: 元数据
            tags: 标签列表

        Returns:
            str: 向量 ID
        """
        vector_id = generate_id(content)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO vectors
            (id, content, embedding, metadata, created_at, tags)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                vector_id,
                content,
                json.dumps(embedding),
                json.dumps(metadata or {}),
                datetime.now().isoformat(),
                json.dumps(tags or []),
            ),
        )

        conn.commit()

        # 更新 FAISS 索引
        if faiss and self._faiss_index:
            with self._lock:
                # 转换为 numpy 数组
                vector_np = np.array([embedding], dtype=np.float32)

                # 如果索引为空，需要先训练
                if self._faiss_index.ntotal == 0:
                    self._faiss_index.train(vector_np)

                # 添加到索引
                self._faiss_index.add(vector_np)
                self._vector_ids.append(vector_id)

        self.logger.debug(f"添加向量成功: ID={vector_id}")
        return vector_id

    def add_vectors_batch(self, items: List[Dict]) -> int:
        """
        批量添加向量

        Args:
            items: 向量数据列表

        Returns:
            int: 添加的向量数量
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        count = 0
        vectors = []
        vector_ids = []

        for item in items:
            vector_id = generate_id(item["content"])

            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO vectors
                    (id, content, embedding, metadata, created_at, tags)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        vector_id,
                        item["content"],
                        json.dumps(item["embedding"]),
                        json.dumps(item.get("metadata", {})),
                        datetime.now().isoformat(),
                        json.dumps(item.get("tags", [])),
                    ),
                )
                count += 1
                vectors.append(item["embedding"])
                vector_ids.append(vector_id)
            except Exception as e:
                self.logger.error(f"添加向量失败：{e}")

        conn.commit()

        # 更新 FAISS 索引
        if faiss and self._faiss_index and vectors:
            with self._lock:
                # 转换为 numpy 数组
                vectors_np = np.array(vectors, dtype=np.float32)

                # 如果索引为空，需要先训练
                if self._faiss_index.ntotal == 0:
                    self._faiss_index.train(vectors_np)

                # 添加到索引
                self._faiss_index.add(vectors_np)
                self._vector_ids.extend(vector_ids)

        self.logger.info(f"✅ 批量添加 {count} 个向量")
        return count

    def search_similar(
        self, query_embedding: List[float], top_k: int = 5, tags_filter: List[str] = None, min_similarity: float = 0.3
    ) -> List[Tuple]:
        """
        搜索相似向量（余弦相似度）

        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            tags_filter: 标签过滤
            min_similarity: 最小相似度阈值

        Returns:
            相似的向量列表（带相似度分数）
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # 转换查询向量
        query_array = np.array(query_embedding, dtype=np.float32)

        # 计算向量范数（用于余弦相似度计算）
        query_norm = np.linalg.norm(query_array)

        results = []

        # 使用 FAISS 进行高效搜索
        if faiss and self._faiss_index and self._faiss_index.ntotal > 0:
            with self._lock:
                # 设置搜索参数
                self._faiss_index.nprobe = 10  # 搜索的聚类中心数量

                # 执行搜索
                distances, indices = self._faiss_index.search(
                    np.array([query_array], dtype=np.float32), min(top_k * 2, self._faiss_index.ntotal)
                )

                # 处理结果
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx < 0 or idx >= len(self._vector_ids):
                        continue

                    vector_id = self._vector_ids[idx]

                    # 获取向量详情
                    cursor.execute(
                        "SELECT content, embedding, metadata, tags FROM vectors WHERE id = ?", (vector_id,)
                    )
                    row = cursor.fetchone()
                    if not row:
                        continue

                    content, embedding_str, metadata_str, tags_str = row

                    # 标签过滤
                    if tags_filter:
                        tags = json.loads(tags_str) if tags_str else []
                        if not any(tag in tags_filter for tag in tags):
                            continue

                    # 计算余弦相似度
                    embedding = json.loads(embedding_str)
                    embedding_array = np.array(embedding, dtype=np.float32)
                    embedding_norm = np.linalg.norm(embedding_array)

                    if query_norm == 0 or embedding_norm == 0:
                        similarity = 0.0
                    else:
                        # FAISS 返回的是内积，需要转换为余弦相似度
                        similarity = float(distance) / (query_norm * embedding_norm)

                    if similarity >= min_similarity:
                        results.append((embedding, json.loads(metadata_str) if metadata_str else {}, similarity))
        else:
            # 回退到传统方法
            # 获取所有向量（带过滤）
            if tags_filter:
                cursor.execute("SELECT id, content, embedding, metadata, tags FROM vectors WHERE tags IS NOT NULL")
            else:
                cursor.execute("SELECT id, content, embedding, metadata, tags FROM vectors")

            rows = cursor.fetchall()

            # 计算相似度
            for row in rows:
                vector_id, content, embedding_str, metadata_str, tags_str = row

                # 标签过滤
                if tags_filter:
                    tags = json.loads(tags_str) if tags_str else []
                    if not any(tag in tags_filter for tag in tags):
                        continue

                # 解析向量
                embedding = json.loads(embedding_str)

                # 计算余弦相似度
                similarity = self._cosine_similarity(query_array, np.array(embedding, dtype=np.float32))

                if similarity >= min_similarity:
                    results.append((embedding, json.loads(metadata_str) if metadata_str else {}, similarity))

        # 按相似度排序（降序：最相似的在前）
        results.sort(key=lambda x: x[2], reverse=True)

        self.logger.debug(f"搜索相似向量成功，返回 {len(results[:top_k])} 条")
        return results[:top_k]

    def search_similar_batch(
        self,
        query_embeddings: List[List[float]],
        top_k: int = 5,
        tags_filter: List[str] = None,
        min_similarity: float = 0.3,
    ) -> List[List[Tuple]]:
        """
        批量搜索相似向量（余弦相似度）

        Args:
            query_embeddings: 查询向量列表
            top_k: 返回数量
            tags_filter: 标签过滤
            min_similarity: 最小相似度阈值

        Returns:
            相似的向量列表列表（带相似度分数）
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        batch_results = []

        # 处理每个查询向量
        for i, query_embedding in enumerate(query_embeddings):
            # 转换查询向量
            query_array = np.array(query_embedding, dtype=np.float32)

            # 计算向量范数（用于余弦相似度计算）
            query_norm = np.linalg.norm(query_array)

            results = []

            # 使用 FAISS 进行高效搜索
            if faiss and self._faiss_index and self._faiss_index.ntotal > 0:
                with self._lock:
                    # 设置搜索参数
                    self._faiss_index.nprobe = 10  # 搜索的聚类中心数量

                    # 执行搜索
                    distances, indices = self._faiss_index.search(
                        np.array([query_array], dtype=np.float32), min(top_k * 2, self._faiss_index.ntotal)
                    )

                    # 处理结果
                    for j, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                        if idx < 0 or idx >= len(self._vector_ids):
                            continue

                        vector_id = self._vector_ids[idx]

                        # 获取向量详情
                        cursor.execute(
                            "SELECT content, embedding, metadata, tags FROM vectors WHERE id = ?", (vector_id,)
                        )
                        row = cursor.fetchone()
                        if not row:
                            continue

                        content, embedding_str, metadata_str, tags_str = row

                        # 标签过滤
                        if tags_filter:
                            tags = json.loads(tags_str) if tags_str else []
                            if not any(tag in tags_filter for tag in tags):
                                continue

                        # 计算余弦相似度
                        embedding = json.loads(embedding_str)
                        embedding_array = np.array(embedding, dtype=np.float32)
                        embedding_norm = np.linalg.norm(embedding_array)

                        if query_norm == 0 or embedding_norm == 0:
                            similarity = 0.0
                        else:
                            # FAISS 返回的是内积，需要转换为余弦相似度
                            similarity = float(distance) / (query_norm * embedding_norm)

                        if similarity >= min_similarity:
                            results.append(
                                (embedding, json.loads(metadata_str) if metadata_str else {}, similarity)
                            )
            else:
                # 回退到传统方法
                # 获取所有向量（带过滤）
                if tags_filter:
                    cursor.execute(
                        "SELECT id, content, embedding, metadata, tags FROM vectors WHERE tags IS NOT NULL"
                    )
                else:
                    cursor.execute("SELECT id, content, embedding, metadata, tags FROM vectors")

                rows = cursor.fetchall()

                # 计算相似度
                for row in rows:
                    vector_id, content, embedding_str, metadata_str, tags_str = row

                    # 标签过滤
                    if tags_filter:
                        tags = json.loads(tags_str) if tags_str else []
                        if not any(tag in tags_filter for tag in tags):
                            continue

                    # 解析向量
                    embedding = json.loads(embedding_str)

                    # 计算余弦相似度
                    similarity = self._cosine_similarity(query_array, np.array(embedding, dtype=np.float32))

                    if similarity >= min_similarity:
                        results.append((embedding, json.loads(metadata_str) if metadata_str else {}, similarity))

            # 按相似度排序
            results.sort(key=lambda x: x[2], reverse=False)

            batch_results.append(results[:top_k])

        self.logger.debug(f"批量搜索相似向量成功，返回 {len(batch_results)} 组结果")
        return batch_results

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度

        Args:
            vec1: 第一个向量
            vec2: 第二个向量

        Returns:
            float: 相似度
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _update_access_count(self, vector_ids: List[str]):
        """
        更新访问计数

        Args:
            vector_ids: 向量 ID 列表
        """
        if not vector_ids:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        for vector_id in vector_ids:
            cursor.execute(
                """
                UPDATE vectors
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
            """,
                (datetime.now().isoformat(), vector_id),
            )

        conn.commit()
        self.logger.debug(f"更新 {len(vector_ids)} 个向量的访问计数")

    # ========== 知识库操作 ==========

    def add_knowledge(
        self,
        title: str,
        content: str,
        embedding: List[float] = None,
        category: str = None,
        metadata: Dict = None,
        source_url: str = None,
        importance: float = 0.5,
    ) -> str:
        """
        添加知识库条目

        Args:
            title: 标题
            content: 内容
            embedding: 向量嵌入
            category: 分类
            metadata: 元数据
            source_url: 源 URL
            importance: 重要性

        Returns:
            str: 知识库条目 ID
        """
        kb_id = generate_id(f"{title}:{content}")

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO knowledge_base
            (id, title, content, embedding, category, metadata, created_at, source_url, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                kb_id,
                title,
                content,
                json.dumps(embedding) if embedding else None,
                category,
                json.dumps(metadata or {}),
                datetime.now().isoformat(),
                source_url,
                importance,
            ),
        )

        conn.commit()
        self.logger.debug(f"添加知识库条目成功: ID={kb_id}")
        return kb_id

    def add_knowledge_batch(self, items: List[Dict]) -> int:
        """
        批量添加知识库

        Args:
            items: 知识库数据列表

        Returns:
            int: 添加的知识库条目数量
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        count = 0
        for item in items:
            kb_id = self._generate_id(f"{item.get('title', '')}:{item.get('content', '')}")

            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO knowledge_base
                    (id, title, content, embedding, category, metadata, created_at, source_url, importance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        kb_id,
                        item.get("title", ""),
                        item.get("content", ""),
                        json.dumps(item.get("embedding")) if item.get("embedding") else None,
                        item.get("category"),
                        json.dumps(item.get("metadata", {})),
                        datetime.now().isoformat(),
                        item.get("source_url"),
                        item.get("importance", 0.5),
                    ),
                )
                count += 1
            except Exception as e:
                self.logger.error(f"添加知识库失败：{e}")

        conn.commit()
        self.logger.info(f"✅ 批量添加 {count} 条知识库")
        return count

    def search_knowledge(
        self, query_embedding: List[float], top_k: int = 5, category_filter: str = None, min_importance: float = 0.0
    ) -> List[Dict]:
        """
        搜索知识库

        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            category_filter: 分类过滤
            min_importance: 最小重要性

        Returns:
            相似的知识库条目列表（带相似度分数）
        """
        try:
            # 验证输入
            validate_list(query_embedding, "query_embedding", min_length=1)
            if len(query_embedding) != self.embedding_dim:
                raise ValidationError(f"query_embedding dimension must be {self.embedding_dim}")
            validate_integer(top_k, "top_k", min_value=1)
            validate_float(min_importance, "min_importance", min_value=0, max_value=1)

            conn = self._get_connection()
            cursor = conn.cursor()

            # 转换查询向量
            query_array = np.array(query_embedding, dtype=np.float32)

            # 获取所有知识库条目
            if category_filter:
                cursor.execute(
                    """
                    SELECT id, title, content, embedding, category, metadata, source_url, importance
                    FROM knowledge_base
                    WHERE category = ? AND importance >= ?
                """,
                    (category_filter, min_importance),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, title, content, embedding, category, metadata, source_url, importance
                    FROM knowledge_base
                    WHERE importance >= ?
                """,
                    (min_importance,),
                )

            rows = cursor.fetchall()

            # 计算相似度
            results = []
            for row in rows:
                kb_id, title, content, embedding_str, category, metadata_str, source_url, importance = row

                if embedding_str:
                    embedding = json.loads(embedding_str)
                    similarity = self._cosine_similarity(query_array, np.array(embedding, dtype=np.float32))

                    results.append(
                        {
                            "id": kb_id,
                            "title": title,
                            "content": content,
                            "category": category,
                            "metadata": json.loads(metadata_str) if metadata_str else {},
                            "source_url": source_url,
                            "importance": importance,
                            "similarity": similarity,
                        }
                    )

            # 按相似度排序
            results.sort(key=lambda x: x["similarity"], reverse=True)

            self.logger.debug(f"搜索知识库成功，返回 {len(results[:top_k])} 条")
            return results[:top_k]
        except Exception as e:
            self.logger.error(f"搜索知识库失败: {e}")
            raise VectorDatabaseError(f"搜索知识库失败: {e}")

    def get_knowledge_stats(self) -> Dict:
        """
        获取知识库统计

        Returns:
            Dict: 统计信息
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # 总数
            cursor.execute("SELECT COUNT(*) FROM knowledge_base")
            total = cursor.fetchone()[0]

            # 分类统计
            cursor.execute("SELECT category, COUNT(*) FROM knowledge_base GROUP BY category")
            categories = dict(cursor.fetchall())

            # 平均重要性
            cursor.execute("SELECT AVG(importance) FROM knowledge_base")
            avg_importance = cursor.fetchone()[0] or 0

            # 最近 7 天添加的条目
            seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute("SELECT COUNT(*) FROM knowledge_base WHERE created_at >= ?", (seven_days_ago,))
            recent_entries = cursor.fetchone()[0]

            return {
                "total_entries": total,
                "categories": categories,
                "average_importance": avg_importance,
                "recent_entries": recent_entries,
            }
        except Exception as e:
            self.logger.error(f"获取知识库统计失败: {e}")
            raise VectorDatabaseError(f"获取知识库统计失败: {e}")

    # ========== 统计与清理 ==========

    def get_stats(self) -> Dict:
        """
        获取数据库统计

        Returns:
            Dict: 统计信息
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # 向量总数
            cursor.execute("SELECT COUNT(*) FROM vectors")
            vector_count = cursor.fetchone()[0]

            # 知识库总数
            cursor.execute("SELECT COUNT(*) FROM knowledge_base")
            knowledge_count = cursor.fetchone()[0]

            # 最常访问
            cursor.execute("""
                SELECT id, content, access_count
                FROM vectors
                ORDER BY access_count DESC
                LIMIT 5
            """)
            top_accessed = cursor.fetchall()

            # 最近 7 天添加的向量
            seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute("SELECT COUNT(*) FROM vectors WHERE created_at >= ?", (seven_days_ago,))
            recent_vectors = cursor.fetchone()[0]

            return {
                "vector_count": vector_count,
                "knowledge_count": knowledge_count,
                "top_accessed": [{"id": r[0], "content": r[1][:50], "access_count": r[2]} for r in top_accessed],
                "recent_vectors": recent_vectors,
            }
        except Exception as e:
            self.logger.error(f"获取数据库统计失败: {e}")
            raise VectorDatabaseError(f"获取数据库统计失败: {e}")

    def cleanup_old_vectors(self, days: int = 30, min_access_count: int = 0) -> int:
        """
        清理旧向量

        Args:
            days: 天数
            min_access_count: 最小访问计数

        Returns:
            int: 删除的向量数量
        """
        try:
            # 验证输入
            validate_integer(days, "days", min_value=1)
            validate_integer(min_access_count, "min_access_count", min_value=0)

            conn = self._get_connection()
            cursor = conn.cursor()

            cutoff_date = datetime.now()
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
            cutoff_str = cutoff_date.isoformat()

            cursor.execute(
                """
                DELETE FROM vectors
                WHERE created_at < ? AND access_count <= ?
            """,
                (cutoff_str, min_access_count),
            )

            deleted = cursor.rowcount
            conn.commit()

            if deleted > 0:
                self.logger.info(f"🗑️ 清理了 {deleted} 个旧向量")

            return deleted
        except Exception as e:
            self.logger.error(f"清理旧向量失败: {e}")
            raise VectorDatabaseError(f"清理旧向量失败: {e}")

    def print_status(self):
        """
        打印状态
        """
        try:
            stats = self.get_stats()
            kb_stats = self.get_knowledge_stats()

            self.logger.info("\n" + "=" * 60)
            self.logger.info("【向量数据库状态】")
            self.logger.info("=" * 60)
            self.logger.info(f"向量总数：{stats['vector_count']}")
            self.logger.info(f"知识库条目：{stats['knowledge_count']}")
            self.logger.info(f"知识库分类：{kb_stats['categories']}")
            self.logger.info(f"平均重要性：{kb_stats['average_importance']:.2f}")
            self.logger.info(f"最近 7 天添加的向量：{stats['recent_vectors']}")
            self.logger.info(f"最近 7 天添加的知识库条目：{kb_stats['recent_entries']}")

            if stats["top_accessed"]:
                self.logger.info("\n最常访问:")
                for i, item in enumerate(stats["top_accessed"], 1):
                    self.logger.info(f"  {i}. {item['content']}... (访问：{item['access_count']}次)")

            self.logger.info("=" * 60)
        except Exception as e:
            self.logger.error(f"打印状态失败: {e}")
            raise VectorDatabaseError(f"打印状态失败: {e}")

    def close(self):
        """
        关闭数据库连接
        """
        try:
            if hasattr(self._local, "conn"):
                self._local.conn.close()
                delattr(self._local, "conn")
                self.logger.debug("关闭向量数据库连接")
        except Exception as e:
            self.logger.error(f"关闭向量数据库连接失败: {e}")
            raise VectorDatabaseError(f"关闭向量数据库连接失败: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ========== 测试需要的方法 ==========

    def add_embedding(self, embedding, metadata=None):
        """
        添加嵌入向量

        Args:
            embedding: 嵌入向量
            metadata: 元数据

        Returns:
            向量ID
        """
        # 使用现有的add_vector方法，但确保content不为空
        content = metadata.get("content", "") if metadata else ""
        if not content:
            # 如果content为空，使用metadata的信息生成一个content
            if metadata:
                content = str(metadata)
            else:
                content = f"embedding_{generate_id('')}"
        return self.add_vector(content, embedding, metadata)

    def add_embeddings(self, embeddings, metadatas):
        """
        批量添加嵌入向量

        Args:
            embeddings: 嵌入向量列表
            metadatas: 元数据列表

        Returns:
            向量ID列表
        """
        vector_ids = []
        for i, embedding in enumerate(embeddings):
            metadata = metadatas[i] if i < len(metadatas) else None
            vector_id = self.add_embedding(embedding, metadata)
            vector_ids.append(vector_id)
        return vector_ids

    def get_embedding(self, vector_id):
        """
        获取嵌入向量

        Args:
            vector_id: 向量ID

        Returns:
            (embedding, metadata) 元组或 None
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT embedding, metadata FROM vectors WHERE id = ?", (vector_id,))
        row = cursor.fetchone()
        if not row:
            return None

        embedding_str, metadata_str = row
        embedding = json.loads(embedding_str)
        metadata = json.loads(metadata_str)
        return (embedding, metadata)

    def update_embedding(self, vector_id, embedding, metadata=None):
        """
        更新嵌入向量

        Args:
            vector_id: 向量ID
            embedding: 新的嵌入向量
            metadata: 新的元数据

        Returns:
            是否更新成功
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # 检查向量是否存在
        cursor.execute("SELECT id FROM vectors WHERE id = ?", (vector_id,))
        row = cursor.fetchone()
        if not row:
            return False

        # 直接更新向量
        cursor.execute(
            """
            UPDATE vectors
            SET embedding = ?, metadata = ?, updated_at = ?
            WHERE id = ?
        """,
            (json.dumps(embedding), json.dumps(metadata or {}), datetime.now().isoformat(), vector_id),
        )
        conn.commit()

        # 更新FAISS索引
        if faiss and self._faiss_index:
            with self._lock:
                # 重新加载向量到FAISS
                self._init_faiss_index()

        return True

    def delete_embedding(self, vector_id):
        """
        删除嵌入向量

        Args:
            vector_id: 向量ID

        Returns:
            是否删除成功
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM vectors WHERE id = ?", (vector_id,))
        conn.commit()

        # 更新FAISS索引
        if faiss and self._faiss_index:
            with self._lock:
                if vector_id in self._vector_ids:
                    idx = self._vector_ids.index(vector_id)
                    # FAISS不支持直接删除，需要重建索引
                    self._vector_ids.remove(vector_id)
                    # 重新加载向量到FAISS
                    self._init_faiss_index()

        return cursor.rowcount > 0

    def get_all_embeddings(self):
        """
        获取所有嵌入向量

        Returns:
            向量列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT embedding, metadata FROM vectors")
        rows = cursor.fetchall()

        embeddings = []
        for row in rows:
            embedding_str, metadata_str = row
            embedding = json.loads(embedding_str)
            metadata = json.loads(metadata_str)
            embeddings.append((embedding, metadata))

        return embeddings

    def count_embeddings(self):
        """
        统计嵌入向量数量

        Returns:
            向量数量
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM vectors")
        count = cursor.fetchone()[0]
        return count

    def clear_embeddings(self):
        """
        清空嵌入向量

        Returns:
            是否清空成功
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM vectors")
        conn.commit()

        # 重置FAISS索引
        if faiss and self._faiss_index:
            with self._lock:
                self._faiss_index.reset()
                self._vector_ids = []

        return True


# 全局实例
vector_store = VectorDatabase()


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("向量数据库测试")
    print("=" * 60)

    db = VectorDatabase()

    # 测试添加向量
    print("\n1. 添加测试向量")
    vector_id = db.add_vector(
        content="测试向量内容", embedding=[0.1] * 768, metadata={"test": "value"}, tags=["test", "example"]
    )
    print(f"   ✅ 添加向量成功: {vector_id}")

    # 测试批量添加
    print("\n2. 批量添加向量")
    items = [
        {"content": "批量测试1", "embedding": [0.2] * 768, "tags": ["batch"]},
        {"content": "批量测试2", "embedding": [0.3] * 768, "tags": ["batch"]},
    ]
    count = db.add_vectors_batch(items)
    print(f"   ✅ 批量添加 {count} 个向量")

    # 测试搜索
    print("\n3. 测试相似度搜索")
    results = db.search_similar(query_embedding=[0.15] * 768, top_k=3, min_similarity=0.1)
    print(f"   ✅ 搜索到 {len(results)} 个相似向量")
    for i, result in enumerate(results, 1):
        print(f"   {i}. 相似度: {result['similarity']:.4f}, 内容: {result['content'][:50]}...")

    # 测试知识库
    print("\n4. 测试知识库")
    kb_id = db.add_knowledge(
        title="测试知识库", content="这是一个测试知识库条目", embedding=[0.4] * 768, category="test", importance=0.8
    )
    print(f"   ✅ 添加知识库成功: {kb_id}")

    # 测试知识库搜索
    kb_results = db.search_knowledge(query_embedding=[0.45] * 768, top_k=3)
    print(f"   ✅ 搜索到 {len(kb_results)} 个相似知识库条目")
    for i, result in enumerate(kb_results, 1):
        print(f"   {i}. 相似度: {result['similarity']:.4f}, 标题: {result['title']}")

    # 测试统计
    print("\n5. 测试统计信息")
    stats = db.get_stats()
    kb_stats = db.get_knowledge_stats()
    print(f"   ✅ 向量总数: {stats['vector_count']}")
    print(f"   ✅ 知识库总数: {stats['knowledge_count']}")
    print(f"   ✅ 知识库分类: {kb_stats['categories']}")

    # 测试清理
    print("\n6. 测试清理")
    deleted = db.cleanup_old_vectors(days=365, min_access_count=0)
    print(f"   ✅ 清理了 {deleted} 个旧向量")

    # 打印状态
    print("\n7. 打印状态")
    db.print_status()

    db.close()
    print("\n✅ 测试完成")
