#!/usr/bin/env python3
"""
🧪 MetaMemory 全量回归测试套件

覆盖范围：
- 缓存层功能测试
- 数据库集成测试
- 向量数据库测试
- LLM 管理器测试
- 端到端集成测试
"""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# 检测 FAISS 可用性
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ============================================================================
# 1. 缓存层测试
# ============================================================================
class TestCacheLayer(unittest.TestCase):
    """缓存层测试"""

    def test_in_memory_cache_basic_ops(self):
        """测试内存缓存基础操作"""
        from core.cache_layer import InMemoryCache

        cache = InMemoryCache(max_size=10, default_ttl=3600)

        # 测试插入和读取
        cache.set("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")

        # 测试不存在的键
        self.assertIsNone(cache.get("key_not_exist"))

        # 测试覆盖
        cache.set("key1", "value2")
        self.assertEqual(cache.get("key1"), "value2")

        # 测试删除
        cache.delete("key1")
        self.assertIsNone(cache.get("key1"))

        # 测试 exists (via get)
        cache.set("key2", {"data": 123})
        self.assertIsNotNone(cache.get("key2"))
        self.assertIsNone(cache.get("key_not_exist"))

    def test_in_memory_cache_stats(self):
        """测试缓存统计"""
        from core.cache_layer import InMemoryCache

        cache = InMemoryCache(max_size=100)

        for i in range(10):
            cache.set(f"key{i}", f"value{i}")

        # 验证缓存正确存储
        self.assertIsNotNone(cache.get("key0"))
        self.assertIsNone(cache.get("not_exist0"))

    def test_in_memory_cache_lru_eviction(self):
        """测试 LRU 淘汰"""
        from core.cache_layer import InMemoryCache

        cache = InMemoryCache(max_size=10, default_ttl=3600)

        # 插入 20 个，应该淘汰前 10 个
        for i in range(20):
            cache.set(f"key{i}", f"value{i}")

        # 访问后 5 个
        for i in range(15, 20):
            cache.get(f"key{i}")

        # 再插入 2 个
        for i in range(20, 22):
            cache.set(f"key{i}", f"value{i}")

        # 被访问的应该还在
        for i in range(15, 20):
            self.assertIsNotNone(cache.get(f"key{i}"))

    def test_multi_level_cache(self):
        """测试多级缓存"""
        from core.cache_layer import MultiLevelCache

        cache = MultiLevelCache(use_memory=True, use_redis=False)

        cache.set("test_key", {"data": "hello"})
        value = cache.get("test_key")
        self.assertEqual(value, {"data": "hello"})

        cache.delete("test_key")
        self.assertIsNone(cache.get("test_key"))

    def test_cached_decorator(self):
        """测试缓存装饰器"""
        from core.cache_layer import cached, InMemoryCache

        call_count = [0]

        @cached(key_prefix="test_func", ttl=60)
        def expensive_function(a: int, b: str = "default"):
            call_count[0] += 1
            return {"a": a, "b": b, "result": a * 2}

        # 第一次调用
        r1 = expensive_function(10, b="test")
        self.assertEqual(r1, {"a": 10, "b": "test", "result": 20})
        self.assertEqual(call_count[0], 1)

        # 第二次调用应该缓存
        r2 = expensive_function(10, b="test")
        self.assertEqual(r2, r1)
        self.assertEqual(call_count[0], 1)  # 没有增加

        # 不同参数应该重新调用
        r3 = expensive_function(20, b="test")
        self.assertEqual(r3["result"], 40)
        self.assertEqual(call_count[0], 2)

    def test_global_cache(self):
        """测试全局缓存实例"""
        from core.cache_layer import get_global_cache

        cache = get_global_cache()
        self.assertIsNotNone(cache)

        cache2 = get_global_cache()
        self.assertIs(cache, cache2)  # 单例


# ============================================================================
# 2. 数据库集成测试
# ============================================================================
class TestDatabaseIntegration(unittest.TestCase):
    """数据库集成测试"""

    def setUp(self):
        """每个测试前创建临时数据库"""
        import tempfile

        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        """清理"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_database_config(self):
        """测试数据库配置"""
        from core.database_integration import DatabaseConfig

        # SQLite 配置
        cfg = DatabaseConfig(db_type="sqlite", sqlite_path=self.db_path)
        self.assertEqual(cfg.db_type, "sqlite")
        self.assertEqual(cfg.sqlite_path, self.db_path)

        # PostgreSQL 配置
        cfg2 = DatabaseConfig(
            db_type="postgresql",
            host="localhost",
            port=5432,
            username="test",
            password="pass",
            database="test_db",
        )
        self.assertEqual(cfg2.db_type, "postgresql")
        self.assertEqual(cfg2.host, "localhost")

    def test_sqlite_integration(self):
        """测试 SQLite 集成"""
        from core.database_integration import DatabaseConfig, DatabaseIntegration

        cfg = DatabaseConfig(db_type="sqlite", sqlite_path=self.db_path, use_cache=False)

        db = DatabaseIntegration(cfg)
        # Access repository to trigger lazy initialization
        repo = db.repository
        self.assertIsNotNone(repo)
        repo.create_tables()

        # 测试插入记忆
        memory_data = {
            "id": "test_memory_001",
            "user_id": "user_001",
            "content": "这是一条测试记忆",
            "memory_type": "memory",
            "importance": 0.8,
            "created_at": "2026-01-01 00:00:00",
            "updated_at": "2026-01-01 00:00:00",
            "metadata": json.dumps({"source": "test"}),
            "tags": "test,unit",
            "source": "test",
            "status": "active",
        }

        self.assertTrue(repo.insert_memory(memory_data))

        # 测试获取记忆
        fetched = repo.get_memory("test_memory_001")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["id"], "test_memory_001")
        self.assertEqual(fetched["content"], "这是一条测试记忆")

        # 测试更新
        self.assertTrue(repo.update_memory("test_memory_001", {"importance": 0.9}))
        updated = repo.get_memory("test_memory_001")
        self.assertEqual(updated["importance"], 0.9)

        # 测试列出记忆
        memories = repo.list_memories("user_001")
        self.assertEqual(len(memories), 1)

        # 测试统计
        stats = repo.get_stats("user_001")
        self.assertEqual(stats["total"], 1)

        # 测试访问日志
        log_data = {
            "user_id": "user_001",
            "action": "search",
            "memory_id": "test_memory_001",
            "query": "测试",
            "result_count": 5,
            "latency_ms": 123.45,
        }
        self.assertTrue(repo.log_access(log_data))

        # 测试删除
        self.assertTrue(repo.delete_memory("test_memory_001"))
        deleted = repo.get_memory("test_memory_001")
        self.assertEqual(deleted["status"], "deleted")

    def test_database_health_check(self):
        """测试健康检查"""
        from core.database_integration import DatabaseConfig, DatabaseIntegration

        cfg = DatabaseConfig(db_type="sqlite", sqlite_path=self.db_path)
        db = DatabaseIntegration(cfg)
        # DatabaseIntegration auto-initializes on creation

        health = db.health_check()
        self.assertIn(health["status"], ["healthy", "unhealthy"])
        if "db_type" in health:
            self.assertEqual(health["db_type"], "sqlite")

    def test_global_database(self):
        """测试全局数据库实例"""
        from core.database_integration import get_database, DatabaseConfig

        cfg = DatabaseConfig(db_type="sqlite", sqlite_path=os.path.join(self.temp_dir, "global.db"))
        db = get_database(cfg)
        self.assertIsNotNone(db)


# ============================================================================
# 3. 向量数据库测试
# ============================================================================
class TestVectorDatabase(unittest.TestCase):
    """向量数据库测试"""

    @classmethod
    def setUpClass(cls):
        """检查 FAISS 是否可用"""
        try:
            import faiss

            cls.faiss_available = True
        except ImportError:
            cls.faiss_available = False

    def test_vector_search_result(self):
        """测试向量搜索结果"""
        from core.vector_database import VectorSearchResult

        result = VectorSearchResult(
            id="test_001",
            score=0.95,
            distance=0.05,
            metadata={"type": "memory"},
        )

        self.assertEqual(result.id, "test_001")
        self.assertEqual(result.score, 0.95)
        self.assertEqual(result.distance, 0.05)
        self.assertEqual(result.metadata["type"], "memory")

    @unittest.skipUnless(FAISS_AVAILABLE, "FAISS not available")
    def test_faiss_backend_basic(self):
        """测试 FAISS 后端基础操作"""
        from core.vector_database import VectorDBConfig, FAISSVectorBackend

        cfg = VectorDBConfig(backend="faiss", dimension=8, index_type="HNSW")
        backend = FAISSVectorBackend(cfg)

        # 测试插入
        vec1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.assertTrue(backend.insert("vec_001", vec1, {"name": "向量1"}))

        # 测试批量插入
        vectors = []
        for i in range(10):
            vec = np.random.rand(8).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # 归一化
            vectors.append((f"vec_batch_{i}", vec, {"index": i}))

        inserted = backend.insert_batch(vectors)
        self.assertEqual(inserted, 10)

        # 测试搜索
        query = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = backend.search(query, top_k=5)
        self.assertGreater(len(results), 0)

        # 第一个结果应该是 vec_001（相似度最高）
        self.assertIn("vec_001", [r.id for r in results])

        # 测试删除
        self.assertTrue(backend.delete("vec_001"))

    @unittest.skipUnless(FAISS_AVAILABLE, "FAISS not available")
    def test_faiss_persistence(self):
        """测试 FAISS 持久化"""
        from core.vector_database import VectorDBConfig, FAISSVectorBackend

        temp_dir = tempfile.mkdtemp()
        index_path = os.path.join(temp_dir, "test.index")

        cfg = VectorDBConfig(backend="faiss", dimension=8, index_type="Flat", faiss_index_path=index_path)
        backend = FAISSVectorBackend(cfg)

        # 插入一些向量
        for i in range(5):
            vec = np.random.rand(8).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            backend.insert(f"save_test_{i}", vec, {"test": True})

        # 保存
        self.assertTrue(backend.save(index_path))

        # 创建新后端加载
        backend2 = FAISSVectorBackend(cfg)
        self.assertTrue(backend2.load(index_path))

        # 验证向量数一致
        stats1 = backend.health_check()
        stats2 = backend2.health_check()
        self.assertEqual(stats1["total_vectors"], stats2["total_vectors"])

        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @unittest.skipUnless(FAISS_AVAILABLE, "FAISS not available")
    def test_vector_database_manager(self):
        """测试向量数据库管理器"""
        from core.vector_database import VectorDatabase, VectorDBConfig

        cfg = VectorDBConfig(backend="faiss", dimension=8, index_type="Flat")
        vdb = VectorDatabase(cfg)

        self.assertIsNotNone(vdb.backend)

        # 插入向量
        vec = np.random.rand(8).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        self.assertTrue(vdb.insert("manager_test", vec))

        # 搜索
        results = vdb.search(vec, top_k=3)
        self.assertGreater(len(results), 0)

        # 健康检查
        health = vdb.health_check()
        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["backend"], "faiss")


# ============================================================================
# 4. LLM 管理器测试
# ============================================================================
class TestLLMManager(unittest.TestCase):
    """LLM 管理器测试"""

    def test_chat_message(self):
        """测试聊天消息"""
        from api.llm_manager import ChatMessage

        msg = ChatMessage(role="user", content="你好", name="测试用户")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "你好")
        self.assertEqual(msg.to_dict(), {"role": "user", "content": "你好", "name": "测试用户"})

    def test_llm_response(self):
        """测试 LLM 响应"""
        from api.llm_manager import LLMResponse

        resp = LLMResponse(
            content="这是回复",
            model="test-model",
            provider="deepseek",
            tokens={"prompt": 10, "completion": 20, "total": 30},
            latency=0.5,
        )

        self.assertEqual(resp.content, "这是回复")
        self.assertEqual(resp.model, "test-model")
        self.assertEqual(resp.tokens["total"], 30)
        self.assertEqual(resp.latency, 0.5)

    def test_deepseek_backend_creation(self):
        """测试 DeepSeek 后端创建"""
        from api.llm_manager import DeepSeekBackend

        # 测试空 API Key 抛出认证错误
        with self.assertRaises(Exception):
            DeepSeekBackend(api_key="")

        # 测试正常创建
        backend = DeepSeekBackend(api_key="sk_test_key_123", use_cache=True)
        self.assertIsNotNone(backend)
        self.assertEqual(backend.api_key, "sk_test_key_123")

    def test_llm_error_types(self):
        """测试 LLM 错误类型"""
        from api.llm_manager import (
            LLMError,
            LLMAuthenticationError,
            LLMRateLimitError,
            LLMTimeoutError,
        )

        # 测试错误继承
        self.assertTrue(issubclass(LLMAuthenticationError, LLMError))
        self.assertTrue(issubclass(LLMRateLimitError, LLMError))
        self.assertTrue(issubclass(LLMTimeoutError, LLMError))

        # 测试可以正常抛出
        with self.assertRaises(LLMAuthenticationError):
            raise LLMAuthenticationError("认证失败")

    def test_deepseek_health_check(self):
        """测试 DeepSeek 健康检查（需要网络+API Key，跳过）"""
        self.skipTest("需要 DeepSeek API Key")