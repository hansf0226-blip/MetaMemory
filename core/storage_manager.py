#!/usr/bin/env python3
"""
🗄️ 统一存储管理系统

功能：
- 统一管理所有存储模块
- 提供一致的接口
- 处理存储的初始化和配置
- 提供存储策略和容错机制
- 支持多存储后端
"""

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.config import get_config
try:
    from core.utils import (
        DatabaseError,
        get_logger,
        validate_dict,
        validate_list,
        validate_required_fields,
    )
except ImportError:
    import logging
    def get_logger(name: str):
        return logging.getLogger(name)

    class DatabaseError(Exception):
        pass

    def validate_dict(data, name="data"):
        if not isinstance(data, dict):
            raise ValueError(f"{name} must be a dict")

    def validate_list(data, name="data", min_length=0):
        if not isinstance(data, list):
            raise ValueError(f"{name} must be a list")
        if len(data) < min_length:
            raise ValueError(f"{name} must have at least {min_length} items")

    def validate_required_fields(data, fields):
        for f in fields:
            if f not in data:
                raise ValueError(f"Missing required field: {f}")

from .hybrid_retrieval import SEMANTIC_WEIGHT, STRUCTURAL_WEIGHT, HybridRetrievalEngine
from .identity_policy import check_memory_policy, get_policy_engine

# 导入存储模块
from .memory_db import MemoryDatabase
from .memory_graph import discover_memory_relations, get_graph_engine
from .mysql_store import MySQLMemoryStore, create_mysql_store
from .vector_store import VectorDatabase

logger = get_logger(__name__)


# 异常定义
class StorageManagerError(DatabaseError):
    """存储管理器异常"""


class StorageManager:
    """
    统一存储管理器

    管理所有存储模块，提供一致的接口
    """

    def __init__(self):
        """
        初始化存储管理器
        """
        self._lock = threading.RLock()
        self._memory_db = None
        self._mysql_store = None
        self._vector_db = None
        self._initialized = False

    def initialize(self):
        """
        初始化所有存储模块
        """
        with self._lock:
            if self._initialized:
                return

            try:
                # 初始化 SQLite 内存数据库
                self._memory_db = MemoryDatabase()
                logger.info("✅ SQLite 内存数据库初始化成功")

                # 初始化 MySQL 存储
                self._mysql_store = create_mysql_store()
                logger.info("✅ MySQL 存储初始化成功")

                # 初始化向量数据库
                embedding_dim = get_config("vector_store.dimension", 768)
                self._vector_db = VectorDatabase(embedding_dim=embedding_dim)
                logger.info("✅ 向量数据库初始化成功")

                self._initialized = True
                logger.info("✅ 存储管理器初始化完成")
            except Exception as e:
                logger.error(f"❌ 存储管理器初始化失败：{e}")
                raise StorageManagerError(f"存储管理器初始化失败：{e}")

    @property
    def memory_db(self) -> MemoryDatabase:
        """
        获取内存数据库实例
        """
        if not self._initialized:
            self.initialize()
        return self._memory_db

    @property
    def mysql_store(self) -> MySQLMemoryStore:
        """
        获取 MySQL 存储实例
        """
        if not self._initialized:
            self.initialize()
        return self._mysql_store

    @property
    def vector_db(self) -> VectorDatabase:
        """
        获取向量数据库实例
        """
        if not self._initialized:
            self.initialize()
        return self._vector_db

    # ========== 记忆操作 ==========

    def create_memory(self, memory_data: Dict[str, Any], write_auth: str = "normal", skip_policy: bool = False) -> Any:
        """
        创建记忆（含 Identity/Policy Layer 校验）

        Args:
            memory_data: 记忆数据字典
            write_auth: 写入权限级别
                - normal: 普通写入（受 BLOCK/WARNING 规则约束）
                - admin: 管理员写入（BLOCK 规则降为 WARNING，可强制写入）
                - system: 系统写入（绕过所有检查）
            skip_policy: 是否跳过策略校验（用于批量导入等可信场景）

        Returns:
            创建的记忆对象

        Raises:
            StorageManagerError: 策略校验 BLOCK 或其他存储错误
        """
        try:
            # 验证输入
            validate_dict(memory_data, "memory_data")
            validate_required_fields(memory_data, ["agent_id", "hexagram", "content"])

            # ===== Identity/Policy Layer 校验 =====
            if not skip_policy:
                policy_result = check_memory_policy(
                    memory_content=memory_data.get("content", ""),
                    agent_id=memory_data["agent_id"],
                    mysql_store=self.mysql_store,
                    write_auth=write_auth,
                )
                if policy_result.is_blocked():
                    logger.warning(
                        f"🛑 记忆写入被策略层阻断：agent={memory_data['agent_id']}, "
                        f"rules={policy_result.triggered_rules}"
                    )
                    raise StorageManagerError(
                        "POLICY_BLOCK: 内容违反核心伦理规则，"
                        f"触发规则：{', '.join(policy_result.triggered_rules)}，"
                        f"建议：{policy_result.suggestion}"
                    )
                if policy_result.is_warning():
                    logger.warning(
                        f"⚠️  记忆写入存在价值观张力：agent={memory_data['agent_id']}, "
                        f"tension={policy_result.triggered_rules}"
                    )

            # 保存到 MySQL
            memory = self.mysql_store.create_memory(memory_data)

            # ===== Memory Graph 关联发现 =====
            if not skip_policy:
                try:
                    # 候选集：从同 agent 取最新 100 条活跃记忆（排除刚创建的那条）
                    candidates_raw = self.mysql_store.get_memories(
                        agent_id=memory_data["agent_id"],
                        limit=100,
                        offset=0,
                        status="active",
                    )
                    # 排除自己
                    new_id = str(memory["id"])
                    candidates = [c for c in candidates_raw if str(getattr(c, "id", c.get("id", ""))) != new_id]
                    new_mem_dict = {
                        "memory_id": new_id,
                        "bagua_type": memory_data.get("bagua_type", ""),
                        "sancai_layer": memory_data.get("sancai_layer", ""),
                        "wuxing": memory_data.get("wuxing", ""),
                    }
                    links = discover_memory_relations(new_mem_dict, candidates, max_relations=10)
                    if links:
                        # 编码关联标签
                        encoded = get_graph_engine().encode_relation_tags(links)
                        self.mysql_store.update_memory(
                            int(new_id.split("-")[-1]) if "-" in new_id else int(new_id),
                            memory_data["agent_id"],
                            {"relations": encoded},
                        )
                        logger.info(f"🔗 关联图谱：发现 {len(links)} 条关联记忆，ID={new_id}")
                except Exception as e:
                    # 关联发现失败不影响写入，仅记录日志
                    logger.warning(f"⚠️  关联图谱发现失败，不阻断写入：{e}")

            # 同时保存到内存数据库（短期记忆）
            short_term_data = {
                "id": memory.get("memory_id") or str(memory.get("id")),
                "content": memory_data.get("content"),
                "category": memory_data.get("bagua_type", "general"),
                "importance": memory_data.get("hot_score", 0.5),
                "created_at": datetime.now().isoformat(),
            }
            self.memory_db.save_short_term(short_term_data)

            # ===== 语义向量存储（Phase 3B: 纯语义，不融合六爻）=====
            try:
                from core.core_yijing import generate_semantic_embedding

                content = memory_data.get("content", "")
                embedding = generate_semantic_embedding(content)  # 不再传入 hex_list
                self.vector_db.add_vector(
                    content=content,
                    embedding=embedding,
                    metadata={
                        "agent_id": memory_data.get("agent_id"),
                        "memory_id": memory.get("memory_id") or str(memory.get("id")),
                        "bagua_type": memory_data.get("bagua_type"),
                        "wuxing": memory_data.get("wuxing"),
                    },
                )
                logger.debug(f"✅ 语义向量已存储：dim={len(embedding)}")
            except Exception as e:
                logger.warning(f"⚠️ 语义向量存储失败（不阻断写入）：{e}")

            logger.info(f"✅ 创建记忆成功：ID={memory['id']}")
            return memory
        except Exception as e:
            logger.error(f"❌ 创建记忆失败：{e}")
            raise StorageManagerError(f"创建记忆失败：{e}")

    def create_memories_batch(
        self, memories_data: List[Dict], write_auth: str = "normal", skip_policy: bool = False
    ) -> List[Any]:
        """
        批量创建记忆（含策略校验）

        Args:
            memories_data: 记忆数据字典列表
            write_auth: 写入权限级别（默认 normal）
            skip_policy: 是否跳过策略校验（用于可信批量导入）

        Returns:
            创建的记忆对象列表
        """
        try:
            # 验证输入
            validate_list(memories_data, "memories_data", min_length=1)

            # ===== Identity/Policy Layer 校验 =====
            if not skip_policy:
                policy_results = get_policy_engine().validate_batch(
                    memories=memories_data,
                    agent_id=memories_data[0].get("agent_id", "default_agent"),
                    mysql_store=self.mysql_store,
                    write_auth=write_auth,
                )
                # 批量检查是否有 BLOCK
                blocked = [r for r in policy_results if r.is_blocked()]
                if blocked:
                    raise StorageManagerError(
                        f"POLICY_BLOCK: 批量写入中有 {len(blocked)} 条被阻断，"
                        f"第一条触发规则：{blocked[0].triggered_rules}"
                    )
                # 记录 WARNING
                warnings = [r for r in policy_results if r.is_warning()]
                if warnings:
                    logger.warning(f"⚠️  批量写入中有 {len(warnings)} 条存在张力警告")

            # 批量保存到 MySQL
            memories = self.mysql_store.create_memories_batch(memories_data)

            # 同时保存到内存数据库（短期记忆）
            short_term_data_list = []
            for i, memory_data in enumerate(memories_data):
                if i < len(memories):
                    memory = memories[i]
                    short_term_data = {
                        "id": memory.get("memory_id") or str(memory.get("id")),
                        "content": memory_data.get("content"),
                        "category": memory_data.get("bagua_type", "general"),
                        "importance": memory_data.get("hot_score", 0.5),
                        "created_at": datetime.now().isoformat(),
                    }
                    short_term_data_list.append(short_term_data)

            # 批量保存到内存数据库
            for short_term_data in short_term_data_list:
                self.memory_db.save_short_term(short_term_data)

            logger.info(f"✅ 批量创建 {len(memories)} 条记忆成功")
            return memories
        except Exception as e:
            logger.error(f"❌ 批量创建记忆失败：{e}")
            raise StorageManagerError(f"批量创建记忆失败：{e}")

    def count_memories(self, agent_id: str = "", status: str = "active") -> int:
        """
        统计记忆数量
        """
        try:
            return self.mysql_store.count_memories(agent_id=agent_id, status=status)
        except Exception as e:
            logger.warn(f"⚠️ 统计记忆失败，返回0: {e}")
            return 0

    def get_memories(
        self,
        agent_id: str,
        limit: int = 100,
        offset: int = 0,
        layer: Optional[str] = None,
        element: Optional[str] = None,
        trigram: Optional[str] = None,
        min_hot: Optional[float] = None,
        status: str = "active",
    ) -> List[Any]:
        """
        查询记忆

        Args:
            agent_id: 租户 ID
            limit: 返回数量限制
            offset: 偏移量
            layer: 三才层筛选
            element: 五行筛选
            trigram: 八卦筛选
            min_hot: 最小热度
            status: 状态筛选

        Returns:
            记忆列表
        """
        try:
            # 从 MySQL 获取
            memories = self.mysql_store.get_memories(
                agent_id=agent_id,
                limit=limit,
                offset=offset,
                layer=layer,
                element=element,
                trigram=trigram,
                min_hot=min_hot,
                status=status,
            )

            return memories
        except Exception as e:
            logger.error(f"❌ 查询记忆失败：{e}")
            raise StorageManagerError(f"查询记忆失败：{e}")

    def update_memory(self, memory_id: int, agent_id: str, updates: Dict[str, Any]) -> Optional[Any]:
        """
        更新记忆

        Args:
            memory_id: 记忆 ID（支持 string 或 int，string 时自动转 int）
            agent_id: 租户 ID
            updates: 更新数据字典

        Returns:
            更新后的记忆对象或 None
        """
        try:
            # 验证输入
            validate_dict(updates, "updates")

            # hexagram 可能是 list [1,0,1,0,1,0]，需转为 string "101010"
            if "hexagram" in updates:
                hx = updates["hexagram"]
                if isinstance(hx, list):
                    updates["hexagram"] = "".join(str(b) for b in hx)
                elif isinstance(hx, str):
                    pass  # already string
                else:
                    updates["hexagram"] = str(hx)

            # API 层传入的 memory_id 是 string，需转为 int
            # 如果是 string 且包含非数字字符，先通过 MySQL 查询映射
            if not isinstance(memory_id, int):
                try:
                    memory_id = int(memory_id)
                except ValueError:
                    # 尝试从缓存或 DB 查找对应的整数 ID
                    for mid_str, mem in self.mysql_store._memory_cache.items():
                        if mem.get("memory_id") == str(memory_id):
                            memory_id = mem.get("id") or int(mid_str)
                            break
                    else:
                        raise StorageManagerError(f"无法将 memory_id '{memory_id}' 转换为整数")

            # 更新 MySQL
            memory = self.mysql_store.update_memory(memory_id, agent_id, updates)

            # 同步更新进程内缓存
            if memory and hasattr(self.mysql_store, "_memory_cache"):
                # 用 id 查找缓存项并更新
                for mid, mem in list(self.mysql_store._memory_cache.items()):
                    if mem.get("id") == memory_id or mem.get("memory_id") == str(memory_id):
                        self.mysql_store._memory_cache[mid] = {**mem, **memory}
                        break

            # 同时更新内存数据库
            if memory:
                short_term_memory = {
                    "id": memory.get("memory_id") or str(memory.get("id")),
                    "content": memory.get("content"),
                    "category": memory.get("bagua_type"),
                    "importance": memory.get("hot_score"),
                    "created_at": memory.get("created_at"),
                    "last_accessed": datetime.now().isoformat(),
                }
                self.memory_db.save_short_term(short_term_memory)

            logger.info(f"✅ 更新记忆成功：ID={memory_id}")
            return memory
        except Exception as e:
            logger.error(f"❌ 更新记忆失败：{e}")
            raise StorageManagerError(f"更新记忆失败：{e}")

    def delete_memory(self, memory_id: int, agent_id: str) -> bool:
        """
        删除记忆

        Args:
            memory_id: 记忆 ID（支持 string 或 int）
            agent_id: 租户 ID

        Returns:
            是否删除成功
        """
        try:
            # API 层传入的 memory_id 是 string，需转为 int
            if not isinstance(memory_id, int):
                try:
                    memory_id = int(memory_id)
                except ValueError:
                    # 尝试从缓存中查找对应的整数 id
                    original_id = memory_id
                    for mid, mem in list(getattr(self.mysql_store, "_memory_cache", {}).items()):
                        if mem.get("memory_id") == str(memory_id):
                            memory_id = mem.get("id") or int(mid)
                            break
                    else:
                        logger.warning(f"无法将 memory_id '{memory_id}' 转换为整数，跳过删除")
                        return False

            # 从 MySQL 删除
            success = self.mysql_store.delete_memory(memory_id, agent_id)

            # 清除缓存
            if success and hasattr(self.mysql_store, "_memory_cache"):
                for mid in list(self.mysql_store._memory_cache.keys()):
                    mem = self.mysql_store._memory_cache.get(mid, {})
                    if mem.get("id") == memory_id:
                        del self.mysql_store._memory_cache[mid]
                        break

            logger.info(f"✅ 删除记忆成功：ID={memory_id}")
            return success
        except Exception as e:
            logger.error(f"❌ 删除记忆失败：{e}")
            raise StorageManagerError(f"删除记忆失败：{e}")

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
        try:
            # 添加到向量数据库
            vector_id = self.vector_db.add_vector(content, embedding, metadata, tags)

            logger.info(f"✅ 添加向量成功：ID={vector_id}")
            return vector_id
        except Exception as e:
            logger.error(f"❌ 添加向量失败：{e}")
            raise StorageManagerError(f"添加向量失败：{e}")

    def add_vectors_batch(self, items: List[Dict]) -> int:
        """
        批量添加向量

        Args:
            items: 向量数据列表

        Returns:
            int: 添加的向量数量
        """
        try:
            # 批量添加到向量数据库
            count = self.vector_db.add_vectors_batch(items)

            logger.info(f"✅ 批量添加 {count} 个向量成功")
            return count
        except Exception as e:
            logger.error(f"❌ 批量添加向量失败：{e}")
            raise StorageManagerError(f"批量添加向量失败：{e}")

    def search_similar(
        self, query_embedding: List[float], top_k: int = 5, tags_filter: List[str] = None, min_similarity: float = 0.3
    ) -> List[Dict]:
        """
        搜索相似向量

        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            tags_filter: 标签过滤
            min_similarity: 最小相似度阈值

        Returns:
            相似的向量列表（带相似度分数）
        """
        try:
            # 从向量数据库搜索
            results = self.vector_db.search_similar(
                query_embedding=query_embedding, top_k=top_k, tags_filter=tags_filter, min_similarity=min_similarity
            )

            logger.info(f"✅ 搜索相似向量成功，返回 {len(results)} 条")
            return results
        except Exception as e:
            logger.error(f"❌ 搜索相似向量失败：{e}")
            raise StorageManagerError(f"搜索相似向量失败：{e}")

    def search_similar_batch(
        self,
        query_embeddings: List[List[float]],
        top_k: int = 5,
        tags_filter: List[str] = None,
        min_similarity: float = 0.3,
    ) -> List[List[Dict]]:
        """
        批量搜索相似向量

        Args:
            query_embeddings: 查询向量列表
            top_k: 返回数量
            tags_filter: 标签过滤
            min_similarity: 最小相似度阈值

        Returns:
            相似的向量列表列表（带相似度分数）
        """
        try:
            # 从向量数据库批量搜索
            results = self.vector_db.search_similar_batch(
                query_embeddings=query_embeddings, top_k=top_k, tags_filter=tags_filter, min_similarity=min_similarity
            )

            logger.info(f"✅ 批量搜索相似向量成功，返回 {len(results)} 组结果")
            return results
        except Exception as e:
            logger.error(f"❌ 批量搜索相似向量失败：{e}")
            raise StorageManagerError(f"批量搜索相似向量失败：{e}")

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
        try:
            # 添加到向量数据库
            kb_id = self.vector_db.add_knowledge(
                title=title,
                content=content,
                embedding=embedding,
                category=category,
                metadata=metadata,
                source_url=source_url,
                importance=importance,
            )

            logger.info(f"✅ 添加知识库条目成功：ID={kb_id}")
            return kb_id
        except Exception as e:
            logger.error(f"❌ 添加知识库条目失败：{e}")
            raise StorageManagerError(f"添加知识库条目失败：{e}")

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
            # 从向量数据库搜索
            results = self.vector_db.search_knowledge(
                query_embedding=query_embedding,
                top_k=top_k,
                category_filter=category_filter,
                min_importance=min_importance,
            )

            logger.info(f"✅ 搜索知识库成功，返回 {len(results)} 条")
            return results
        except Exception as e:
            logger.error(f"❌ 搜索知识库失败：{e}")
            raise StorageManagerError(f"搜索知识库失败：{e}")

    # ========== 混合检索（Hybrid Retrieval）==========

    def search_memories(
        self,
        agent_id: str,
        keyword: str = "",
        query_hexagram: Optional[List[int]] = None,
        query_vector: Optional[List[float]] = None,
        limit: int = 5,
        bagua_type: Optional[str] = None,
        sancai_layer: Optional[str] = None,
        min_hot_score: float = 0.0,
        use_hybrid: bool = True,
        structural_weight: float = STRUCTURAL_WEIGHT,
        semantic_weight: float = SEMANTIC_WEIGHT,
    ) -> List[Dict[str, Any]]:
        """
        记忆检索（统一入口，支持混合检索 + 关键词过滤）

        默认启用混合检索：结构相似度（40%）+ 向量相似度（60%）
        当 use_hybrid=False 或缺少向量时，降级为纯向量检索

        Args:
            agent_id: Agent ID
            keyword: 关键词过滤（可选）
            query_hexagram: 六爻查询向量（6位 0/1 列表）
            query_vector: 768-dim 语义向量
            limit: 返回数量
            bagua_type: 八卦类型过滤
            sancai_layer: 三才层级过滤
            min_hot_score: 最低热度
            use_hybrid: 是否启用混合检索（默认 True）
            structural_weight: 结构距离权重（默认 0.4）
            semantic_weight: 向量相似度权重（默认 0.6）

        Returns:
            检索结果列表（按混合评分降序）
        """
        try:
            # 无六爻且无向量 → 降级为 MySQL 关键词查询
            if not query_hexagram and not query_vector:
                results = self.mysql_store.get_memories(
                    agent_id=agent_id,
                    limit=limit,
                    offset=0,
                    layer=sancai_layer,
                    trigram=bagua_type,
                    min_hot=min_hot_score,
                )
                logger.info(f"✅ 关键词检索：agent={agent_id}，返回 {len(results)} 条")
                return [
                    {
                        "memory_id": r.get("memory_id") or str(r.get("id")),
                        "agent_id": r.get("agent_id"),
                        "bagua_type": r.get("bagua_type"),
                        "sancai_layer": r.get("sancai_layer"),
                        "wuxing": r.get("wuxing"),
                        "content": r.get("content"),
                        "hot_score": r.get("hot_score"),
                        "hexagram": r.get("hexagram"),
                        "similarity": 1.0,
                    }
                    for r in results
                ]

            if not use_hybrid or not query_hexagram or not query_vector:
                # 纯向量检索模式（降级路径）
                if query_vector:
                    vector_results = self.vector_db.search_similar(
                        query_embedding=query_vector, top_k=limit, min_similarity=0.05
                    )
                    # vector_results 格式: [(embedding, metadata_dict, similarity), ...]
                    # 建立 memory_id → similarity 映射
                    vec_similarity = {}
                    for vec_item in vector_results:
                        metadata = vec_item[1]  # metadata_dict
                        sim = vec_item[2]       # similarity float
                        mid = metadata.get("memory_id", "")
                        if mid:
                            vec_similarity[mid] = sim

                    # 用 memory_ids 查 MySQL — get_memories 拿全部再内存匹配
                    # 因为向量库存的 memory_id 是 uuid 字符串，而 MySQL get_memory 用的是自增 id
                    memory_ids = list(vec_similarity.keys())
                    all_mems = self.mysql_store.get_memories(
                        agent_id=agent_id if agent_id else None,
                        limit=1000,
                        offset=0,
                    )
                    # 建立 uuid → 记忆 映射
                    mem_by_uuid = {}
                    for m in all_mems:
                        mid = m.get("memory_id") or str(m.get("id", ""))
                        mem_by_uuid[mid] = m
                        # 同时用自增 id 做 key（兼容不同的匹配方式）
                        int_id = str(m.get("id", ""))
                        if int_id not in mem_by_uuid:
                            mem_by_uuid[int_id] = m

                    full_results = []
                    for mid in memory_ids:
                        m = mem_by_uuid.get(mid)
                        if m:
                            full_results.append(m)

                    # 合并向量相似度到结果
                    results = []
                    for r in full_results:
                        mid = r.get("memory_id") or str(r.get("id", ""))
                        sim = vec_similarity.get(mid, r.get("hot_score", 0.0))
                        if sim < 0.05:
                            continue
                        results.append({
                            "memory_id": mid,
                            "agent_id": r.get("agent_id"),
                            "bagua_type": r.get("bagua_type"),
                            "sancai_layer": r.get("sancai_layer"),
                            "wuxing": r.get("wuxing"),
                            "content": r.get("content"),
                            "hot_score": r.get("hot_score"),
                            "hexagram": r.get("hexagram"),
                            "similarity": round(sim, 4),
                        })
                    # 按相似度降序排列
                    results.sort(key=lambda x: x["similarity"], reverse=True)
                    results = results[:limit]
                    logger.info(f"✅ 向量检索：agent={agent_id}，返回 {len(results)} 条")
                    return results

            # ===== 混合检索（主路径）=====
            engine = HybridRetrievalEngine()
            hybrid_results = engine.retrieve(
                query_hexagram=query_hexagram,
                query_vector=query_vector,
                mysql_store=self.mysql_store,
                top_k=limit,
                agent_id=agent_id,
                bagua_filter=bagua_type,
                layer_filter=sancai_layer,
                min_hot_score=min_hot_score,
                structural_weight=structural_weight,
                semantic_weight=semantic_weight,
            )

            results = [r.to_dict() for r in hybrid_results]
            logger.info(
                f"✅ 混合检索：agent={agent_id}，top_k={limit}，"
                f"structural={structural_weight}，semantic={semantic_weight}"
            )
            return results

        except Exception as e:
            logger.error(f"❌ 记忆检索失败：{e}")
            raise StorageManagerError(f"记忆检索失败：{e}")

    # ========== 关联图谱查询 ==========

    def get_related_memories(
        self,
        memory_id: str,
        agent_id: str,
        top_k: int = 10,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        查询指定记忆的关联记忆

        Args:
            memory_id: 记忆 ID
            agent_id: Agent ID
            top_k: 返回数量上限
            min_confidence: 最低关联置信度

        Returns:
            关联记忆列表
        """
        try:
            results = get_graph_engine().get_related(
                memory_id=memory_id,
                mysql_store=self.mysql_store,
                agent_id=agent_id,
                top_k=top_k,
                min_confidence=min_confidence,
            )
            return [
                {
                    "memory_id": r.get("memory_id"),
                    "bagua_type": r.get("bagua_type"),
                    "sancai_layer": r.get("sancai_layer"),
                    "wuxing": r.get("wuxing"),
                    "content": r.get("content"),
                    "hot_score": r.get("hot_score"),
                    "relation_types": r.get("relation_types"),
                    "confidence": round(r.get("max_confidence", 0), 3),
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"❌ 关联查询失败：{e}")
            return []

    def get_all_agent_ids(self) -> List[str]:
        """
        获取所有有记忆的 Agent ID 列表（去重）
        用于定时任务多租户扫描

        Returns:
            agent_id 列表
        """
        try:
            return self.mysql_store.get_all_agent_ids()
        except Exception as e:
            logger.error(f"❌ 获取 Agent 列表失败：{e}")
            return []

    def get_memory(self, memory_id: str, agent_id: str = "") -> Optional[Dict]:
        """
        获取记忆（支持字符串 memory_id）

        Args:
            memory_id: 记忆 ID（字符串，如 "test_123"）
            agent_id: Agent ID（可选）

        Returns:
            记忆字典或 None
        """
        try:
            # 优先用字符串 memory_id 查询（不依赖自增 ID）
            memory = self.mysql_store.get_memory_by_memory_id(memory_id, agent_id)

            # 更新访问计数（内存数据库）
            if memory:
                short_term_memory = {
                    "id": memory.get("memory_id") or str(memory.get("id")),
                    "content": memory.get("content"),
                    "category": memory.get("bagua_type"),
                    "importance": memory.get("hot_score"),
                    "created_at": memory.get("created_at"),
                    "last_accessed": datetime.now().isoformat(),
                }
                self.memory_db.save_short_term(short_term_memory)

            return memory
        except Exception as e:
            logger.error(f"❌ 获取记忆失败：{str(e)}")
            return None

    # ========== 统计与分析 ==========

    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """
        获取 Agent 统计信息

        Args:
            agent_id: 租户 ID

        Returns:
            统计信息字典
        """
        try:
            # 从 MySQL 获取统计信息
            stats = self.mysql_store.get_agent_stats(agent_id)

            # 添加内存数据库统计
            memory_stats = self.memory_db.get_stats()
            stats["memory_db_stats"] = memory_stats

            # 添加向量数据库统计
            vector_stats = self.vector_db.get_stats()
            stats["vector_db_stats"] = vector_stats

            return stats
        except Exception as e:
            logger.error(f"❌ 获取统计信息失败：{e}")
            raise StorageManagerError(f"获取统计信息失败：{e}")

    def get_system_stats(self) -> Dict[str, Any]:
        """
        获取系统统计信息

        Returns:
            系统统计信息字典
        """
        try:
            stats = {
                "memory_db": self.memory_db.get_stats(),
                "vector_db": self.vector_db.get_stats(),
                "timestamp": datetime.now().isoformat(),
            }

            return stats
        except Exception as e:
            logger.error(f"❌ 获取系统统计信息失败：{e}")
            raise StorageManagerError(f"获取系统统计信息失败：{e}")

    # ========== 清理操作 ==========

    def clean_expired_memories(self, agent_id: Optional[str] = None) -> int:
        """
        清理过期记忆

        Args:
            agent_id: 可选，不传则清理所有

        Returns:
            清理数量
        """
        try:
            # 从 MySQL 清理
            count = self.mysql_store.clean_expired_memories(agent_id)

            logger.info(f"✅ 清理 {count} 条过期记忆成功")
            return count
        except Exception as e:
            logger.error(f"❌ 清理过期记忆失败：{e}")
            raise StorageManagerError(f"清理过期记忆失败：{e}")

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
            # 从向量数据库清理
            count = self.vector_db.cleanup_old_vectors(days, min_access_count)

            logger.info(f"✅ 清理 {count} 个旧向量成功")
            return count
        except Exception as e:
            logger.error(f"❌ 清理旧向量失败：{e}")
            raise StorageManagerError(f"清理旧向量失败：{e}")

    # ========== 工具方法 ==========

    def health_check(self) -> Dict[str, bool]:
        """
        健康检查

        Returns:
            各存储模块的健康状态
        """
        try:
            if not self._initialized:
                self.initialize()

            health = {
                "memory_db": True,  # SQLite 总是可用
                "mysql_store": True,  # MySQL 连接在初始化时已检查
                "vector_db": True,  # 向量数据库在初始化时已检查
            }

            return health
        except Exception as e:
            logger.error(f"❌ 健康检查失败：{e}")
            return {"memory_db": False, "mysql_store": False, "vector_db": False}

    def close(self):
        """
        关闭所有存储连接
        """
        try:
            if self._memory_db:
                self._memory_db.close()
            if self._vector_db:
                self._vector_db.close()

            logger.info("✅ 关闭所有存储连接成功")
        except Exception as e:
            logger.error(f"❌ 关闭存储连接失败：{e}")
            raise StorageManagerError(f"关闭存储连接失败：{e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 单例
_storage_manager = None
_storage_manager_lock = threading.RLock()


def get_storage_manager() -> StorageManager:
    """
    获取存储管理器实例

    Returns:
        StorageManager 实例
    """
    global _storage_manager
    with _storage_manager_lock:
        if _storage_manager is None:
            _storage_manager = StorageManager()
            _storage_manager.initialize()
    return _storage_manager


def get_memory_db() -> MemoryDatabase:
    """
    获取内存数据库实例

    Returns:
        MemoryDatabase 实例
    """
    return get_storage_manager().memory_db


def get_mysql_store() -> MySQLMemoryStore:
    """
    获取 MySQL 存储实例

    Returns:
        MySQLMemoryStore 实例
    """
    return get_storage_manager().mysql_store


def get_vector_db() -> VectorDatabase:
    """
    获取向量数据库实例

    Returns:
        VectorDatabase 实例
    """
    return get_storage_manager().vector_db
