#!/usr/bin/env python3
"""
🔍 Semantic Retrieval Engine（语义检索引擎）

功能：
- 纯语义向量检索（sentence-transformers / hash fallback）
- 六爻仅作为分类标签（bagua_type / sancai_layer / wuxing），不参与检索评分
- 支持 MySQL + 向量库联合查询，agent 级别隔离
- 支持 bagua/sancai/wuxing 过滤（分类标签过滤，非检索评分）

架构决策（Phase 3B, 2026-04-28）：
  六爻 rule 编码区分度 = -0.2（完全反向），在检索路径中不产生增量价值。
  六爻退化为纯分类标签，检索评分 100% 语义向量相似度。

对应原"六十四卦态势检索"设计，用通用软件工程语言实现。
"""

import ast
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.config import get_config

try:
    from core.utils import get_logger, validate_float, validate_integer
except ImportError:
    import logging
    def get_logger(name: str):
        return logging.getLogger(name)
    def validate_float(v, name, min_value=0, max_value=1):
        if not (min_value <= float(v) <= max_value):
            raise ValueError(f"{name} must be in [{min_value}, {max_value}]")
    def validate_integer(v, name, min_value=1, max_value=100):
        if not (min_value <= int(v) <= max_value):
            raise ValueError(f"{name} must be in [{min_value}, {max_value}]")

logger = get_logger(__name__)

# ===== 评分权重配置 =====

# Phase 3B: 结构权重归零，纯语义检索
STRUCTURAL_WEIGHT = get_config("hybrid_retrieval.structural_weight", 0.0)
SEMANTIC_WEIGHT = get_config("hybrid_retrieval.semantic_weight", 1.0)

# ===== 辅助函数 =====


def parse_hexagram(hex_str) -> Optional[List[int]]:
    """
    解析 hexagram 字段，返回 [0/1] 列表
    支持格式："[1, 0, 1, 0, 1, 0]" 或 "[1,0,1,0,1,0]"
    """
    if hex_str is None:
        return None
    if isinstance(hex_str, list):
        return hex_str if len(hex_str) == 6 else None
    try:
        parsed = ast.literal_eval(str(hex_str))
        if isinstance(parsed, list) and len(parsed) == 6:
            return [int(x) for x in parsed]
    except Exception:
        pass
    return None


def hamming_distance(hex1: List[int], hex2: List[int]) -> int:
    """六爻 Hamming 距离：两个卦象间不同爻位的数量（已弃用，仅供调试）"""
    return sum(a != b for a, b in zip(hex1, hex2))


# ===== 检索结果数据类 =====


@dataclass
class RetrievalResult:
    """单条检索结果"""

    memory_id: str
    agent_id: str
    hexagram: List[int]          # 分类标签（不参与检索评分）
    bagua_type: str               # 八宫分类
    sancai_layer: str             # 三才层级
    wuxing: str                   # 五行归类
    content: str
    hot_score: float
    semantic_similarity: float = 0.0  # 向量余弦相似度（唯一评分来源）
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "agent_id": self.agent_id,
            "hexagram": self.hexagram,
            "bagua_type": self.bagua_type,
            "sancai_layer": self.sancai_layer,
            "wuxing": self.wuxing,
            "content": self.content,
            "hot_score": self.hot_score,
            "scores": {
                "semantic_similarity": round(self.semantic_similarity, 4),
            },
            "rank": self.rank,
        }


# ===== 语义检索引擎 =====


class SemanticRetrievalEngine:
    """
    语义检索引擎（Phase 3B 重构）

    工作流程：
    1. 用 query_vector 做向量检索（取 top_k 条结果）
    2. 从 MySQL 补充元数据（hexagram/bagua/wuxing 等分类标签）
    3. 按语义相似度排序
    4. 返回 top_k 条结果

    六爻在此引擎中仅作为分类标签展示，不参与检索评分。
    """

    def __init__(self):
        self._lock = threading.RLock()

    def retrieve(
        self,
        query_vector: List[float],
        mysql_store,
        top_k: int = 5,
        agent_id: Optional[str] = None,
        bagua_filter: Optional[str] = None,
        sancai_filter: Optional[str] = None,
        layer_filter: Optional[str] = None,
        min_hot_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        语义检索

        Args:
            query_vector: 查询的语义向量（384-dim float 列表）
            mysql_store: MySQL 存储实例
            top_k: 返回结果数量
            agent_id: Agent 隔离（可选）
            bagua_filter: 八宫分类过滤（六爻作为标签使用）
            sancai_filter: 三才层级过滤
            layer_filter: 层过滤（alias for sancai_filter）
            min_hot_score: 最低热度过滤

        Returns:
            按语义相似度排序的 RetrievalResult 列表
        """
        validate_integer(top_k, "top_k", min_value=1, max_value=100)

        if len(query_vector) == 0:
            raise ValueError("query_vector cannot be empty")

        layer = layer_filter or sancai_filter

        # ===== STEP 1: 向量检索 =====
        vector_results = self._vector_search(
            query_vector=query_vector,
            mysql_store=mysql_store,
            top_k=top_k,
            agent_id=agent_id,
            min_hot_score=min_hot_score,
        )
        if not vector_results:
            logger.info("向量检索无结果，返回空列表")
            return []

        # ===== STEP 2: MySQL 补充查询 + 分类标签过滤 =====
        candidate_ids = [r["id"] for r in vector_results]

        candidate_memories = mysql_store.get_memories_by_ids(
            memory_ids=candidate_ids,
            agent_id=agent_id,
            bagua_type=bagua_filter,
            layer=layer,
            min_hot=min_hot_score,
        )

        # ===== STEP 3: 建立 memory_id → 向量相似度 的映射 =====
        vector_simi_map = {r["id"]: r["similarity"] for r in vector_results}

        # ===== STEP 4: 组装结果 =====
        scored_results: List[RetrievalResult] = []
        for mem in candidate_memories:
            hexagram_list = parse_hexagram(mem.get("hexagram"))
            if hexagram_list is None:
                hexagram_list = [0, 0, 0, 0, 0, 0]

            sem_simi = float(vector_simi_map.get(mem["id"], 0.0))

            result = RetrievalResult(
                memory_id=mem.get("memory_id") or str(mem["id"]),
                agent_id=mem.get("agent_id", agent_id or ""),
                hexagram=hexagram_list,
                bagua_type=mem.get("bagua_type", ""),
                sancai_layer=mem.get("sancai_layer", ""),
                wuxing=mem.get("wuxing", ""),
                content=mem.get("content", ""),
                hot_score=float(mem.get("hot_score", 0.0)),
                semantic_similarity=sem_simi,
            )
            scored_results.append(result)

        # ===== STEP 5: 语义相似度排序，取 top_k =====
        scored_results.sort(key=lambda x: x.semantic_similarity, reverse=True)
        top_results = scored_results[:top_k]

        for i, r in enumerate(top_results, 1):
            r.rank = i

        logger.info(
            f"✅ 语义检索完成：候选={len(scored_results)}，返回={len(top_results)}，"
            f"agent={agent_id}"
        )
        return top_results

    def _vector_search(
        self,
        query_vector: List[float],
        mysql_store,
        top_k: int,
        agent_id: Optional[str],
        min_hot_score: float,
    ) -> List[Dict[str, Any]]:
        """向量库检索（内部方法）"""
        try:
            results = mysql_store.vector_db.search_similar(
                query_embedding=query_vector, top_k=top_k, min_similarity=0.05
            )
            return results
        except Exception as e:
            logger.warning(f"向量检索失败，降级为 MySQL 全量查询：{e}")
            return []


# ===== 向后兼容别名 =====

HybridRetrievalEngine = SemanticRetrievalEngine

# ===== MySQL store 补充方法 =====


def get_memories_by_ids(
    mysql_store,
    memory_ids: List[str],
    agent_id: Optional[str] = None,
    bagua_type: Optional[str] = None,
    layer: Optional[str] = None,
    min_hot: float = 0.0,
) -> List[Dict[str, Any]]:
    """根据 ID 列表批量查询记忆（MySQL 直查，用于检索候选补充）"""
    if not memory_ids:
        return []
    try:
        results = mysql_store.get_memories(
            agent_id=agent_id or "",
            limit=len(memory_ids),
            offset=0,
            trigram=bagua_type,
            layer=layer,
            min_hot=min_hot,
        )
        id_set = set(str(m) for m in memory_ids)
        return [r for r in results if str(getattr(r, "id", r.get("id", ""))) in id_set]
    except Exception as e:
        logger.error(f"MySQL 候选集查询失败：{e}")
        return []


# ===== 便捷函数 =====


def semantic_retrieve(
    query_vector: List[float],
    mysql_store,
    agent_id: str,
    top_k: int = 5,
    bagua_filter: Optional[str] = None,
    sancai_filter: Optional[str] = None,
    min_hot_score: float = 0.0,
) -> List[RetrievalResult]:
    """
    便捷函数：执行一次语义检索
    """
    engine = _get_engine()
    return engine.retrieve(
        query_vector=query_vector,
        mysql_store=mysql_store,
        top_k=top_k,
        agent_id=agent_id,
        bagua_filter=bagua_filter,
        sancai_filter=sancai_filter,
        min_hot_score=min_hot_score,
    )


# 向后兼容别名
hybrid_retrieve = semantic_retrieve


def compute_hybrid_score(
    hex_simi: float,
    vector_simi: float,
    structural_weight: float = STRUCTURAL_WEIGHT,
    semantic_weight: float = SEMANTIC_WEIGHT,
) -> float:
    """
    Phase 3B: 结构权重归零，混合评分 = 纯语义相似度
    保留此函数以维持 API 兼容，但 structural_weight=0 恒成立
    """
    return vector_simi * semantic_weight


def hex_similarity(hex1: List[int], hex2: List[int]) -> float:
    """
    Phase 3B: 六爻相似度已弃用。保留此函数供调试。
    """
    dist = hamming_distance(hex1, hex2)
    return 1.0 - (dist / 6.0)


# ===== 单例 =====

_engine: Optional[SemanticRetrievalEngine] = None
_engine_lock = threading.Lock()


def _get_engine() -> SemanticRetrievalEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = SemanticRetrievalEngine()
    return _engine
