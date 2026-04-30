#!/usr/bin/env python3
"""
🕸️ Memory Graph（记忆关联图谱）

功能：
- 基于"天然卦对"关系自动建立记忆间的关联链接
- 存储时自动触发关联写入（relations 字段）
- 提供关联查询接口
- 关联类型：NATURAL_PAIR（天生卦对）/ SAME_BAGUA（同卦象）/ SAME_LAYER（同层级）/ CROSS_REFERENCE（互引）

对应原"卦变关联"设计，以通用软件工程语言实现。
"""

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from core.config import get_config
from core.utils import get_logger
logger = get_logger(__name__)


# ===== 关联类型 =====


class RelationType(Enum):
    """关联类型"""

    NATURAL_PAIR = "natural_pair"  # 天然卦对（如 乾↔坤、泰↔否）
    SAME_BAGUA = "same_bagua"  # 同卦象（同属乾/坤/震...）
    SAME_LAYER = "same_layer"  # 同三才层级
    SAME_WUXING = "same_wuxing"  # 同五行属性
    SEMANTIC_SIM = "semantic_sim"  # 语义关联（内容相似度高）


# ===== 天然卦对定义 =====

# 互为关联的卦象对（对立/转化关系）
# 来源：易经卦象的对卦/错卦/综卦关系简化为最核心的互译对
NATURAL_HEXAGRAM_PAIRS = {
    # 对立卦（阴阳完全相反）
    ("qian", "kun"),  # 乾为天 ↔ 坤为地
    # 错卦对（各爻皆反）
    ("qian", "kun"),
    ("zhen", "xun"),  # 震 ↔ 巽
    ("kan", "li"),  # 坎 ↔ 离
    ("gen", "dui"),  # 艮 ↔ 兑
    # 核心转化对（泰与否、损益、既济与未济）
    ("kun", "qian"),  # 泰 ↔ 否（坤↔乾 同上归一）
    ("li", "kan"),  # 既济 ↔ 未济（离↔坎）
}

# 建立 pair → True 的快速查找
_NATURAL_PAIR_SET = {frozenset({a, b}) for a, b in NATURAL_HEXAGRAM_PAIRS}


def are_natural_pair(bagua_a: str, bagua_b: str) -> bool:
    """判断两个卦象是否为天然关联对"""
    return frozenset({bagua_a, bagua_b}) in _NATURAL_PAIR_SET


# ===== 同组卦象分类 =====

# 同一象限的卦象（功能相近）
BAGUA_GROUPS = {
    "core": ["qian", "kun"],  # 核心层
    "event": ["zhen", "xun"],  # 事件/触发
    "perception": ["kan", "li"],  # 感知/显现
    "foundation": ["gen", "dui"],  # 基础/交互
}


def get_bagua_group(bagua: str) -> Optional[str]:
    """获取卦象所属象限组"""
    for group_name, members in BAGUA_GROUPS.items():
        if bagua in members:
            return group_name
    return None


# ===== 关联结果数据类 =====


@dataclass
class RelationLink:
    """单条关联记录"""

    from_memory_id: str
    to_memory_id: str
    relation_type: RelationType
    confidence: float  # 关联置信度 0.0-1.0
    reason: str  # 关联原因描述


@dataclass
class RelatedMemory:
    """关联记忆（含关联元信息）"""

    memory_id: str
    bagua_type: str
    sancai_layer: str
    wuxing: str
    content: str
    hot_score: float
    relation_types: List[str] = field(default_factory=list)
    max_confidence: float = 0.0


# ===== 核心关联引擎 =====


class MemoryGraphEngine:
    """
    记忆关联图谱引擎

    工作流程：
    1. 存储触发：当新记忆存储时，自动分析并写入关联
    2. 关联发现：根据 NATURAL_PAIR / SAME_BAGUA / SAME_LAYER / SAME_WUXING 四种规则发现关联
    3. 关联写入：将关联结果写入 relations JSON 字段（双向写入）
    4. 关联查询：给定记忆，返回其所有关联记忆
    """

    # 关联规则置信度（用于混合检索排序加成）
    CONFIDENCE_WEIGHTS = {
        RelationType.NATURAL_PAIR: 0.9,
        RelationType.SAME_BAGUA: 0.7,
        RelationType.SAME_LAYER: 0.5,
        RelationType.SAME_WUXING: 0.4,
        RelationType.SEMANTIC_SIM: 0.3,
    }

    # 各规则在 relations 字段中的前缀（用于标识来源）
    TYPE_PREFIX = {
        RelationType.NATURAL_PAIR: "NP:",
        RelationType.SAME_BAGUA: "SB:",
        RelationType.SAME_LAYER: "SL:",
        RelationType.SAME_WUXING: "SW:",
        RelationType.SEMANTIC_SIM: "SS:",
    }

    def __init__(self):
        self._lock = threading.RLock()
        self._relation_cache: Dict[str, List[str]] = {}
        self._cache_max_size = get_config("memory_graph.cache_size", 5000)

    # ===== 关联发现 =====

    def discover_relations(
        self,
        new_memory: Dict[str, Any],
        candidate_memories: List[Dict[str, Any]],
        max_relations_per_memory: int = 10,
    ) -> List[RelationLink]:
        """
        给定一条新记忆，从候选集中发现关联

        Args:
            new_memory: 新记忆数据（需包含 bagua_type / sancai_layer / wuxing / memory_id）
            candidate_memories: 候选记忆列表
            max_relations_per_memory: 每条记忆最多保留的关联数

        Returns:
            RelationLink 列表
        """
        with self._lock:
            if not candidate_memories:
                return []

            new_bagua = new_memory.get("bagua_type", "")
            new_layer = new_memory.get("sancai_layer", "")
            new_wuxing = new_memory.get("wuxing", "")
            new_id = str(new_memory.get("memory_id") or new_memory.get("id", ""))

            links: List[RelationLink] = []

            for cand in candidate_memories:
                cand_id = str(cand.get("memory_id") or cand.get("id", ""))
                if cand_id == new_id:
                    continue

                cand_bagua = cand.get("bagua_type", "")
                cand_layer = cand.get("sancai_layer", "")
                cand_wuxing = cand.get("wuxing", "")

                # NATURAL_PAIR（最高优先级）
                if new_bagua and cand_bagua and are_natural_pair(new_bagua, cand_bagua):
                    links.append(
                        RelationLink(
                            from_memory_id=new_id,
                            to_memory_id=cand_id,
                            relation_type=RelationType.NATURAL_PAIR,
                            confidence=self.CONFIDENCE_WEIGHTS[RelationType.NATURAL_PAIR],
                            reason=f"天然卦对关联：{new_bagua}↔{cand_bagua}",
                        )
                    )
                    continue  # 天然对已够强，跳过后续

                # SAME_BAGUA（同组卦象）
                if new_bagua and cand_bagua == new_bagua:
                    links.append(
                        RelationLink(
                            from_memory_id=new_id,
                            to_memory_id=cand_id,
                            relation_type=RelationType.SAME_BAGUA,
                            confidence=self.CONFIDENCE_WEIGHTS[RelationType.SAME_BAGUA],
                            reason=f"同卦象关联：{new_bagua}",
                        )
                    )

                # SAME_LAYER（同层级）
                if new_layer and cand_layer == new_layer:
                    links.append(
                        RelationLink(
                            from_memory_id=new_id,
                            to_memory_id=cand_id,
                            relation_type=RelationType.SAME_LAYER,
                            confidence=self.CONFIDENCE_WEIGHTS[RelationType.SAME_LAYER],
                            reason=f"同层级关联：{new_layer}",
                        )
                    )

                # SAME_WUXING（同五行）
                if new_wuxing and cand_wuxing == new_wuxing:
                    links.append(
                        RelationLink(
                            from_memory_id=new_id,
                            to_memory_id=cand_id,
                            relation_type=RelationType.SAME_WUXING,
                            confidence=self.CONFIDENCE_WEIGHTS[RelationType.SAME_WUXING],
                            reason=f"同五行关联：{new_wuxing}",
                        )
                    )

            # 按置信度截断
            links.sort(key=lambda x: x.confidence, reverse=True)
            return links[:max_relations_per_memory]

    # ===== 关联写入 =====

    def encode_relation_tags(self, links: List[RelationLink]) -> List[str]:
        """
        将 RelationLink 列表编码为 relations JSON 字符串列表
        格式："TYPE:memory_id:confidence:reason"
        """
        return [
            f"{self.TYPE_PREFIX[link.relation_type]}{link.to_memory_id}:" f"{link.confidence:.2f}:{link.reason}"
            for link in links
        ]

    def decode_relation_tags(self, tags: List[str]) -> List[RelationLink]:
        """解析 relations JSON 字符串为 RelationLink 列表"""
        links = []
        for tag in tags:
            if not tag or ":" not in tag:
                continue
            parts = tag.split(":", 3)
            if len(parts) < 2:
                continue
            type_prefix = parts[0]
            to_id = parts[1]
            confidence = float(parts[2]) if len(parts) > 2 and parts[2] else 0.5
            reason = parts[3] if len(parts) > 3 else ""

            # 反查 relation_type
            rel_type = None
            for rt, prefix in self.TYPE_PREFIX.items():
                if type_prefix == prefix:
                    rel_type = rt
                    break
            if rel_type is None:
                continue

            links.append(
                RelationLink(
                    from_memory_id="",
                    to_memory_id=to_id,
                    relation_type=rel_type,
                    confidence=confidence,
                    reason=reason,
                )
            )
        return links

    # ===== 关联查询 =====

    def get_related(
        self,
        memory_id: str,
        mysql_store,
        agent_id: Optional[str] = None,
        top_k: int = 10,
        min_confidence: float = 0.0,
    ) -> List[RelatedMemory]:
        """
        查询指定记忆的所有关联记忆

        Args:
            memory_id: 记忆 ID
            mysql_store: MySQL 存储实例
            agent_id: Agent ID（用于补充查询）
            top_k: 返回数量上限
            min_confidence: 最低置信度过滤

        Returns:
            RelatedMemory 列表（按置信度降序）
        """
        with self._lock:
            try:
                # 从 MySQL 读取该记忆的 relations 字段（使用字符串 memory_id）
                memory = mysql_store.get_memory_by_memory_id(memory_id, agent_id or "")
                if not memory:
                    return []

                # 支持 dict 或对象
                if hasattr(memory, "relations"):
                    rel_tags = memory.relations or []
                elif isinstance(memory, dict):
                    rel_tags = memory.get("relations", [])
                else:
                    rel_tags = []

                if not rel_tags:
                    return []

                # 解析关联标签
                links = self.decode_relation_tags(rel_tags)
                if not links:
                    return []

                # 过滤置信度
                links = [l for l in links if l.confidence >= min_confidence]

                # 按 memory_id 去重，保留最高置信度
                best_link: Dict[str, RelationLink] = {}
                for link in links:
                    key = link.to_memory_id
                    if key not in best_link or link.confidence > best_link[key].confidence:
                        best_link[key] = link

                # 批量查询关联记忆的详情
                related_ids = list(best_link.keys())
                if not related_ids:
                    return []

                related_records = mysql_store.get_memories_by_ids(
                    memory_ids=related_ids,
                    agent_id=agent_id,
                )

                # 构建 RelatedMemory 对象
                results: List[RelatedMemory] = []
                for rec in related_records:
                    rec_id = str(rec.get("memory_id") or rec.get("id", ""))
                    if rec_id not in best_link:
                        continue
                    link = best_link[rec_id]
                    results.append(
                        RelatedMemory(
                            memory_id=rec_id,
                            bagua_type=rec.get("bagua_type", ""),
                            sancai_layer=rec.get("sancai_layer", ""),
                            wuxing=rec.get("wuxing", ""),
                            content=rec.get("content", ""),
                            hot_score=float(rec.get("hot_score", 0.0)),
                            relation_types=[link.relation_type.value],
                            max_confidence=link.confidence,
                        )
                    )

                # 按置信度排序
                results.sort(key=lambda x: x.max_confidence, reverse=True)
                return results[:top_k]

            except Exception as e:
                logger.error(f"关联查询失败：{e}")
                return []

    # ===== 缓存管理 =====

    def invalidate_cache(self, memory_id: str = None):
        """清除缓存"""
        with self._lock:
            if memory_id and memory_id in self._relation_cache:
                del self._relation_cache[memory_id]
            elif memory_id is None:
                self._relation_cache.clear()


# ===== 单例 =====

_graph_engine: Optional[MemoryGraphEngine] = None
_graph_lock = threading.Lock()


def get_graph_engine() -> MemoryGraphEngine:
    global _graph_engine
    if _graph_engine is None:
        with _graph_lock:
            if _graph_engine is None:
                _graph_engine = MemoryGraphEngine()
    return _graph_engine


# ===== 便捷函数 =====


def discover_memory_relations(
    new_memory: Dict[str, Any],
    candidate_memories: List[Dict[str, Any]],
    max_relations: int = 10,
) -> List[RelationLink]:
    """发现并返回关联列表（供存储时调用）"""
    return get_graph_engine().discover_relations(new_memory, candidate_memories, max_relations)


def get_related_memories(
    memory_id: str,
    mysql_store,
    agent_id: Optional[str] = None,
    top_k: int = 10,
) -> List[RelatedMemory]:
    """查询关联记忆"""
    return get_graph_engine().get_related(memory_id, mysql_store, agent_id, top_k)
