#!/usr/bin/env python3
"""
🌟 I Ching V2.0 完整集成系统

整合 IMA V2.0 规范的核心功能：
- 六爻统一编码
- 五行调度引擎
- 自愈机制
- 多租户支持

这是 Alpha-Conscious-Agent 的 V2.0 完整版本
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from five_elements_scheduler import FiveElementsScheduler, SchedulerDecision
from yao_encoding import Element, Layer, MemoryEncoding, Trigram, YaoEncoder

logger = logging.getLogger(__name__)


# ========== V2.0 完整记忆系统 ==========


class IChingMemorySystemV2:
    """
    易经记忆系统 V2.0

    完整实现 IMA V2.0 规范的核心功能：
    1. 六爻统一编码（6 维向量）
    2. 五行调度引擎（相生相克）
    3. 自愈机制（时位校验）
    4. 多租户隔离（Agent ID）
    """

    def __init__(self, agent_id: str = "default"):
        """
        初始化记忆系统 V2.0

        Args:
            agent_id: Agent 标识（多租户隔离）
        """
        self.agent_id = agent_id
        self.encoder = YaoEncoder()
        self.scheduler = FiveElementsScheduler()

        # 内存存储（生产环境应使用 Redis/PG/Milvus）
        self.memories: Dict[str, MemoryEncoding] = {}
        self.yao_history: Dict[str, List[Dict]] = {}  # 爻变历史

        # 统计
        self.stats = {
            "total_memories": 0,
            "total_yao_changes": 0,
            "total_self_heals": 0,
            "forgotten_count": 0,
        }

        logger.info(f"✅ 易经记忆系统 V2.0 已初始化 (Agent ID: {agent_id})")

    def add_memory(
        self, content: str, trigram: str, layer: str, element: str, weight: float = 0.5, hot: float = 0.5
    ) -> MemoryEncoding:
        """
        添加新记忆（使用六爻统一编码）

        Args:
            content: 记忆内容
            trigram: 八卦分类（乾/坤/震/巽/坎/离/艮/兑）
            layer: 三才分层（天/人/地）
            element: 五行状态（木/火/土/金/水）
            weight: 重要性权重 (0-1)
            hot: 热度 (0-1)

        Returns:
            MemoryEncoding 编码后的记忆
        """
        # 使用六爻编码器创建记忆
        encoding = self.encoder.encode(
            content=content,
            trigram=Trigram(trigram),
            layer=Layer(layer),
            element=Element(element),
            weight=weight,
            hot=hot,
            agent_id=self.agent_id,
        )

        # 存储记忆
        self.memories[encoding.id] = encoding

        # 初始化爻变历史
        self.yao_history[encoding.id] = [
            {
                "timestamp": datetime.now().isoformat(),
                "action": "create",
                "hexagram": encoding.hexagram,
                "reason": "记忆创建",
            }
        ]

        self.stats["total_memories"] += 1

        logger.info(f"📝 新记忆已添加：{encoding.id} ({encoding.name})")
        return encoding

    def schedule_memories(self) -> List[SchedulerDecision]:
        """
        调度所有记忆（五行流转）

        Returns:
            调度决策列表
        """
        decisions = []

        for mem_id, memory in list(self.memories.items()):
            # 计算冗余度（简化版本）
            redundancy = self._calculate_redundancy(memory)

            # 执行调度
            decision = self.scheduler.schedule(memory, redundancy)

            # 应用决策
            updated = self.scheduler.apply_decision(memory, decision)

            # 如果状态变化，记录爻变历史
            if decision.old_state != decision.new_state:
                self._record_yao_change(mem_id, decision)

            # 更新记忆
            self.memories[mem_id] = updated
            decisions.append(decision)

        return decisions

    def _calculate_redundancy(self, memory: MemoryEncoding) -> float:
        """计算记忆冗余度"""
        # 简化版本：查找相似记忆
        similar_count = 0
        for other_id, other in self.memories.items():
            if other_id != memory.id:
                similarity = self.encoder.calculate_similarity(memory, other)
                if similarity > 0.8:  # 80% 相似度阈值
                    similar_count += 1

        # 冗余度 = 相似记忆数 / 总记忆数
        return min(1.0, similar_count / max(1, len(self.memories) - 1))

    def _record_yao_change(self, memory_id: str, decision: SchedulerDecision):
        """记录爻变历史"""
        if memory_id not in self.yao_history:
            self.yao_history[memory_id] = []

        self.yao_history[memory_id].append(
            {
                "timestamp": decision.timestamp,
                "action": decision.action,
                "old_state": decision.old_state,
                "new_state": decision.new_state,
                "reason": decision.reason,
                "weight_change": decision.weight_change,
                "hot_change": decision.hot_change,
            }
        )

        self.stats["total_yao_changes"] += 1

    def self_heal(self) -> List[Dict]:
        """
        自愈机制：时位校验 + 自动修复

        检查所有记忆的"当位/失位"状态，自动修复冲突

        Returns:
            修复记录列表
        """
        heal_records = []

        for mem_id, memory in list(self.memories.items()):
            issues = self._check_position(memory)

            if issues:
                # 执行修复
                repair_action = self._repair_memory(memory, issues)

                heal_records.append(
                    {
                        "memory_id": mem_id,
                        "issues": issues,
                        "repair_action": repair_action,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                self.stats["total_self_heals"] += 1

        return heal_records

    def _check_position(self, memory: MemoryEncoding) -> List[str]:
        """
        检查记忆的时位状态

        返回失位问题列表
        """
        issues = []

        # 规则 1：地层记忆不应有高热度和高权重
        if memory.layer == "地" and memory.hot > 0.8:
            issues.append("地层记忆热度过高（应≤0.8）")

        if memory.layer == "地" and memory.weight > 0.7:
            issues.append("地层记忆权重过高（应≤0.7）")

        # 规则 2：天层记忆不应有低权重
        if memory.layer == "天" and memory.weight < 0.3:
            issues.append("天层记忆权重过低（应≥0.3）")

        # 规则 3：过期记忆应进入闭藏状态
        if memory.expire_time:
            try:
                expire = datetime.strptime(memory.expire_time, "%Y-%m-%d %H:%M:%S")
                if datetime.now() > expire and memory.element != "水":
                    issues.append("记忆已过期但未进入闭藏状态")
            except (ValueError, TypeError):
                pass

        # 规则 4：五行状态与热度不匹配
        if memory.element == "火" and memory.hot < 0.5:
            issues.append("火状态记忆热度不足（应≥0.5）")

        if memory.element == "水" and memory.hot > 0.3:
            issues.append("水状态记忆热度过高（应≤0.3）")

        return issues

    def _repair_memory(self, memory: MemoryEncoding, issues: List[str]) -> str:
        """
        修复记忆问题

        Returns:
            修复动作描述
        """
        repairs = []

        for issue in issues:
            if "地层记忆热度过高" in issue:
                memory.hot = min(memory.hot, 0.6)
                repairs.append("降低地层记忆热度")

            if "地层记忆权重过高" in issue:
                memory.weight = min(memory.weight, 0.5)
                repairs.append("降低地层记忆权重")

            if "天层记忆权重过低" in issue:
                memory.weight = max(memory.weight, 0.5)
                repairs.append("提升天层记忆权重")

            if "已过期但未进入闭藏" in issue:
                memory.element = "水"
                memory.hot = min(memory.hot, 0.2)
                repairs.append("转入闭藏状态")

            if "火状态记忆热度不足" in issue:
                memory.element = "土"  # 转为稳定状态
                repairs.append("转为土状态")

            if "水状态记忆热度过高" in issue:
                memory.hot = min(memory.hot, 0.2)
                repairs.append("降低水状态热度")

        # 记录修复到爻变历史
        if repairs:
            self._record_yao_change(
                memory.id,
                SchedulerDecision(
                    memory_id=memory.id,
                    old_state=memory.element,
                    new_state=memory.element,
                    action="self_heal",
                    reason="; ".join(repairs),
                    weight_change=0.0,
                    hot_change=0.0,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )

        return "; ".join(repairs) if repairs else "无需修复"

    def get_yao_history(self, memory_id: str) -> List[Dict]:
        """获取记忆的爻变历史"""
        return self.yao_history.get(memory_id, [])

    def get_memory(self, memory_id: str) -> Optional[MemoryEncoding]:
        """获取记忆"""
        return self.memories.get(memory_id)

    def search_memories(
        self,
        trigram: Optional[str] = None,
        layer: Optional[str] = None,
        element: Optional[str] = None,
        min_hot: float = 0.0,
    ) -> List[MemoryEncoding]:
        """
        搜索记忆（支持多维度筛选）

        Args:
            trigram: 八卦筛选
            layer: 三才层筛选
            element: 五行筛选
            min_hot: 最小热度

        Returns:
            匹配的记忆列表
        """
        results = []

        for mem in self.memories.values():
            if trigram and mem.trigram != trigram:
                continue
            if layer and mem.layer != layer:
                continue
            if element and mem.element != element:
                continue
            if mem.hot < min_hot:
                continue
            results.append(mem)

        return results

    def forget_old_memories(self, age_days: int = 365) -> int:
        """
        遗忘老旧记忆

        Args:
            age_days: 年龄阈值（天）

        Returns:
            遗忘的记忆数量
        """
        forgotten = 0

        for mem_id, memory in list(self.memories.items()):
            age = self.scheduler._calculate_age_days(memory.timestamp)

            if age > age_days and memory.hot < 0.2:
                # 遗忘：移除记忆
                del self.memories[mem_id]
                forgotten += 1
                self.stats["forgotten_count"] += 1

        logger.info(f"🍂 已遗忘 {forgotten} 个老旧记忆")
        return forgotten

    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        # 五行分布
        element_dist = {}
        layer_dist = {}

        for mem in self.memories.values():
            element_dist[mem.element] = element_dist.get(mem.element, 0) + 1
            layer_dist[mem.layer] = layer_dist.get(mem.layer, 0) + 1

        # 计算五行能量场
        field = self.scheduler.calculate_field(list(self.memories.values()))

        return {
            **self.stats,
            "total_memories": len(self.memories),
            "element_distribution": element_dist,
            "layer_distribution": layer_dist,
            "five_element_field": field.to_dict(),
        }

    def export_memories(self) -> str:
        """导出所有记忆为 JSON"""
        data = {
            "agent_id": self.agent_id,
            "export_time": datetime.now().isoformat(),
            "stats": self.get_stats(),
            "memories": [m.to_dict() for m in self.memories.values()],
            "yao_history": self.yao_history,
        }
        return json.dumps(data, ensure_ascii=False, indent=2)


# ========== 便捷函数 ==========


def create_memory_system(agent_id: str = "default") -> IChingMemorySystemV2:
    """创建记忆系统 V2.0 实例"""
    return IChingMemorySystemV2(agent_id)


# ========== 命令行测试 ==========

if __name__ == "__main__":
    print("=" * 80)
    print("🌟 易经记忆系统 V2.0 完整测试")
    print("=" * 80)

    # 创建系统
    system = IChingMemorySystemV2(agent_id="alpha-001")

    # 测试 1：添加记忆
    print("\n【测试 1】添加记忆（六爻编码）")
    mem1 = system.add_memory(
        content="用户核心需求：需要 7 天无理由退货", trigram="乾", layer="天", element="金", weight=0.95, hot=0.8
    )
    print(f"ID: {mem1.id}")
    print(f"卦名：{mem1.name}")
    print(f"六爻：{mem1.hexagram}")

    mem2 = system.add_memory(
        content="临时对话：用户询问价格", trigram="巽", layer="地", element="木", weight=0.3, hot=0.9
    )
    print(f"ID: {mem2.id}")
    print(f"卦名：{mem2.name}")

    # 测试 2：搜索记忆
    print("\n【测试 2】搜索记忆")
    results = system.search_memories(layer="天")
    print(f"天层记忆数：{len(results)}")

    results = system.search_memories(min_hot=0.5)
    print(f"热度>0.5 的记忆数：{len(results)}")

    # 测试 3：五行调度
    print("\n【测试 3】五行调度")
    decisions = system.schedule_memories()
    print(f"调度决策数：{len(decisions)}")
    for d in decisions[:3]:
        print(f"  - {d.memory_id[:20]}... : {d.action} ({d.reason})")

    # 测试 4：自愈机制
    print("\n【测试 4】自愈机制")
    # 先制造一个问题记忆
    problem_mem = system.add_memory(
        content="问题记忆", trigram="坤", layer="地", element="土", weight=0.9, hot=0.95  # 地层权重过高  # 地层热度过高
    )

    heal_records = system.self_heal()
    print(f"修复记录数：{len(heal_records)}")
    for record in heal_records:
        print(f"  - {record['memory_id'][:20]}...")
        print(f"    问题：{record['issues']}")
        print(f"    修复：{record['repair_action']}")

    # 测试 5：爻变历史
    print("\n【测试 5】爻变历史")
    history = system.get_yao_history(problem_mem.id)
    print(f"爻变历史数：{len(history)}")
    for h in history[:3]:
        print(f"  - {h['timestamp']}: {h['action']} - {h.get('reason', '')}")

    # 测试 6：系统统计
    print("\n【测试 6】系统统计")
    stats = system.get_stats()
    print(f"总记忆数：{stats['total_memories']}")
    print(f"爻变次数：{stats['total_yao_changes']}")
    print(f"自愈次数：{stats['total_self_heals']}")
    print(f"五行分布：{stats['element_distribution']}")
    print(f"三才分布：{stats['layer_distribution']}")
    print(f"五行能量场平衡度：{stats['five_element_field']['balance_score']:.2%}")

    # 测试 7：导出
    print("\n【测试 7】导出记忆")
    json_export = system.export_memories()
    print(f"导出 JSON 长度：{len(json_export)} 字符")

    print("\n" + "=" * 80)
    print("✅ 易经记忆系统 V2.0 所有测试完成！")
    print("=" * 80)
