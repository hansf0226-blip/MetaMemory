#!/usr/bin/env python3
"""
🔄 五行调度引擎 - Five Elements Scheduler

基于 IMA V2.0 规范的五行生克调控引擎：
- 五行状态：木 (生成)→火 (活跃)→土 (稳定)→金 (精炼)→水 (闭藏)
- 相生调控：木生火→火生土→土生金→金生水→水生木
- 相克调控：木克土→土克水→水克火→火克金→金克木
- 动态平衡：负反馈调节，防止记忆过度膨胀/遗忘

核心算法：
1. 根据热度、时间、冗余度判断五行状态
2. 应用相生规则进行正向流转
3. 应用相克规则进行负反馈调节
4. 触发爻变/卦变更新记忆状态
"""

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple

from yao_encoding import Element, Layer, MemoryEncoding, Trigram

logger = logging.getLogger(__name__)


# ========== 五行状态枚举 ==========


class FiveElementState(Enum):
    """五行状态"""

    MU = "木"  # 生成期 - 新记忆
    HUO = "火"  # 活跃期 - 高频访问
    TU = "土"  # 稳定期 - 中期沉淀
    JIN = "金"  # 精炼期 - 去冗余
    SHUI = "水"  # 闭藏期 - 冷存储/遗忘


# ========== 相生相克关系 ==========

# 相生关系：A→B 表示 A 生 B
GENERATION_CYCLE = {
    FiveElementState.MU: FiveElementState.HUO,  # 木生火
    FiveElementState.HUO: FiveElementState.TU,  # 火生土
    FiveElementState.TU: FiveElementState.JIN,  # 土生金
    FiveElementState.JIN: FiveElementState.SHUI,  # 金生水
    FiveElementState.SHUI: FiveElementState.MU,  # 水生木
}

# 相克关系：A→B 表示 A 克 B
RESTRAINT_CYCLE = {
    FiveElementState.MU: FiveElementState.TU,  # 木克土
    FiveElementState.TU: FiveElementState.SHUI,  # 土克水
    FiveElementState.SHUI: FiveElementState.HUO,  # 水克火
    FiveElementState.HUO: FiveElementState.JIN,  # 火克金
    FiveElementState.JIN: FiveElementState.MU,  # 金克木
}

# 五行状态触发条件
STATE_THRESHOLDS = {
    FiveElementState.MU: {"hot_min": 0.0, "hot_max": 0.5, "age_days_max": 1, "description": "生成期 - 新输入信息"},
    FiveElementState.HUO: {"hot_min": 0.8, "hot_max": 1.0, "age_days_max": 1, "description": "活跃期 - 高频访问"},
    FiveElementState.TU: {
        "hot_min": 0.3,
        "hot_max": 0.7,
        "age_days_min": 1,
        "age_days_max": 30,
        "description": "稳定期 - 中期沉淀",
    },
    FiveElementState.JIN: {
        "hot_min": 0.0,
        "hot_max": 0.3,
        "age_days_min": 7,
        "redundancy_max": 0.7,
        "description": "精炼期 - 去冗余",
    },
    FiveElementState.SHUI: {"hot_min": 0.0, "hot_max": 0.2, "age_days_min": 30, "description": "闭藏期 - 冷存储"},
}


# ========== 数据结构 ==========


@dataclass
class SchedulerDecision:
    """调度决策结果"""

    memory_id: str
    old_state: str
    new_state: str
    action: str  # "generate", "activate", "stabilize", "refine", "hide", "forget"
    reason: str
    weight_change: float
    hot_change: float
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FiveElementField:
    """五行能量场"""

    mu_energy: float  # 木能量 - 生成力
    huo_energy: float  # 火能量 - 活跃度
    tu_energy: float  # 土能量 - 稳定度
    jin_energy: float  # 金能量 - 精炼度
    shui_energy: float  # 水能量 - 闭藏度
    balance_score: float  # 平衡度 (0-1)

    def to_dict(self) -> Dict:
        return asdict(self)


# ========== 五行调度引擎 ==========


class FiveElementsScheduler:
    """
    五行调度引擎

    核心功能：
    1. 状态判断：根据热度/时间/冗余度判断当前五行状态
    2. 相生流转：正向能量流转（生成→活跃→稳定→精炼→闭藏）
    3. 相克调控：负反馈调节（防止过度）
    4. 决策生成：生成调度决策（更新/转移/遗忘）
    """

    def __init__(self):
        self.decision_history: List[SchedulerDecision] = []
        self.stats = {
            "total_scheduled": 0,
            "state_changes": 0,
            "forgotten": 0,
            "refined": 0,
        }

    def schedule(self, memory: MemoryEncoding, redundancy: float = 0.0) -> SchedulerDecision:
        """
        调度单个记忆

        Args:
            memory: 记忆编码
            redundancy: 冗余度 (0-1)

        Returns:
            SchedulerDecision 决策结果
        """
        self.stats["total_scheduled"] += 1

        # 获取当前状态
        current_state = FiveElementState(memory.element)

        # 计算记忆年龄（天）
        age_days = self._calculate_age_days(memory.timestamp)

        # 获取热度
        hot = memory.hot

        # 判断目标状态
        target_state = self._determine_state(hot, age_days, redundancy)

        # 生成调度决策
        decision = self._generate_decision(memory, current_state, target_state, redundancy)

        # 记录决策
        self.decision_history.append(decision)

        # 更新统计
        if current_state != target_state:
            self.stats["state_changes"] += 1
        if decision.action == "forget":
            self.stats["forgotten"] += 1
        elif decision.action == "refine":
            self.stats["refined"] += 1

        return decision

    def _calculate_age_days(self, timestamp: str) -> float:
        """计算记忆年龄（天）"""
        try:
            create_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            delta = datetime.now() - create_time
            return delta.total_seconds() / 86400
        except (ValueError, TypeError):
            return 0

    def _determine_state(self, hot: float, age_days: float, redundancy: float) -> FiveElementState:
        """
        根据热度、时间、冗余度判断目标状态

        状态机规则（基于 IMA V2.0 规范）：
        - 木 (生成): hot < 0.5, age < 1 天
        - 火 (活跃): hot > 0.8, age < 1 天
        - 土 (稳定): 0.3 < hot < 0.7, 1 < age < 30 天
        - 金 (精炼): hot < 0.3 或 redundancy > 0.7
        - 水 (闭藏): hot < 0.2, age > 30 天
        """
        # 检查是否应该遗忘
        if age_days > 365 and hot < 0.1:
            return FiveElementState.SHUI  # 超过 1 年且低热度 → 闭藏（待遗忘）

        # 检查活跃状态
        if hot > 0.8 and age_days < 1:
            return FiveElementState.HUO

        # 检查稳定状态
        if 0.3 < hot < 0.7 and 1 < age_days < 30:
            return FiveElementState.TU

        # 检查精炼状态
        if hot < 0.3 or redundancy > 0.7:
            return FiveElementState.JIN

        # 检查闭藏状态
        if hot < 0.2 and age_days > 30:
            return FiveElementState.SHUI

        # 默认生成状态
        return FiveElementState.MU

    def _generate_decision(
        self, memory: MemoryEncoding, current: FiveElementState, target: FiveElementState, redundancy: float
    ) -> SchedulerDecision:
        """生成调度决策"""

        # 判断动作类型
        action, reason = self._determine_action(current, target, redundancy, memory.hot)

        # 计算权重和热度变化
        weight_change, hot_change = self._calculate_changes(current, target, action)

        return SchedulerDecision(
            memory_id=memory.id,
            old_state=current.value,
            new_state=target.value,
            action=action,
            reason=reason,
            weight_change=weight_change,
            hot_change=hot_change,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _determine_action(
        self, current: FiveElementState, target: FiveElementState, redundancy: float, hot: float
    ) -> Tuple[str, str]:
        """确定调度动作"""

        # 相生流转（正向）
        if GENERATION_CYCLE.get(current) == target:
            actions = {
                FiveElementState.MU: ("generate", "新记忆生成，进入编码流程"),
                FiveElementState.HUO: ("activate", "记忆活跃度提升，增加检索优先级"),
                FiveElementState.TU: ("stabilize", "记忆进入稳定期，整合为结构化经验"),
                FiveElementState.JIN: ("refine", "记忆精炼，压缩冗余信息"),
                FiveElementState.SHUI: ("hide", "记忆进入冷存储，降低检索优先级"),
            }
            return actions.get(target, ("update", "状态更新"))

        # 相克调控（负反馈）
        if RESTRAINT_CYCLE.get(current) == target:
            if current == FiveElementState.HUO and target == FiveElementState.JIN:
                return ("cool_down", "记忆过度活跃，触发火克金冷却机制")
            elif current == FiveElementState.MU and target == FiveElementState.JIN:
                return ("filter", "记忆生成过多，触发金克木过滤机制")
            elif current == FiveElementState.TU and target == FiveElementState.SHUI:
                return ("archive", "记忆过度沉淀，触发土克水归档机制")
            return ("restrain", "相克调控，防止过度")

        # 状态不变
        if current == target:
            return ("maintain", "状态稳定，无需调整")

        # 其他状态转换
        return ("transition", f"状态转换：{current.value}→{target.value}")

    def _calculate_changes(
        self, current: FiveElementState, target: FiveElementState, action: str
    ) -> Tuple[float, float]:
        """计算权重和热度变化"""

        # 相生流转的增益
        if GENERATION_CYCLE.get(current) == target:
            return (0.05, 0.1)  # 权重 +5%, 热度 +10%

        # 相克调控的衰减
        if RESTRAINT_CYCLE.get(current) == target:
            return (-0.1, -0.2)  # 权重 -10%, 热度 -20%

        # 遗忘动作
        if action == "forget":
            return (-0.5, -0.5)

        # 默认
        return (0.0, 0.0)

    def apply_decision(self, memory: MemoryEncoding, decision: SchedulerDecision) -> MemoryEncoding:
        """
        应用调度决策到记忆

        Args:
            memory: 原记忆
            decision: 调度决策

        Returns:
            更新后的记忆
        """
        # 更新五行状态
        memory.element = decision.new_state

        # 更新权重和热度
        memory.weight = max(0.0, min(1.0, memory.weight + decision.weight_change))
        memory.hot = max(0.0, min(1.0, memory.hot + decision.hot_change))

        # 根据动作类型进行特殊处理
        if decision.action == "hide":
            # 冷存储：降低热度
            memory.hot = max(0.0, memory.hot * 0.5)
        elif decision.action == "activate":
            # 激活：提升热度
            memory.hot = min(1.0, memory.hot * 1.5)
        elif decision.action == "refine":
            # 精炼：提升权重，降低热度
            memory.weight = min(1.0, memory.weight + 0.1)
            memory.hot = max(0.0, memory.hot * 0.7)

        return memory

    def calculate_field(self, memories: List[MemoryEncoding]) -> FiveElementField:
        """
        计算五行能量场

        Args:
            memories: 记忆列表

        Returns:
            FiveElementField 能量场
        """
        # 统计各五行的能量
        energies = {e: 0.0 for e in FiveElementState}

        for mem in memories:
            try:
                state = FiveElementState(mem.element)
                # 能量 = 权重 × 热度
                energies[state] += mem.weight * mem.hot
            except (ValueError, TypeError):
                continue

        # 归一化
        total = sum(energies.values()) or 1.0
        normalized = {k: v / total for k, v in energies.items()}

        # 计算平衡度（越接近 0.2 越平衡）
        ideal = 0.2
        variance = sum((v - ideal) ** 2 for v in normalized.values()) / 5
        balance = max(0.0, 1.0 - variance * 5)

        return FiveElementField(
            mu_energy=normalized[FiveElementState.MU],
            huo_energy=normalized[FiveElementState.HUO],
            tu_energy=normalized[FiveElementState.TU],
            jin_energy=normalized[FiveElementState.JIN],
            shui_energy=normalized[FiveElementState.SHUI],
            balance_score=balance,
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取调度统计"""
        return {
            **self.stats,
            "decision_history_count": len(self.decision_history),
        }

    def clear_history(self):
        """清空决策历史"""
        self.decision_history = []


# ========== 便捷函数 ==========


def create_scheduler() -> FiveElementsScheduler:
    """创建调度器实例"""
    return FiveElementsScheduler()


def schedule_memory(memory: MemoryEncoding, redundancy: float = 0.0) -> Tuple[MemoryEncoding, SchedulerDecision]:
    """
    快速调度记忆的便捷函数

    示例:
        updated_mem, decision = schedule_memory(memory, redundancy=0.3)
    """
    scheduler = FiveElementsScheduler()
    decision = scheduler.schedule(memory, redundancy)
    updated = scheduler.apply_decision(memory, decision)
    return updated, decision


# ========== 命令行测试 ==========

if __name__ == "__main__":
    from yao_encoding import Element, Layer, Trigram, YaoEncoder

    print("=" * 80)
    print("🔄 五行调度引擎测试")
    print("=" * 80)

    encoder = YaoEncoder()
    scheduler = FiveElementsScheduler()

    # 测试 1：新记忆（木状态）
    print("\n【测试 1】新记忆调度（木→火）")
    mem1 = encoder.encode(
        content="新输入的对话内容", trigram=Trigram.XUN, layer=Layer.DI, element=Element.MU, hot=0.9, weight=0.5
    )
    print(f"初始状态：{mem1.element}, 热度：{mem1.hot}")

    decision1 = scheduler.schedule(mem1, redundancy=0.1)
    print(f"决策：{decision1.action}")
    print(f"原因：{decision1.reason}")
    print(f"状态变化：{decision1.old_state}→{decision1.new_state}")

    # 测试 2：活跃记忆（火状态）
    print("\n【测试 2】活跃记忆调度（火→土）")
    mem2 = encoder.encode(
        content="高频访问的知识", trigram=Trigram.LI, layer=Layer.REN, element=Element.HUO, hot=0.85, weight=0.7
    )
    decision2 = scheduler.schedule(mem2, redundancy=0.2)
    print(f"决策：{decision2.action}")
    print(f"原因：{decision2.reason}")

    # 测试 3：冗余记忆（金状态）
    print("\n【测试 3】冗余记忆调度（金克木过滤）")
    mem3 = encoder.encode(
        content="重复的冗余信息", trigram=Trigram.QIAN, layer=Layer.TIAN, element=Element.JIN, hot=0.2, weight=0.3
    )
    decision3 = scheduler.schedule(mem3, redundancy=0.8)
    print(f"决策：{decision3.action}")
    print(f"原因：{decision3.reason}")

    # 测试 4：五行能量场
    print("\n【测试 4】五行能量场计算")
    memories = [mem1, mem2, mem3]
    field = scheduler.calculate_field(memories)
    print(f"木能量：{field.mu_energy:.2%}")
    print(f"火能量：{field.huo_energy:.2%}")
    print(f"土能量：{field.tu_energy:.2%}")
    print(f"金能量：{field.jin_energy:.2%}")
    print(f"水能量：{field.shui_energy:.2%}")
    print(f"平衡度：{field.balance_score:.2%}")

    # 测试 5：调度统计
    print("\n【测试 5】调度统计")
    stats = scheduler.get_stats()
    print(f"总调度次数：{stats['total_scheduled']}")
    print(f"状态变化：{stats['state_changes']}")
    print(f"遗忘记忆：{stats['forgotten']}")
    print(f"精炼记忆：{stats['refined']}")

    print("\n" + "=" * 80)
    print("✅ 所有测试完成！")
    print("=" * 80)
