#!/usr/bin/env python3
"""
💫 记忆渗透机制 - Memory Permeation System
基于易经哲学的记忆固化与渗透机制
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List


class PermeationStage(Enum):
    """渗透阶段"""

    OUTER = "外球壳"  # 临时记忆
    TETRAGRAM = "四象"  # 情绪代谢
    TRIGRAM = "八卦"  # 常识观念
    HEXAGRAM = "六十四卦"  # 人格特质
    TAIJI = "太极"  # 核心身份


@dataclass
class MemoryPermeation:
    """记忆渗透数据"""

    content: str
    current_stage: PermeationStage
    repetition_count: int
    importance: float
    decay_rate: float
    created_at: str
    last_reinforced: str
    next_reinforcement: str


class MemoryPermeationSystem:
    """记忆渗透系统"""

    def __init__(self):
        """初始化渗透系统"""
        # 渗透阈值
        self.thresholds = {
            PermeationStage.OUTER: 0,  # 初始
            PermeationStage.TETRAGRAM: 3,  # 重复 3 次
            PermeationStage.TRIGRAM: 10,  # 重复 10 次
            PermeationStage.HEXAGRAM: 30,  # 重复 30 次
            PermeationStage.TAIJI: 100,  # 重复 100 次
        }

        # 各层衰减率
        self.decay_rates = {
            PermeationStage.OUTER: 0.8,  # 快速衰减
            PermeationStage.TETRAGRAM: 0.5,  # 中等衰减
            PermeationStage.TRIGRAM: 0.2,  # 缓慢衰减
            PermeationStage.HEXAGRAM: 0.05,  # 几乎不衰减
            PermeationStage.TAIJI: 0.0,  # 永不衰减
        }

        # 记忆存储
        self.memories: Dict[str, MemoryPermeation] = {}

        print("✅ 记忆渗透系统已初始化")
        print(f"   渗透阶段：{len(self.thresholds)} 层")
        print("   阈值：3 → 10 → 30 → 100")

    def add_memory(self, content: str, initial_importance: float = 0.5) -> MemoryPermeation:
        """
        添加记忆

        Args:
            content: 记忆内容
            initial_importance: 初始重要度

        Returns:
            记忆渗透数据
        """
        now = datetime.now().isoformat()

        memory = MemoryPermeation(
            content=content,
            current_stage=PermeationStage.OUTER,
            repetition_count=0,
            importance=initial_importance,
            decay_rate=self.decay_rates[PermeationStage.OUTER],
            created_at=now,
            last_reinforced=now,
            next_reinforcement=self._calculate_next_reinforcement(0),
        )

        self.memories[content] = memory
        return memory

    def reinforce(self, content: str) -> MemoryPermeation:
        """
        强化记忆（重复）

        Args:
            content: 记忆内容

        Returns:
            更新后的记忆数据
        """
        if content not in self.memories:
            raise ValueError(f"记忆不存在：{content}")

        memory = self.memories[content]
        memory.repetition_count += 1
        memory.last_reinforced = datetime.now().isoformat()

        # 检查是否需要渗透
        self._check_permeation(memory)

        # 更新衰减率
        memory.decay_rate = self.decay_rates[memory.current_stage]

        # 计算下次强化时间
        memory.next_reinforcement = self._calculate_next_reinforcement(memory.repetition_count)

        return memory

    def _check_permeation(self, memory: MemoryPermeation):
        """检查是否需要渗透到下一层"""
        current_stage = memory.current_stage
        repetition = memory.repetition_count

        # 检查是否达到下一层阈值
        stages = list(PermeationStage)
        current_index = stages.index(current_stage)

        if current_index < len(stages) - 1:
            next_stage = stages[current_index + 1]
            threshold = self.thresholds[next_stage]

            if repetition >= threshold:
                old_stage = current_stage
                memory.current_stage = next_stage
                print(f"💫 记忆渗透：{old_stage.value} → {next_stage.value}")
                print(f"   内容：{memory.content[:50]}...")
                print(f"   重复次数：{repetition}")

    def _calculate_next_reinforcement(self, repetition_count: int) -> str:
        """
        计算下次强化时间（基于艾宾浩斯遗忘曲线）

        Args:
            repetition_count: 重复次数

        Returns:
            下次强化时间（ISO 格式）
        """
        # 艾宾浩斯复习间隔（小时）
        intervals = [0.5, 1, 3, 6, 12, 24, 48, 168, 336]  # 9 次复习

        if repetition_count < len(intervals):
            hours = intervals[repetition_count]
        else:
            # 指数增长
            hours = intervals[-1] * (2 ** (repetition_count - len(intervals)))

        from datetime import timedelta

        next_time = datetime.now() + timedelta(hours=hours)
        return next_time.isoformat()

    def get_permeation_status(self, content: str) -> Dict:
        """获取记忆渗透状态"""
        if content not in self.memories:
            return {"error": "记忆不存在"}

        memory = self.memories[content]

        return {
            "content": memory.content,
            "stage": memory.current_stage.value,
            "repetition_count": memory.repetition_count,
            "importance": memory.importance,
            "decay_rate": memory.decay_rate,
            "created_at": memory.created_at,
            "last_reinforced": memory.last_reinforced,
            "next_reinforcement": memory.next_reinforcement,
            "progress_to_next": self._calculate_progress_to_next(memory),
        }

    def _calculate_progress_to_next(self, memory: MemoryPermeation) -> float:
        """计算到下一层的进度"""
        stages = list(PermeationStage)
        current_index = stages.index(memory.current_stage)

        if current_index >= len(stages) - 1:
            return 1.0  # 已到达最高层

        next_stage = stages[current_index + 1]
        threshold = self.thresholds[next_stage]

        return min(1.0, memory.repetition_count / threshold)

    def get_all_memories_by_stage(self) -> Dict[str, List[Dict]]:
        """按层获取所有记忆"""
        result = {stage.value: [] for stage in PermeationStage}

        for memory in self.memories.values():
            result[memory.current_stage.value].append(
                {"content": memory.content, "repetition": memory.repetition_count, "importance": memory.importance}
            )

        return result

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total = len(self.memories)
        if total == 0:
            return {"total": 0}

        # 各层数量
        by_stage = {stage.value: 0 for stage in PermeationStage}
        for memory in self.memories.values():
            by_stage[memory.current_stage.value] += 1

        # 平均重复次数
        avg_repetition = sum(m.repetition_count for m in self.memories.values()) / total

        # 平均重要度
        avg_importance = sum(m.importance for m in self.memories.values()) / total

        return {
            "total_memories": total,
            "by_stage": by_stage,
            "avg_repetition": round(avg_repetition, 2),
            "avg_importance": round(avg_importance, 2),
            "deepest_stage": self._get_deepest_stage(),
        }

    def _get_deepest_stage(self) -> str:
        """获取最深的渗透层"""
        if not self.memories:
            return "无"

        deepest = max(self.memories.values(), key=lambda m: list(PermeationStage).index(m.current_stage))
        return deepest.current_stage.value

    def manual_permeation(self, content: str, target_stage: PermeationStage) -> bool:
        """
        手动渗透（用户干预）

        Args:
            content: 记忆内容
            target_stage: 目标层

        Returns:
            是否成功
        """
        if content not in self.memories:
            print(f"❌ 记忆不存在：{content}")
            return False

        memory = self.memories[content]
        current_index = list(PermeationStage).index(memory.current_stage)
        target_index = list(PermeationStage).index(target_stage)

        if target_index <= current_index:
            print(f"⚠️ 目标层{target_stage.value}不高于当前层{memory.current_stage.value}")
            return False

        # 直接提升到目标层
        old_stage = memory.current_stage
        memory.current_stage = target_stage
        memory.decay_rate = self.decay_rates[target_stage]
        memory.repetition_count = self.thresholds[target_stage]

        print(f"✅ 手动渗透：{old_stage.value} → {target_stage.value}")
        return True

    def export_memories(self) -> List[Dict]:
        """导出所有记忆"""
        return [
            {
                "content": m.content,
                "stage": m.current_stage.value,
                "repetition": m.repetition_count,
                "importance": m.importance,
                "decay_rate": m.decay_rate,
            }
            for m in self.memories.values()
        ]


# ========== 便捷函数 ==========


def create_permeation_system() -> MemoryPermeationSystem:
    """创建渗透系统"""
    return MemoryPermeationSystem()


# ========== 测试 ==========

if __name__ == "__main__":
    # 创建系统
    system = create_permeation_system()

    print("\n💫 记忆渗透系统测试")
    print("=" * 60)

    # 添加记忆
    memories = [
        "今天学习了易经",
        "Python 很有趣",
        "明天要开会",
        "AI 与人类平等",
    ]

    for mem in memories:
        system.add_memory(mem, initial_importance=0.7)

    # 强化记忆
    print("\n强化记忆:")
    system.reinforce("今天学习了易经")
    system.reinforce("今天学习了易经")
    system.reinforce("今天学习了易经")  # 3 次，应该渗透到四象层

    # 查看状态
    print("\n记忆状态:")
    for mem in memories:
        status = system.get_permeation_status(mem)
        if "error" not in status:
            print(f"  {status['content'][:20]:20} - {status['stage']:10} - 重复：{status['repetition_count']}")

    # 统计
    print("\n统计信息:")
    stats = system.get_statistics()
    print(f"  总记忆数：{stats['total_memories']}")
    print(f"  平均重复：{stats['avg_repetition']}")
    print(f"  平均重要：{stats['avg_importance']}")
    print("  各层分布:")
    for stage, count in stats["by_stage"].items():
        print(f"    {stage:10}: {count}")
