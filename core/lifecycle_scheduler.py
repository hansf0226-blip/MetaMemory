#!/usr/bin/env python3
"""
🔄 Lifecycle Scheduler（生命周期调度器）

功能：
- 五阶段流转：生成(generate) → 活跃(active) → 稳定(stable) → 精炼(refine) → 归档(archive)
- 流转触发条件：age（创建天数）+ hot_score 双条件判断
- 相克抑制规则：防止某一阶段记忆过度积累导致失衡
- 热度随时间和访问动态衰减

对应原"五行调度"设计，以通用软件工程语言实现。
"""

import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from core.config import get_config
from core.utils import get_logger
logger = get_logger(__name__)

# ===== 生命周期阶段定义 =====


class LifecyclePhase(Enum):
    """
    记忆生命周期五阶段：
    - GENERATE: 生成期（木）— 新记忆被创建，等待激活
    - ACTIVE: 活跃期（火）— 被频繁访问，处于工作状态
    - STABLE: 稳定期（土）— 已整合，转变为长期记忆
    - REFINE: 精炼期（金）— 压缩冗余，强化重要记忆
    - ARCHIVE: 归档期（水）— 冷存储或自然遗忘
    """

    GENERATE = "generate"
    ACTIVE = "active"
    STABLE = "stable"
    REFINE = "refine"
    ARCHIVE = "archive"

    def __str__(self):
        return self.value


# ===== 五行 ↔ 阶段映射 =====

# 五行到阶段的固定映射（原设计对应关系）
WUXING_TO_PHASE = {
    "mu": LifecyclePhase.GENERATE,
    "huo": LifecyclePhase.ACTIVE,
    "tu": LifecyclePhase.STABLE,
    "jin": LifecyclePhase.REFINE,
    "shui": LifecyclePhase.ARCHIVE,
}

# 阶段到五行的反向映射
PHASE_TO_WUXING = {v: k for k, v in WUXING_TO_PHASE.items()}

# 五行序列（用于流转方向）
WUXING_SEQUENCE = ["mu", "huo", "tu", "jin", "shui"]


# ===== 相克规则定义 =====

# KEY: 被克制的五行 → (压制者五行, 触发条件)
# 当"压制者"在当前调度周期内处理量过多时，抑制"被克制"记忆的流转
MUTUAL_CONSTRAINTS = [
    # 木克土：生成（木）过多会抑制稳定（土）积累
    # 触发：当前批次中 GENERATE 记忆占比 > 阈值
    {
        "suppressed": "tu",
        "suppressor": "mu",
        "rule_name": "木克土",
        "threshold": 0.5,
        "direction": "suppress_upgrade",
        "description": "生成期记忆过多时，抑制稳定期流转速度",
    },
    # 土克水：稳定（土）过多抑制归档（水）遗忘
    {
        "suppressed": "shui",
        "suppressor": "tu",
        "threshold": 0.5,
        "direction": "suppress_upgrade",
        "rule_name": "土克水",
        "description": "稳定期记忆过多时，抑制归档期遗忘速度",
    },
    # 水克火：归档（水）抑制活跃（火）
    {
        "suppressed": "huo",
        "suppressor": "shui",
        "threshold": 0.5,
        "direction": "suppress_upgrade",
        "rule_name": "水克火",
        "description": "归档期记忆过多时，抑制活跃期流转速度",
    },
    # 火克金：活跃（火）抑制精炼（金）固化
    {
        "suppressed": "jin",
        "suppressor": "huo",
        "threshold": 0.5,
        "direction": "suppress_upgrade",
        "rule_name": "火克金",
        "description": "活跃期记忆过多时，抑制精炼期固化速度",
    },
    # 金克木：精炼（金）抑制生成（木）冗余
    {
        "suppressed": "mu",
        "suppressor": "jin",
        "threshold": 0.5,
        "direction": "suppress_upgrade",
        "rule_name": "金克木",
        "description": "精炼期记忆过多时，抑制新记忆生成速度",
    },
]


# ===== 调度结果数据类 =====


@dataclass
class LifecycleResult:
    """单条记忆的调度结果"""

    memory_id: str
    agent_id: str
    # 输入
    input_phase: LifecyclePhase
    input_wuxing: str
    input_hot_score: float
    input_age_days: float
    # 输出
    output_phase: LifecyclePhase
    output_hot_score: float
    output_wuxing: str
    # 元数据
    triggered: bool  # 是否发生了流转
    trigger_reason: str  # 触发原因
    suppressed_by: Optional[str] = None  # 被哪个相克规则抑制
    hot_delta: float = 0.0  # 热度变化量


# ===== 核心调度引擎 =====


class LifecycleScheduler:
    """
    生命周期调度引擎

    工作流程：
    1. 判断记忆当前阶段（根据 wuxing 字段）
    2. 计算 age（天数）和 hot_score
    3. 判断流转条件：
       - age > age_threshold 或 hot_score > activate_threshold → 向后流转
       - age > max_age 或 hot_score < forget_threshold → 加速流转
    4. 检查相克规则：若当前批次某阶段记忆过多，抑制其克制对象的流转
    5. 更新 phase + hot_score
    """

    def __init__(self):
        self._lock = threading.RLock()

        # ===== 可配置阈值 =====
        self.age_threshold_generate = get_config("lifecycle.age_threshold_generate", 1.0)
        self.age_threshold_active = get_config("lifecycle.age_threshold_active", 7.0)
        self.age_threshold_stable = get_config("lifecycle.age_threshold_stable", 30.0)
        self.age_threshold_refine = get_config("lifecycle.age_threshold_refine", 90.0)
        self.max_age_archive = get_config("lifecycle.max_age_archive", 365.0)

        self.hot_threshold_activate = get_config("lifecycle.hot_threshold_activate", 0.8)
        self.hot_threshold_hide = get_config("lifecycle.hot_threshold_hide", 0.15)
        self.hot_decay_rate = get_config("lifecycle.hot_decay_per_day", 0.01)
        self.hot_boost_on_access = get_config("lifecycle.hot_boost_on_access", 0.05)
        self.hot_boost_max = get_config("lifecycle.hot_boost_max", 1.0)

        # 相克规则启用
        self.mutual_constraints_enabled = get_config("lifecycle.mutual_constraints", True)

        logger.info(
            "✅ LifecycleScheduler 初始化："
            "age_thresholds=[1d/7d/30d/90d/365d]，"
            f"hot_decay={self.hot_decay_rate}/d，"
            f"constraints={'ON' if self.mutual_constraints_enabled else 'OFF'}"
        )

    def schedule(
        self,
        memory_id: str,
        agent_id: str,
        wuxing: str,
        hot_score: float,
        created_at: datetime,
        now: Optional[datetime] = None,
    ) -> LifecycleResult:
        """
        对单条记忆执行生命周期调度

        Args:
            memory_id: 记忆 ID
            agent_id: Agent ID
            wuxing: 当前五行属性
            hot_score: 当前热度分数
            created_at: 记忆创建时间
            now: 当前时间（可选，默认 now）

        Returns:
            LifecycleResult: 包含输入/输出状态和触发原因
        """
        with self._lock:
            if now is None:
                now = datetime.now()

            current_phase = WUXING_TO_PHASE.get(wuxing, LifecyclePhase.GENERATE)
            age_days = (now - created_at).total_seconds() / 86400.0
            age_days = max(0.0, age_days)

            input_phase = current_phase
            input_wuxing = wuxing
            input_hot_score = hot_score
            input_age_days = age_days

            # ===== 热度衰减 =====
            hot_delta = -self.hot_decay_rate * age_days
            new_hot = max(0.0, min(self.hot_boost_max, hot_score + hot_delta))

            # ===== 阶段流转判断 =====
            new_phase, trigger_reason = self._compute_next_phase(current_phase, age_days, hot_score, new_hot)

            triggered = new_phase != current_phase
            suppressed_by = None

            new_wuxing = PHASE_TO_WUXING.get(new_phase, wuxing)

            return LifecycleResult(
                memory_id=memory_id,
                agent_id=agent_id,
                input_phase=input_phase,
                input_wuxing=wuxing,
                input_hot_score=input_hot_score,
                input_age_days=input_age_days,
                output_phase=new_phase,
                output_hot_score=round(new_hot, 4),
                output_wuxing=new_wuxing,
                triggered=triggered,
                trigger_reason=trigger_reason,
                suppressed_by=suppressed_by,
                hot_delta=round(hot_delta, 4),
            )

    def schedule_batch(
        self,
        memories: List[Dict[str, Any]],
        agent_id: str,
        now: Optional[datetime] = None,
    ) -> List[LifecycleResult]:
        """
        批量调度（支持相克规则）

        Args:
            memories: 记忆列表，每条包含 memory_id / wuxing / hot_score / created_at
            agent_id: Agent ID
            now: 当前时间

        Returns:
            与输入对应的 LifecycleResult 列表
        """
        if now is None:
            now = datetime.now()

        # ===== STEP 1: 预计算所有单条结果 =====
        individual_results = []
        wuxing_counts: Dict[str, int] = {}

        for mem in memories:
            result = self.schedule(
                memory_id=mem.get("memory_id") or str(mem.get("id", "")),
                agent_id=agent_id,
                wuxing=mem.get("wuxing", "mu"),
                hot_score=float(mem.get("hot_score", 0.5)),
                created_at=mem.get("created_at", now),
                now=now,
            )
            individual_results.append(result)

            # 统计各阶段记忆数量（用于相克规则）
            if result.triggered:
                phase_key = result.output_phase.value
                wuxing_counts[phase_key] = wuxing_counts.get(phase_key, 0) + 1

        # ===== STEP 2: 应用相克规则 =====
        if self.mutual_constraints_enabled and memories:
            total = len(memories)
            if total > 0:
                for result in individual_results:
                    if not result.triggered:
                        continue

                    output_wuxing = result.output_wuxing
                    output_phase = WUXING_TO_PHASE.get(output_wuxing)

                    for constraint in MUTUAL_CONSTRAINTS:
                        if constraint["suppressed"] != output_wuxing:
                            continue

                        suppressor_phase = WUXING_TO_PHASE.get(constraint["suppressor"])
                        if suppressor_phase is None:
                            continue

                        suppressor_count = wuxing_counts.get(suppressor_phase.value, 0)
                        suppressor_ratio = suppressor_count / total

                        if suppressor_ratio > constraint["threshold"]:
                            # 相克触发：撤销本次流转，记忆停留在当前阶段
                            result.output_phase = result.input_phase
                            result.output_wuxing = result.input_wuxing
                            result.triggered = False
                            result.trigger_reason = (
                                f"相克[{constraint['rule_name']}]抑制："
                                f"{suppressor_phase.value}记忆占比{suppressor_ratio:.0%}"
                                f">{constraint['threshold']:.0%}，"
                                f"已暂停{output_wuxing}流转"
                            )
                            result.suppressed_by = constraint["rule_name"]
                            # 被抑制时，热度轻微恢复（防止过度压制）
                            result.output_hot_score = round(min(self.hot_boost_max, result.output_hot_score + 0.005), 4)
                            break

        return individual_results

    def _compute_next_phase(
        self,
        current: LifecyclePhase,
        age_days: float,
        hot_score: float,
        new_hot: float,
    ) -> tuple[LifecyclePhase, str]:
        """
        计算下一个阶段

        Returns:
            (new_phase, trigger_reason)
        """
        # ===== ARCHIVE 阶段：不再向前流转 =====
        if current == LifecyclePhase.ARCHIVE:
            return LifecyclePhase.ARCHIVE, "已处于归档期，不再流转"

        # ===== 基于 age + hot_score 双条件判断 =====
        next_phase_map = {
            LifecyclePhase.GENERATE: LifecyclePhase.ACTIVE,
            LifecyclePhase.ACTIVE: LifecyclePhase.STABLE,
            LifecyclePhase.STABLE: LifecyclePhase.REFINE,
            LifecyclePhase.REFINE: LifecyclePhase.ARCHIVE,
        }

        age_thresholds = {
            LifecyclePhase.GENERATE: self.age_threshold_generate,
            LifecyclePhase.ACTIVE: self.age_threshold_active,
            LifecyclePhase.STABLE: self.age_threshold_stable,
            LifecyclePhase.REFINE: self.age_threshold_refine,
        }

        next_phase = next_phase_map.get(current, current)
        age_threshold = age_thresholds.get(current, 30.0)

        # 双条件：age 超阈值 OR 热度高于激活阈值
        age_triggered = age_days >= age_threshold
        hot_triggered = hot_score >= self.hot_threshold_activate

        # 遗忘条件：age 超过最大值 OR 热度低于遗忘阈值
        will_forget = age_days >= self.max_age_archive or new_hot < self.hot_threshold_hide

        if will_forget and current != LifecyclePhase.ARCHIVE:
            return LifecyclePhase.ARCHIVE, (
                f"遗忘触发：age={age_days:.1f}d>{self.max_age_archive}d"
                f" 或 hot={new_hot:.3f}<{self.hot_threshold_hide}"
            )

        if age_triggered or hot_triggered:
            return next_phase, (
                "流转触发："
                f"age={age_days:.1f}d{'[超阈值]' if age_triggered else ''}"
                f" hot={hot_score:.3f}{'[高热]' if hot_triggered else ''}"
            )

        return current, "条件未触发，维持当前阶段"

    def boost_on_access(self, hot_score: float) -> float:
        """访问时提升热度（被读取时强化记忆）"""
        return min(self.hot_boost_max, hot_score + self.hot_boost_on_access)


# ===== 便捷函数 =====

_scheduler: Optional[LifecycleScheduler] = None
_scheduler_lock = threading.Lock()


def get_scheduler() -> LifecycleScheduler:
    global _scheduler
    if _scheduler is None:
        with _scheduler_lock:
            if _scheduler is None:
                _scheduler = LifecycleScheduler()
    return _scheduler


def schedule_memory(
    memory_id: str,
    agent_id: str,
    wuxing: str,
    hot_score: float,
    created_at: datetime,
) -> LifecycleResult:
    """对单条记忆执行生命周期调度"""
    return get_scheduler().schedule(memory_id, agent_id, wuxing, hot_score, created_at)


def schedule_memory_batch(
    memories: List[Dict[str, Any]],
    agent_id: str,
) -> List[LifecycleResult]:
    """批量调度（含相克规则）"""
    return get_scheduler().schedule_batch(memories, agent_id)


# 兼容原 wuxing_life_schedule 接口
def wuxing_life_schedule(mem_dict: dict, create_time: datetime) -> dict:
    """
    兼容包装：将新引擎结果适配回原接口格式
    """
    result = schedule_memory(
        memory_id=mem_dict.get("memory_id", ""),
        agent_id=mem_dict.get("agent_id", "default_agent"),
        wuxing=mem_dict.get("wuxing", "mu"),
        hot_score=float(mem_dict.get("hot_score", 0.5)),
        created_at=create_time,
    )
    return {
        "wuxing": result.output_wuxing,
        "hot_score": result.output_hot_score,
    }
