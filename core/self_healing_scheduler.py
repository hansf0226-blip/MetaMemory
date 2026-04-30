#!/usr/bin/env python3
"""
🔧 Self-Healing Scheduler（自愈调度器）

功能：
- 定期扫描所有记忆，检查时位匹配状态（position_status）
- 对"失位"记忆自动修正（八卦与三才层级不匹配）
- 对"正常"记忆跳过不必要的重复校验
- 支持多租户批量处理

对应原"时位自愈加强"设计，以通用软件工程语言实现。
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.config import get_config
from core.utils import get_logger
logger = get_logger(__name__)

# ===== 状态定义 =====


class PositionStatus:
    """位置状态常量"""

    NORMAL = "正常"  # 时位匹配，无需修正
    MISALIGN = "失位"  # 时位不匹配，需要修正
    UNKNOWN = "待检"  # 状态未知，需要首次检查
    FIXED = "已修正"  # 已修正（可标记用于观察）


# ===== 自愈结果数据类 =====


@dataclass
class HealResult:
    """单条记忆的自愈结果"""

    memory_id: str
    agent_id: str
    bagua_type: str
    original_layer: str
    corrected_layer: Optional[str]
    status: str
    confidence: float
    action: str  # "checked_ok" / "auto_fixed" / "user_kept"
    reason: str
    timestamp: str


@dataclass
class HealBatchSummary:
    """批量自愈汇总"""

    total_checked: int
    auto_fixed: int
    user_kept: int
    checked_ok: int
    errors: int
    by_bagua: Dict[str, int] = field(default_factory=dict)
    duration_ms: float = 0.0


# ===== 时位规则映射 =====

# 八卦 → 标准三才层级
BAGUA_TO_STANDARD_LAYER = {
    # 核心层：乾坤 → 天
    "qian": "tien",
    "kun": "tien",
    # 活跃层：震巽 → 人
    "zhen": "ren",
    "xun": "ren",
    # 感知层：坎离 → 人（亦归人层）
    "kan": "ren",
    "li": "ren",
    # 基础层：艮兑 → 地
    "gen": "di",
    "dui": "di",
}

LAYER_DISPLAY = {
    "tien": "天层",
    "ren": "人层",
    "di": "地层",
}


def get_standard_layer(bagua: str) -> Optional[str]:
    return BAGUA_TO_STANDARD_LAYER.get(bagua)


# ===== 核心自愈引擎 =====


class SelfHealingEngine:
    """
    自愈调度引擎

    工作流程：
    1. 扫描：按租户分批读取记忆（支持增量跳过）
    2. 预过滤：position_status == "正常" 的记忆直接跳过
    3. 校验：对未知/失位状态执行 position_check_auto
    4. 修正：
       - status == "normal" → 更新为"正常"，跳过
       - status == "fix" → 自动修正层级，写入 DB
       - status == "suggest" → 标记为"建议修正"但保持原层级（尊重用户意图）
    5. 记录：YaoChangeHistory 写入变化记录
    """

    def __init__(self):
        self._lock = threading.RLock()

        # 跳过已确认正常的记忆，减少不必要的重复校验
        # 相当于缓存"最近校验过且正常的记忆"
        self._recent_ok_cache: Dict[str, datetime] = {}
        self._cache_ttl_seconds = get_config("self_healing.ok_cache_ttl_seconds", 86400)
        self._cache_max_entries = get_config("self_healing.ok_cache_max", 50000)

        # 修正置信度阈值：低于此值不自动修正（需人工确认）
        self._min_auto_fix_confidence = get_config("self_healing.min_auto_fix_conf", 0.85)

        logger.info(
            "✅ SelfHealingEngine 初始化："
            f"ok_cache_ttl={self._cache_ttl_seconds}s，"
            f"min_auto_fix_conf={self._min_auto_fix_confidence}"
        )

    def check_layer_alignment(
        self,
        bagua_type: str,
        current_layer: str,
        user_intent: bool = False,
    ) -> Dict[str, Any]:
        """
        检查八卦与三才层级的匹配状态

        Returns:
            dict: {
                "status": "normal" | "suggest" | "fix",
                "expected_layer": str,
                "confidence": float,
                "reason": str
            }
        """
        with self._lock:
            if not bagua_type or not current_layer:
                return {
                    "status": "unknown",
                    "expected_layer": None,
                    "confidence": 0.0,
                    "reason": "缺少八卦类型或层级数据",
                }

            expected_layer = get_standard_layer(bagua_type)
            if expected_layer is None:
                return {
                    "status": "unknown",
                    "expected_layer": None,
                    "confidence": 0.0,
                    "reason": f"未知八卦类型：{bagua_type}",
                }

            if current_layer == expected_layer:
                return {
                    "status": "normal",
                    "expected_layer": expected_layer,
                    "confidence": 1.0,
                    "reason": f"八卦 {bagua_type} 与层级 {LAYER_DISPLAY.get(current_layer, current_layer)} 匹配",
                }

            # 不匹配：根据 user_intent 决定建议还是强制修正
            base_conf = 0.9
            if user_intent:
                confidence = max(0.5, base_conf - 0.3)
                status = "suggest"
                reason = (
                    f"用户指定层级 {LAYER_DISPLAY.get(current_layer, current_layer)}，"
                    f"但八卦 {bagua_type} 标准为 {LAYER_DISPLAY.get(expected_layer, expected_layer)}"
                )
            else:
                confidence = base_conf
                status = "fix"
                reason = (
                    f"八卦 {bagua_type} 应位于 {LAYER_DISPLAY.get(expected_layer, expected_layer)}，"
                    f"当前位于 {LAYER_DISPLAY.get(current_layer, current_layer)}（不匹配）"
                )

            return {
                "status": status,
                "expected_layer": expected_layer,
                "confidence": confidence,
                "reason": reason,
                "original_layer": current_layer,
            }

    def should_skip(self, memory: Dict[str, Any]) -> bool:
        """
        判断该记忆是否应该跳过校验（基于缓存和状态）
        """
        memory_id = str(memory.get("id", ""))
        position_status = memory.get("position_status", "")

        # 已明确标记为"正常"的记忆 → 跳过的条件：
        # 缓存中有且未过期
        if position_status == PositionStatus.NORMAL:
            if memory_id in self._recent_ok_cache:
                cached_at = self._recent_ok_cache[memory_id]
                age = (datetime.now() - cached_at).total_seconds()
                if age < self._cache_ttl_seconds:
                    return True
        return False

    def mark_as_ok(self, memory_id: str):
        """将记忆标记为最近校验正常"""
        with self._lock:
            if len(self._recent_ok_cache) >= self._cache_max_entries:
                # 清空最老的10%
                sorted_items = sorted(self._recent_ok_cache.items(), key=lambda x: x[1])
                drop_count = len(sorted_items) // 10
                for key, _ in sorted_items[:drop_count]:
                    del self._recent_ok_cache[key]

            self._recent_ok_cache[memory_id] = datetime.now()

    def heal_memory(
        self,
        memory: Dict[str, Any],
        user_intent: bool = False,
    ) -> HealResult:
        """
        对单条记忆执行自愈检查

        Args:
            memory: 记忆数据字典
            user_intent: 用户是否明确指定了层级

        Returns:
            HealResult
        """
        memory_id = str(memory.get("id", ""))
        agent_id = memory.get("agent_id", "default_agent")
        bagua_type = memory.get("bagua_type", "")
        current_layer = memory.get("sancai_layer", "")

        # 检查是否应该跳过
        if self.should_skip(memory):
            return HealResult(
                memory_id=memory_id,
                agent_id=agent_id,
                bagua_type=bagua_type,
                original_layer=current_layer,
                corrected_layer=None,
                status="skipped",
                confidence=1.0,
                action="checked_ok",
                reason="缓存命中，跳过重复校验",
                timestamp=datetime.now().isoformat(),
            )

        # 执行校验
        check = self.check_layer_alignment(bagua_type, current_layer, user_intent)

        if check["status"] == "normal":
            self.mark_as_ok(memory_id)
            return HealResult(
                memory_id=memory_id,
                agent_id=agent_id,
                bagua_type=bagua_type,
                original_layer=current_layer,
                corrected_layer=None,
                status=PositionStatus.NORMAL,
                confidence=check["confidence"],
                action="checked_ok",
                reason=check["reason"],
                timestamp=datetime.now().isoformat(),
            )

        elif check["status"] == "fix":
            # 置信度低于阈值时不自动修正
            if check["confidence"] < self._min_auto_fix_confidence:
                return HealResult(
                    memory_id=memory_id,
                    agent_id=agent_id,
                    bagua_type=bagua_type,
                    original_layer=current_layer,
                    corrected_layer=None,
                    status=PositionStatus.MISALIGN,
                    confidence=check["confidence"],
                    action="low_confidence_hold",
                    reason=f"修正置信度 {check['confidence']:.2f} < {self._min_auto_fix_confidence}，需人工确认",
                    timestamp=datetime.now().isoformat(),
                )

            # 执行自动修正
            new_layer = check["expected_layer"]
            return HealResult(
                memory_id=memory_id,
                agent_id=agent_id,
                bagua_type=bagua_type,
                original_layer=current_layer,
                corrected_layer=new_layer,
                status=PositionStatus.FIXED,
                confidence=check["confidence"],
                action="auto_fixed",
                reason=f"自动修正：{LAYER_DISPLAY.get(current_layer, current_layer)} → {LAYER_DISPLAY.get(new_layer, new_layer)}",
                timestamp=datetime.now().isoformat(),
            )

        else:  # "suggest" 或 "unknown"
            return HealResult(
                memory_id=memory_id,
                agent_id=agent_id,
                bagua_type=bagua_type,
                original_layer=current_layer,
                corrected_layer=None,
                status=PositionStatus.MISALIGN,
                confidence=check["confidence"],
                action="user_kept",
                reason=check["reason"],
                timestamp=datetime.now().isoformat(),
            )

    def heal_batch(
        self,
        memories: List[Dict[str, Any]],
        agent_id: str,
        user_intent: bool = False,
    ) -> List[HealResult]:
        """批量自愈"""
        return [self.heal_memory(mem, user_intent) for mem in memories]

    def apply_result_to_storage(
        self,
        result: HealResult,
        storage_manager,
    ) -> bool:
        """
        将自愈结果写入存储

        Returns:
            bool: 是否执行了写操作
        """
        try:
            if result.action == "auto_fixed" and result.corrected_layer:
                storage_manager.update_memory(
                    memory_id=(
                        int(result.memory_id.split("-")[-1]) if "-" in result.memory_id else int(result.memory_id)
                    ),
                    agent_id=result.agent_id,
                    updates={
                        "sancai_layer": result.corrected_layer,
                        "position_status": PositionStatus.FIXED,
                        "position_fixed_at": result.timestamp,
                        "position_confidence": result.confidence,
                    },
                )
                return True

            elif result.action == "checked_ok":
                storage_manager.update_memory(
                    memory_id=(
                        int(result.memory_id.split("-")[-1]) if "-" in result.memory_id else int(result.memory_id)
                    ),
                    agent_id=result.agent_id,
                    updates={
                        "position_status": PositionStatus.NORMAL,
                        "position_confidence": result.confidence,
                    },
                )
                return True

            return False

        except Exception as e:
            logger.warning(f"自愈结果写入失败 memory={result.memory_id}: {e}")
            return False


# ===== 便捷函数 =====

_healer: Optional[SelfHealingEngine] = None
_healer_lock = threading.Lock()


def get_healer() -> SelfHealingEngine:
    global _healer
    if _healer is None:
        with _healer_lock:
            if _healer is None:
                _healer = SelfHealingEngine()
    return _healer


def heal_memory(memory: Dict[str, Any], user_intent: bool = False) -> HealResult:
    return get_healer().heal_memory(memory, user_intent)
