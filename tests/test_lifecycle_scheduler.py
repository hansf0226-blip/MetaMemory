"""
测试记忆生命周期调度器 (API 同步 v7.0)
"""
import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.lifecycle_scheduler import (
    LifecycleScheduler,
    LifecyclePhase,
    LifecycleResult,
    get_scheduler,
    schedule_memory,
    schedule_memory_batch,
    wuxing_life_schedule,
)


# ─── LifecyclePhase 枚举 ─────────────────────────────────────

class TestLifecyclePhase:
    """测试生命周期阶段枚举"""

    def test_phase_values(self):
        """阶段值"""
        assert LifecyclePhase.GENERATE.value == "generate"
        assert LifecyclePhase.ACTIVE.value == "active"
        assert LifecyclePhase.STABLE.value == "stable"
        assert LifecyclePhase.REFINE.value == "refine"
        assert LifecyclePhase.ARCHIVE.value == "archive"

    def test_phase_count(self):
        """5 个阶段（易经五态）"""
        phases = list(LifecyclePhase)
        assert len(phases) == 5


# ─── LifecycleResult ──────────────────────────────────────────

class TestLifecycleResult:
    """测试生命周期结果"""

    def test_result_fields(self):
        """结果字段"""
        r = LifecycleResult(
            memory_id="mem-001",
            agent_id="test-agent",
            input_phase=LifecyclePhase.ACTIVE,
            input_wuxing="金",
            input_hot_score=0.5,
            input_age_days=10.0,
            output_phase=LifecyclePhase.STABLE,
            output_hot_score=0.4,
            output_wuxing="金",
            triggered=True,
            trigger_reason="age_threshold",
        )
        assert r.memory_id == "mem-001"
        assert r.agent_id == "test-agent"
        assert r.output_phase == LifecyclePhase.STABLE


# ─── LifecycleScheduler ───────────────────────────────────────

class TestLifecycleScheduler:
    """测试生命周期调度器"""

    def test_scheduler_initialization(self):
        """调度器初始化"""
        scheduler = LifecycleScheduler()
        assert scheduler is not None

    def test_singleton(self):
        """单例模式"""
        s1 = get_scheduler()
        s2 = get_scheduler()
        assert s1 is s2

    def test_schedule_returns_result(self):
        """调度返回 LifecycleResult"""
        scheduler = LifecycleScheduler()
        result = scheduler.schedule(
            memory_id="mem-001",
            agent_id="test-agent",
            wuxing="金",
            hot_score=0.5,
            created_at=datetime.now(),
        )
        assert isinstance(result, LifecycleResult)

    def test_schedule_batch(self):
        """批量调度"""
        scheduler = LifecycleScheduler()
        memories = [
            {"wuxing": "金", "hot_score": 0.5, "created_at": datetime.now()},
            {"wuxing": "木", "hot_score": 0.3, "created_at": datetime.now() - timedelta(days=30)},
        ]
        results = scheduler.schedule_batch(
            agent_id="test-agent",
            memories=memories,
        )
        assert isinstance(results, list)
        assert len(results) == 2

    def test_boost_on_access(self):
        """访问提升热度"""
        scheduler = LifecycleScheduler()
        boosted = scheduler.boost_on_access(0.5)
        assert boosted >= 0.5


# ─── 便捷函数 ─────────────────────────────────────────────────

class TestConvenienceFunctions:
    """测试便捷调度函数"""

    def test_schedule_memory(self):
        """schedule_memory 函数"""
        result = schedule_memory(
            memory_id="mem-001",
            agent_id="test-agent",
            wuxing="金",
            hot_score=0.5,
            created_at=datetime.now(),
        )
        assert isinstance(result, LifecycleResult)

    def test_wuxing_life_schedule(self):
        """wuxing_life_schedule 函数"""
        result = wuxing_life_schedule(
            {"hot_score": 0.5},
            create_time=datetime.now(),
        )
        assert isinstance(result, dict)


# ─── 已移除功能（保留占位） ───────────────────────────────────

@pytest.mark.skip(reason="v7.0: calculate_decay_rate 已合并到 LifecycleScheduler")
class TestCalculateDecayRate:
    def test_basic_decay_rate(self): pass
    def test_high_importance_decay_slower(self): pass
    def test_old_memory_decay_faster(self): pass
    def test_decay_rate_bounds(self): pass

@pytest.mark.skip(reason="v7.0: should_awake_memory 已合并")
class TestShouldAwakeMemory:
    def test_high_access_count_awake(self): pass
    def test_recently_accessed_not_awake(self): pass
    def test_low_importance_no_awake(self): pass

@pytest.mark.skip(reason="v7.0: MemoryLifecycleManager 已合并为 LifecycleScheduler")
class TestMemoryLifecycleManager:
    def test_manager_initialization(self): pass
    def test_apply_decay_to_memory(self): pass
    def test_no_decay_when_disabled(self): pass
    def test_consolidate_similar_memories(self): pass
    def test_cleanup_weak_memories(self): pass
    def test_awake_dormant_memories(self): pass

@pytest.mark.skip(reason="v7.0: MemoryPriority 已合并")
class TestMemoryPriority:
    def test_priority_calculation(self): pass
    def test_importance_affects_priority(self): pass

@pytest.mark.skip(reason="v7.0: 边缘情况待新测试覆盖")
class TestEdgeCases:
    def test_empty_memory_list(self): pass
    def test_zero_strength_memory(self): pass
    def test_future_created_memory(self): pass
