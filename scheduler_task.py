from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.intelligent_decay import IntelligentDecay
from core.lifecycle_scheduler import LifecycleScheduler, schedule_memory_batch
from core.self_healing_scheduler import PositionStatus, get_healer
from core.storage_manager import get_storage_manager
from logger import logger

scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")
PAGE_SIZE = 200

# 初始化存储管理器
storage_manager = get_storage_manager()

# 初始化智能衰减实例
# 初始化生命周期调度引擎
lifecycle_scheduler = LifecycleScheduler()

# 初始化自愈调度引擎
self_healer = get_healer()

decay_engine = IntelligentDecay()


def get_all_enable_agent():
    """获取所有有记忆的 Agent 列表（从数据库真实查询）"""
    try:
        all_agent_ids = storage_manager.mysql_store.get_all_agent_ids()
        logger.info(f"【Agent列表】共发现 {len(all_agent_ids)} 个租户：{all_agent_ids}")
        return all_agent_ids
    except Exception as e:
        logger.warning(f"【Agent列表查询失败】降级为默认：{str(e)}")
        return ["default_agent"]


async def global_wuxing_daily_task():
    """多租户每日生命周期调度（含相克规则）"""
    try:
        logger.info(f"【多租户每日生命周期调度】{datetime.now()}")
        agent_ids = get_all_enable_agent()
        total_processed = 0
        total_triggered = 0
        stats_by_phase = {}

        for aid in agent_ids:
            offset = 0
            while True:
                memories = storage_manager.get_memories(agent_id=aid, limit=PAGE_SIZE, offset=offset)
                if not memories:
                    break

                # 批量调度（含相克规则）
                results = schedule_memory_batch(memories, aid)

                for result in results:
                    # 仅当有实际变化时才写回
                    if result.triggered or result.output_hot_score != result.input_hot_score:
                        storage_manager.update_memory(
                            result.memory_id,
                            aid,
                            {
                                "wuxing": result.output_wuxing,
                                "hot_score": result.output_hot_score,
                            },
                        )
                        total_triggered += 1
                        phase_key = result.output_phase.value
                        stats_by_phase[phase_key] = stats_by_phase.get(phase_key, 0) + 1

                    total_processed += 1

                if len(memories) < PAGE_SIZE:
                    break
                offset += PAGE_SIZE

        logger.info(
            "✅ 生命周期调度完成：" f"处理={total_processed}条，触发流转={total_triggered}条，" f"分布={stats_by_phase}"
        )
    except Exception as e:
        logger.error(f"生命周期调度异常：{str(e)}")


async def global_position_heal_task():
    """多租户全局自愈（基于 SelfHealingEngine）"""
    try:
        logger.info(f"【多租户全局自愈】{datetime.now()}")
        agent_ids = get_all_enable_agent()
        summary_stats = {"checked": 0, "auto_fixed": 0, "user_kept": 0, "skipped": 0}

        for aid in agent_ids:
            offset = 0
            while True:
                memories = storage_manager.get_memories(agent_id=aid, limit=PAGE_SIZE, offset=offset)
                if not memories:
                    break

                # 过滤：已标记正常的记忆直接跳过（减少重复校验）
                to_check = [m for m in memories if m.get("position_status") != PositionStatus.NORMAL]
                if len(to_check) < len(memories):
                    summary_stats["skipped"] += len(memories) - len(to_check)

                results = self_healer.heal_batch(to_check, aid)

                for result in results:
                    if result.action == "auto_fixed":
                        self_healer.apply_result_to_storage(result, storage_manager)
                        summary_stats["auto_fixed"] += 1
                    elif result.action == "user_kept":
                        summary_stats["user_kept"] += 1

                    summary_stats["checked"] += 1

                if len(memories) < PAGE_SIZE:
                    break
                offset += PAGE_SIZE

        logger.info(
            "✅ 全局自愈完成："
            f"校验={summary_stats['checked']}条，"
            f"自动修正={summary_stats['auto_fixed']}条，"
            f"保留={summary_stats['user_kept']}条，"
            f"跳过={summary_stats['skipped']}条（已确认正常）"
        )
    except Exception as e:
        logger.error(f"自愈任务异常：{str(e)}")


async def global_intelligent_decay_task():
    """多租户智能衰减任务"""
    try:
        logger.info(f"【多租户智能衰减】{datetime.now()}")
        agent_ids = get_all_enable_agent()
        total_processed = 0
        status_stats = {"hot": 0, "active": 0, "normal": 0, "archived": 0, "forgotten": 0}

        for aid in agent_ids:
            offset = 0
            while True:
                memories = storage_manager.get_memories(agent_id=aid, limit=PAGE_SIZE, offset=offset)
                if not memories:
                    break
                items = []
                for mem in memories:
                    item = {
                        "id": mem.get("id"),
                        "created_at": mem.get("created_at"),
                        "access_count": int(mem.get("access_count", 0)),
                        "last_accessed": mem.get("last_accessed"),
                        "importance": float(mem.get("importance", 0.5)),
                        "emotion_intensity": float(mem.get("emotion_intensity", 0.5)),
                        "memory_type": mem.get("memory_type", "normal"),
                        "content": mem.get("content", ""),
                        "vitality": float(mem.get("vitality", 1.0)),
                    }
                    items.append(item)

                if items:
                    batch_stats = decay_engine.batch_decay(items)
                    for status, count in batch_stats.items():
                        status_stats[status] = status_stats.get(status, 0) + count

                    for item in items:
                        storage_manager.update_memory(
                            item["id"],
                            aid,
                            {
                                "vitality": item.get("vitality", 0.0),
                                "status": item.get("status", "normal"),
                                "last_decay": item.get("last_decay"),
                            },
                        )
                    total_processed += len(items)

                if len(memories) < PAGE_SIZE:
                    break
                offset += PAGE_SIZE

        logger.info(f"✅ 智能衰减完成，共处理记忆：{total_processed} 条")
        logger.info(
            f"   状态分布：热门={status_stats['hot']} | 活跃={status_stats['active']} | 正常={status_stats['normal']} | 归档={status_stats['archived']} | 遗忘={status_stats['forgotten']}"
        )

        # 打印衰减状态
        decay_engine.print_status()
    except Exception as e:
        logger.error(f"智能衰减任务异常：{str(e)}")


def start_scheduler():
    """启动定时任务"""
    scheduler.add_job(global_wuxing_daily_task, "cron", hour=0, minute=0)
    scheduler.add_job(global_position_heal_task, "interval", hours=12)
    scheduler.add_job(global_intelligent_decay_task, "interval", hours=1)
    scheduler.start()
    logger.info("✅ 多租户定时任务已启动")


def shutdown_scheduler():
    """关闭定时任务"""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("✅ 定时任务已关闭")
