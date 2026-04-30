"""
智能衰减机制 - 记忆/情绪的智能淡化系统
支持：多维度衰减计算、自然遗忘、选择性保留、衰减追踪
"""

import logging
import math
import os
import sqlite3
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 尝试加载配置管理器（支持外部配置覆盖）
from core.config import get_config
from core.utils import get_logger


class IntelligentDecay:
    """智能衰减机制"""

    # ===== 外部可配置参数（通过 config_manager 覆盖） =====

    def __init__(self, db_path: str = "data/decay_stats.db"):
        # ===== 配置（通过 config_manager 读取，支持默认值） =====
        self.decay_factors = {
            "base_decay": {
                "core": get_config("decay.base_decay.core", 0.02) if get_config else 0.02,
                "important": get_config("decay.base_decay.important", 0.05) if get_config else 0.05,
                "normal": get_config("decay.base_decay.normal", 0.10) if get_config else 0.10,
                "temporary": get_config("decay.base_decay.temporary", 0.20) if get_config else 0.20,
                "chat": get_config("decay.base_decay.chat", 0.15) if get_config else 0.15,
                "task": get_config("decay.base_decay.task", 0.12) if get_config else 0.12,
                "knowledge": get_config("decay.base_decay.knowledge", 0.08) if get_config else 0.08,
                "rule": get_config("decay.base_decay.rule", 0.03) if get_config else 0.03,
            },
            "access_boost": {
                "base": get_config("decay.access_boost.base", 0.15) if get_config else 0.15,
                "recent": get_config("decay.access_boost.recent", 0.25) if get_config else 0.25,
                "frequent": get_config("decay.access_boost.frequent", 0.30) if get_config else 0.30,
            },
            "time_window": {
                "day": get_config("decay.time_window.day", 1.0) if get_config else 1.0,
                "week": get_config("decay.time_window.week", 0.8) if get_config else 0.8,
                "month": get_config("decay.time_window.month", 0.6) if get_config else 0.6,
                "quarter": get_config("decay.time_window.quarter", 0.4) if get_config else 0.4,
                "year": get_config("decay.time_window.year", 0.2) if get_config else 0.2,
            },
            "emotion_weight": get_config("decay.emotion_weight", 0.3) if get_config else 0.3,
            "relevance_weight": get_config("decay.relevance_weight", 0.2) if get_config else 0.2,
            "importance_weight": get_config("decay.importance_weight", 0.4) if get_config else 0.4,
        }
        self.thresholds = {
            "forget": get_config("decay.thresholds.forget", 0.1) if get_config else 0.1,
            "archive": get_config("decay.thresholds.archive", 0.3) if get_config else 0.3,
            "active": get_config("decay.thresholds.active", 0.6) if get_config else 0.6,
            "hot": get_config("decay.thresholds.hot", 0.8) if get_config else 0.8,
        }
        self.access_thresholds = {
            "frequent": get_config("decay.access_thresholds.frequent", 5) if get_config else 5,
            "recent": get_config("decay.access_thresholds.recent", 7) if get_config else 7,
        }

        # 衰减追踪

        # 衰减追踪
        self.decay_history = deque(maxlen=1000)  # 固定长度队列，避免内存泄漏
        self.total_decayed = 0
        self.total_forgotten = 0
        self.total_archived = 0

        # 数据库配置
        self.db_path = db_path
        self._init_db()
        self._load_stats()

        logger.info("🍂 智能衰减机制已初始化")
        logger.info(f"   核心记忆衰减：{self.decay_factors['base_decay']['core']*100:.1f}% / 天")
        logger.info(f"   重要记忆衰减：{self.decay_factors['base_decay']['important']*100:.1f}% / 天")
        logger.info(f"   普通记忆衰减：{self.decay_factors['base_decay']['normal']*100:.1f}% / 天")
        logger.info(f"   临时记忆衰减：{self.decay_factors['base_decay']['temporary']*100:.1f}% / 天")
        logger.info(f"   访问提升：{self.decay_factors['access_boost']['base']*100:.0f}% / 次")
        logger.info(f"   遗忘阈值：{self.thresholds['forget']*100:.0f}%")

    # ========== 数据库操作 ==========

    def _init_db(self):
        """初始化数据库"""
        # 确保数据目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # 连接数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建统计数据表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS decay_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE,
            value INTEGER,
            updated_at TEXT
        )
        """)

        # 初始化默认值
        default_stats = {"total_decayed": 0, "total_forgotten": 0, "total_archived": 0}

        for key, value in default_stats.items():
            cursor.execute(
                """
            INSERT OR IGNORE INTO decay_stats (key, value, updated_at)
            VALUES (?, ?, ?)
            """,
                (key, value, datetime.now().isoformat()),
            )

        conn.commit()
        conn.close()

    def _load_stats(self):
        """从数据库加载统计数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT key, value FROM decay_stats")
        for row in cursor.fetchall():
            key, value = row
            if key == "total_decayed":
                self.total_decayed = value
            elif key == "total_forgotten":
                self.total_forgotten = value
            elif key == "total_archived":
                self.total_archived = value

        conn.close()

    def _save_stats(self):
        """将统计数据保存到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {
            "total_decayed": self.total_decayed,
            "total_forgotten": self.total_forgotten,
            "total_archived": self.total_archived,
        }

        for key, value in stats.items():
            cursor.execute(
                """
            UPDATE decay_stats
            SET value = ?, updated_at = ?
            WHERE key = ?
            """,
                (value, datetime.now().isoformat(), key),
            )

        conn.commit()
        conn.close()

    # ========== 辅助函数 ==========

    def _get_memory_type(self, item: Dict) -> str:
        """获取记忆类型"""
        memory_type = item.get("type", "normal")
        # 兼容旧版记忆类型
        if "memory_type" in item:
            memory_type = item["memory_type"]
        return memory_type

    def _get_decay_rate(self, item: Dict) -> float:
        """根据记忆类型获取衰减率"""
        memory_type = self._get_memory_type(item)
        base_decay = self.decay_factors["base_decay"]

        # 优先使用具体类型的衰减率
        if memory_type in base_decay:
            return base_decay[memory_type]
        # 否则使用默认衰减率
        return base_decay["normal"]

    def _get_time_window_weight(self, item: Dict) -> float:
        """根据时间窗口获取权重"""
        last_accessed = item.get("last_accessed", item.get("created_at"))
        if not last_accessed:
            return 1.0

        days_since_access = (datetime.now() - datetime.fromisoformat(last_accessed)).days

        if days_since_access <= 1:
            return self.decay_factors["time_window"]["day"]
        elif days_since_access <= 7:
            return self.decay_factors["time_window"]["week"]
        elif days_since_access <= 30:
            return self.decay_factors["time_window"]["month"]
        elif days_since_access <= 90:
            return self.decay_factors["time_window"]["quarter"]
        elif days_since_access <= 365:
            return self.decay_factors["time_window"]["year"]
        else:
            return 0.1  # 超过1年，权重很低

    def _get_access_boost(self, item: Dict) -> float:
        """计算访问提升"""
        access_count = item.get("access_count", 0)
        base_boost = self.decay_factors["access_boost"]["base"]

        # 计算访问频率提升
        boost = base_boost * math.log1p(access_count)

        # 最近访问额外提升
        last_accessed = item.get("last_accessed")
        if last_accessed:
            days_since_access = (datetime.now() - datetime.fromisoformat(last_accessed)).days
            if days_since_access <= self.access_thresholds["recent"]:
                boost += self.decay_factors["access_boost"]["recent"]

        # 频繁访问额外提升
        if access_count >= self.access_thresholds["frequent"]:
            boost += self.decay_factors["access_boost"]["frequent"]

        return min(1.0, boost)  # 最大提升不超过1.0

    # ========== 衰减计算 ==========

    def calculate_decay(self, item: Dict) -> float:
        """
        计算项目的当前活力值（综合考虑多维度）

        Args:
            item: 记忆/情绪项目（包含 created_at, importance, access_count 等）

        Returns:
            活力值 (0-1)
        """
        # 1. 时间衰减（指数衰减，基于记忆类型）
        time_elapsed = (datetime.now() - datetime.fromisoformat(item["created_at"])).total_seconds() / 86400  # 转换为天
        decay_rate = self._get_decay_rate(item)
        time_factor = math.exp(-decay_rate * time_elapsed)

        # 2. 访问提升（基于访问频率和时间窗口）
        access_factor = self._get_access_boost(item)

        # 3. 情绪权重
        emotion_intensity = item.get("emotion_intensity", 0.5)
        emotion_factor = emotion_intensity * self.decay_factors["emotion_weight"]

        # 4. 相关性权重（如果有标签匹配）
        relevance = item.get("relevance", 0.5)
        relevance_factor = relevance * self.decay_factors["relevance_weight"]

        # 5. 重要度权重（基础权重）
        importance = item.get("importance", 0.5)
        importance_factor = importance * self.decay_factors["importance_weight"]

        # 6. 记忆长度因素（适中长度的记忆更容易保留）
        content_length = len(item.get("content", ""))
        if content_length < 10:
            length_factor = 0.7  # 太短的记忆可能不重要
        elif content_length > 1000:
            length_factor = 0.8  # 太长的记忆可能难以记忆
        else:
            length_factor = 1.0  # 适中长度的记忆

        # 7. 时间窗口权重（最近访问的记忆更活跃）
        time_window_factor = self._get_time_window_weight(item)

        # 8. 类型因素（不同类型的记忆有不同的衰减特性）
        memory_type = self._get_memory_type(item)
        type_factors = {
            "core": 1.2,  # 核心记忆
            "important": 1.1,  # 重要记忆
            "normal": 1.0,  # 普通记忆
            "temporary": 0.8,  # 临时记忆
            "chat": 0.9,  # 聊天记忆
            "task": 1.0,  # 任务记忆
            "knowledge": 1.1,  # 知识记忆
            "rule": 1.2,  # 规则记忆
            "text": 1.0,  # 文本记忆
            "image": 1.1,  # 图像记忆
            "audio": 1.0,  # 音频记忆
            "video": 0.9,  # 视频记忆（占用空间大）
        }
        type_factor = type_factors.get(memory_type, 1.0)

        # 综合计算 - 多因素加权
        vitality = (
            time_factor * 0.3  # 时间因素占 30%
            + access_factor * 0.25  # 访问因素占 25%
            + emotion_factor * 0.1  # 情绪因素占 10%
            + relevance_factor * 0.05  # 相关性因素占 5%
            + importance_factor * 0.2  # 重要度因素占 20%
            + length_factor * 0.025  # 长度因素占 2.5%
            + time_window_factor * 0.05  # 时间窗口因素占 5%
            + type_factor * 0.025  # 类型因素占 2.5%
        )

        # 归一化到 0-1
        vitality = min(1.0, max(0.0, vitality))

        return vitality

    def decay_item(self, item: Dict) -> Tuple[float, str]:
        """
        对项目执行衰减，返回新活力值和状态

        Args:
            item: 记忆/情绪项目

        Returns:
            (新活力值，状态变更)
        """
        old_vitality = item.get("vitality", 1.0)
        new_vitality = self.calculate_decay(item)

        # 确定状态
        if new_vitality < self.thresholds["forget"]:
            status = "forgotten"
            self.total_forgotten += 1
        elif new_vitality < self.thresholds["archive"]:
            status = "archived"
            self.total_archived += 1
        elif new_vitality > self.thresholds["hot"]:
            status = "hot"
        elif new_vitality > self.thresholds["active"]:
            status = "active"
        else:
            status = "normal"

        # 记录衰减历史
        history_item = {
            "item_id": item.get("id", "unknown"),
            "memory_type": self._get_memory_type(item),
            "old_vitality": old_vitality,
            "new_vitality": new_vitality,
            "status_change": status,
            "timestamp": datetime.now().isoformat(),
            "access_count": item.get("access_count", 0),
            "importance": item.get("importance", 0.5),
        }
        self.decay_history.append(history_item)

        self.total_decayed += 1

        # 更新项目
        item["vitality"] = new_vitality
        item["last_decay"] = datetime.now().isoformat()
        item["status"] = status

        # 保存统计数据到数据库
        self._save_stats()

        return new_vitality, status

    # ========== 批量衰减 ==========

    def batch_decay(self, items: List[Dict]) -> Dict[str, int]:
        """
        批量衰减项目

        Args:
            items: 项目列表

        Returns:
            状态统计
        """
        stats = {"hot": 0, "active": 0, "normal": 0, "archived": 0, "forgotten": 0}

        for item in items:
            _, status = self.decay_item(item)
            stats[status] = stats.get(status, 0) + 1

        logger.info(f"🍂 批量衰减完成：{len(items)} 个项目")
        logger.info(
            f"   热门：{stats['hot']} | 活跃：{stats['active']} | 正常：{stats['normal']} | 归档：{stats['archived']} | 遗忘：{stats['forgotten']}"
        )

        return stats

    # ========== 访问增强 ==========

    def access_boost(self, item: Dict) -> float:
        """
        访问提升（记忆被访问时增强活力）

        Args:
            item: 项目

        Returns:
            提升后的活力值
        """
        # 增加访问计数
        item["access_count"] = item.get("access_count", 0) + 1
        item["last_accessed"] = datetime.now().isoformat()

        # 重新计算活力
        new_vitality = self.calculate_decay(item)
        item["vitality"] = new_vitality

        # 更新状态
        if new_vitality < self.thresholds["forget"]:
            status = "forgotten"
        elif new_vitality < self.thresholds["archive"]:
            status = "archived"
        elif new_vitality > self.thresholds["hot"]:
            status = "hot"
        elif new_vitality > self.thresholds["active"]:
            status = "active"
        else:
            status = "normal"

        item["status"] = status

        # 如果从归档/遗忘恢复到活跃，记录
        old_status = item.get("status", "active")
        if old_status in ["archived", "forgotten"] and new_vitality > self.thresholds["active"]:
            logger.info(f"✨ 记忆恢复：{item.get('content', '')[:30]}... (活力：{new_vitality:.2f})")

        return new_vitality

    # ========== 智能遗忘 ==========

    def intelligent_forget(self, items: List[Dict], force_keep: List[str] = None) -> List[Dict]:
        """
        智能遗忘（自动筛选应遗忘的项目）

        Args:
            items: 项目列表
            force_keep: 强制保留的项目 ID 列表

        Returns:
            应遗忘的项目列表
        """
        force_keep = force_keep or []
        to_forget = []

        for item in items:
            # 强制保留的跳过
            if item.get("id") in force_keep:
                continue

            # 高重要度的保留
            if item.get("importance", 0) > 0.8:
                continue

            # 最近访问的保留
            last_accessed = item.get("last_accessed")
            if last_accessed:
                days_since_access = (datetime.now() - datetime.fromisoformat(last_accessed)).days
                if days_since_access < self.access_thresholds["recent"]:  # 7 天内访问过
                    continue

            # 频繁访问的保留
            if item.get("access_count", 0) >= self.access_thresholds["frequent"]:
                continue

            # 核心/规则记忆保留
            memory_type = self._get_memory_type(item)
            if memory_type in ["core", "rule", "knowledge"]:
                continue

            # 计算活力
            vitality = self.calculate_decay(item)

            # 低于遗忘阈值的标记为遗忘
            if vitality < self.thresholds["forget"]:
                to_forget.append(item)

        if to_forget:
            logger.info(f"🍂 智能遗忘：{len(to_forget)} 个项目应被遗忘")

        return to_forget

    # ========== 选择性保留 ==========

    def selective_preserve(self, items: List[Dict], preserve_criteria: Dict = None) -> List[Dict]:
        """
        选择性保留（筛选应重点保留的项目）

        Args:
            items: 项目列表
            preserve_criteria: 保留标准

        Returns:
            应保留的项目列表
        """
        preserve_criteria = preserve_criteria or {
            "min_importance": 0.7,
            "min_access_count": self.access_thresholds["frequent"],
            "recent_days": 30,
            "high_emotion": 0.8,
        }

        to_preserve = []

        for item in items:
            should_preserve = False

            # 高重要度
            if item.get("importance", 0) >= preserve_criteria["min_importance"]:
                should_preserve = True

            # 高频访问
            if item.get("access_count", 0) >= preserve_criteria["min_access_count"]:
                should_preserve = True

            # 近期访问
            last_accessed = item.get("last_accessed")
            if last_accessed:
                days_since = (datetime.now() - datetime.fromisoformat(last_accessed)).days
                if days_since <= preserve_criteria["recent_days"]:
                    should_preserve = True

            # 高情绪强度
            if item.get("emotion_intensity", 0) >= preserve_criteria["high_emotion"]:
                should_preserve = True

            # 核心记忆类型
            memory_type = self._get_memory_type(item)
            if memory_type in ["core", "rule", "knowledge"]:
                should_preserve = True

            if should_preserve:
                to_preserve.append(item)

        if to_preserve:
            logger.info(f"🔒 选择性保留：{len(to_preserve)} 个项目")

        return to_preserve

    # ========== 衰减追踪 ==========

    def get_decay_stats(self) -> Dict:
        """获取衰减统计"""
        return {
            "total_decayed": self.total_decayed,
            "total_forgotten": self.total_forgotten,
            "total_archived": self.total_archived,
            "forget_rate": self.total_forgotten / max(self.total_decayed, 1),
            "archive_rate": self.total_archived / max(self.total_decayed, 1),
            "history_count": len(self.decay_history),
        }

    def get_vitality_distribution(self, items: List[Dict]) -> Dict[str, int]:
        """获取活力分布"""
        distribution = {
            "very_high": 0,  # 0.8-1.0
            "high": 0,  # 0.6-0.8
            "medium": 0,  # 0.4-0.6
            "low": 0,  # 0.2-0.4
            "very_low": 0,  # 0.0-0.2
        }

        for item in items:
            vitality = item.get("vitality", self.calculate_decay(item))

            if vitality >= 0.8:
                distribution["very_high"] += 1
            elif vitality >= 0.6:
                distribution["high"] += 1
            elif vitality >= 0.4:
                distribution["medium"] += 1
            elif vitality >= 0.2:
                distribution["low"] += 1
            else:
                distribution["very_low"] += 1

        return distribution

    def get_type_based_decay_stats(self, items: List[Dict]) -> Dict[str, Dict]:
        """获取基于类型的衰减统计"""
        type_stats = {}

        for item in items:
            memory_type = self._get_memory_type(item)
            if memory_type not in type_stats:
                type_stats[memory_type] = {
                    "count": 0,
                    "total_vitality": 0,
                    "average_vitality": 0,
                    "status_distribution": {"hot": 0, "active": 0, "normal": 0, "archived": 0, "forgotten": 0},
                }

            type_stats[memory_type]["count"] += 1
            vitality = item.get("vitality", self.calculate_decay(item))
            type_stats[memory_type]["total_vitality"] += vitality

            # 确定状态
            if vitality < self.thresholds["forget"]:
                status = "forgotten"
            elif vitality < self.thresholds["archive"]:
                status = "archived"
            elif vitality > self.thresholds["hot"]:
                status = "hot"
            elif vitality > self.thresholds["active"]:
                status = "active"
            else:
                status = "normal"

            type_stats[memory_type]["status_distribution"][status] += 1

        # 计算平均活力
        for memory_type in type_stats:
            if type_stats[memory_type]["count"] > 0:
                type_stats[memory_type]["average_vitality"] = (
                    type_stats[memory_type]["total_vitality"] / type_stats[memory_type]["count"]
                )

        return type_stats

    def print_status(self):
        """打印状态"""
        stats = self.get_decay_stats()

        logger.info("\n" + "=" * 60)
        logger.info("【智能衰减机制】")
        logger.info("=" * 60)
        logger.info(f"总衰减次数：{stats['total_decayed']}")
        logger.info(f"遗忘数量：{stats['total_forgotten']}")
        logger.info(f"归档数量：{stats['total_archived']}")
        logger.info(f"遗忘率：{stats['forget_rate']*100:.1f}%")
        logger.info(f"归档率：{stats['archive_rate']*100:.1f}%")
        logger.info(f"衰减历史记录：{stats['history_count']} 条")

        logger.info("\n衰减因子:")
        logger.info("  基础衰减率:")
        for memory_type, rate in self.decay_factors["base_decay"].items():
            logger.info(f"    {memory_type:15} {rate*100:.1f}%/天")

        logger.info("  访问提升:")
        for boost_type, value in self.decay_factors["access_boost"].items():
            logger.info(f"    {boost_type:15} {value*100:.0f}%")

        logger.info("  时间窗口权重:")
        for window, weight in self.decay_factors["time_window"].items():
            logger.info(f"    {window:15} {weight:.2f}")

        logger.info("\n阈值:")
        for threshold, value in self.thresholds.items():
            logger.info(f"  {threshold:20} {value:.2f}")

        logger.info("=" * 60)

    def print_decay_history(self, limit: int = 20):
        """打印衰减历史"""
        logger.info("\n【最近衰减历史】")

        if not self.decay_history:
            logger.info("  无衰减记录")
            return

        for i, record in enumerate(self.decay_history[-limit:], 1):
            time_str = record["timestamp"].split("T")[1].split(".")[0]
            old_v = record["old_vitality"]
            new_v = record["new_vitality"]
            status = record["status_change"]
            memory_type = record.get("memory_type", "normal")

            # 状态图标
            if status == "forgotten":
                emoji = "🗑️"
            elif status == "archived":
                emoji = "📦"
            elif status == "hot":
                emoji = "🔥"
            elif status == "active":
                emoji = "✨"
            else:
                emoji = "📝"

            logger.info(f"  {i}. [{time_str}] {emoji} [{memory_type}] {old_v:.2f} → {new_v:.2f} ({status})")
