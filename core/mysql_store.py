#!/usr/bin/env python3
"""
🗄️ MySQL 存储层 - 多租户记忆存储

IMA V2.0 规范 - 生产级存储实现
功能：CRUD 操作 + 多租户隔离 + 爻变历史追踪 + 安全验证
"""

import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from sqlalchemy import create_engine, func, or_, text
    from sqlalchemy.orm import Session, sessionmaker
except ImportError:
    create_engine = func = or_ = text = None
    Session = sessionmaker = None

from core.config import get_config
from core.utils import (
    DatabaseError,
    ValidationError,
    get_logger,
    validate_dict,
    validate_float,
    validate_integer,
    validate_list,
    validate_required_fields,
    validate_string,
)

from .models import Base, TaijiCoreConfig, YaoChangeHistory, YijingMemory

logger = get_logger(__name__)


# 异常定义
class MySQLStoreError(DatabaseError):
    """MySQL 存储异常"""


# ========== 数据库连接管理 ==========


class DatabaseManager:
    """
    数据库管理器

    负责连接池管理、会话管理、事务管理
    """

    def __init__(self, database_url: str = None, pool_size: int = None, max_overflow: int = None):
        """
        初始化数据库

        Args:
            database_url: 数据库连接 URL
                示例：mysql+pymysql://user:***@localhost:3306/yijing_memory_db
            pool_size: 连接池大小
            max_overflow: 最大溢出连接数
        """
        # 从配置文件读取连接参数
        if database_url is None:
            host = get_config("database.mysql.host", "localhost")
            port = get_config("database.mysql.port", 3306)
            user = get_config("database.mysql.user", "root")
            password = get_config("database.mysql.password", "password")
            database = get_config("database.mysql.database", "memory_system")
            database_url = f"mysql+pymysql://{user}:***@{host}:{port}/{database}"

        if pool_size is None:
            pool_size = get_config("database.mysql.pool_size", 10)

        if max_overflow is None:
            max_overflow = get_config("database.mysql.max_overflow", 20)

        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow

        try:
            # 创建引擎
            self.engine = create_engine(
                database_url,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_pre_ping=True,  # 连接前检查
                pool_recycle=3600,  # 1 小时回收
                echo=False,  # 生产环境关闭 SQL 日志
            )

            # 创建会话工厂
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.SessionLocal = SessionLocal

            logger.info(f"✅ 数据库已初始化：{database_url[:50]}...")
        except Exception as e:
            logger.error(f"❌ 数据库初始化失败：{e}")
            raise MySQLStoreError(f"数据库初始化失败：{e}")

    def create_tables(self):
        """创建所有表"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("✅ 数据库表已创建")
        except Exception as e:
            logger.error(f"❌ 创建数据库表失败：{e}")
            raise MySQLStoreError(f"创建数据库表失败：{e}")

    def drop_tables(self):
        """删除所有表"""
        try:
            Base.metadata.drop_all(self.engine)
            logger.warning("⚠️ 数据库表已删除")
        except Exception as e:
            logger.error(f"❌ 删除数据库表失败：{e}")
            raise MySQLStoreError(f"删除数据库表失败：{e}")

    @contextmanager
    def get_session(self) -> Session:
        """获取数据库会话 (上下文管理器)"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"❌ 数据库事务失败：{e}")
            raise MySQLStoreError(f"数据库事务失败：{e}")
        finally:
            session.close()

    def health_check(self) -> bool:
        """健康检查"""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            logger.debug("✅ 数据库健康检查通过")
            return True
        except Exception as e:
            logger.error(f"❌ 数据库健康检查失败：{e}")
            return False


# ========== MySQL 存储实现 ==========


class MySQLMemoryStore:
    # 进程内内存缓存（绕过 SQLite session 隔离问题）
    _memory_cache: Dict[str, Dict[str, Any]] = {}

    """
    MySQL 记忆存储

    实现完整的 CRUD 操作，支持多租户隔离
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self._lock = threading.RLock()  # 线程安全锁

    # ---------- 创建 ----------

    def create_memory(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建记忆

        Args:
            memory_data: 记忆数据字典

        Returns:
            创建的记忆对象
        """
        try:
            # 验证输入
            validate_dict(memory_data, "memory_data")
            validate_required_fields(memory_data, ["agent_id", "hexagram", "content"])
            validate_string(memory_data.get("agent_id"), "agent_id", min_length=1)
            # 兼容处理：hexagram 可能是 list（[1,0,1,0,1,0]）或 string（"101010"）
            if isinstance(memory_data.get("hexagram"), list):
                memory_data["hexagram"] = "".join(str(b) for b in memory_data["hexagram"])
            validate_string(memory_data.get("hexagram"), "hexagram", min_length=1)
            validate_string(memory_data.get("content"), "content", min_length=1)
            if "hot" in memory_data:
                validate_float(memory_data.get("hot"), "hot", min_value=0, max_value=1)
                memory_data["hot_score"] = memory_data.pop("hot")
            if "weight" in memory_data:
                validate_float(memory_data.get("weight"), "weight", min_value=0, max_value=1)
            if "layer" in memory_data:
                memory_data["sancai_layer"] = memory_data.pop("layer")
            if "element" in memory_data:
                memory_data["wuxing"] = memory_data.pop("element")
            if "trigram" in memory_data:
                memory_data["bagua_type"] = memory_data.pop("trigram")
            if "memory_type" in memory_data:
                # 移除 memory_type 字段，因为 YijingMemory 模型中没有这个字段
                memory_data.pop("memory_type")
            if "priority" in memory_data:
                # 移除 priority 字段，因为 YijingMemory 模型中没有这个字段
                memory_data.pop("priority")
            # 处理 datetime 类型转换
            from datetime import datetime

            if "expire_time" in memory_data and memory_data["expire_time"]:
                if isinstance(memory_data["expire_time"], str):
                    try:
                        # 尝试解析字符串为 datetime 对象
                        memory_data["expire_time"] = datetime.strptime(memory_data["expire_time"], "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        # 如果解析失败，使用当前时间
                        memory_data["expire_time"] = datetime.now()
            if "last_accessed_at" in memory_data and memory_data["last_accessed_at"]:
                if isinstance(memory_data["last_accessed_at"], str):
                    try:
                        # 尝试解析字符串为 datetime 对象
                        memory_data["last_accessed_at"] = datetime.strptime(
                            memory_data["last_accessed_at"], "%Y-%m-%d %H:%M:%S"
                        )
                    except ValueError:
                        # 如果解析失败，使用当前时间
                        memory_data["last_accessed_at"] = datetime.now()

            # 确保 memory_id 存在
            if "memory_id" not in memory_data or memory_data["memory_id"] is None:
                from core.utils import generate_id

                memory_data["memory_id"] = generate_id(f"{memory_data.get('agent_id')}:{memory_data.get('content')}")

            # 确保 bagua_type 存在
            if "bagua_type" not in memory_data or memory_data["bagua_type"] is None:
                memory_data["bagua_type"] = "乾"

            # 确保 sancai_layer 存在
            if "sancai_layer" not in memory_data or memory_data["sancai_layer"] is None:
                memory_data["sancai_layer"] = "天"

            # 确保 wuxing 存在
            if "wuxing" not in memory_data or memory_data["wuxing"] is None:
                memory_data["wuxing"] = "金"

            # 确保 position_status 存在
            if "position_status" not in memory_data or memory_data["position_status"] is None:
                memory_data["position_status"] = "当位"

            with self.db.get_session() as session:
                memory = YijingMemory(**memory_data)
                session.add(memory)
                session.flush()  # 获取 ID
                session.commit()  # 确保写入持久化

                memory_dict = {
                    "id": memory.id,
                    "memory_id": memory.memory_id,
                    "agent_id": memory.agent_id,
                    "content": memory.content,
                    "hexagram": memory.hexagram,
                    "bagua_type": memory.bagua_type,
                    "sancai_layer": memory.sancai_layer,
                    "wuxing": memory.wuxing,
                    "hot_score": memory.hot_score,
                    "position_status": memory.position_status,
                    "created_at": memory.created_at.isoformat() if memory.created_at else None,
                    "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                }
                logger.info(
                    f"📝 创建记忆：ID={memory.id}, agent={memory.agent_id}, memory_id={memory.memory_id}, engine={self.db.engine}"
                )
                # 写入缓存（key: memory_id）
                self._memory_cache[memory_dict["memory_id"]] = memory_dict
                return memory_dict
        except Exception as e:
            logger.error(f"❌ 创建记忆失败：{e}")
            raise MySQLStoreError(f"创建记忆失败：{e}")

    def create_memories_batch(self, memories_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        批量创建记忆

        Args:
            memories_data: 记忆数据字典列表

        Returns:
            创建的记忆对象列表
        """
        try:
            # 验证输入
            validate_list(memories_data, "memories_data", min_length=1)
            from core.utils import generate_id

            for memory_data in memories_data:
                validate_dict(memory_data, "memory_data")
                validate_required_fields(memory_data, ["agent_id", "hexagram", "content"])
                validate_string(memory_data.get("agent_id"), "agent_id", min_length=1)
                validate_string(memory_data.get("hexagram"), "hexagram", min_length=1)
                validate_string(memory_data.get("content"), "content", min_length=1)
                if "hot" in memory_data:
                    validate_float(memory_data.get("hot"), "hot", min_value=0, max_value=1)
                    memory_data["hot_score"] = memory_data.pop("hot")
                if "weight" in memory_data:
                    validate_float(memory_data.get("weight"), "weight", min_value=0, max_value=1)
                if "layer" in memory_data:
                    memory_data["sancai_layer"] = memory_data.pop("layer")
                if "element" in memory_data:
                    memory_data["wuxing"] = memory_data.pop("element")
                if "trigram" in memory_data:
                    memory_data["bagua_type"] = memory_data.pop("trigram")
                if "memory_type" in memory_data:
                    # 移除 memory_type 字段，因为 YijingMemory 模型中没有这个字段
                    memory_data.pop("memory_type")
                if "priority" in memory_data:
                    # 移除 priority 字段，因为 YijingMemory 模型中没有这个字段
                    memory_data.pop("priority")

                # 确保 memory_id 存在
                if "memory_id" not in memory_data or memory_data["memory_id"] is None:
                    memory_data["memory_id"] = generate_id(
                        f"{memory_data.get('agent_id')}:{memory_data.get('content')}"
                    )

                # 确保 bagua_type 存在
                if "bagua_type" not in memory_data or memory_data["bagua_type"] is None:
                    memory_data["bagua_type"] = "乾"

                # 确保 sancai_layer 存在
                if "sancai_layer" not in memory_data or memory_data["sancai_layer"] is None:
                    memory_data["sancai_layer"] = "天"

                # 确保 wuxing 存在
                if "wuxing" not in memory_data or memory_data["wuxing"] is None:
                    memory_data["wuxing"] = "金"

                # 确保 position_status 存在
                if "position_status" not in memory_data or memory_data["position_status"] is None:
                    memory_data["position_status"] = "当位"

            with self.db.get_session() as session:
                memories = [YijingMemory(**data) for data in memories_data]
                session.add_all(memories)
                session.flush()

                # 清理过期记忆（后台维护，不阻塞主事务）
                if memories_data:
                    agent_id = memories_data[0].get("agent_id")
                    if agent_id:
                        try:
                            self.clean_expired_memories(agent_id)
                        except Exception as cleanup_err:
                            logger.warning(f"清理过期记忆失败（不影响写入）：{cleanup_err}")

                memories_list = [
                    {
                        "id": m.id,
                        "memory_id": m.memory_id,
                        "agent_id": m.agent_id,
                        "content": m.content,
                        "hexagram": m.hexagram,
                        "bagua_type": m.bagua_type,
                        "sancai_layer": m.sancai_layer,
                        "wuxing": m.wuxing,
                        "hot_score": m.hot_score,
                        "position_status": m.position_status,
                        "created_at": m.created_at.isoformat() if m.created_at else None,
                        "updated_at": m.updated_at.isoformat() if m.updated_at else None,
                    }
                    for m in memories
                ]
                logger.info(f"📝 批量创建 {len(memories)} 条记忆")
                return memories_list
        except Exception as e:
            logger.error(f"❌ 批量创建记忆失败：{e}")
            raise MySQLStoreError(f"批量创建记忆失败：{e}")

    # ---------- 读取 ----------

    def get_memory(self, memory_id: int, agent_id: str) -> Optional[Dict]:
        """
        获取记忆 (带租户校验)

        Args:
            memory_id: 记忆 ID
            agent_id: 租户 ID

        Returns:
            记忆字典或 None
        """
        try:
            # 验证输入
            validate_integer(memory_id, "memory_id", min_value=1)
            validate_string(agent_id, "agent_id", min_length=1)

            with self.db.get_session() as session:
                memory = (
                    session.query(YijingMemory)
                    .filter(YijingMemory.id == memory_id, YijingMemory.agent_id == agent_id)
                    .first()
                )
                if not memory:
                    return None

                # 更新访问时间和访问计数
                memory.last_accessed_at = datetime.now()
                if hasattr(memory, "access_count"):
                    memory.access_count = (memory.access_count or 0) + 1
                else:
                    # 如果没有 access_count 字段，添加一个
                    memory.access_count = 1

                # 提升热度
                # 基础热度提升
                base_boost = 0.1
                # 最近访问额外提升
                if memory.last_accessed_at:
                    days_since_access = (
                        (datetime.now() - memory.last_accessed_at).days if memory.last_accessed_at else 0
                    )
                    if days_since_access <= 7:
                        base_boost += 0.05
                # 频繁访问额外提升
                if hasattr(memory, "access_count") and memory.access_count >= 5:
                    base_boost += 0.05

                # 计算新热度
                new_hot = min(1.0, memory.hot_score + base_boost)
                memory.hot_score = new_hot

                # 转换为字典
                memory_dict = {
                    "id": memory.id,
                    "agent_id": memory.agent_id,
                    "memory_id": memory.memory_id,
                    "hexagram": memory.hexagram,
                    "bagua_type": memory.bagua_type,
                    "hexagram_num": memory.hexagram_num,
                    "sancai_layer": memory.sancai_layer,
                    "wuxing": memory.wuxing,
                    "position_status": memory.position_status,
                    "content": memory.content,
                    "weight": memory.weight,
                    "hot": memory.hot_score,
                    "hot_score": memory.hot_score,
                    "redundancy": memory.redundancy,
                    "last_accessed_at": memory.last_accessed_at,
                    "expire_time": memory.expire_time,
                    "yao_history": memory.yao_history,
                    "relations": memory.relations,
                    "source": memory.source,
                    "category": memory.category,
                    "tags": memory.tags,
                    "status": memory.status,
                    "created_at": memory.created_at,
                    "updated_at": memory.updated_at,
                    "access_count": memory.access_count,
                }
                return memory_dict
        except Exception as e:
            logger.error(f"❌ 获取记忆失败：{e}")
            raise MySQLStoreError(f"获取记忆失败：{e}")

    def get_memory_by_memory_id(self, memory_id: str, agent_id: str = "") -> Optional[Dict]:
        """
        通过字符串 memory_id 获取记忆（不依赖自增 ID）
        使用 session 查询（与 create/get_memories 保持一致）
        """
        try:
            validate_string(memory_id, "memory_id", min_length=1)
            # 先从缓存读（绕过 session 隔离问题）
            cached = self._memory_cache.get(memory_id)
            if cached:
                # 跳过已删除的记录
                if cached.get("status") == "deleted":
                    return None
                logger.info(f"🔍 get_memory_by_memory_id cache HIT: {memory_id}")
                return cached
            logger.info(f"🔍 get_memory_by_memory_id cache MISS: {memory_id}")
            with self.db.get_session() as session:
                query = session.query(YijingMemory).filter(
                    YijingMemory.memory_id == memory_id,
                    # 过滤已删除的记录
                    or_(YijingMemory.status.is_(None), YijingMemory.status != "deleted"),
                )
                if agent_id:
                    query = query.filter(YijingMemory.agent_id == agent_id)
                memory = query.first()
                if not memory:
                    return None
                # 更新访问
                memory.last_accessed_at = datetime.now()
                memory.hot_score = min(1.0, memory.hot_score + 0.1)
                if hasattr(memory, "access_count"):
                    memory.access_count = (memory.access_count or 0) + 1
                session.commit()
                return {
                    "id": memory.id,
                    "agent_id": memory.agent_id,
                    "memory_id": memory.memory_id,
                    "hexagram": memory.hexagram,
                    "bagua_type": memory.bagua_type,
                    "hexagram_num": memory.hexagram_num,
                    "sancai_layer": memory.sancai_layer,
                    "wuxing": memory.wuxing,
                    "position_status": memory.position_status,
                    "content": memory.content,
                    "weight": memory.weight,
                    "hot": memory.hot_score,
                    "hot_score": memory.hot_score,
                    "redundancy": memory.redundancy,
                    "last_accessed_at": memory.last_accessed_at,
                    "expire_time": memory.expire_time,
                    "yao_history": memory.yao_history,
                    "relations": memory.relations,
                    "source": memory.source,
                    "category": memory.category,
                    "tags": memory.tags,
                    "status": memory.status,
                    "created_at": memory.created_at,
                    "updated_at": memory.updated_at,
                    "access_count": getattr(memory, "access_count", 0),
                }
        except Exception as e:
            logger.error(f"❌ 获取记忆失败：{e}")
            raise MySQLStoreError(f"获取记忆失败：{e}")

    def get_memories_by_ids(
        self,
        memory_ids: List[str],
        agent_id: Optional[str] = None,
        bagua_type: Optional[str] = None,
        layer: Optional[str] = None,
        min_hot: float = 0.0,
    ) -> List[Dict]:
        """
        根据 ID 列表批量查询记忆（用于混合检索候选集补充）

        Args:
            memory_ids: 记忆 ID 列表
            agent_id: 租户 ID（可选）
            bagua_type: 八卦类型过滤
            layer: 三才层级过滤
            min_hot: 最低热度过滤

        Returns:
            符合条件的记忆列表
        """
        if not memory_ids:
            return []
        try:
            with self.db.get_session() as session:
                query = session.query(YijingMemory)
                # IN 查询
                str_ids = [str(i) for i in memory_ids]
                query = query.filter(YijingMemory.id.in_(str_ids))
                if agent_id:
                    query = query.filter(YijingMemory.agent_id == agent_id)
                if bagua_type:
                    query = query.filter(YijingMemory.bagua_type == bagua_type)
                if layer:
                    query = query.filter(YijingMemory.sancai_layer == layer)
                if min_hot > 0:
                    query = query.filter(YijingMemory.hot_score >= min_hot)
                memories = query.limit(len(memory_ids) * 2).all()
                result = []
                for m in memories:
                    result.append(
                        {
                            "id": m.id,
                            "memory_id": m.memory_id,
                            "agent_id": m.agent_id,
                            "hexagram": m.hexagram,
                            "bagua_type": m.bagua_type,
                            "sancai_layer": m.sancai_layer,
                            "wuxing": m.wuxing,
                            "hot_score": m.hot_score,
                            "content": m.content,
                            "weight": m.weight,
                            "is_core": m.is_core,
                            "position_status": m.position_status,
                            "importance": m.importance,
                            "vitality": m.vitality,
                            "status": m.status,
                            "created_at": m.created_at,
                            "updated_at": m.updated_at,
                        }
                    )
                return result
        except Exception as e:
            logger.error(f"❌ ID批量查询失败：{e}")
            return []

    def get_memories(
        self,
        agent_id: str,
        limit: int = 100,
        offset: int = 0,
        layer: Optional[str] = None,
        element: Optional[str] = None,
        trigram: Optional[str] = None,
        min_hot: Optional[float] = None,
        status: str = "active",
    ) -> List[Dict]:
        """
        查询记忆 (支持多条件筛选)

        Args:
            agent_id: 租户 ID
            limit: 返回数量限制
            offset: 偏移量
            layer: 三才层筛选
            element: 五行筛选
            trigram: 八卦筛选
            min_hot: 最小热度
            status: 状态筛选

        Returns:
            记忆字典列表
        """
        try:
            with self.db.get_session() as session:
                query = session.query(YijingMemory)
                if agent_id:
                    query = query.filter(YijingMemory.agent_id == agent_id)
                if status and status != "all":
                    query = query.filter(YijingMemory.status == status)

                # 条件筛选
                if layer:
                    query = query.filter(YijingMemory.sancai_layer == layer)
                if element:
                    query = query.filter(YijingMemory.wuxing == element)
                if trigram:
                    query = query.filter(YijingMemory.bagua_type == trigram)
                if min_hot is not None:
                    query = query.filter(YijingMemory.hot_score >= min_hot)

                # 分页
                query = query.order_by(YijingMemory.created_at.desc())
                query = query.offset(offset).limit(limit)

                memory_objects = query.all()

                # 转换为字典列表
                memories = []
                for memory in memory_objects:
                    memory_dict = {
                        "id": memory.id,
                        "agent_id": memory.agent_id,
                        "memory_id": memory.memory_id,
                        "hexagram": memory.hexagram,
                        "bagua_type": memory.bagua_type,
                        "hexagram_num": memory.hexagram_num,
                        "sancai_layer": memory.sancai_layer,
                        "wuxing": memory.wuxing,
                        "position_status": memory.position_status,
                        "content": memory.content,
                        "weight": memory.weight,
                        "hot": memory.hot_score,
                        "hot_score": memory.hot_score,
                        "redundancy": memory.redundancy,
                        "last_accessed_at": memory.last_accessed_at,
                        "expire_time": memory.expire_time,
                        "yao_history": memory.yao_history,
                        "relations": memory.relations,
                        "source": memory.source,
                        "category": memory.category,
                        "tags": memory.tags,
                        "status": memory.status,
                        "created_at": memory.created_at,
                        "updated_at": memory.updated_at,
                    }
                    memories.append(memory_dict)

                logger.debug(f"✅ 查询到 {len(memories)} 条记忆")
                return memories
        except Exception as e:
            logger.error(f"❌ 查询记忆失败：{e}")
            raise MySQLStoreError(f"查询记忆失败：{e}")

    def count_memories(self, agent_id: str, status: str = "active") -> int:
        """
        统计记忆数量

        Args:
            agent_id: 租户 ID
            status: 状态筛选

        Returns:
            记忆数量
        """
        try:
            with self.db.get_session() as session:
                query = session.query(YijingMemory)
                if status and status != "all":
                    query = query.filter(YijingMemory.status == status)
                if agent_id:
                    query = query.filter(YijingMemory.agent_id == agent_id)
                return query.count()
        except Exception as e:
            logger.warn(f"⚠️ 统计失败: {e}, 返回0")
            return 0

        # 验证输入
        validate_string(agent_id, "agent_id", min_length=1)
        if status not in ["active", "deleted", "archived"]:
            raise ValidationError("status must be one of: active, deleted, archived")

        with self.db.get_session() as session:
            count = (
                session.query(YijingMemory)
                .filter(YijingMemory.agent_id == agent_id, YijingMemory.status == status)
                .count()
            )
        return count

    # ---------- 更新 ----------

    def update_memory(self, memory_id: int, agent_id: str, updates: Dict[str, Any]) -> Optional[Dict]:
        """
        更新记忆

        Args:
            memory_id: 记忆 ID
            agent_id: 租户 ID
            updates: 更新数据字典

        Returns:
            更新后的记忆字典或 None
        """
        try:
            # 验证输入
            validate_integer(memory_id, "memory_id", min_value=1)
            validate_string(agent_id, "agent_id", min_length=1)
            validate_dict(updates, "updates")

            with self.db.get_session() as session:
                memory = (
                    session.query(YijingMemory)
                    .filter(YijingMemory.id == memory_id, YijingMemory.agent_id == agent_id)
                    .first()
                )

                if not memory:
                    return None

                # 更新字段
                for key, value in updates.items():
                    if key == "hot":
                        # 处理 hot 字段，转换为 hot_score
                        if hasattr(memory, "hot_score"):
                            setattr(memory, "hot_score", value)
                    elif hasattr(memory, key):
                        setattr(memory, key, value)

                session.flush()
                logger.info(f"✏️ 更新记忆：ID={memory_id}")

                # 转换为字典
                memory_dict = {
                    "id": memory.id,
                    "agent_id": memory.agent_id,
                    "memory_id": memory.memory_id,
                    "hexagram": memory.hexagram,
                    "bagua_type": memory.bagua_type,
                    "hexagram_num": memory.hexagram_num,
                    "sancai_layer": memory.sancai_layer,
                    "wuxing": memory.wuxing,
                    "position_status": memory.position_status,
                    "content": memory.content,
                    "weight": memory.weight,
                    "hot": memory.hot_score,
                    "hot_score": memory.hot_score,
                    "redundancy": memory.redundancy,
                    "last_accessed_at": memory.last_accessed_at,
                    "expire_time": memory.expire_time,
                    "yao_history": memory.yao_history,
                    "relations": memory.relations,
                    "source": memory.source,
                    "category": memory.category,
                    "tags": memory.tags,
                    "status": memory.status,
                    "created_at": memory.created_at,
                    "updated_at": memory.updated_at,
                }
                return memory_dict
        except Exception as e:
            logger.error(f"❌ 更新记忆失败：{e}")
            raise MySQLStoreError(f"更新记忆失败：{e}")

    def update_memory_hot(self, memory_id: int, agent_id: str, new_hot: float) -> bool:
        """
        更新记忆热度

        Args:
            memory_id: 记忆 ID
            agent_id: 租户 ID
            new_hot: 新热度值

        Returns:
            是否更新成功
        """
        try:
            # 验证输入
            validate_float(new_hot, "new_hot", min_value=0, max_value=1)

            return self.update_memory(memory_id, agent_id, {"hot_score": new_hot}) is not None
        except Exception as e:
            logger.error(f"❌ 更新记忆热度失败：{e}")
            raise MySQLStoreError(f"更新记忆热度失败：{e}")

    def update_memory_weight(self, memory_id: int, agent_id: str, new_weight: float) -> bool:
        """
        更新记忆权重

        Args:
            memory_id: 记忆 ID
            agent_id: 租户 ID
            new_weight: 新权重值

        Returns:
            是否更新成功
        """
        try:
            # 验证输入
            validate_float(new_weight, "new_weight", min_value=0, max_value=1)

            return self.update_memory(memory_id, agent_id, {"weight": new_weight}) is not None
        except Exception as e:
            logger.error(f"❌ 更新记忆权重失败：{e}")
            raise MySQLStoreError(f"更新记忆权重失败：{e}")

    # ---------- 删除 ----------

    def delete_memory(self, memory_id: int, agent_id: str) -> bool:
        """
        删除记忆 (软删除)

        Args:
            memory_id: 记忆 ID
            agent_id: 租户 ID

        Returns:
            是否删除成功
        """
        try:
            # 验证输入
            validate_integer(memory_id, "memory_id", min_value=1)
            validate_string(agent_id, "agent_id", min_length=1)

            result = self.update_memory(memory_id, agent_id, {"status": "deleted"})
            if result:
                # 软删除后从缓存清除
                for cid, mem in list(self._memory_cache.items()):
                    if mem.get("memory_id") == str(memory_id):
                        del self._memory_cache[cid]
                        break
                logger.info(f"🗑️ 软删除记忆：ID={memory_id}")
            return result is not None
        except Exception as e:
            logger.error(f"❌ 删除记忆失败：{e}")
            raise MySQLStoreError(f"删除记忆失败：{e}")

    def hard_delete_memory(self, memory_id: int, agent_id: str) -> bool:
        """
        物理删除记忆

        Args:
            memory_id: 记忆 ID
            agent_id: 租户 ID

        Returns:
            是否删除成功
        """
        try:
            # 验证输入
            validate_integer(memory_id, "memory_id", min_value=1)
            validate_string(agent_id, "agent_id", min_length=1)

            with self.db.get_session() as session:
                count = (
                    session.query(YijingMemory)
                    .filter(YijingMemory.id == memory_id, YijingMemory.agent_id == agent_id)
                    .delete()
                )
                session.flush()
                logger.info(f"🗑️ 物理删除记忆：ID={memory_id}")
                return count > 0
        except Exception as e:
            logger.error(f"❌ 物理删除记忆失败：{e}")
            raise MySQLStoreError(f"物理删除记忆失败：{e}")

    # ---------- 爻变历史 ----------

    def record_yao_change(self, change_data: Dict[str, Any]) -> YaoChangeHistory:
        """
        记录爻变历史

        Args:
            change_data: 爻变数据字典

        Returns:
            爻变历史对象
        """
        try:
            # 验证输入
            validate_dict(change_data, "change_data")
            validate_required_fields(change_data, ["memory_id", "agent_id", "old_hexagram", "new_hexagram"])
            validate_integer(change_data.get("memory_id"), "memory_id", min_value=1)
            validate_string(change_data.get("agent_id"), "agent_id", min_length=1)
            validate_string(change_data.get("old_hexagram"), "old_hexagram", min_length=1)
            validate_string(change_data.get("new_hexagram"), "new_hexagram", min_length=1)

            # 移除不支持的字段
            if "change_reason" in change_data:
                change_data.pop("change_reason")

            with self.db.get_session() as session:
                change = YaoChangeHistory(**change_data)
                session.add(change)
                session.flush()

                # 同时更新记忆的爻变历史 JSON
                memory = session.query(YijingMemory).filter(YijingMemory.id == change.memory_id).first()
                if memory:
                    if not memory.yao_history:
                        memory.yao_history = []
                    memory.yao_history.append(change.to_dict())

                logger.info(f"📜 记录爻变：memory_id={change.memory_id}, type={change.change_type}")
                return change
        except Exception as e:
            logger.error(f"❌ 记录爻变失败：{e}")
            raise MySQLStoreError(f"记录爻变失败：{e}")

    def get_yao_history(self, memory_id: int, agent_id: str, limit: int = 50) -> List[YaoChangeHistory]:
        """
        获取爻变历史

        Args:
            memory_id: 记忆 ID
            agent_id: 租户 ID
            limit: 返回数量限制

        Returns:
            爻变历史列表
        """
        try:
            # 验证输入
            validate_integer(memory_id, "memory_id", min_value=1)
            validate_string(agent_id, "agent_id", min_length=1)
            validate_integer(limit, "limit", min_value=1)

            with self.db.get_session() as session:
                history = (
                    session.query(YaoChangeHistory)
                    .filter(YaoChangeHistory.memory_id == memory_id, YaoChangeHistory.agent_id == agent_id)
                    .order_by(YaoChangeHistory.created_at.desc())
                    .limit(limit)
                    .all()
                )

                logger.debug(f"✅ 获取到 {len(history)} 条爻变历史")
                return history
        except Exception as e:
            logger.error(f"❌ 获取爻变历史失败：{e}")
            raise MySQLStoreError(f"获取爻变历史失败：{e}")

    # ---------- 统计与分析 ----------

    def get_all_agent_ids(self) -> List[str]:
        """
        获取所有有记忆的 Agent ID 列表（去重）
        用于定时任务多租户扫描

        Returns:
            agent_id 列表
        """
        try:
            with self.db.get_session() as session:
                result = session.query(YijingMemory.agent_id).filter(YijingMemory.status == "active").distinct().all()
                agent_ids = list(set([row[0] for row in result]))
                logger.debug(f"【Agent去重】共 {len(agent_ids)} 个唯一租户")
                return agent_ids
        except Exception as e:
            logger.error(f"❌ 获取 Agent 列表失败：{e}")
            return []

    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """
        获取 Agent 统计信息

        Args:
            agent_id: 租户 ID

        Returns:
            统计信息字典
        """
        try:
            # 验证输入
            validate_string(agent_id, "agent_id", min_length=1)

            with self.db.get_session() as session:
                # 总记忆数
                total = (
                    session.query(YijingMemory)
                    .filter(YijingMemory.agent_id == agent_id, YijingMemory.status == "active")
                    .count()
                )

                # 按三才统计
                layer_dist = dict(
                    session.query(YijingMemory.sancai_layer, func.count(YijingMemory.id))
                    .filter(YijingMemory.agent_id == agent_id, YijingMemory.status == "active")
                    .group_by(YijingMemory.sancai_layer)
                    .all()
                )

                # 按五行统计
                element_dist = dict(
                    session.query(YijingMemory.wuxing, func.count(YijingMemory.id))
                    .filter(YijingMemory.agent_id == agent_id, YijingMemory.status == "active")
                    .group_by(YijingMemory.wuxing)
                    .all()
                )

                # 按八卦统计
                trigram_dist = dict(
                    session.query(YijingMemory.bagua_type, func.count(YijingMemory.id))
                    .filter(YijingMemory.agent_id == agent_id, YijingMemory.status == "active")
                    .group_by(YijingMemory.bagua_type)
                    .all()
                )

                # 平均热度/权重
                avg_stats = (
                    session.query(func.avg(YijingMemory.hot_score), func.avg(YijingMemory.weight))
                    .filter(YijingMemory.agent_id == agent_id, YijingMemory.status == "active")
                    .first()
                )

                # 状态分布
                status_dist = dict(
                    session.query(YijingMemory.status, func.count(YijingMemory.id))
                    .filter(YijingMemory.agent_id == agent_id)
                    .group_by(YijingMemory.status)
                    .all()
                )

                # 爻变历史数量
                yao_change_count = session.query(YaoChangeHistory).filter(YaoChangeHistory.agent_id == agent_id).count()

                return {
                    "total": total,
                    "total_memories": total,
                    "layer_distribution": layer_dist,
                    "element_distribution": element_dist,
                    "trigram_distribution": trigram_dist,
                    "status_distribution": status_dist,
                    "average_hot": float(avg_stats[0]) if avg_stats[0] else 0,
                    "average_weight": float(avg_stats[1]) if avg_stats[1] else 0,
                    "yao_change_count": yao_change_count,
                }
        except Exception as e:
            logger.error(f"❌ 获取统计信息失败：{e}")
            raise MySQLStoreError(f"获取统计信息失败：{e}")

    # ---------- 过期清理 ----------

    def clean_expired_memories(self, agent_id: Optional[str] = None) -> int:
        """
        清理过期记忆

        Args:
            agent_id: 可选，不传则清理所有

        Returns:
            清理数量
        """
        try:
            # 验证输入
            if agent_id is not None:
                validate_string(agent_id, "agent_id", min_length=1)

            with self.db.get_session() as session:
                # 使用 idx_expire_time 索引
                query = session.query(YijingMemory).filter(
                    YijingMemory.expire_time < datetime.now(), YijingMemory.status == "active"
                )

                if agent_id:
                    # 使用 idx_agent_status 索引
                    query = query.filter(YijingMemory.agent_id == agent_id)

                # 批量更新，提高效率
                count = query.update({"status": "archived"}, synchronize_session=False)
                session.flush()

                if count > 0:
                    logger.info(f"🧹 清理 {count} 条过期记忆")

                return count
        except Exception as e:
            logger.error(f"❌ 清理过期记忆失败：{e}")
            raise MySQLStoreError(f"清理过期记忆失败：{e}")

    # ---------- 核心配置（Identity/Policy Layer） ----------

    def get_core_config(self, agent_id: str) -> Optional["TaijiCoreConfig"]:
        """
        获取指定 Agent 的核心配置（太极核心配置表）

        Args:
            agent_id: Agent ID

        Returns:
            TaijiCoreConfig 对象，或 None（未配置过）
        """
        try:
            with self.db.get_session() as session:
                config = session.query(TaijiCoreConfig).filter(TaijiCoreConfig.agent_id == agent_id).first()
                # detach from session so caller can use it outside session context
                if config:
                    session.expunge(config)
                return config
        except Exception as e:
            logger.error(f"❌ 获取核心配置失败：{e}")
            return None

    def save_core_config(self, agent_id: str, config_data: Dict[str, Any]) -> TaijiCoreConfig:
        """
        保存或更新核心配置（仅 agent_id 唯一）

        Args:
            agent_id: Agent ID
            config_data: 核心配置字典

        Returns:
            保存后的 TaijiCoreConfig 对象
        """
        try:
            validate_string(agent_id, "agent_id")
            with self.db.get_session() as session:
                existing = session.query(TaijiCoreConfig).filter(TaijiCoreConfig.agent_id == agent_id).first()

                if existing:
                    # 更新已有配置
                    for key in [
                        "identity",
                        "core_values",
                        "ethical_rules",
                        "encoding_mode",
                        "auto_schedule",
                        "auto_self_heal",
                        "hot_threshold_activate",
                        "hot_threshold_hide",
                        "age_threshold_forget",
                    ]:
                        if key in config_data:
                            setattr(existing, key, config_data[key])
                    existing.updated_at = datetime.now()
                    session.commit()
                    session.refresh(existing)
                    session.expunge(existing)
                    logger.info(f"✅ 核心配置已更新：agent_id={agent_id}")
                    return existing
                else:
                    # 新建配置
                    new_config = TaijiCoreConfig(
                        agent_id=agent_id,
                        **{
                            k: v
                            for k, v in config_data.items()
                            if k
                            in [
                                "identity",
                                "core_values",
                                "ethical_rules",
                                "encoding_mode",
                                "auto_schedule",
                                "auto_self_heal",
                                "hot_threshold_activate",
                                "hot_threshold_hide",
                                "age_threshold_forget",
                            ]
                        },
                    )
                    session.add(new_config)
                    session.commit()
                    session.refresh(new_config)
                    session.expunge(new_config)
                    logger.info(f"✅ 核心配置已创建：agent_id={agent_id}")
                    return new_config
        except Exception as e:
            logger.error(f"❌ 保存核心配置失败：{e}")
            raise MySQLStoreError(f"保存核心配置失败：{e}")

    # ---------- 辅助方法 ----------


# ========== 便捷函数 ==========


def create_mysql_store(database_url: str = None) -> MySQLMemoryStore:
    """
    创建 MySQL 存储实例

    Args:
        database_url: 数据库连接 URL，默认从配置文件读取

    Returns:
        MySQLMemoryStore 实例
    """
    try:
        db_manager = DatabaseManager(database_url)
        db_manager.create_tables()
        return MySQLMemoryStore(db_manager)
    except Exception as e:
        logger.warning(f"MySQL 连接失败，将使用 SQLite 作为备选: {e}")
        # 使用 SQLite 作为备选
        sqlite_url = "sqlite:///memory_system.db"
        db_manager = DatabaseManager(sqlite_url)
        db_manager.create_tables()
        return MySQLMemoryStore(db_manager)


# 全局实例
mysql_store = create_mysql_store()


# ========== 测试 ==========

if __name__ == "__main__":
    print("=" * 80)
    print("🗄️ MySQL 存储层测试")
    print("=" * 80)

    # 使用 SQLite 测试 (无需 MySQL)
    store = create_mysql_store("sqlite:///test_mysql_store.db")

    # 测试创建
    print("\n【测试 1】创建记忆")
    memory = store.create_memory(
        {
            "agent_id": "test-001",
            "hexagram": "111111",
            "trigram": "乾",
            "hexagram_num": 1,
            "layer": "天",
            "element": "金",
            "content": "测试记忆内容",
            "weight": 0.9,
            "hot": 0.8,
        }
    )
    print(f"✅ 创建成功：ID={memory.id}")

    # 测试查询
    print("\n【测试 2】查询记忆")
    memories = store.get_memories("test-001", limit=10)
    print(f"✅ 查询到 {len(memories)} 条记忆")

    # 测试统计
    print("\n【测试 3】统计信息")
    stats = store.get_agent_stats("test-001")
    print(f"✅ 总记忆数：{stats['total_memories']}")
    print(f"✅ 三才分布：{stats['layer_distribution']}")
    print(f"✅ 五行分布：{stats['element_distribution']}")
    print(f"✅ 八卦分布：{stats['trigram_distribution']}")
    print(f"✅ 状态分布：{stats['status_distribution']}")
    print(f"✅ 平均热度：{stats['average_hot']}")
    print(f"✅ 平均权重：{stats['average_weight']}")
    print(f"✅ 爻变历史数：{stats['yao_change_count']}")

    # 测试更新
    print("\n【测试 4】更新热度")
    store.update_memory_hot(memory.id, "test-001", 0.95)
    updated = store.get_memory(memory.id, "test-001")
    print(f"✅ 更新后热度：{updated.hot}")

    # 测试爻变记录
    print("\n【测试 5】记录爻变")
    store.record_yao_change(
        {
            "memory_id": memory.id,
            "agent_id": "test-001",
            "old_hexagram": "111111",
            "new_hexagram": "111110",
            "changed_positions": [0],
            "change_type": "yao_change",
            "action": "activate",
            "reason": "热度提升触发",
            "old_state": "金",
            "new_state": "火",
            "weight_change": 0.05,
            "hot_change": 0.15,
        }
    )
    history = store.get_yao_history(memory.id, "test-001")
    print(f"✅ 爻变历史数：{len(history)}")

    # 测试清理过期记忆
    print("\n【测试 6】清理过期记忆")
    count = store.clean_expired_memories("test-001")
    print(f"✅ 清理过期记忆数：{count}")

    print("\n" + "=" * 80)
    print("✅ MySQL 存储层测试完成！")
    print("=" * 80)
