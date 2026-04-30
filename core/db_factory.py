#!/usr/bin/env python3
"""
🏭 数据库工厂 - 统一数据库连接管理

提供多种数据库后端的统一接口，支持：
- SQLite（开发/小规模）
- PostgreSQL（生产推荐）
- MySQL（现有支持）
- 自动降级策略
"""

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """数据库异常"""


@dataclass
class DatabaseConfig:
    """数据库配置"""

    db_type: str = "sqlite"  # sqlite, postgresql, mysql
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: str = "memory_system"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    sqlite_path: Optional[str] = None


class DatabaseBackend(ABC):
    """数据库后端抽象基类"""

    @abstractmethod
    def get_engine(self) -> Engine:
        """获取数据库引擎"""
        pass

    @abstractmethod
    def get_session(self) -> Session:
        """获取数据库会话"""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """测试连接"""
        pass

    @abstractmethod
    def get_type(self) -> str:
        """获取数据库类型"""
        pass


class SQLiteBackend(DatabaseBackend):
    """SQLite 后端"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[Callable] = None
        self._scoped_session: Optional[scoped_session] = None
        self._lock = threading.RLock()

    def get_engine(self) -> Engine:
        with self._lock:
            if self._engine is None:
                db_path = self.config.sqlite_path or str(
                    Path(__file__).parent.parent / "data" / "memory_system.db"
                )
                Path(db_path).parent.mkdir(exist_ok=True)

                self._engine = create_engine(
                    f"sqlite:///{db_path}",
                    echo=self.config.echo,
                    connect_args={"check_same_thread": False},
                    pool_pre_ping=True,
                )
                logger.info(f"✅ SQLite 引擎已初始化: {db_path}")

            return self._engine

    def get_session(self) -> Session:
        with self._lock:
            if self._scoped_session is None:
                engine = self.get_engine()
                self._session_factory = sessionmaker(bind=engine)
                self._scoped_session = scoped_session(self._session_factory)

            return self._scoped_session()

    def test_connection(self) -> bool:
        try:
            session = self.get_session()
            session.execute("SELECT 1")
            session.close()
            return True
        except Exception as e:
            logger.error(f"❌ SQLite 连接测试失败: {e}")
            return False

    def get_type(self) -> str:
        return "sqlite"


class PostgreSQLBackend(DatabaseBackend):
    """PostgreSQL 后端 - 生产推荐"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[Callable] = None
        self._scoped_session: Optional[scoped_session] = None
        self._lock = threading.RLock()

    def _build_url(self) -> str:
        """构建连接 URL"""
        host = self.config.host or "localhost"
        port = self.config.port or 5432
        user = self.config.username or "postgres"
        password = self.config.password or ""
        db = self.config.database

        if password:
            return f"postgresql://{user}:***@{host}:{port}/{db}"
        else:
            return f"postgresql://{user}@{host}:{port}/{db}"

    def get_engine(self) -> Engine:
        with self._lock:
            if self._engine is None:
                url = self._build_url()

                self._engine = create_engine(
                    url,
                    echo=self.config.echo,
                    pool_size=self.config.pool_size,
                    max_overflow=self.config.max_overflow,
                    pool_timeout=self.config.pool_timeout,
                    pool_recycle=self.config.pool_recycle,
                    pool_pre_ping=True,
                )
                logger.info(f"✅ MySQL 引擎已初始化: {self.config.host}:{self.config.port}/{self.config.database}")

            return self._engine

    def get_session(self) -> Session:
        with self._lock:
            if self._scoped_session is None:
                engine = self.get_engine()
                self._session_factory = sessionmaker(bind=engine)
                self._scoped_session = scoped_session(self._session_factory)

            return self._scoped_session()

    def test_connection(self) -> bool:
        try:
            session = self.get_session()
            session.execute("SELECT 1")
            session.close()
            return True
        except Exception as e:
            logger.error(f"❌ MySQL 连接测试失败: {e}")
            return False

    def get_type(self) -> str:
        return "mysql"


class DatabaseFactory:
    """数据库工厂 - 统一创建和管理数据库连接"""

    _backends: Dict[str, Type[DatabaseBackend]] = {
        "sqlite": SQLiteBackend,
        "postgresql": PostgreSQLBackend,
        "postgres": PostgreSQLBackend,
        "mysql": MySQLBackend,
    }

    def __init__(self):
        self._instances: Dict[str, DatabaseBackend] = {}
        self._lock = threading.RLock()
        self._primary_backend: Optional[str] = None
        self._fallback_chain: list[str] = []

    def register_backend(self, name: str, backend_class: Type[DatabaseBackend]):
        """注册新的数据库后端"""
        self._backends[name.lower()] = backend_class
        logger.info(f"✅ 已注册数据库后端: {name}")

    def create(self, config: DatabaseConfig) -> DatabaseBackend:
        """创建数据库后端实例"""
        backend_class = self._backends.get(config.db_type.lower())
        if not backend_class:
            raise DatabaseError(f"不支持的数据库类型: {config.db_type}")

        with self._lock:
            key = f"{config.db_type}:{config.host or 'local'}:{config.database}"
            if key in self._instances:
                return self._instances[key]

            backend = backend_class(config)
            self._instances[key] = backend
            return backend

    def setup_primary(self, config: DatabaseConfig) -> bool:
        """设置主数据库，带自动降级"""
        db_types_to_try = [config.db_type, "sqlite"]  # 降级链

        for db_type in db_types_to_try:
            try:
                test_config = DatabaseConfig(**{**config.__dict__, "db_type": db_type})
                backend = self.create(test_config)

                if backend.test_connection():
                    self._primary_backend = db_type
                    logger.info(f"✅ 主数据库已设置为: {db_type}")
                    return True
                else:
                    logger.warning(f"⚠️ {db_type} 连接测试失败，尝试下一个")
            except Exception as e:
                logger.warning(f"⚠️ {db_type} 初始化失败: {e}")

        logger.error("❌ 所有数据库后端都失败了")
        return False

    def get_primary(self) -> Optional[DatabaseBackend]:
        """获取主数据库后端"""
        if self._primary_backend is None:
            logger.warning("⚠️ 主数据库未设置，使用 SQLite 默认")
            # 自动创建默认 SQLite
            return self.create(DatabaseConfig(db_type="sqlite"))

        for key, backend in self._instances.items():
            if key.startswith(f"{self._primary_backend}:"):
                return backend

        return None

    def get_session(self) -> Session:
        """快捷获取主数据库会话"""
        backend = self.get_primary()
        if not backend:
            raise DatabaseError("没有可用的数据库后端")
        return backend.get_session()

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        primary = self.get_primary()
        if not primary:
            return {"status": "error", "message": "没有可用的数据库后端"}

        return {
            "status": "ok" if primary.test_connection() else "error",
            "type": primary.get_type(),
            "available_backends": list(self._instances.keys()),
        }


# ===== 全局工厂实例 =====

_global_factory: Optional[DatabaseFactory] = None
_factory_lock = threading.Lock()


def get_db_factory() -> DatabaseFactory:
    """获取全局数据库工厂实例"""
    global _global_factory

    if _global_factory is not None:
        return _global_factory

    with _factory_lock:
        if _global_factory is None:
            _global_factory = DatabaseFactory()

            # 尝试从配置初始化主数据库
            try:
                from core.config import get_config

                db_type = get_config("database.type", "sqlite")

                if db_type == "sqlite":
                    config = DatabaseConfig(
                        db_type="sqlite",
                        sqlite_path=get_config("database.sqlite.path"),
                    )
                elif db_type in ["postgresql", "postgres"]:
                    config = DatabaseConfig(
                        db_type="postgresql",
                        host=get_config("database.postgresql.host", "localhost"),
                        port=get_config("database.postgresql.port", 5432),
                        username=get_config("database.postgresql.user", "postgres"),
                        password=get_config("database.postgresql.password"),
                        database=get_config("database.postgresql.database", "memory_system"),
                        pool_size=get_config("database.pool_size", 10),
                    )
                elif db_type == "mysql":
                    config = DatabaseConfig(
                        db_type="mysql",
                        host=get_config("database.mysql.host", "localhost"),
                        port=get_config("database.mysql.port", 3306),
                        username=get_config("database.mysql.user", "root"),
                        password=get_config("database.mysql.password"),
                        database=get_config("database.mysql.database", "memory_system"),
                        pool_size=get_config("database.pool_size", 10),
                    )
                else:
                    config = DatabaseConfig(db_type="sqlite")

                _global_factory.setup_primary(config)

            except Exception as e:
                logger.warning(f"⚠️ 从配置初始化数据库失败: {e}，使用默认 SQLite")
                _global_factory.setup_primary(DatabaseConfig(db_type="sqlite"))

    return _global_factory
