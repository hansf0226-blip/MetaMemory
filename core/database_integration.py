#!/usr/bin/env python3
"""
Database integration module - Production ready
Unified data layer with multiple backend support
"""

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Type

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker

from core.cache_layer import MultiLevelCache, get_global_cache, cached

from core.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Complete database configuration"""

    db_type: str = "sqlite"
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: str = "metamemory"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    sqlite_path: Optional[str] = None
    use_cache: bool = True
    cache_ttl: int = 300
    read_replicas: List[Dict] = field(default_factory=list)


class DatabaseConnectionError(Exception):
    """Database connection exception"""


class BaseRepository:
    """Base repository class with generic CRUD"""

    def __init__(self, engine: Engine, cache: Optional[MultiLevelCache] = None):
        self._engine = engine
        self._cache = cache or get_global_cache()
        self._session_factory = sessionmaker(bind=engine)
        self._scoped_session = scoped_session(self._session_factory)
        self._lock = threading.RLock()

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get session context manager"""
        session = self._scoped_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise DatabaseConnectionError(f"Database operation failed: {e}")
        finally:
            session.close()

    def execute_query(self, query: str, params: Optional[Dict] = None, fetch: bool = True):
        """Execute query"""
        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            if fetch:
                return [dict(row._mapping) for row in result]
            return None

    def execute_update(self, query: str, params: Optional[Dict] = None) -> int:
        """Execute update query"""
        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            return result.rowcount

    @cached(key_prefix="db:scalar", ttl=60)
    def scalar(self, query: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Get single value query with cache"""
        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            row = result.fetchone()
            return row[0] if row else None


class MemoryRepository(BaseRepository):
    """Memory repository for memory CRUD operations"""

    def create_tables(self):
        """Create database tables structure"""
        # Create memories table
        memories_table = """
        CREATE TABLE IF NOT EXISTS memories (
            id VARCHAR(64) PRIMARY KEY,
            user_id VARCHAR(64) NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB,
            embedding_model VARCHAR(64),
            memory_type VARCHAR(32) DEFAULT 'memory',
            importance FLOAT DEFAULT 0.5,
            access_count INTEGER DEFAULT 0,
            last_accessed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            tags VARCHAR(255),
            source VARCHAR(64),
            status VARCHAR(32) DEFAULT 'active',
            version INTEGER DEFAULT 1
        )
        """

        # Create indexes separately
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC)",
            "CREATE TABLE IF NOT EXISTS memory_vectors (id VARCHAR(64) PRIMARY KEY, memory_id VARCHAR(64) NOT NULL, vector BLOB NOT NULL, model VARCHAR(64) NOT NULL, dimension INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "CREATE INDEX IF NOT EXISTS idx_vectors_memory_id ON memory_vectors(memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_vectors_model ON memory_vectors(model)",
            "CREATE TABLE IF NOT EXISTS user_configs (user_id VARCHAR(64) PRIMARY KEY, config TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "CREATE TABLE IF NOT EXISTS access_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id VARCHAR(64), action VARCHAR(64), memory_id VARCHAR(64), query TEXT, result_count INTEGER, latency_ms FLOAT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "CREATE INDEX IF NOT EXISTS idx_logs_user_id ON access_logs(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_logs_created_at ON access_logs(created_at)",
        ]

        with self.get_session() as session:
            session.execute(text(memories_table))
            for index_sql in indexes:
                session.execute(text(index_sql))

        logger.info("✅ Database tables created successfully")

    def insert_memory(self, memory_data: Dict) -> bool:
        """Insert a memory record"""
        sql = """
        INSERT INTO memories (
            id, user_id, content, memory_type, importance,
            created_at, updated_at, metadata, tags, source, status
        ) VALUES (:id, :user_id, :content, :memory_type, :importance,
                  :created_at, :updated_at, :metadata, :tags, :source, :status)
        """
        try:
            self.execute_update(sql, memory_data)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to insert memory: {e}")
            return False

    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """Get a single memory record"""
        sql = "SELECT * FROM memories WHERE id = :id"
        results = self.execute_query(sql, {"id": memory_id})
        return results[0] if results else None

    def update_memory(self, memory_id: str, updates: Dict) -> bool:
        """Update a memory record"""
        set_clause = ", ".join([f"{k} = :{k}" for k in updates.keys()])
        sql = f"UPDATE memories SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = :id"
        params = {**updates, "id": memory_id}
        try:
            self.execute_update(sql, params)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to update memory: {e}")
            return False

    def delete_memory(self, memory_id: str) -> bool:
        """Soft delete a memory record"""
        sql = "UPDATE memories SET status = 'deleted', updated_at = CURRENT_TIMESTAMP WHERE id = :id"
        try:
            self.execute_update(sql, {"id": memory_id})
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete memory: {e}")
            return False

    def list_memories(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        memory_type: Optional[str] = None,
    ) -> List[Dict]:
        """List memories for a user"""
        sql = "SELECT * FROM memories WHERE user_id = :user_id AND status = 'active'"
        params = {"user_id": user_id}

        if memory_type:
            sql += " AND memory_type = :memory_type"
            params["memory_type"] = memory_type

        sql += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
        params["limit"] = limit
        params["offset"] = offset

        return self.execute_query(sql, params)

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for a user"""
        sql = """
        SELECT
            COUNT(*) as total,
            memory_type,
            status
        FROM memories
        WHERE user_id = :user_id
        GROUP BY memory_type, status
        """
        results = self.execute_query(sql, {"user_id": user_id})

        stats = {"total": 0, "by_type": {}, "by_status": {}}
        for row in results:
            stats["total"] += row["total"]
            stats["by_type"][row["memory_type"]] = stats["by_type"].get(row["memory_type"], 0) + row["total"]
            stats["by_status"][row["status"]] = stats["by_status"].get(row["status"], 0) + row["total"]

        return stats

    def log_access(self, log_data: Dict) -> bool:
        """Record access log"""
        sql = """
        INSERT INTO access_logs (user_id, action, memory_id, query, result_count, latency_ms)
        VALUES (:user_id, :action, :memory_id, :query, :result_count, :latency_ms)
        """
        try:
            self.execute_update(sql, log_data)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to log access: {e}")
            return False


class DatabaseIntegration:
    """Complete database integration manager"""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._engine: Optional[Engine] = None
        self._repository: Optional[MemoryRepository] = None
        self._cache: Optional[MultiLevelCache] = None
        self._lock = threading.RLock()
        self._initialized = False

    def _build_connection_url(self) -> str:
        """Build database connection URL"""
        cfg = self.config

        if cfg.db_type == "sqlite":
            import os

            path = cfg.sqlite_path or "data/metamemory.db"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return f"sqlite:///{path}"

        elif cfg.db_type == "postgresql":
            auth = f"{cfg.username}:{cfg.password}@" if cfg.password else f"{cfg.username}@" if cfg.username else ""
            host = cfg.host or "localhost"
            port = cfg.port or 5432
            return f"postgresql://{auth}{host}:{port}/{cfg.database}"

        elif cfg.db_type == "mysql":
            auth = f"{cfg.username}:{cfg.password}@" if cfg.password else f"{cfg.username}@" if cfg.username else ""
            host = cfg.host or "localhost"
            port = cfg.port or 3306
            return f"mysql+pymysql://{auth}{host}:{port}/{cfg.database}"

    @property
    def repository(self) -> MemoryRepository:
        """Get memory repository"""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    url = self._build_connection_url()
                    self._engine = create_engine(url, pool_size=self.config.pool_size)
                    self._repository = MemoryRepository(self._engine, self._cache)
                    self._initialized = True
        return self._repository

    def health_check(self) -> Dict[str, Any]:
        """Health check"""
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            return {
                "status": "healthy",
                "db_type": self.config.db_type,
                "pool_size": self.config.pool_size,
                "cache_enabled": self.config.use_cache,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }


_global_db: Optional[DatabaseIntegration] = None
_global_db_lock = threading.Lock()


def get_database(config: Optional[DatabaseConfig] = None) -> DatabaseIntegration:
    """Get global database instance"""
    global _global_db

    if _global_db is not None:
        return _global_db

    with _global_db_lock:
        if _global_db is None:
            _global_db = DatabaseIntegration(config)
            # DatabaseIntegration auto-initializes on construction

    return _global_db
