#!/usr/bin/env python3
"""
🗄️ 缓存层模块 — 内存缓存（精简版）

提供 InMemoryCache 和 MultiLevelCache，去掉未使用的 RedisCache 抽象基类。

当前项目不使用 Redis 分布式缓存，后续需要时再恢复。
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheError(Exception):
    """缓存异常"""
    pass


@dataclass
class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0
    size: int = 0
    max_size: int = 1000


class InMemoryCache:
    """内存缓存 — 基于字典 + LRU 淘汰"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
        self.stats = CacheStats(max_size=max_size)
        self.logger = logging.getLogger(__name__)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                return None
            # 检查过期
            if key in self._expiry and time.time() > self._expiry[key]:
                self._remove(key)
                self.stats.misses += 1
                return None
            # 更新访问顺序
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            self.stats.hits += 1
            return self._cache[key]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            if key in self._cache:
                if key in self._access_order:
                    self._access_order.remove(key)
            elif len(self._cache) >= self.max_size:
                self._evict()
            self._cache[key] = value
            if ttl is not None:
                self._expiry[key] = time.time() + ttl
            elif self.default_ttl > 0:
                self._expiry[key] = time.time() + self.default_ttl
            self._access_order.append(key)
            self.stats.size = len(self._cache)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._remove(key)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._expiry.clear()
            self._access_order.clear()
            self.stats.size = 0

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": self.stats.hits / (self.stats.hits + self.stats.misses)
                if (self.stats.hits + self.stats.misses) > 0
                else 0,
            }

    def _remove(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            self._expiry.pop(key, None)
            if key in self._access_order:
                self._access_order.remove(key)
            return True
        return False

    def _evict(self) -> None:
        if self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)
            self._expiry.pop(oldest, None)


class MultiLevelCache:
    """多级缓存 — 当前只使用内存层"""

    def __init__(self, use_memory: bool = True, use_redis: bool = False,
                 memory_max_size: int = 10000, memory_ttl: int = 86400,
                 redis_config: Optional[dict] = None, redis_ttl: int = 3600):
        self._memory_cache = InMemoryCache(max_size=memory_max_size, default_ttl=memory_ttl) if use_memory else None

    def get(self, key: str) -> Optional[Any]:
        if self._memory_cache:
            val = self._memory_cache.get(key)
            if val is not None:
                return val
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if self._memory_cache:
            self._memory_cache.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        if self._memory_cache:
            return self._memory_cache.delete(key)
        return False

    def clear(self) -> None:
        if self._memory_cache:
            self._memory_cache.clear()

    def get_stats(self) -> dict:
        stats = {}
        if self._memory_cache:
            stats["memory"] = self._memory_cache.get_stats()
        return stats


def cached(key_prefix: str, ttl: int = 300, cache_instance: Optional[MultiLevelCache] = None):
    """函数结果缓存装饰器"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            cache = cache_instance or get_global_cache()
            cache_key = f"{key_prefix}:{str(args)}:{str(sorted(kwargs.items()))}"
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            result = func(*args, **kwargs)
            try:
                cache.set(cache_key, result, ttl)
            except Exception:
                pass
            return result
        return wrapper
    return decorator


_global_cache: Optional[MultiLevelCache] = None
_global_cache_lock = threading.RLock()


def get_global_cache() -> MultiLevelCache:
    global _global_cache
    if _global_cache is None:
        with _global_cache_lock:
            if _global_cache is None:
                _global_cache = MultiLevelCache(use_memory=True, use_redis=False)
    return _global_cache
