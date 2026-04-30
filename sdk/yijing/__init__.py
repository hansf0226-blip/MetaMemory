#!/usr/bin/env python3
"""
易经 Agent 记忆系统 - Python SDK

提供简单易用的 Python 客户端，方便集成到各类 AI Agent 中
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import requests


class YijingError(Exception):
    """基础错误类"""


class AuthenticationError(YijingError):
    """认证错误"""


class APIError(YijingError):
    """API 错误"""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")


class RateLimitError(YijingError):
    """速率限制错误"""


# ============================================================================
# 数据模型
# ============================================================================


class Memory:
    """记忆对象"""

    def __init__(self, data: Dict[str, Any]):
        self.id = data.get("id")
        self.memory_id = data.get("memory_id")
        self.agent_id = data.get("agent_id")
        self.hexagram = data.get("hexagram", [])
        self.bagua_type = data.get("bagua_type")
        self.sancai_layer = data.get("sancai_layer")
        self.wuxing = data.get("wuxing")
        self.content = data.get("content")
        self.hot_score = data.get("hot_score")
        self.create_time = data.get("create_time")
        self.update_time = data.get("update_time")

    def __repr__(self):
        return f"<Memory {self.memory_id}>"


class SearchResult:
    """搜索结果"""

    def __init__(self, data: Dict[str, Any]):
        self.memory_id = data.get("memory_id")
        self.bagua_type = data.get("bagua_type")
        self.wuxing = data.get("wuxing")
        self.content = data.get("content")
        self.similarity = data.get("similarity", 0.0)

    def __repr__(self):
        return f"<SearchResult {self.memory_id} ({self.similarity:.2f})>"


class Agent:
    """Agent 对象"""

    def __init__(self, data: Dict[str, Any]):
        self.agent_id = data.get("agent_id")
        self.agent_name = data.get("agent_name")
        self.status = data.get("status")
        self.memory_count = data.get("memory_count")
        self.create_time = data.get("create_time")

    def __repr__(self):
        return f"<Agent {self.agent_id}>"


# ============================================================================
# SDK 客户端
# ============================================================================


class YijingClient:
    """
    易经记忆系统客户端

    使用示例:
        client = YijingClient(api_key="your_key")
        memory = client.memories.create(content="测试")
        results = client.memories.search("关键词")
    """

    def __init__(
        self, api_key: str, base_url: str = "http://localhost:8000", timeout: int = 30, agent_id: str = "default_agent"
    ):
        """
        初始化客户端

        Args:
            api_key: API Key
            base_url: API 基础 URL
            timeout: 请求超时时间（秒）
            agent_id: 默认 Agent ID
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.agent_id = agent_id
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": self.api_key, "Content-Type": "application/json"})

        # 子模块
        self.memories = MemoryManager(self)
        self.agents = AgentManager(self)
        self.keys = KeyManager(self)

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """
        发送 HTTP 请求

        Args:
            method: HTTP 方法
            path: 请求路径
            **kwargs: 其他参数

        Returns:
            dict: 响应数据

        Raises:
            AuthenticationError: 认证失败
            APIError: API 错误
            RateLimitError: 速率限制
        """
        url = f"{self.base_url}{path}"

        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)

            # 处理错误
            if response.status_code == 401:
                raise AuthenticationError("API Key 无效或缺失")
            elif response.status_code == 403:
                raise AuthenticationError("API Key 已被禁用或过期")
            elif response.status_code == 429:
                raise RateLimitError("请求过于频繁")
            elif response.status_code >= 500:
                raise APIError(response.status_code, "服务器内部错误")

            # 返回数据
            return response.json()

        except requests.exceptions.Timeout:
            raise APIError(0, "请求超时")
        except requests.exceptions.ConnectionError:
            raise APIError(0, "无法连接到服务器")

    def health_check(self) -> bool:
        """健康检查"""
        try:
            response = self._request("GET", "/health")
            return response.get("status") == "ok"
        except Exception:
            return False


# ============================================================================
# 记忆管理
# ============================================================================


class MemoryManager:
    """记忆管理器"""

    def __init__(self, client: YijingClient):
        self.client = client

    def create(
        self, content: str, memory_type: str = "chat", memory_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> Memory:
        """
        创建记忆

        Args:
            content: 记忆内容
            memory_type: 记忆类型
            memory_id: 记忆 ID（可选，自动生成）
            agent_id: Agent ID（可选，使用客户端默认）

        Returns:
            Memory: 创建的记忆对象
        """
        # 先编码
        encode_data = self.client._request(
            "POST", "/api/v1/memory/encode/hexagram", json={"content": content, "memory_type": memory_type}
        )

        # 生成记忆 ID
        if not memory_id:
            memory_id = f"mem_{int(datetime.now().timestamp())}"

        # 创建记忆
        data = self.client._request(
            "POST",
            "/api/v1/memories",
            json={
                "memory_id": memory_id,
                "hexagram": encode_data["hexagram"],
                "bagua_type": encode_data["bagua_type"],
                "sancai_layer": encode_data["sancai_layer"],
                "wuxing": encode_data["wuxing"],
                "content": content,
                "hot_score": 0.8,
                "agent_id": agent_id or self.client.agent_id,
            },
        )

        return Memory(data.get("data", data))

    def get(self, memory_id: str) -> Memory:
        """获取记忆详情"""
        data = self.client._request("GET", f"/api/v1/memories/{memory_id}")
        return Memory(data.get("data", {}))

    def list(self, page: int = 1, page_size: int = 20, agent_id: Optional[str] = None) -> List[Memory]:
        """获取记忆列表"""
        data = self.client._request(
            "GET",
            "/api/v1/memories",
            params={"page": page, "page_size": page_size, "agent_id": agent_id or self.client.agent_id},
        )
        return [Memory(m) for m in data.get("list", [])]

    def search(self, query: str, top_k: int = 5, agent_id: Optional[str] = None) -> List[SearchResult]:
        """搜索记忆"""
        data = self.client._request(
            "GET",
            "/api/v1/memories/search",
            params={"content": query, "top_k": top_k, "agent_id": agent_id or self.client.agent_id},
        )
        return [SearchResult(r) for r in data.get("data", [])]

    def update(self, memory_id: str, content: Optional[str] = None, hot_score: Optional[float] = None) -> bool:
        """更新记忆"""
        data = self.client._request(
            "PUT", f"/api/v1/memories/{memory_id}", json={"content": content, "hot_score": hot_score}
        )
        return data.get("code") == 200

    def delete(self, memory_id: str) -> bool:
        """删除记忆"""
        data = self.client._request("DELETE", f"/api/v1/memories/{memory_id}")
        return data.get("code") == 200


# ============================================================================
# Agent 管理
# ============================================================================


class AgentManager:
    """Agent 管理器"""

    def __init__(self, client: YijingClient):
        self.client = client

    def create(self, agent_id: str, agent_name: str, remark: str = "") -> Agent:
        """创建 Agent"""
        data = self.client._request(
            "POST", "/api/v1/agents", json={"agent_id": agent_id, "agent_name": agent_name, "remark": remark}
        )
        return Agent(data.get("data", {"agent_id": agent_id}))

    def get(self, agent_id: str) -> Agent:
        """获取 Agent 信息"""
        data = self.client._request("GET", f"/api/v1/agents/{agent_id}/info")
        return Agent(data.get("data", {}))

    def list(self) -> List[Agent]:
        """获取 Agent 列表"""
        data = self.client._request("GET", "/api/v1/agents")
        return [Agent(a) for a in data.get("list", [])]


# ============================================================================
# API Key 管理
# ============================================================================


class KeyManager:
    """API Key 管理器"""

    def __init__(self, client: YijingClient):
        self.client = client

    def create(
        self,
        name: str = "",
        agent_id: str = "default_agent",
        rate_limit: int = 1000,
        expires_days: Optional[int] = None,
    ) -> str:
        """
        创建 API Key

        Returns:
            str: 新生成的 API Key（只返回一次）
        """
        data = self.client._request(
            "POST",
            "/api/v1/keys",
            json={"name": name, "agent_id": agent_id, "rate_limit": rate_limit, "expires_days": expires_days},
        )
        return data.get("data", {}).get("api_key", "")

    def list(self, page: int = 1, page_size: int = 20) -> List[Dict[str, Any]]:
        """获取 API Key 列表"""
        data = self.client._request("GET", "/api/v1/keys", params={"page": page, "page_size": page_size})
        return data.get("list", [])

    def delete(self, key_id: int) -> bool:
        """删除 API Key"""
        data = self.client._request("DELETE", f"/api/v1/keys/{key_id}")
        return data.get("code") == 200
