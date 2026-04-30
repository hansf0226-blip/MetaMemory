"""
MetaMemory SDK — 通用 Python 客户端

一行接入主动记忆:

    from sdk.metamemory import MetaMemory
    mm = MetaMemory()

    # 对话完自动记忆
    mm.remember("用户: 我喜欢川菜\n助手: 推荐麻婆豆腐", agent_id="user_123")

    # 新对话前加载上下文
    ctx = mm.recall("今天想吃什么", agent_id="user_123")
    print(ctx)  # → "相关历史记忆:\n- 用户喜欢川菜..."
"""
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class MetaMemory:
    """MetaMemory 通用客户端"""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    # ─── 主动记忆（核心） ───

    def remember(self, conversation: str, agent_id: str = "default") -> Dict[str, Any]:
        """
        主动记忆：对话完调用一次即可

        自动:
        1. LLM 提取关键信息
        2. 存入记忆库
        3. 加载历史上下文

        Args:
            conversation: 对话文本（可含多轮）
            agent_id: 用户/租户标识

        Returns:
            {"extracted": [...], "stored": N, "context": {...}}
        """
        resp = self.session.post(
            f"{self.base}/api/v1/memory/auto",
            json={"conversation": conversation, "agent_id": agent_id},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["data"]

    def recall(self, query: str, agent_id: str = "default", top_k: int = 10) -> str:
        """
        回忆：新对话前调用，获取历史上下文

        Args:
            query: 当前话题/查询
            agent_id: 用户标识
            top_k: 返回条数

        Returns:
            格式化的上下文文本（可直接注入 system prompt）
        """
        resp = self.session.get(
            f"{self.base}/api/v1/memory/context",
            params={"query": query, "agent_id": agent_id, "top_k": top_k},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["data"].get("context_text", "")

    # ─── 原始 API（向下兼容） ───

    def search(self, content: str, agent_id: str = "", top_k: int = 5) -> List[dict]:
        """语义检索"""
        resp = self.session.get(
            f"{self.base}/api/v1/memories/search",
            params={"content": content, "agent_id": agent_id, "top_k": top_k},
        )
        return resp.json().get("data", [])

    def save(self, agent_id: str, content: str, **kwargs) -> dict:
        """手动存储记忆"""
        data = {"agent_id": agent_id, "content": content, **kwargs}
        resp = self.session.post(f"{self.base}/api/v1/memories", json=data)
        return resp.json()

    def stats(self) -> dict:
        """系统统计"""
        return self.session.get(f"{self.base}/api/v1/stats").json()

    def health(self) -> bool:
        """健康检查"""
        try:
            resp = self.session.get(f"{self.base}/health", timeout=5)
            return resp.json().get("status") == "ok"
        except Exception:
            return False


# ─── 便捷函数 ───

_default_client: Optional[MetaMemory] = None


def remember(conversation: str, agent_id: str = "default") -> dict:
    """一行记忆"""
    global _default_client
    if _default_client is None:
        _default_client = MetaMemory()
    return _default_client.remember(conversation, agent_id)


def recall(query: str, agent_id: str = "default") -> str:
    """一行回忆"""
    global _default_client
    if _default_client is None:
        _default_client = MetaMemory()
    return _default_client.recall(query, agent_id)
