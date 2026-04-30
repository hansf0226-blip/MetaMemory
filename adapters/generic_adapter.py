"""
MetaMemory 通用适配器 — 适配 OpenAI / 类 OpenAI 接口的 Agent

用法:
    from metamemory_adapter import MetaMemoryAdapter

    adapter = MetaMemoryAdapter(agent_id="my_agent")
    
    # 对话前加载记忆
    context = adapter.load_context("用户刚才问了什么")
    
    # 对话后保存
    adapter.remember("用户: 我想吃川菜", "助手: 推荐麻婆豆腐")
"""

import uuid
import time
from typing import List, Dict, Optional


class MetaMemoryAdapter:
    """通用 MetaMemory 适配器 — 适配任何 HTTP 可访问的 Agent"""

    def __init__(
        self,
        agent_id: str,
        base_url: str = "http://localhost:8000",
        auto_load: bool = True,
        auto_save: bool = True,
        search_top_k: int = 5,
    ):
        self.agent_id = agent_id
        self.base_url = base_url.rstrip("/")
        self.auto_load = auto_load
        self.auto_save = auto_save
        self.search_top_k = search_top_k
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"

    # ─── 核心 API ───

    def save(self, content: str, memory_type: str = "chat", hot_score: float = 0.7) -> str:
        """保存单条记忆"""
        try:
            import requests
            memory_id = f"{self.session_id}_{int(time.time() * 1000)}"
            resp = requests.post(
                f"{self.base_url}/api/v1/memories",
                json={
                    "agent_id": self.agent_id,
                    "memory_id": memory_id,
                    "content": content,
                    "hot_score": hot_score,
                    # 后端旧字段（必填但适配器用默认值填充）
                    "hexagram": "000000",
                    "bagua_type": "auto",
                    "sancai_layer": "unknown",
                    "wuxing": "unknown",
                },
                timeout=5,
            )
            return memory_id
        except Exception as e:
            return f"error: {e}"

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """语义搜索"""
        try:
            import requests
            resp = requests.get(
                f"{self.base_url}/api/v1/memories/search",
                params={
                    "content": query,
                    "top_k": top_k or self.search_top_k,
                    "agent_id": self.agent_id,
                },
                timeout=10,
            )
            return resp.json().get("data", [])
        except Exception:
            return []

    def remember(self, user_msg: str, assistant_msg: str = ""):
        """保存一轮对话"""
        if user_msg:
            self.save(f"[用户] {user_msg}", "chat", 0.8)
        if assistant_msg:
            self.save(f"[助手] {assistant_msg[:300]}", "chat", 0.6)

    def load_context(self, query: str, top_k: int = None) -> str:
        """加载相关上下文（文本格式，可直接注入 prompt）"""
        results = self.search(query, top_k)
        if not results:
            return "（无相关历史记忆）"

        lines = ["--- 历史相关记忆 ---"]
        for i, r in enumerate(results, 1):
            sim = r.get("similarity", 0)
            content = r.get("content", "")[:200]
            lines.append(f"{i}. [匹配度:{sim:.0%}] {content}")
        return "\n".join(lines)

    def inject_into_messages(self, messages: List[Dict], query: str = "") -> List[Dict]:
        """在 OpenAI 消息列表中注入记忆上下文"""
        if not query and messages:
            query = messages[-1].get("content", "")

        context = self.load_context(query)
        if "无相关" not in context:
            messages.insert(0, {"role": "system", "content": context})
        return messages

    # ─── 统计 ───

    def stats(self) -> Dict:
        """获取记忆统计"""
        try:
            import requests
            resp = requests.get(f"{self.base_url}/api/v1/stats", timeout=5)
            return resp.json().get("data", {})
        except Exception:
            return {}


# ─── 快捷工厂 ───


def create_adapter(agent_id: str, base_url: str = "http://localhost:8000", **kwargs) -> MetaMemoryAdapter:
    """快捷创建适配器"""
    return MetaMemoryAdapter(agent_id=agent_id, base_url=base_url, **kwargs)
