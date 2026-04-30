"""
MetaMemory Hermes Agent 集成 — 自动记忆工具

用法: 在 Hermes Agent 的 toolsets 注册即可
    from metamemory_hermes import register_metamemory_tools
    register_metamemory_tools(registry, agent_id="hermes", base_url="http://localhost:8000")
"""

import time
import uuid


def create_metamemory_tools(agent_id: str, base_url: str = "http://localhost:8000"):
    """创建 MetaMemory 工具集，可直接注册到 Hermes toolsets"""

    session_id = f"hermes_{uuid.uuid4().hex[:8]}"
    base = base_url.rstrip("/")

    def save_memory(content: str, memory_type: str = "chat", hot_score: float = 0.7):
        """保存记忆到 MetaMemory

        Args:
            content: 要保存的记忆内容
            memory_type: 类型 (chat/task/knowledge/preference)
            hot_score: 热度 (0-1)
        """
        try:
            import requests

            memory_id = f"{session_id}_{int(time.time() * 1000)}"
            resp = requests.post(
                f"{base}/api/v1/memories",
                json={
                    "agent_id": agent_id,
                    "memory_id": memory_id,
                    "content": f"[{memory_type}] {content}",
                    "hot_score": hot_score,
                    "hexagram": "000000",
                    "bagua_type": "auto",
                    "sancai_layer": "unknown",
                    "wuxing": "unknown",
                },
                timeout=5,
            )
            data = resp.json()
            return f"✅ 已保存记忆: {memory_id}"
        except Exception as e:
            return f"❌ 保存失败: {e}"

    def search_memory(query: str, top_k: int = 5):
        """搜索相关记忆

        Args:
            query: 搜索关键词
            top_k: 返回数量 (默认5)
        """
        try:
            import requests

            resp = requests.get(
                f"{base}/api/v1/memories/search",
                params={"content": query, "top_k": top_k, "agent_id": agent_id},
                timeout=10,
            )
            results = resp.json().get("data", [])
            if not results:
                return "未找到相关记忆。"

            lines = []
            for r in results:
                sim = r.get("similarity", 0)
                content = r.get("content", "")[:150]
                lines.append(f"[匹配度:{sim:.0%}] {content}")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ 检索失败: {e}"

    def auto_remember(user_message: str, assistant_response: str):
        """自动保存本轮对话 (用户消息 + AI 回复)

        Args:
            user_message: 用户消息
            assistant_response: AI 回复
        """
        save_memory(f"用户: {user_message}", "chat", 0.8)
        save_memory(f"助手: {assistant_response[:200]}", "chat", 0.6)
        return "✅ 对话已自动记录"

    def recall_context(query: str = ""):
        """调用前自动加载最近相关记忆上下文

        如果 query 为空，则加载最近 5 条记忆
        """
        if query:
            return search_memory(query, top_k=5)
        else:
            return search_memory("", top_k=5)

    return {
        "save_memory": save_memory,
        "search_memory": search_memory,
        "auto_remember": auto_remember,
        "recall_context": recall_context,
    }
