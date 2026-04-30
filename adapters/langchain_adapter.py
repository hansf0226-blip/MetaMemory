"""
MetaMemory LangChain 适配器 — 一行代码接入记忆系统

用法:
    from metamemory_langchain import MetaMemoryChatMessageHistory

    memory = MetaMemoryChatMessageHistory(
        agent_id="my_agent",
        base_url="http://localhost:8000"
    )

    # 自动存每条消息，自动检索历史上下文
    chain = ConversationalRetrievalChain.from_llm(
        llm, retriever, memory=memory
    )
"""

import uuid
import time
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class MetaMemoryChatMessageHistory:
    """LangChain 兼容的对话记忆后端，自动同步到 MetaMemory"""

    def __init__(
        self,
        agent_id: str,
        base_url: str = "http://localhost:8000",
        auto_search: bool = True,
        search_top_k: int = 5,
        session_id: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.base_url = base_url.rstrip("/")
        self.auto_search = auto_search
        self.search_top_k = search_top_k
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self._messages: List[Any] = []

    @property
    def messages(self) -> List[Any]:
        return self._messages

    def add_message(self, message: Any) -> None:
        """添加消息，自动存入 MetaMemory"""
        self._messages.append(message)
        self._save_to_metamemory(message)

    def add_user_message(self, content: str) -> None:
        from langchain_core.messages import HumanMessage
        self.add_message(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        from langchain_core.messages import AIMessage
        self.add_message(AIMessage(content=content))

    def clear(self) -> None:
        self._messages = []

    def _save_to_metamemory(self, message: Any) -> None:
        """存入 MetaMemory"""
        try:
            import requests
            content = message.content if hasattr(message, "content") else str(message)
            msg_type = type(message).__name__

            memory_id = f"{self.session_id}_{int(time.time() * 1000)}"
            requests.post(
                f"{self.base_url}/api/v1/memories",
                json={
                    "agent_id": self.agent_id,
                    "memory_id": memory_id,
                    "content": f"[{msg_type}] {content}",
                    "hot_score": 0.7,
                    "hexagram": "000000",
                    "bagua_type": "auto",
                    "sancai_layer": "unknown",
                    "wuxing": "unknown",
                },
                timeout=5,
            )
        except Exception as e:
            logger.warning(f"MetaMemory 存储失败: {e}")

    def search_context(self, query: str, top_k: int = None) -> List[Dict]:
        """检索历史上下文"""
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
        except Exception as e:
            logger.warning(f"MetaMemory 检索失败: {e}")
            return []

    def get_context_for_query(self, query: str) -> str:
        """获取查询相关的历史上下文文本"""
        results = self.search_context(query)
        if not results:
            return ""
        lines = []
        for r in results:
            sim = r.get("similarity", 0)
            content = r.get("content", "")[:200]
            lines.append(f"[相似度:{sim:.2f}] {content}")
        return "\n".join(lines)


class MetaMemoryConversationBufferMemory:
    """LangChain ConversationBufferMemory 的 MetaMemory 后端"""

    def __init__(
        self,
        agent_id: str,
        base_url: str = "http://localhost:8000",
        search_top_k: int = 5,
        memory_key: str = "history",
        return_messages: bool = True,
    ):
        self.chat_memory = MetaMemoryChatMessageHistory(
            agent_id=agent_id,
            base_url=base_url,
            search_top_k=search_top_k,
        )
        self.memory_key = memory_key
        self.return_messages = return_messages

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs.get("input", "") or inputs.get("query", "")
        context = self.chat_memory.get_context_for_query(query)
        return {self.memory_key: context}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", "") or outputs.get("response", "")
        if user_input:
            self.chat_memory.add_user_message(user_input)
        if ai_output:
            self.chat_memory.add_ai_message(ai_output)

    def clear(self) -> None:
        self.chat_memory.clear()
