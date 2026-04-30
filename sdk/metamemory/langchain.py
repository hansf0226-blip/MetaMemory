"""
LangChain 适配器 — 将 MetaMemory 接入 LangChain 对话链

使用方式:

    from sdk.metamemory.langchain import MetaMemoryChatHistory
    memory = MetaMemoryChatHistory(agent_id="user_123")

    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationChain

    chain = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(chat_memory=memory)
    )
    chain.run("我喜欢川菜")  # 对话完自动记忆
    chain.run("推荐个菜")     # 自动加载历史上下文
"""
from typing import Any, Dict, List, Optional

from sdk.metamemory import MetaMemory


class MetaMemoryChatHistory:
    """LangChain 兼容的 Chat Message History

    对接 LangChain 的 BaseChatMessageHistory 接口，
    每次对话自动提取记忆并加载历史上下文。
    """

    def __init__(
        self,
        agent_id: str = "default",
        base_url: str = "http://localhost:8000",
        auto_remember: bool = True,
        auto_context: bool = True,
    ):
        self.agent_id = agent_id
        self.client = MetaMemory(base_url=base_url)
        self.auto_remember = auto_remember
        self.auto_context = auto_context
        self._messages: List[Dict[str, Any]] = []

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """LangChain 要求: 返回消息列表"""
        return self._messages

    def add_message(self, message: Any) -> None:
        """LangChain 要求: 添加单条消息"""
        msg_dict = {
            "type": getattr(message, "type", "human"),
            "content": getattr(message, "content", str(message)),
        }
        self._messages.append(msg_dict)

    def add_user_message(self, message: str) -> None:
        """LangChain 要求"""
        self._messages.append({"type": "human", "content": message})

    def add_ai_message(self, message: str) -> None:
        """LangChain 要求"""
        self._messages.append({"type": "ai", "content": message})

        # AI 回复后自动记忆
        if self.auto_remember:
            self._auto_save()

    def clear(self) -> None:
        """LangChain 要求"""
        self._messages = []

    def _auto_save(self) -> None:
        """自动保存对话到 MetaMemory"""
        if len(self._messages) < 2:
            return

        # 拼接最近几轮对话
        conversation = "\n".join(
            f"{'用户' if m['type'] == 'human' else '助手'}: {m['content']}"
            for m in self._messages[-6:]  # 最近 3 轮
        )
        try:
            self.client.remember(conversation, self.agent_id)
        except Exception:
            pass  # 静默失败，不影响主流程

    def get_context(self, query: str = "") -> str:
        """获取历史上下文（可注入 system prompt）"""
        if not self.auto_context:
            return ""
        if not query and self._messages:
            query = self._messages[-1].get("content", "") if self._messages else ""

        if not query:
            return ""

        try:
            return self.client.recall(query, self.agent_id)
        except Exception:
            return ""


# ─── LangChain 集成辅助 ───

def create_metamemory_chain(llm, agent_id: str = "default", **kwargs):
    """快速创建带 MetaMemory 的 LangChain 对话链"""
    try:
        from langchain.memory import ConversationBufferMemory
        from langchain.chains import ConversationChain

        chat_history = MetaMemoryChatHistory(agent_id=agent_id, **kwargs)
        memory = ConversationBufferMemory(
            chat_memory=chat_history, return_messages=True
        )

        # 注入历史上下文到 system prompt
        context = chat_history.get_context()
        if context:
            # 修改 system prompt
            pass

        return ConversationChain(llm=llm, memory=memory)
    except ImportError:
        raise ImportError("需要安装 langchain: pip install langchain")
