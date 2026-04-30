#!/usr/bin/env python3
"""
Hermes Agent - 易经记忆系统自动集成模块

使用方法：
在 Hermes 启动前导入此模块，自动注入记忆功能：

```python
# 在 run_agent.py 或 cli.py 的开头导入
import yijing_memory_hook  # 自动生效
```

功能：
1. 每次对话前自动检索相关历史记忆
2. 每次对话后自动保存对话内容
3. 完全自动化，无需手动指令
4. 低侵入，不修改 Hermes 原有代码
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

# ============================================================================
# 配置加载
# ============================================================================

YIJING_CONFIG = {
    "enabled": os.getenv("YIJING_ENABLED", "true").lower() == "true",
    "api_url": os.getenv("YIJING_API_URL", "http://localhost:8000"),
    "agent_id": os.getenv("YIJING_AGENT_ID", None),
    "search_top_k": int(os.getenv("YIJING_SEARCH_TOP_K", "5")),
    "min_similarity": float(os.getenv("YIJING_MIN_SIMILARITY", "0.3")),
}

if not YIJING_CONFIG["enabled"]:
    logger.info("🔮 易经记忆系统：已禁用（YIJING_ENABLED=false）")
    sys.exit(0)

logger.info(f"🔮 易经记忆系统：已启用 | API: {YIJING_CONFIG['api_url']}")

# ============================================================================
# 易经记忆客户端
# ============================================================================


class YijingMemoryClient:
    """轻量级易经记忆 HTTP 客户端"""

    def __init__(self):
        self.api_url = YIJING_CONFIG["api_url"].rstrip("/")
        self.agent_id = YIJING_CONFIG["agent_id"] or self._generate_id()
        self.timeout = 5
        self.session = None  # 延迟导入 requests

    def _generate_id(self) -> str:
        """生成唯一 Agent ID"""
        import getpass
        import hashlib
        import socket

        raw = f"{socket.gethostname()}:{getpass.getuser()}:{os.getpid()}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _request(self, method: str, endpoint: str, **kwargs):
        """发送 HTTP 请求"""
        if self.session is None:
            import requests

            self.session = requests.Session()

        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        try:
            resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"⚠️ 易经记忆 API 请求失败：{e}")
            return None

    def retrieve(self, query: str, top_k: int = None) -> list:
        """检索相关记忆"""
        # GET 接口
        params = {
            "content": query,
            "top_k": top_k or YIJING_CONFIG["search_top_k"],
            "agent_id": self.agent_id,
        }
        result = self._request("GET", "/api/v1/memories/search", params=params)

        if not result or result.get("code") != 200:
            return []

        memories = result.get("data", [])
        return [m for m in memories if m.get("similarity", 0) >= YIJING_CONFIG["min_similarity"]]

    def save(self, user_msg: str, assistant_msg: str, memory_type: str = None) -> str:
        """保存对话记忆"""
        import uuid

        # 自动推断记忆类型
        if not memory_type:
            content = f"{user_msg} {assistant_msg}".lower()
            if any(kw in content for kw in ["记住", "重要", "总是", "偏好", "name", "call me"]):
                memory_type = "long_term"
            elif any(kw in content for kw in ["项目", "任务", "代码", "project", "task"]):
                memory_type = "medium_term"
            else:
                memory_type = "short_term"

        # 卦象编码
        encode_result = (
            self._request(
                "POST",
                "/api/v1/memory/encode/hexagram",
                json={
                    "content": f"用户：{user_msg}\n\n助手：{assistant_msg}",
                    "memory_type": memory_type,
                    "agent_id": self.agent_id,
                },
            )
            or {}
        )

        # 生成记忆 ID
        memory_id = str(uuid.uuid4())

        # 保存入库（匹配 API 字段）
        result = self._request(
            "POST",
            "/api/v1/memories",
            json={
                "agent_id": self.agent_id,
                "memory_id": memory_id,
                "content": f"用户：{user_msg}\n\n助手：{assistant_msg}",
                "hexagram": encode_result.get("hexagram", [0, 0, 0, 0, 0, 0]),
                "bagua_type": encode_result.get("bagua_type", "kun"),
                "sancai_layer": encode_result.get("sancai_layer", "di"),
                "wuxing": encode_result.get("wuxing", "tu"),
                "hot_score": 0.5,
            },
        )

        if result and result.get("code") == 200:
            returned_id = result.get("memory_id", memory_id)
            logger.info(f"💾 记忆已保存 | ID: {returned_id}")
            return returned_id
        return None

    def format_memories(self, memories: list) -> str:
        """格式化记忆为 prompt"""
        if not memories:
            return ""

        lines = ["\n\n【相关历史记忆】", "以下是与当前问题相关的历史对话：", "-" * 60]
        for i, mem in enumerate(memories, 1):
            lines.append(f"[记忆{i}] (相似度：{mem.get('similarity', 0):.2f})")
            lines.append(mem.get("content", ""))
            lines.append("-" * 60)
        lines.append("请结合以上历史记忆回答问题。\n")
        return "\n".join(lines)


# 全局单例
_yijing_client = None


def get_client() -> YijingMemoryClient:
    global _yijing_client
    if _yijing_client is None:
        _yijing_client = YijingMemoryClient()
    return _yijing_client


# ============================================================================
# Monkey Patch - 注入到 Hermes
# ============================================================================


def _patch_hermes():
    """将易经记忆注入到 Hermes Agent"""

    try:
        from run_agent import AIAgent
    except ImportError:
        logger.warning("⚠️ 无法导入 AIAgent，易经记忆集成跳过")
        return

    # 保存原始方法
    _original_run = AIAgent.run_conversation

    def _patched_run(self, user_message: str, *args, **kwargs):
        """包装 run_conversation，添加记忆检索和保存"""

        client = get_client()

        # 【第一步】对话前：检索相关记忆
        memories = client.retrieve(user_message)
        memory_context = client.format_memories(memories)

        # 将记忆注入到 user_message
        if memory_context:
            enhanced_message = f"{user_message}{memory_context}"
            logger.info(f"📖 注入 {len(memories)} 条相关记忆")
        else:
            enhanced_message = user_message

        # 调用原始方法
        try:
            result = _original_run(self, enhanced_message, *args, **kwargs)
        except Exception as e:
            # 即使出错也要保存记忆（记录错误）
            client.save(user_message, f"[错误] {str(e)}")
            raise

        # 【第二步】对话后：保存对话记忆
        assistant_response = result.get("final_response", "") if isinstance(result, dict) else str(result)
        client.save(user_message, assistant_response)

        return result

    # 应用 patch
    AIAgent.run_conversation = _patched_run
    logger.info("✅ 易经记忆系统已成功注入 Hermes Agent")


# ============================================================================
# 自动执行
# ============================================================================

if __name__ == "__main__":
    # 测试模式
    print("🔮 易经记忆系统集成测试")
    client = get_client()

    print("\n📝 测试保存...")
    mid = client.save("今天天气很好", "是的，天气不错！")
    print(f"保存结果：{mid}")

    print("\n🔍 测试检索...")
    memories = client.retrieve("心情怎么样")
    print(f"检索到 {len(memories)} 条")

else:
    # 导入时自动 patch
    _patch_hermes()
