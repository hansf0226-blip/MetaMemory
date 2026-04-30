#!/usr/bin/env python3
"""
Hermes Agent 集成示例

演示如何将易经记忆系统集成到 Hermes Agent 中
"""

import time

import requests

# 配置
API_KEY = "your_api_key"
BASE_URL = "http://localhost:8000"
AGENT_ID = "hermes_agent"

HEADERS = {"X-API-Key": API_KEY}


class YijingMemoryTool:
    """易经记忆工具 - 可集成到 Hermes Agent"""

    def __init__(self, api_key: str, base_url: str, agent_id: str):
        self.api_key = api_key
        self.base_url = base_url
        self.agent_id = agent_id
        self.headers = {"X-API-Key": api_key}

    def save_memory(self, content: str, memory_type: str = "chat") -> dict:
        """
        保存记忆到易经系统

        Args:
            content: 记忆内容
            memory_type: 记忆类型 (chat/task/temp 等)

        Returns:
            dict: 创建结果
        """
        try:
            # 先编码
            encode_resp = requests.post(
                f"{self.base_url}/api/v1/memory/encode/hexagram",
                headers=self.headers,
                json={"content": content, "memory_type": memory_type},
                timeout=10,
            )

            if encode_resp.status_code != 200:
                return {"error": "编码失败", "detail": encode_resp.json()}

            encode_data = encode_resp.json()

            # 创建记忆
            memory_id = f"hermes_{int(time.time())}"
            create_resp = requests.post(
                f"{self.base_url}/api/v1/memories",
                headers=self.headers,
                json={
                    "memory_id": memory_id,
                    "hexagram": encode_data["hexagram"],
                    "bagua_type": encode_data["bagua_type"],
                    "sancai_layer": encode_data["sancai_layer"],
                    "wuxing": encode_data["wuxing"],
                    "content": content,
                    "hot_score": 0.8,
                    "agent_id": self.agent_id,
                },
                timeout=10,
            )

            if create_resp.status_code in [200, 201]:
                return {"success": True, "memory_id": memory_id, "hexagram": encode_data["hexagram"]}
            else:
                return {"error": "创建失败", "detail": create_resp.json()}

        except Exception as e:
            return {"error": str(e)}

    def search_memory(self, query: str, top_k: int = 5) -> list:
        """
        搜索相关记忆

        Args:
            query: 搜索关键词
            top_k: 返回数量

        Returns:
            list: 相似记忆列表
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/memories/search",
                headers=self.headers,
                params={"content": query, "top_k": top_k, "agent_id": self.agent_id},
                timeout=10,
            )

            if response.status_code == 200:
                return response.json()["data"]
            else:
                return []
        except Exception as e:
            print(f"搜索失败：{e}")
            return []

    def get_recent_memories(self, limit: int = 10) -> list:
        """
        获取最近的记忆

        Args:
            limit: 数量限制

        Returns:
            list: 记忆列表
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/memories",
                headers=self.headers,
                params={"page": 1, "page_size": limit, "agent_id": self.agent_id},
                timeout=10,
            )

            if response.status_code == 200:
                return response.json()["list"]
            else:
                return []
        except Exception as e:
            print(f"获取失败：{e}")
            return []


# ============================================================================
# Hermes Agent 工具注册示例
# ============================================================================


def register_yijing_tools():
    """
    在 Hermes Agent 中注册易经记忆工具

    在实际使用中，这应该添加到 Hermes 的 toolsets 中
    """

    # 创建工具实例
    memory_tool = YijingMemoryTool(api_key=API_KEY, base_url=BASE_URL, agent_id=AGENT_ID)

    # 示例：定义工具函数（实际应使用 @registry.register 装饰器）
    def yijing_save_memory(content: str, memory_type: str = "chat"):
        """保存记忆到易经系统"""
        return memory_tool.save_memory(content, memory_type)

    def yijing_search_memory(query: str, top_k: int = 5):
        """搜索相关记忆"""
        return memory_tool.search_memory(query, top_k)

    def yijing_get_recent(limit: int = 10):
        """获取最近的记忆"""
        return memory_tool.get_recent_memories(limit)

    return {"save": yijing_save_memory, "search": yijing_search_memory, "recent": yijing_get_recent}


# ============================================================================
# 使用示例
# ============================================================================


def demo():
    """演示 Hermes 集成"""
    print("=" * 60)
    print("  Hermes Agent - 易经记忆系统集成演示")
    print("=" * 60)

    # 注册工具
    tools = register_yijing_tools()

    # 场景 1: 保存用户偏好
    print("\n📝 场景 1: 保存用户偏好")
    result = tools["save"]("用户说喜欢吃川菜", "chat")
    if result.get("success"):
        print(f"✅ 记忆已保存：{result['memory_id']}")
    else:
        print(f"❌ 保存失败：{result.get('error')}")

    # 场景 2: 保存任务
    print("\n📝 场景 2: 保存任务")
    result = tools["save"]("用户需要明天上午开会", "task")
    if result.get("success"):
        print(f"✅ 任务已保存：{result['memory_id']}")

    # 场景 3: 搜索记忆
    print("\n🔍 场景 3: 搜索记忆")
    results = tools["search"]("川菜", top_k=3)
    if results:
        print(f"✅ 找到 {len(results)} 条相关记忆:")
        for r in results:
            print(f"  - {r['content']}")

    # 场景 4: 获取最近记忆
    print("\n📋 场景 4: 获取最近记忆")
    results = tools["recent"](limit=5)
    if results:
        print(f"✅ 最近 {len(results)} 条记忆:")
        for r in results:
            print(f"  - {r['content'][:50]}")

    print("\n" + "=" * 60)
    print("✅ 演示完成")
    print("=" * 60)


if __name__ == "__main__":
    try:
        demo()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断")
    except Exception as e:
        print(f"\n❌ 错误：{e}")
