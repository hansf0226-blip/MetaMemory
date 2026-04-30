#!/usr/bin/env python3
"""
易经 Agent 记忆系统 - 基础用法示例

本示例演示如何使用易经记忆系统的基础功能
"""

import time

import requests

# 配置
API_KEY = "your_api_key"  # 替换为你的 API Key
BASE_URL = "http://localhost:8000"

HEADERS = {"X-API-Key": API_KEY}


def print_section(title: str):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def health_check():
    """健康检查"""
    print_section("1. 健康检查")

    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码：{response.status_code}")
    print(f"响应：{response.json()}")

    return response.status_code == 200


def encode_hexagram():
    """卦象编码"""
    print_section("2. 卦象编码")

    content = "用户说喜欢吃川菜"
    print(f"内容：{content}")

    response = requests.post(
        f"{BASE_URL}/api/v1/memory/encode/hexagram", headers=HEADERS, json={"content": content, "memory_type": "chat"}
    )

    if response.status_code == 200:
        data = response.json()
        print("✅ 编码成功")
        print(f"  六爻：{data['hexagram']}")
        print(f"  八卦：{data['bagua_type']}")
        print(f"  五行：{data['wuxing']}")
        print(f"  三才：{data['sancai_layer']}")
        return data
    else:
        print(f"❌ 编码失败：{response.json()}")
        return None


def create_memory():
    """创建记忆"""
    print_section("3. 创建记忆")

    memory_id = f"test_memory_{int(time.time())}"
    print(f"记忆 ID: {memory_id}")

    response = requests.post(
        f"{BASE_URL}/api/v1/memories",
        headers=HEADERS,
        json={
            "memory_id": memory_id,
            "hexagram": [1, 0, 1, 0, 1, 0],
            "bagua_type": "kan",
            "sancai_layer": "ren",
            "wuxing": "shui",
            "content": "用户偏好川菜口味",
            "hot_score": 0.8,
            "agent_id": "default_agent",
        },
    )

    if response.status_code in [200, 201]:
        print("✅ 记忆创建成功")
        print(f"响应：{response.json()}")
        return memory_id
    else:
        print(f"❌ 创建失败：{response.json()}")
        return None


def get_memory(memory_id: str):
    """获取记忆详情"""
    print_section("4. 获取记忆详情")

    response = requests.get(f"{BASE_URL}/api/v1/memories/{memory_id}", headers=HEADERS)

    if response.status_code == 200:
        data = response.json()["data"]
        print("✅ 获取成功")
        print(f"  内容：{data['content']}")
        print(f"  八卦：{data['bagua_type']}")
        print(f"  五行：{data['wuxing']}")
        print(f"  热度：{data['hot_score']}")
        return data
    else:
        print(f"❌ 获取失败：{response.json()}")
        return None


def search_memory():
    """搜索记忆"""
    print_section("5. 搜索记忆")

    response = requests.get(
        f"{BASE_URL}/api/v1/memories/search",
        headers=HEADERS,
        params={"content": "川菜", "top_k": 5, "agent_id": "default_agent"},
    )

    if response.status_code == 200:
        results = response.json()["data"]
        print(f"✅ 搜索到 {len(results)} 条结果")
        for i, r in enumerate(results, 1):
            print(f"  {i}. 相似度：{r.get('similarity', 'N/A'):.2f}, 内容：{r['content'][:50]}")
        return results
    else:
        print(f"❌ 搜索失败：{response.json()}")
        return []


def update_memory(memory_id: str):
    """更新记忆"""
    print_section("6. 更新记忆")

    response = requests.put(
        f"{BASE_URL}/api/v1/memories/{memory_id}",
        headers=HEADERS,
        json={
            "hexagram": [0, 1, 0, 1, 0, 1],
            "bagua_type": "li",
            "sancai_layer": "ren",
            "wuxing": "huo",
            "content": "用户偏好川菜口味（已更新）",
            "hot_score": 0.9,
        },
    )

    if response.status_code == 200:
        print("✅ 更新成功")
        return True
    else:
        print(f"❌ 更新失败：{response.json()}")
        return False


def delete_memory(memory_id: str):
    """删除记忆"""
    print_section("7. 删除记忆")

    response = requests.delete(f"{BASE_URL}/api/v1/memories/{memory_id}", headers=HEADERS)

    if response.status_code == 200:
        print("✅ 删除成功")
        return True
    else:
        print(f"❌ 删除失败：{response.json()}")
        return False


def main():
    """主函数"""
    print("\n" + "📦" * 30)
    print("  易经 Agent 记忆系统 - 基础用法示例")
    print("📦" * 30)

    # 1. 健康检查
    if not health_check():
        print("\n❌ 服务未启动，请先启动服务")
        return

    # 2. 卦象编码
    encode_result = encode_hexagram()
    if not encode_result:
        print("\n⚠️  编码失败，继续执行...")

    # 3. 创建记忆
    memory_id = create_memory()
    if not memory_id:
        print("\n⚠️  创建失败，可能是 ID 已存在")
        memory_id = f"test_memory_{int(time.time())}"

    # 等待一下
    time.sleep(1)

    # 4. 获取记忆详情
    get_memory(memory_id)

    # 5. 搜索记忆
    search_memory()

    # 6. 更新记忆
    update_memory(memory_id)

    # 7. 删除记忆
    delete_memory(memory_id)

    print_section("示例完成")
    print("✅ 所有操作已完成")
    print("\n更多示例请查看 EXAMPLES.md")
    print("📦" * 30 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断")
    except Exception as e:
        print(f"\n❌ 错误：{e}")
