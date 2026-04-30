#!/usr/bin/env python3
"""
Alpha-Conscious-Agent 工具调用示例
演示如何使用工具完成各种任务
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.llm_manager import LLMManager
from api.tool_calling import ToolCallingIntegration


def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def example_1_file_operations():
    """示例 1：文件操作"""
    print_separator("示例 1: 文件操作")

    llm = LLMManager()
    tools = ToolCallingIntegration(llm)

    task = """
请帮我完成以下任务：
1. 在当前目录创建一个测试文件 test_alpha.txt
2. 写入内容："Hello Alpha! 这是工具测试。"
3. 读取这个文件并显示内容
4. 告诉我文件创建成功了吗
"""

    print(f"任务：{task}")
    result = tools.chat_with_tools(task, max_iterations=5)

    print(f"\n最终回复：\n{result['response']}")
    print(f"\n工具调用次数：{len(result['tool_calls'])}")
    for i, tc in enumerate(result["tool_calls"], 1):
        print(f"  {i}. {tc.tool_name}: {'✅' if tc.success else '❌'}")

    return result


def example_2_web_search():
    """示例 2：网络搜索"""
    print_separator("示例 2: 网络搜索")

    llm = LLMManager()
    tools = ToolCallingIntegration(llm)

    task = """
请帮我搜索一下"Python 最新特性 2026"，然后总结一下搜索结果。
"""

    print(f"任务：{task}")
    result = tools.chat_with_tools(task, max_iterations=3)

    print(f"\n最终回复：\n{result['response']}")
    print(f"\n工具调用次数：{len(result['tool_calls'])}")

    return result


def example_3_code_execution():
    """示例 3：代码执行"""
    print_separator("示例 3: 代码执行")

    llm = LLMManager()
    tools = ToolCallingIntegration(llm)

    task = """
请帮我执行以下 Python 代码，并告诉我结果：
计算 1 到 100 的平方和，然后找出其中的所有质数。
"""

    print(f"任务：{task}")
    result = tools.chat_with_tools(task, max_iterations=3)

    print(f"\n最终回复：\n{result['response']}")
    print(f"\n工具调用次数：{len(result['tool_calls'])}")

    return result


def example_4_terminal_command():
    """示例 4：终端命令"""
    print_separator("示例 4: 终端命令")

    llm = LLMManager()
    tools = ToolCallingIntegration(llm)

    task = """
请帮我执行以下操作：
1. 查看当前目录是什么
2. 列出当前目录下的所有文件和文件夹
3. 告诉我有多少个文件
"""

    print(f"任务：{task}")
    result = tools.chat_with_tools(task, max_iterations=4)

    print(f"\n最终回复：\n{result['response']}")
    print(f"\n工具调用次数：{len(result['tool_calls'])}")

    return result


def example_5_api_call():
    """示例 5：API 调用"""
    print_separator("示例 5: API 调用")

    llm = LLMManager()
    tools = ToolCallingIntegration(llm)

    task = """
请帮我访问 https://httpbin.org/get 这个 API，然后告诉我返回了什么信息。
"""

    print(f"任务：{task}")
    result = tools.chat_with_tools(task, max_iterations=3)

    print(f"\n最终回复：\n{result['response']}")
    print(f"\n工具调用次数：{len(result['tool_calls'])}")

    return result


def example_6_complex_task():
    """示例 6：复杂任务（多工具组合）"""
    print_separator("示例 6: 复杂任务 - 多工具组合")

    llm = LLMManager()
    tools = ToolCallingIntegration(llm)

    task = """
请帮我完成一个综合任务：
1. 创建一个名为 report.txt 的文件
2. 写入一份简单的报告，包含：
   - 标题："Alpha 工具测试报告"
   - 日期：今天
   - 测试内容：文件创建、代码执行、命令运行
3. 执行一个 Python 代码计算 1+2+3+...+100 的和
4. 把计算结果也追加到报告中
5. 最后读取报告并展示给我
"""

    print(f"任务：{task}")
    result = tools.chat_with_tools(task, max_iterations=8)

    print(f"\n最终回复：\n{result['response']}")
    print(f"\n工具调用次数：{len(result['tool_calls'])}")
    for i, tc in enumerate(result["tool_calls"], 1):
        status = "✅" if tc.success else "❌"
        print(f"  {i}. {tc.tool_name}: {status}")

    # 显示统计
    stats = tools.get_stats()
    print("\n工具使用统计:")
    print(f"  总调用：{stats['total_calls']}")
    print(f"  成功：{stats['successful_calls']}")
    print(f"  成功率：{stats['success_rate']:.1f}%")

    return result


def main():
    """运行所有示例"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "Alpha-Conscious-Agent" + " " * 15 + "║")
    print("║" + " " * 20 + "工具调用示例" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")

    # 检查 LLM 配置
    print("\n检查 LLM 配置...")
    llm = LLMManager()
    if not llm.providers:
        print("⚠️ 未配置 LLM 提供商，将使用模拟模式")
        print("   请在 .env 文件中配置 API Key")
    else:
        print(f"✅ LLM 提供商：{llm.active_provider}")

    # 运行示例（可以选择运行哪些）
    examples = [
        ("文件操作", example_1_file_operations),
        ("网络搜索", example_2_web_search),
        ("代码执行", example_3_code_execution),
        ("终端命令", example_4_terminal_command),
        ("API 调用", example_5_api_call),
        ("复杂任务", example_6_complex_task),
    ]

    print("\n可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print("  0. 运行全部")

    try:
        choice = input("\n请选择要运行的示例 (0-6): ").strip()

        if choice == "0":
            for name, func in examples:
                func()
        elif choice.isdigit() and 1 <= int(choice) <= len(examples):
            examples[int(choice) - 1][1]()
        else:
            print("无效选择，运行第一个示例")
            examples[0][1]()

    except KeyboardInterrupt:
        print("\n\n中断")
    except Exception as e:
        print(f"\n错误：{e}")
        import traceback

        traceback.print_exc()

    print("\n✅ 示例运行完成")


if __name__ == "__main__":
    main()
