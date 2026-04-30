#!/usr/bin/env python3
"""
工具调用集成模块 - 实现与 LLM 的工具调用集成

功能：
- 与 LLM 集成，支持工具调用
- 提供各种工具，如文件操作、网络搜索、代码执行等
- 支持多轮工具调用和结果处理
- 记录工具调用统计信息
"""

import json
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

import requests

from api.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class ToolCall:
    """
    工具调用类

    表示一次工具调用及其结果
    """

    def __init__(self, tool_name: str, arguments: Dict[str, Any], result: Any, success: bool):
        self.tool_name = tool_name
        self.arguments = arguments
        self.result = result
        self.success = success


class ToolCallingIntegration:
    """
    工具调用集成

    与 LLM 集成，支持工具调用
    """

    def __init__(self, llm: LLMManager):
        """
        初始化工具调用集成

        Args:
            llm: LLM 管理器实例
        """
        self.llm = llm
        self.tools = self._register_tools()
        self.tool_calls: List[ToolCall] = []
        self.stats = {"total_calls": 0, "successful_calls": 0, "failed_calls": 0, "success_rate": 0.0}

    def _register_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        注册工具

        Returns:
            工具字典
        """
        return {
            "file_operations": {
                "name": "file_operations",
                "description": "文件操作工具，用于创建、读取、写入文件",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["create", "read", "write", "append"],
                            "description": "操作类型",
                        },
                        "file_path": {"type": "string", "description": "文件路径"},
                        "content": {"type": "string", "description": "文件内容，仅用于 write 和 append 操作"},
                    },
                    "required": ["operation", "file_path"],
                },
                "function": self._file_operations,
            },
            "web_search": {
                "name": "web_search",
                "description": "网络搜索工具，用于搜索网络信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索查询"},
                        "top_k": {"type": "integer", "default": 3, "description": "返回结果数量"},
                    },
                    "required": ["query"],
                },
                "function": self._web_search,
            },
            "code_execution": {
                "name": "code_execution",
                "description": "代码执行工具，用于执行 Python 代码",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "要执行的 Python 代码"}},
                    "required": ["code"],
                },
                "function": self._code_execution,
            },
            "terminal_command": {
                "name": "terminal_command",
                "description": "终端命令工具，用于执行终端命令",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "要执行的终端命令"},
                        "cwd": {"type": "string", "default": ".", "description": "工作目录"},
                    },
                    "required": ["command"],
                },
                "function": self._terminal_command,
            },
            "api_call": {
                "name": "api_call",
                "description": "API 调用工具，用于访问外部 API",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "API URL"},
                        "method": {
                            "type": "string",
                            "default": "GET",
                            "enum": ["GET", "POST", "PUT", "DELETE"],
                            "description": "HTTP 方法",
                        },
                        "headers": {"type": "object", "description": "HTTP 头"},
                        "data": {"type": "object", "description": "请求数据，仅用于 POST 和 PUT 方法"},
                    },
                    "required": ["url"],
                },
                "function": self._api_call,
            },
        }

    def _file_operations(self, operation: str, file_path: str, content: str = "") -> Dict[str, Any]:
        """
        文件操作

        Args:
            operation: 操作类型
            file_path: 文件路径
            content: 文件内容

        Returns:
            操作结果
        """
        try:
            if operation == "create":
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return {"success": True, "message": f"文件已创建: {file_path}"}

            elif operation == "read":
                if not os.path.exists(file_path):
                    return {"success": False, "error": f"文件不存在: {file_path}"}
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return {"success": True, "content": content}

            elif operation == "write":
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return {"success": True, "message": f"文件已写入: {file_path}"}

            elif operation == "append":
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(content)
                return {"success": True, "message": f"内容已追加到文件: {file_path}"}

            else:
                return {"success": False, "error": f"未知操作: {operation}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _web_search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        网络搜索

        Args:
            query: 搜索查询
            top_k: 返回结果数量

        Returns:
            搜索结果
        """
        try:
            # 这里应该实现实际的网络搜索
            # 为了演示，我们返回模拟结果
            results = []
            for i in range(top_k):
                results.append(
                    {
                        "title": f"搜索结果 {i+1} for {query}",
                        "snippet": f"这是关于 {query} 的搜索结果 {i+1} 的摘要",
                        "url": f"https://example.com/search?q={query}&result={i+1}",
                    }
                )
            return {"success": True, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _code_execution(self, code: str) -> Dict[str, Any]:
        """
        代码执行

        Args:
            code: 要执行的 Python 代码

        Returns:
            执行结果
        """
        try:
            # 创建临时文件执行代码
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_file = f.name

            # 执行代码
            result = subprocess.run(["python3", temp_file], capture_output=True, text=True, timeout=30)

            # 清理临时文件
            os.unlink(temp_file)

            return {"success": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _terminal_command(self, command: str, cwd: str = ".") -> Dict[str, Any]:
        """
        终端命令

        Args:
            command: 要执行的终端命令
            cwd: 工作目录

        Returns:
            执行结果
        """
        try:
            result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=30)

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _api_call(
        self, url: str, method: str = "GET", headers: Dict[str, str] = None, data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        API 调用

        Args:
            url: API URL
            method: HTTP 方法
            headers: HTTP 头
            data: 请求数据

        Returns:
            API 响应
        """
        try:
            headers = headers or {}

            if method == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=10)
            elif method == "PUT":
                response = requests.put(url, headers=headers, json=data, timeout=10)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers, timeout=10)
            else:
                return {"success": False, "error": f"未知 HTTP 方法: {method}"}

            return {
                "success": True,
                "status_code": response.status_code,
                "content": response.text,
                "headers": dict(response.headers),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _format_tools_for_llm(self) -> str:
        """
        格式化工具信息，用于 LLM

        Returns:
            格式化的工具信息
        """
        tools_info = []
        for tool_name, tool_info in self.tools.items():
            tools_info.append(f"{tool_name}: {tool_info['description']}")
        return "\n".join(tools_info)

    def _parse_tool_call(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """
        解析 LLM 的工具调用请求

        Args:
            llm_response: LLM 的响应

        Returns:
            工具调用信息
        """
        # 这里应该实现更复杂的解析逻辑
        # 为了演示，我们使用简单的解析
        try:
            # 查找工具调用模式
            if "tool_call" in llm_response:
                # 模拟解析结果
                return {
                    "tool_name": "file_operations",
                    "arguments": {"operation": "create", "file_path": "test.txt", "content": "Hello World"},
                }
        except Exception as e:
            logger.error(f"解析工具调用失败: {e}")
        return None

    def chat_with_tools(self, task: str, max_iterations: int = 5) -> Dict[str, Any]:
        """
        与工具聊天

        Args:
            task: 任务描述
            max_iterations: 最大迭代次数

        Returns:
            聊天结果
        """
        messages = [
            {
                "role": "system",
                "content": f"你是一个助手，能够使用以下工具来完成任务:\n{self._format_tools_for_llm()}\n\n当需要使用工具时，请以 tool_call: {{'tool_name': '工具名称', 'arguments': {{参数}}}} 的格式输出。",
            },
            {"role": "user", "content": task},
        ]

        for i in range(max_iterations):
            # 调用 LLM
            llm_response = self.llm.chat(messages)
            messages.append({"role": "assistant", "content": llm_response})

            # 解析工具调用
            tool_call_info = self._parse_tool_call(llm_response)

            if not tool_call_info:
                # 没有工具调用，结束对话
                return {"response": llm_response, "tool_calls": self.tool_calls}

            # 执行工具调用
            tool_name = tool_call_info["tool_name"]
            arguments = tool_call_info["arguments"]

            if tool_name in self.tools:
                tool_function = self.tools[tool_name]["function"]
                result = tool_function(**arguments)

                # 记录工具调用
                success = result.get("success", False)
                tool_call = ToolCall(tool_name, arguments, result, success)
                self.tool_calls.append(tool_call)

                # 更新统计
                self.stats["total_calls"] += 1
                if success:
                    self.stats["successful_calls"] += 1
                else:
                    self.stats["failed_calls"] += 1
                self.stats["success_rate"] = (
                    self.stats["successful_calls"] / self.stats["total_calls"] if self.stats["total_calls"] > 0 else 0.0
                )

                # 将工具执行结果添加到对话
                messages.append({"role": "tool", "content": f"工具 {tool_name} 执行结果: {json.dumps(result)}"})
            else:
                # 未知工具
                messages.append({"role": "tool", "content": f"未知工具: {tool_name}"})

        # 达到最大迭代次数
        return {"response": "任务执行超时", "tool_calls": self.tool_calls}

    def get_stats(self) -> Dict[str, Any]:
        """
        获取工具调用统计信息

        Returns:
            统计信息
        """
        return self.stats

    def clear_stats(self):
        """
        清空统计信息
        """
        self.stats = {"total_calls": 0, "successful_calls": 0, "failed_calls": 0, "success_rate": 0.0}
        self.tool_calls = []
