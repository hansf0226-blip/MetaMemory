"""
行动执行模块 - 执行 Agent 的决策和行动
"""

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List


class ActionExecutor:
    """Agent 的行动执行器"""

    def __init__(self):
        self.actions: List[Dict] = []  # 行动历史
        self.pending_actions: List[Dict] = []  # 待执行行动
        self.registered_actions: Dict[str, Callable] = {}  # 注册的行动函数
        self.execution_log: List[Dict] = []  # 执行日志

    def register_action(self, name: str, func: Callable):
        """注册行动函数"""
        self.registered_actions[name] = func

    def queue_action(self, action_type: str, params: Dict = None, priority: int = 5):
        """将行动加入队列"""
        action = {
            "id": f"action_{datetime.now().timestamp()}",
            "type": action_type,
            "params": params or {},
            "priority": priority,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
        }
        self.pending_actions.append(action)
        self.pending_actions.sort(key=lambda x: x["priority"], reverse=True)

    async def execute_action(self, action: Dict) -> Any:
        """执行单个行动"""
        action_type = action["type"]

        if action_type not in self.registered_actions:
            result = {"error": f"未知行动类型：{action_type}"}
        else:
            try:
                func = self.registered_actions[action_type]
                if asyncio.iscoroutinefunction(func):
                    result = await func(**action["params"])
                else:
                    result = func(**action["params"])
            except Exception as e:
                result = {"error": str(e)}

        # 记录执行结果
        result_dict = result if isinstance(result, dict) else {}
        has_error = "error" in result_dict
        self.actions.append(
            {
                **action,
                "result": result,
                "executed_at": datetime.now().isoformat(),
                "status": "failed" if has_error else "completed",
            }
        )

        # 记录日志
        self.execution_log.append(
            {
                "action": action_type,
                "status": "error" if has_error else "success",
                "timestamp": datetime.now().isoformat(),
            }
        )

        return result

    async def execute_pending(self, max_count: int = 5):
        """执行待处理行动"""
        executed = []

        for _ in range(min(max_count, len(self.pending_actions))):
            if self.pending_actions:
                action = self.pending_actions.pop(0)
                action["status"] = "executing"
                result = await self.execute_action(action)
                executed.append({"action": action, "result": result})

        return executed

    def get_action_summary(self) -> str:
        """获取行动摘要"""
        return """
【行动系统】
待执行：{len(self.pending_actions)}
已执行：{len(self.actions)}
日志记录：{len(self.execution_log)}

【最近行动】
{chr(10).join(f"• {a['type']} - {a['status']} ({a.get('result', {})})" for a in self.actions[-5:]) if self.actions else '无'}

【执行日志】
{chr(10).join(f"• {log['action']} - {log['status']}" for log in self.execution_log[-10:]) if self.execution_log else '无'}
"""

    def to_dict(self) -> Dict:
        """序列化"""
        return {
            "actions": self.actions[-50:],
            "pending_actions": self.pending_actions,
            "execution_log": self.execution_log[-100:],
        }
