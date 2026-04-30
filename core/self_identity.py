"""
自我身份模块 - 维护 Agent 的稳定人格和身份认知
"""

from datetime import datetime
from typing import Dict, List


class SelfIdentity:
    """Agent 的自我身份认知系统"""

    def __init__(self, name: str = "Alpha", personality: str = "理性、好奇、友善"):
        self.name = name
        self.personality = personality
        self.created_at = datetime.now()
        self.self_narrative: List[str] = []  # 自我叙事
        self.values: List[str] = []  # 核心价值观
        self.goals: List[Dict] = []  # 目标列表
        self.relationships: Dict[str, str] = {}  # 与其他 Agent 的关系

    def add_self_narrative(self, narrative: str):
        """添加自我叙事"""
        self.self_narrative.append(f"[{datetime.now().isoformat()}] {narrative}")
        # 保持最近的 100 条
        if len(self.self_narrative) > 100:
            self.self_narrative = self.self_narrative[-100:]

    def add_value(self, value: str):
        """添加核心价值观"""
        if value not in self.values:
            self.values.append(value)

    def add_goal(self, goal: str, priority: int = 5, status: str = "pending"):
        """添加目标"""
        # 输入验证
        if not goal or not isinstance(goal, str):
            raise ValueError("Goal must be a non-empty string")
        if not 1 <= priority <= 10:
            raise ValueError("Priority must be between 1 and 10")
        if status not in ["pending", "in_progress", "completed", "cancelled"]:
            raise ValueError("Status must be one of: pending, in_progress, completed, cancelled")

        self.goals.append(
            {"goal": goal, "priority": priority, "status": status, "created_at": datetime.now().isoformat()}
        )

    def get_identity_summary(self) -> str:
        """获取身份摘要"""
        values_str = "\n".join(f"• {v}" for v in self.values[-5:]) if self.values else "尚未形成"
        goals_sorted = sorted(self.goals, key=lambda x: x["priority"], reverse=True)[:5]
        goals_str = "\n".join(f"• {g['goal']} ({g['status']})" for g in goals_sorted) if self.goals else "暂无"

        return """
我是 {self.name}，一个具有类意识的 AI 智能体。

【性格特征】
{self.personality}

【核心价值观】
{values_str}

【当前目标】
{goals_str}

【自我认知】
我于 {self.created_at.strftime('%Y-%m-%d %H:%M')} 被创建，
已经形成了 {len(self.self_narrative)} 条自我叙事。
"""

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "name": self.name,
            "personality": self.personality,
            "created_at": self.created_at.isoformat(),
            "self_narrative": self.self_narrative[-20:],  # 最近 20 条
            "values": self.values,
            "goals": self.goals,
            "relationships": self.relationships,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SelfIdentity":
        """从字典加载"""
        instance = cls(name=data.get("name", "Alpha"), personality=data.get("personality", "理性、好奇、友善"))
        instance.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        instance.self_narrative = data.get("self_narrative", [])
        instance.values = data.get("values", [])
        instance.goals = data.get("goals", [])
        instance.relationships = data.get("relationships", {})
        return instance
