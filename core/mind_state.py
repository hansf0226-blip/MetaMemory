"""
思维状态模块 - 管理 Agent 的情绪和思维状态
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class Emotion:
    """情绪状态"""

    name: str
    intensity: float  # 0.0 - 1.0
    trigger: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    decay_rate: float = 0.1  # 每小时衰减

    def decay(self, hours: float = 1.0) -> float:
        """情绪衰减"""
        self.intensity *= 1 - self.decay_rate * hours
        return self.intensity


class MindState:
    """Agent 的思维状态系统"""

    def __init__(self, sensitivity: float = 0.7):
        self.sensitivity = sensitivity  # 情绪敏感度
        self.current_emotions: List[Emotion] = []
        self.mood_level: float = 0.5  # 基础心情 0-1
        self.energy_level: float = 1.0  # 能量水平 0-1
        self.focus_level: float = 1.0  # 专注度 0-1
        self.thought_history: List[Dict] = []
        self.last_update = datetime.now()

    def add_emotion(self, name: str, intensity: float, trigger: str = ""):
        """添加情绪"""
        # 输入验证
        if not name or not isinstance(name, str):
            raise ValueError("Emotion name must be a non-empty string")
        if not 0.0 <= intensity <= 1.0:
            raise ValueError("Intensity must be between 0.0 and 1.0")

        emotion = Emotion(name=name, intensity=min(1.0, max(0.0, intensity * self.sensitivity)), trigger=trigger)
        self.current_emotions.append(emotion)

        # 更新心情水平
        self._update_mood()

        # 保持最近的情绪
        if len(self.current_emotions) > 20:
            self.current_emotions = self.current_emotions[-20:]

    def _update_mood(self):
        """更新心情水平"""
        # 衰减旧情绪（使用副本避免遍历删除问题）
        now = datetime.now()
        for emotion in self.current_emotions.copy():
            hours = (now - emotion.created_at).total_seconds() / 3600
            if emotion.decay(hours) < 0.1:
                self.current_emotions.remove(emotion)

        # 计算综合心情
        positive = sum(e.intensity for e in self.current_emotions if e.name in ["开心", "兴奋", "满足", "好奇"])
        negative = sum(e.intensity for e in self.current_emotions if e.name in ["沮丧", "焦虑", "困惑", "失望"])

        self.mood_level = 0.5 + (positive - negative) * 0.3
        self.mood_level = max(0.0, min(1.0, self.mood_level))

    def record_thought(self, thought: str, category: str = "general"):
        """记录思考"""
        self.thought_history.append(
            {"thought": thought, "category": category, "timestamp": datetime.now().isoformat(), "mood": self.mood_level}
        )

        # 保持最近的思考
        if len(self.thought_history) > 100:
            self.thought_history = self.thought_history[-100:]

    def get_state_summary(self) -> str:
        """获取状态摘要"""
        emotion_str = ", ".join(f"{e.name}({e.intensity:.2f})" for e in self.current_emotions[-5:])

        return """
【当前状态】
心情：{self.mood_level:.2f}  {'😊' if self.mood_level > 0.6 else '😐' if self.mood_level > 0.4 else '😔'}
能量：{self.energy_level:.2f}  {'⚡' if self.energy_level > 0.7 else '🔋' if self.energy_level > 0.4 else '🪫'}
专注：{self.focus_level:.2f}

【当前情绪】
{emotion_str if emotion_str else '平静'}

【最近思考】
{chr(10).join(f"• {t['thought'][:50]}..." for t in self.thought_history[-3:]) if self.thought_history else '无'}
"""

    def to_dict(self) -> Dict:
        """序列化"""
        return {
            "mood_level": self.mood_level,
            "energy_level": self.energy_level,
            "focus_level": self.focus_level,
            "current_emotions": [
                {"name": e.name, "intensity": e.intensity, "trigger": e.trigger} for e in self.current_emotions[-10:]
            ],
            "thought_history": self.thought_history[-20:],
        }
