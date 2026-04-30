"""
学习引擎 - 管理经验学习和能力提升

功能：
- 记录和分析经历
- 提取成功和失败模式
- 能力指标管理
- 记忆提炼
- 决策反思
"""

import os
import sys
from datetime import datetime
from typing import Dict, List

# 确保正确的导入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from api.llm_manager import LLMManager
except ImportError:
    LLMManager = None


class LearningEngine:
    """学习引擎 - 管理经验学习和能力提升"""

    def __init__(self, aggressive_mode: bool = True, llm_manager=None):
        self.aggressive_mode = aggressive_mode  # 激进模式

        # 经验存储
        self.experiences: List[Dict] = []
        self.success_patterns: List[Dict] = []
        self.failure_patterns: List[Dict] = []

        # 能力指标（0-100 分制）
        self.abilities: Dict[str, float] = {
            "decision_quality": 50.0,  # 决策质量
            "learning_speed": 50.0,  # 学习速度
            "adaptation_rate": 50.0,  # 适应率
            "memory_efficiency": 50.0,  # 记忆效率
            "thinking_depth": 50.0,  # 思考深度
            "emotional_intelligence": 50.0,  # 情绪智力
            "creativity": 50.0,  # 创造力
        }

        # 能力历史（用于趋势分析）
        self.ability_history: Dict[str, List[float]] = {k: [v] for k, v in self.abilities.items()}

        # 学习统计
        self.total_learnings = 0
        self.total_reflections = 0
        self.deep_analyses = 0
        self.last_evolution_time = None

        # LLM 集成 - 使用传入的实例，或新建
        if llm_manager is not None:
            self.llm = llm_manager
        else:
            self.llm = LLMManager()

        # 外部系统引用
        self.memory_ref = None  # 记忆系统引用
        self.identity_ref = None  # 身份系统引用
        self.mind_state_ref = None  # 思维状态引用

        # 配置
        self.config = {
            "auto_consolidate": True,  # 自动记忆提炼
            "deep_analysis_threshold": 0,  # 深度分析阈值
            "evolution_frequency": 1,  # 进化频率
            "ability_boost_rate": 0.15,  # 能力提升率
        }

        # 初始化信息
        print("📚 学习引擎已初始化")
        print(f"   能力维度：{len(self.abilities)} 个")
        print(f"   自动记忆提炼：{'启用' if self.config['auto_consolidate'] else '禁用'}")
        print(f"   深度分析：{'每次经历' if self.aggressive_mode else '按需'}")
        print(f"   进化频率：每{self.config['evolution_frequency']}个周期")

    # ========== 经验学习 ==========

    def record_experience(
        self, event_type: str, context: Dict, outcome: str, reward: float = 0.0, metadata: Dict = None
    ):
        """记录经历并分析"""
        experience = {
            "type": event_type,
            "context": context,
            "outcome": outcome,
            "reward": reward,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "processed": False,
            "analyzed": False,
        }

        self.experiences.append(experience)
        self.total_learnings += 1

        # 分析经验
        if reward > 0.3:
            self._extract_success_pattern(experience)
        elif reward < -0.3:
            self._extract_failure_pattern(experience)

        # 深度分析
        if self.aggressive_mode:
            self._deep_analyze_experience(experience)

        # 实时更新能力
        self._update_abilities_realtime(experience)

        # 输出信息
        emoji = "🎯" if reward > 0 else "⚠️" if reward < 0 else "📝"
        print(f"{emoji} 学习：{event_type} (奖励：{reward:+.2f})")

    def _extract_success_pattern(self, experience: Dict):
        """从成功经历中提取模式"""
        pattern = {
            "type": experience["type"],
            "context_summary": str(experience["context"])[:100],
            "outcome": experience["outcome"],
            "reward": experience["reward"],
            "timestamp": datetime.now().isoformat(),
            "frequency": 1,
            "key_factors": self._identify_key_factors(experience),
        }

        # 合并相似模式
        for p in self.success_patterns:
            if p["type"] == pattern["type"] and p["outcome"] == pattern["outcome"]:
                p["frequency"] += 1
                return

        self.success_patterns.append(pattern)
        print(f"   ✨ 成功模式：{experience['type']} (关键因素：{', '.join(pattern['key_factors'])})")

    def _extract_failure_pattern(self, experience: Dict):
        """从失败经历中提取模式"""
        pattern = {
            "type": experience["type"],
            "context_summary": str(experience["context"])[:100],
            "outcome": experience["outcome"],
            "reward": experience["reward"],
            "timestamp": datetime.now().isoformat(),
            "frequency": 1,
            "root_causes": [],
            "lessons_learned": False,
        }

        for p in self.failure_patterns:
            if p["type"] == pattern["type"] and p["outcome"] == pattern["outcome"]:
                p["frequency"] += 1
                return

        self.failure_patterns.append(pattern)
        print(f"   ⚠️ 失败模式：{experience['type']}")

    def _identify_key_factors(self, experience: Dict) -> List[str]:
        """识别成功的关键因素"""
        factors = []

        reward = experience["reward"]
        if reward > 0.7:
            factors.append("高质量决策")
        if reward > 0.5:
            factors.append("有效执行")

        context = experience.get("context", {})
        if "thought" in context:
            factors.append("深度思考")
        if "memory" in context:
            factors.append("记忆整合")

        return factors if factors else ["未知因素"]

    # ========== 深度分析 ==========

    def _deep_analyze_experience(self, experience: Dict):
        """使用 LLM 进行深度分析"""
        if not self.llm.providers:
            return

        self.deep_analyses += 1

        # 构建分析提示
        prompt = """请分析以下经历，提取关键洞察：

经历类型：{experience['type']}
结果：{experience['outcome']}
奖励值：{experience['reward']} (+1 为成功，-1 为失败)
上下文：{json.dumps(experience['context'], ensure_ascii=False)[:200]}

请回答：
1. 成功/失败的根本原因是什么？
2. 应该强化/改进什么行为？
3. 一条具体的改进建议。

用简洁的 JSON 格式回复：{{"root_cause": "...", "reinforce": "...", "improvement": "..."}}"""

        try:
            messages = [{"role": "user", "content": prompt}]
            analysis = self.llm.chat(messages, temperature=0.3, max_tokens=300)

            if analysis:
                # 存储分析结果
                experience["llm_analysis"] = analysis
                experience["analyzed"] = True

                # 提取教训
                if experience["reward"] < 0:
                    self._extract_lesson(experience, analysis)

                print("   🧠 深度分析完成")
                # 打印分析结果预览
                if analysis:
                    preview = analysis.strip()[:300]
                    if len(analysis) > 300:
                        preview += "..."
                    print(f"   💡 分析结果：\n{preview}")
        except Exception as e:
            print(f"   ⚠️ 分析失败：{e}")

    def _extract_lesson(self, experience: Dict, analysis: str):
        """从分析中提取教训"""
        lesson = {
            "source_experience": experience,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
            "applied": False,
        }

        # 添加到失败模式
        for p in self.failure_patterns:
            if p["type"] == experience["type"]:
                p["lessons_learned"] = True
                p["lesson"] = lesson
                break

    # ========== 能力管理 ==========

    def _update_abilities_realtime(self, experience: Dict):
        """实时更新能力指标"""
        reward = experience["reward"]
        event_type = experience["type"]

        # 决策质量
        if event_type == "decision":
            old = self.abilities["decision_quality"]
            delta = reward * self.config["ability_boost_rate"] * 10
            self.abilities["decision_quality"] = min(100, max(0, old + delta))

        # 学习速度
        self.abilities["learning_speed"] = min(100, self.total_learnings / 5)

        # 适应率
        if len(self.experiences) > 0:
            success_rate = len(self.success_patterns) / len(self.experiences)
            self.abilities["adaptation_rate"] = min(100, success_rate * 150)

        # 思考深度
        if event_type == "thinking" and reward > 0:
            self.abilities["thinking_depth"] = min(100, self.abilities["thinking_depth"] + reward * 2)

        # 情绪智力
        if event_type == "interaction":
            self.abilities["emotional_intelligence"] = min(100, self.abilities["emotional_intelligence"] + reward * 3)

        # 创造力
        if event_type == "thinking" and reward > 0.5:
            self.abilities["creativity"] = min(100, self.abilities["creativity"] + 2)

        # 记录历史
        for key in self.ability_history:
            self.ability_history[key].append(self.abilities[key])

    def update_abilities(self):
        """更新能力指标"""
        # 基于所有经历进行批量更新
        if not self.experiences:
            return

        print("📈 更新能力指标...")

        # 计算平均奖励
        recent = self.experiences[-10:]
        avg_reward = sum(e["reward"] for e in recent) / len(recent)

        # 全局调整
        for ability in self.abilities:
            old = self.abilities[ability]
            delta = avg_reward * self.config["ability_boost_rate"] * 5
            self.abilities[ability] = min(100, max(0, old + delta))

        # 记录历史
        for key in self.ability_history:
            self.ability_history[key].append(self.abilities[key])

        # 打印变化
        for ability, value in self.abilities.items():
            print(f"   {ability}: {value:.1f}")

    # ========== 记忆提炼 ==========

    def consolidate_memories(self):
        """智能记忆提炼"""
        if not self.memory_ref:
            print("⚠️ 记忆系统未连接")
            return

        print("🧠 记忆提炼...")

        # 获取短期记忆
        short_term = self.memory_ref.get_all_short_term(limit=20)

        if len(short_term) < 3:
            print("   短期记忆不足，跳过提炼")
            return

        # 提取主题
        if self.llm.providers:
            themes = self._extract_themes_with_llm(short_term)
        else:
            themes = self._extract_themes_with_keywords(short_term)

        # 创建长期记忆
        for theme in themes:
            self.memory_ref.add_long_term_memory(content=theme["summary"], tags=theme["tags"])
            print(f"   ✅ 长期记忆：{theme['summary'][:40]}...")

        self.total_reflections += 1
        print(f"✅ 记忆提炼完成（{len(themes)} 个主题）")

    def _extract_themes_with_llm(self, memories: List[Dict]) -> List[Dict]:
        """使用 LLM 提取记忆主题"""
        memory_contents = "\n".join([m["content"] for m in memories[:10]])

        prompt = """分析以下记忆内容，提取 2-3 个核心主题：

{memory_contents}

请用 JSON 格式回复：[{{"summary": "...", "tags": ["tag1", "tag2"]}}]"""

        try:
            messages = [{"role": "user", "content": prompt}]
            result = self.llm.chat(messages, temperature=0.3, max_tokens=500)

            if result:
                # 简单解析
                themes = []
                # 实际项目中应该完善 JSON 解析
                themes.append({"summary": "从记忆中提取的核心洞察", "tags": ["学习", "成长"]})
                return themes
        except Exception as e:
            print(f"   ⚠️ LLM 主题提取失败：{e}")

        return self._extract_themes_with_keywords(memories)

    def _extract_themes_with_keywords(self, memories: List[Dict]) -> List[Dict]:
        """关键词提取主题（降级方案）"""
        keywords = {}
        for memory in memories:
            content = memory.get("content", "")
            words = content.replace("的", " ").replace("了", " ").split()
            for word in words:
                if len(word) >= 2:
                    keywords[word] = keywords.get(word, 0) + 1

        themes = []
        for word, count in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:3]:
            if count >= 2:
                themes.append(
                    {
                        "summary": f"关于 {word} 的经验总结",
                        "tags": [word],
                        "importance": min(1.0, count / len(memories)),
                    }
                )

        return themes

    # ========== 决策反思 ==========

    def reflect_on_decision(self, decision: Dict, outcome: str, reward: float):
        """反思决策"""
        print(f"🤔 反思：{decision.get('action', 'unknown')} → {outcome}")

        self.record_experience(event_type="decision", context=decision, outcome=outcome, reward=reward)

        # 低奖励触发深度反思
        if reward < 0:
            self._deep_reflection(decision, outcome, reward)

    def _deep_reflection(self, decision: Dict, outcome: str, reward: float):
        """深度反思"""
        print("   🔍 深度反思中...")

        # 使用 LLM 分析
        if self.llm.providers:
            prompt = """分析这次决策失败的原因：

决策：{json.dumps(decision, ensure_ascii=False)[:200]}
结果：{outcome}
奖励：{reward}

请提供 3 条具体的改进建议。"""

            try:
                messages = [{"role": "user", "content": prompt}]
                analysis = self.llm.chat(messages, temperature=0.3, max_tokens=400)

                if analysis:
                    print(f"   📝 改进建议：{analysis[:100]}...")
            except Exception as e:
                print(f"   ⚠️ 反思失败：{e}")

    # ========== 状态管理 ==========

    def get_status(self) -> Dict:
        """获取状态"""
        return {
            "total_experiences": len(self.experiences),
            "total_learnings": self.total_learnings,
            "total_reflections": self.total_reflections,
            "deep_analyses": self.deep_analyses,
            "success_patterns": len(self.success_patterns),
            "failure_patterns": len(self.failure_patterns),
            "abilities": self.abilities.copy(),
            "last_evolution": self.last_evolution_time.isoformat() if self.last_evolution_time else None,
        }

    def print_status(self):
        """打印状态"""
        status = self.get_status()

        print("\n" + "=" * 60)
        print("【学习引擎状态】")
        print("=" * 60)
        print(f"总经历数：{status['total_experiences']}")
        print(f"总学习次数：{status['total_learnings']}")
        print(f"深度分析：{status['deep_analyses']} 次")
        print(f"成功模式：{status['success_patterns']} 个")
        print(f"失败模式：{status['failure_patterns']} 个")

        print("\n【能力指标】")
        for ability, value in status["abilities"].items():
            bar_len = int(value / 10)
            bar = "█" * bar_len + "░" * (10 - bar_len)

            # 趋势箭头
            history = self.ability_history.get(ability, [value])
            if len(history) >= 2:
                trend = "↑" if history[-1] > history[-2] else "↓" if history[-1] < history[-2] else "→"
            else:
                trend = "→"

            print(f"  {ability:20} [{bar}] {value:5.1f} {trend}")

        print("=" * 60)

    # ========== 外部系统连接 ==========

    def set_memory_ref(self, memory):
        """设置记忆系统引用"""
        self.memory_ref = memory
        print("✅ 学习引擎已连接记忆系统")

    def set_identity_ref(self, identity):
        """设置身份系统引用"""
        self.identity_ref = identity
        print("✅ 学习引擎已连接身份系统")

    def set_mind_state_ref(self, mind_state):
        """设置思维状态引用"""
        self.mind_state_ref = mind_state
        print("✅ 学习引擎已连接思维状态系统")
