"""
主动记忆引擎 — LLM 驱动的自动记忆提取与上下文加载

流程:
  对话文本 → LLM 提取关键信息 → 自动存储 → 返回上下文
"""
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── 记忆提取提示词 ───

EXTRACTION_PROMPT = """你是一个记忆提取助手。从以下对话中提取需要长期记住的关键信息。

规则:
1. 只提取事实、偏好、决定、承诺——忽略寒暄和废话
2. 每条记忆用一句话概括，不超过50字
3. 给每条记忆标注类型: preference(偏好), fact(事实), decision(决定), plan(计划), skill(技能)
4. 只输出JSON数组，不要任何解释

对话内容:
{conversation}

输出格式（严格JSON数组）:
[
  {{"content": "用户喜欢川菜，尤其是麻婆豆腐", "type": "preference", "importance": 0.9}},
  {{"content": "用户计划下周五去成都出差", "type": "plan", "importance": 0.7}}
]

JSON:"""


class AutoMemoryEngine:
    """主动记忆引擎"""

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.enabled = self.config.get("auto_memory", {}).get("enabled", True)

    def _load_config(self, config_path: str = None) -> dict:
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    def _get_llm_config(self) -> dict:
        return self.config.get("auto_memory", {}).get("llm", {})

    def _get_context_config(self) -> dict:
        return self.config.get("auto_memory", {}).get("context", {})

    # ─── LLM 调用 ───

    def _call_llm(self, prompt: str) -> Optional[str]:
        """调用 LLM（OpenAI 兼容 API）"""
        llm = self._get_llm_config()
        api_key = llm.get("api_key", "")
        base_url = llm.get("base_url", "https://api.siliconflow.cn/v1")
        model = llm.get("model", "Qwen/Qwen3-8B")
        temperature = llm.get("temperature", 0.3)
        max_tokens = llm.get("max_tokens", 1024)

        # 解析 ${VAR} 占位符
        if api_key.startswith("${") and api_key.endswith("}"):
            import os
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, "")

        if not api_key:
            logger.warning("未配置 LLM API Key，使用规则提取")
            return None

        try:
            import requests
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return None

    # ─── 记忆提取 ───

    def extract_memories(self, conversation: str, agent_id: str = "default") -> List[dict]:
        """
        从对话中提取关键记忆

        Args:
            conversation: 对话文本（可含多轮）
            agent_id: 租户 ID

        Returns:
            提取的记忆列表 [{"content": "...", "type": "...", "importance": 0.8}, ...]
        """
        if not self.enabled:
            return []

        prompt = EXTRACTION_PROMPT.format(conversation=conversation[-4000:])
        response = self._call_llm(prompt)

        if not response:
            # LLM 不可用，回退到简单规则
            return self._rule_extract(conversation)

        # 解析 JSON
        try:
            text = response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            memories = json.loads(text)
            if isinstance(memories, list):
                return memories
        except json.JSONDecodeError:
            logger.warning(f"LLM 输出非 JSON，回退规则提取: {response[:100]}...")

        return self._rule_extract(conversation)

    def _rule_extract(self, conversation: str) -> List[dict]:
        """规则降级：按句号/换行拆分 >3 字的句子"""
        import re
        sentences = re.split(r'[。！？\n]', conversation)
        memories = []
        for s in sentences:
            s = s.strip()
            if len(s) > 6 and len(s) < 100:
                memories.append({"content": s, "type": "fact", "importance": 0.3})
        return memories[:5]

    # ─── 自动存储 ───

    def auto_store(self, conversation: str, agent_id: str = "default") -> Dict[str, Any]:
        """
        自动提取并存储记忆

        Returns:
            {"extracted": [...], "stored": 5, "memories": [...]}
        """
        from core.storage_manager import get_storage_manager
        sm = get_storage_manager()

        extracted = self.extract_memories(conversation, agent_id)
        stored = []

        for mem in extracted:
            try:
                mid = f"auto_{agent_id}_{uuid.uuid4().hex[:8]}"
                sm.create_memory({
                    "agent_id": agent_id,
                    "memory_id": mid,
                    "content": mem.get("content", ""),
                    "bagua_type": self._type_to_bagua(mem.get("type", "fact")),
                    "wuxing": self._type_to_wuxing(mem.get("type", "fact")),
                    "hot_score": mem.get("importance", 0.5),
                })
                stored.append(mid)
            except Exception as e:
                logger.warning(f"存储失败 {mem.get('content','')[:30]}: {e}")

        logger.info(f"自动存储: 提取 {len(extracted)} 条，成功 {len(stored)} 条")
        return {
            "extracted": extracted,
            "stored": len(stored),
            "memory_ids": stored,
        }

    # ─── 上下文加载 ───

    def load_context(self, query: str, agent_id: str = "default") -> Dict[str, Any]:
        """
        根据当前查询加载相关历史记忆

        Returns:
            {"memories": [...], "context_text": "..."} 可直接注入 system prompt
        """
        ctx_cfg = self._get_context_config()
        top_k = ctx_cfg.get("top_k", 10)

        from core.storage_manager import get_storage_manager
        sm = get_storage_manager()

        results = sm.search_memories(agent_id=agent_id, keyword=query, limit=top_k)

        memories = []
        lines = []
        for r in results:
            sim = r.get("similarity", 0)
            if sim < 0.3:
                continue
            content = r.get("content", "")
            memories.append({
                "memory_id": r.get("memory_id"),
                "content": content,
                "similarity": round(sim, 3),
            })
            lines.append(f"- [{sim:.0%}] {content}")

        context_text = "相关历史记忆:\n" + "\n".join(lines) if lines else ""

        logger.info(f"上下文加载: 查询='{query[:30]}...', 命中 {len(memories)} 条")
        return {
            "memories": memories,
            "context_text": context_text,
        }

    # ─── 一站式：对话 → 提取+存储+返回上下文 ───

    def process_conversation(
        self, conversation: str, agent_id: str = "default"
    ) -> Dict[str, Any]:
        """
        一站式处理对话：提取记忆 → 自动存储 → 加载上下文

        这是外部调用的主入口
        """
        # 1. 自动提取+存储
        store_result = self.auto_store(conversation, agent_id)

        # 2. 加载上下文（用对话最后一句作为查询）
        query = conversation.split("\n")[-1][-200:] if conversation else ""
        context = self.load_context(query, agent_id)

        return {
            **store_result,
            "context": context,
        }

    # ─── 类型映射 ───

    @staticmethod
    def _type_to_bagua(mem_type: str) -> str:
        mapping = {
            "preference": "离", "fact": "乾", "decision": "震",
            "plan": "巽", "skill": "坎", "knowledge": "艮",
        }
        return mapping.get(mem_type, "坤")

    @staticmethod
    def _type_to_wuxing(mem_type: str) -> str:
        mapping = {
            "preference": "火", "fact": "金", "decision": "木",
            "plan": "水", "skill": "土", "knowledge": "土",
        }
        return mapping.get(mem_type, "土")


# 单例
_engine: Optional[AutoMemoryEngine] = None


def get_auto_memory() -> AutoMemoryEngine:
    global _engine
    if _engine is None:
        _engine = AutoMemoryEngine()
    return _engine
