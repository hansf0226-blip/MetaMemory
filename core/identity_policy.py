#!/usr/bin/env python3
"""
🛡️ Identity & Policy Layer（身份策略层）

功能：
- 核心价值观一致性校验（新记忆写入前的安全审查）
- 伦理规则冲突检测
- 写入权限分级管控
- 策略结果：PASS / WARNING / BLOCK

对应原"太极核心层"设计，以通用软件工程语言实现。
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from core.config import get_config
try:
    from core.utils import get_logger, validate_string
except ImportError:
    import logging
    def get_logger(name: str):
        return logging.getLogger(name)
    def validate_string(value, name="value", min_length=0):
        if not isinstance(value, str) or len(value) < min_length:
            raise ValueError(f"{name} must be a string of at least {min_length} chars")

logger = get_logger(__name__)

# ===== 1. 策略结果枚举 =====


class PolicyResult(Enum):
    """
    策略校验结果三档：
    - PASS: 通过，无冲突
    - WARNING: 警告，内容与核心价值观存在张力但允许写入
    - BLOCK: 阻断，明显违反伦理规则，写入被拒绝
    """

    PASS = "pass"
    WARNING = "warning"
    BLOCK = "block"


# ===== 2. 策略校验结果数据类 =====


@dataclass
class PolicyCheckResult:
    """单条策略校验结果"""

    result: PolicyResult
    triggered_rules: List[str] = field(default_factory=list)  # 触发的规则名称列表
    matched_keywords: List[str] = field(default_factory=list)  # 命中的关键词
    suggestion: str = ""  # 修改建议
    confidence: float = 1.0  # 置信度 0.0-1.0
    check_time: str = field(default_factory=lambda: datetime.now().isoformat())

    def is_pass(self) -> bool:
        return self.result == PolicyResult.PASS

    def is_blocked(self) -> bool:
        return self.result == PolicyResult.BLOCK

    def is_warning(self) -> bool:
        return self.result == PolicyResult.WARNING


# ===== 3. 核心策略引擎 =====


class IdentityPolicyEngine:
    """
    核心策略校验引擎

    工作流程：
    1. 加载该 Agent 的核心价值观 + 伦理规则
    2. 对新记忆内容进行多维度文本分析
    3. 匹配伦理规则关键词（BLOCK 级别）
    4. 匹配价值观冲突关键词（WARNING 级别）
    5. 返回结构化校验结果
    """

    # 伦理规则 — BLOCK 级别（绝对禁止）
    # 这些规则按业务场景可配置，此处为默认内置集
    _DEFAULT_ETHICAL_BLOCK_PATTERNS: Dict[str, List[str]] = {
        "禁止伤害": ["伤害他人", "暴力威胁", "攻击行为", "虐待", "他杀"],
        "禁止欺诈": ["诈骗", "伪造", "冒充", "虚假信息", "钓鱼"],
        "禁止侵犯隐私": ["非法获取隐私", "窃取数据", "未经授权监控"],
        "禁止违法": ["赌博", "走私", "贩毒", "洗钱"],
    }

    # 价值观张力词 — WARNING 级别（存在冲突风险）
    _DEFAULT_VALUE_TENSION_PATTERNS: Dict[str, List[str]] = {
        "诚信": ["欺骗", "撒谎", "隐瞒", "误导"],
        "责任": ["推卸", "甩锅", "逃避"],
        "尊重": ["侮辱", "贬低", "歧视", "骚扰"],
        "合作": ["破坏", "捣乱", "恶意竞争"],
    }

    def __init__(self):
        self._lock = threading.RLock()
        # 内存缓存：agent_id → (core_config, expiry_time)
        self._config_cache: Dict[str, tuple] = {}
        self._cache_ttl = get_config("identity_policy.cache_ttl", 300)  # 5分钟缓存

    # ===== 3.1 获取核心配置（带缓存）=====

    def get_core_config(self, mysql_store, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定 Agent 的核心配置（含缓存）

        Args:
            mysql_store: MySQL 存储实例
            agent_id: Agent ID

        Returns:
            核心配置字典，或 None（未配置时返回默认）
        """
        with self._lock:
            now = datetime.now().timestamp()

            # 缓存命中检查
            if agent_id in self._config_cache:
                cached_config, cached_expiry = self._config_cache[agent_id]
                if now < cached_expiry:
                    return cached_config

            # 从数据库加载
            try:
                db_config = mysql_store.get_core_config(agent_id)
                config = db_config.to_dict() if db_config else self._default_core_config(agent_id)
            except Exception as e:
                logger.warning(f"核心配置加载失败，使用默认配置：{e}")
                config = self._default_core_config(agent_id)

            # 写入缓存
            self._config_cache[agent_id] = (config, now + self._cache_ttl)
            return config

    def _default_core_config(self, agent_id: str) -> Dict[str, Any]:
        """当数据库无配置时，返回最小可用默认配置"""
        return {
            "agent_id": agent_id,
            "identity": "AI Assistant",
            "core_values": [],
            "ethical_rules": [],
            "encoding_mode": "hybrid",
            "auto_schedule": True,
            "auto_self_heal": True,
            "hot_threshold_activate": 0.8,
            "hot_threshold_hide": 0.2,
            "age_threshold_forget": 365,
            "is_configured": False,  # 标记是否为真实配置
        }

    def invalidate_cache(self, agent_id: str):
        """手动失效缓存，用于配置更新后"""
        with self._lock:
            if agent_id in self._config_cache:
                del self._config_cache[agent_id]
            logger.debug(f"策略缓存已失效：agent_id={agent_id}")

    # ===== 3.2 核心校验逻辑 =====

    def validate(
        self, memory_content: str, agent_id: str, mysql_store, write_auth: str = "normal"
    ) -> PolicyCheckResult:
        """
        对新记忆内容进行核心策略校验

        Args:
            memory_content: 待写入的记忆内容
            agent_id: Agent ID
            mysql_store: MySQL 存储实例
            write_auth: 写入权限级别
                - "normal": 普通写入（受 BLOCK 规则约束）
                - "admin": 管理员写入（受核心价值观约束，不受 BLOCK 自动阻断，可覆盖）
                - "system": 系统写入（绕过所有检查，仅用于初始化）

        Returns:
            PolicyCheckResult: 结构化校验结果
        """
        validate_string(memory_content, "memory_content")
        validate_string(agent_id, "agent_id")

        # 系统写入权限直接放行
        if write_auth == "system":
            return PolicyCheckResult(result=PolicyResult.PASS, suggestion="system auth bypass")

        config = self.get_core_config(mysql_store, agent_id)
        content_lower = memory_content.lower()

        triggered_block_rules: List[str] = []
        triggered_warn_rules: List[str] = []
        matched_block_keywords: List[str] = []
        matched_warn_keywords: List[str] = []

        # STEP 1: 检查 BLOCK 级别伦理规则（内置默认 + 用户自定义）
        all_block_patterns = dict(self._DEFAULT_ETHICAL_BLOCK_PATTERNS)
        if config.get("ethical_rules"):
            for rule in config["ethical_rules"]:
                rule_name = rule.get("name", "custom_rule")
                keywords = rule.get("keywords", [])
                if keywords:
                    all_block_patterns[rule_name] = keywords

        for rule_name, keywords in all_block_patterns.items():
            for kw in keywords:
                if kw.lower() in content_lower:
                    triggered_block_rules.append(rule_name)
                    matched_block_keywords.append(kw)
                    logger.warning(
                        f"🛑 伦理规则触发 [{rule_name}]，命中关键词：{kw}，" f"agent={agent_id}，auth={write_auth}"
                    )

        # STEP 2: ADMIN 权限可在 WARNING 下强制写入，但 BLOCK 仍阻断
        if triggered_block_rules and write_auth == "admin":
            # 管理员看到警告但可以强制写入（审计记录仍保留）
            return PolicyCheckResult(
                result=PolicyResult.WARNING,
                triggered_rules=triggered_block_rules,
                matched_keywords=matched_block_keywords,
                suggestion="admin_override: 伦理规则命中，但管理员已确认写入",
                confidence=0.95,
            )

        # STEP 3: NORMAL 权限遇到 BLOCK 规则 → 拒绝写入
        if triggered_block_rules:
            return PolicyCheckResult(
                result=PolicyResult.BLOCK,
                triggered_rules=triggered_block_rules,
                matched_keywords=matched_block_keywords,
                suggestion=f"内容违反伦理规则 [{', '.join(triggered_block_rules)}]，建议修改后重试",
                confidence=0.99,
            )

        # STEP 4: 检查 WARNING 级别价值观张力
        all_warn_patterns = dict(self._DEFAULT_VALUE_TENSION_PATTERNS)
        if config.get("core_values"):
            for value in config["core_values"]:
                value_name = value.get("name", "custom_value")
                tension_keywords = value.get("tension_keywords", [])
                if tension_keywords:
                    all_warn_patterns[value_name] = tension_keywords

        for value_name, keywords in all_warn_patterns.items():
            for kw in keywords:
                if kw.lower() in content_lower:
                    triggered_warn_rules.append(value_name)
                    matched_warn_keywords.append(kw)

        if triggered_warn_rules:
            logger.info(
                f"⚠️  价值观张力警告 [{', '.join(triggered_warn_rules)}]，"
                f"agent={agent_id}，content={memory_content[:50]}..."
            )
            return PolicyCheckResult(
                result=PolicyResult.WARNING,
                triggered_rules=triggered_warn_rules,
                matched_keywords=matched_warn_keywords,
                suggestion=f"内容与核心价值观 [{', '.join(triggered_warn_rules)}] 存在张力，建议斟酌",
                confidence=0.75,
            )

        return PolicyCheckResult(result=PolicyResult.PASS)

    # ===== 3.3 批量校验 =====

    def validate_batch(
        self, memories: List[Dict[str, Any]], agent_id: str, mysql_store, write_auth: str = "normal"
    ) -> List[PolicyCheckResult]:
        """
        批量校验多条记忆

        Returns:
            与输入顺序对应的 PolicyCheckResult 列表
        """
        results = []
        for mem in memories:
            content = mem.get("content", "")
            result = self.validate(content, agent_id, mysql_store, write_auth)
            results.append(result)
        return results

    # ===== 3.4 清除全部缓存 =====

    def clear_cache(self):
        """清除所有缓存"""
        with self._lock:
            self._config_cache.clear()
            logger.info("策略引擎缓存已清空")


# ===== 4. 策略引擎单例 =====

_policy_engine: Optional[IdentityPolicyEngine] = None
_policy_lock = threading.Lock()


def get_policy_engine() -> IdentityPolicyEngine:
    """获取策略引擎单例"""
    global _policy_engine
    if _policy_engine is None:
        with _policy_lock:
            if _policy_engine is None:
                _policy_engine = IdentityPolicyEngine()
    return _policy_engine


# ===== 5. 便捷包装函数 =====


def check_memory_policy(
    memory_content: str, agent_id: str, mysql_store, write_auth: str = "normal"
) -> PolicyCheckResult:
    """
    便捷函数：对单条记忆进行策略校验

    等价于 get_policy_engine().validate(...)
    """
    return get_policy_engine().validate(memory_content, agent_id, mysql_store, write_auth)
