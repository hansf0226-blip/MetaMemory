"""
测试身份策略引擎 (API 同步 v7.0)
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.identity_policy import (
    IdentityPolicyEngine,
    PolicyCheckResult,
    PolicyResult,
    check_memory_policy,
    get_policy_engine,
)


# ─── PolicyResult 枚举 ────────────────────────────────────────

class TestPolicyResult:
    """测试策略结果枚举"""

    def test_policy_result_values(self):
        """策略结果值"""
        assert PolicyResult.PASS.value == "pass"
        assert PolicyResult.BLOCK.value == "block"
        assert PolicyResult.WARNING.value == "warning"

    def test_policy_result_members(self):
        """策略结果成员"""
        members = {m.name for m in PolicyResult}
        assert "PASS" in members
        assert "BLOCK" in members
        assert "WARNING" in members


# ─── PolicyCheckResult ────────────────────────────────────────

class TestPolicyCheckResult:
    """测试策略检查结果"""

    def test_pass_result(self):
        """通过结果"""
        r = PolicyCheckResult(result=PolicyResult.PASS)
        assert r.is_pass() is True
        assert r.is_blocked() is False
        assert r.is_warning() is False

    def test_block_result(self):
        """阻止结果"""
        r = PolicyCheckResult(result=PolicyResult.BLOCK)
        assert r.is_blocked() is True
        assert r.is_pass() is False

    def test_warning_result(self):
        """警告结果"""
        r = PolicyCheckResult(result=PolicyResult.WARNING)
        assert r.is_warning() is True
        assert r.is_blocked() is False


# ─── IdentityPolicyEngine ─────────────────────────────────────

class TestIdentityPolicyEngine:
    """测试身份策略引擎"""

    def test_engine_initialization(self):
        """引擎初始化"""
        engine = IdentityPolicyEngine()
        assert engine is not None

    def test_singleton(self):
        """单例模式"""
        e1 = get_policy_engine()
        e2 = get_policy_engine()
        assert e1 is e2

    def test_validate_returns_result(self):
        """验证返回 PolicyCheckResult"""
        engine = IdentityPolicyEngine()
        result = engine.validate(
            memory_content="test content",
            agent_id="test-agent",
            mysql_store=None,
        )
        assert isinstance(result, PolicyCheckResult)

    def test_cache_clear(self):
        """清空缓存"""
        engine = IdentityPolicyEngine()
        engine.clear_cache()

    def test_default_config(self):
        """默认配置"""
        engine = IdentityPolicyEngine()
        config = engine._default_core_config("test-agent")
        assert isinstance(config, dict)


# ─── check_memory_policy 便捷函数 ─────────────────────────────

class TestCheckMemoryPolicy:
    """测试便捷策略检查函数"""

    def test_basic_policy_check(self):
        """基本策略检查"""
        result = check_memory_policy(
            memory_content="普通内容",
            agent_id="test",
            mysql_store=None,
        )
        assert isinstance(result, PolicyCheckResult)


# ─── 已移除功能（保留占位） ───────────────────────────────────

@pytest.mark.skip(reason="v7.0: MemoryAccessPolicy 已合并到 IdentityPolicyEngine")
class TestMemoryAccessPolicy:
    def test_policy_creation(self): pass
    def test_default_policy(self): pass
    def test_check_basic_access(self): pass
    def test_role_denied(self): pass

@pytest.mark.skip(reason="v7.0: PrivacyLevel 枚举已移除")
class TestPrivacyLevel:
    def test_privacy_level_values(self): pass
    def test_privacy_order(self): pass

@pytest.mark.skip(reason="v7.0: classify_privacy_level 已移除")
class TestPrivacyClassification:
    def test_classify_public(self): pass
    def test_classify_internal(self): pass
    def test_classify_confidential(self): pass
    def test_classify_secret(self): pass

@pytest.mark.skip(reason="v7.0: redact_sensitive_info 已移除")
class TestSensitiveInfoRedaction:
    def test_redact_phone_number(self): pass
    def test_redact_email(self): pass
    def test_redact_id_card(self): pass
    def test_redact_bank_card(self): pass
    def test_no_sensitive_info(self): pass
    def test_mixed_content(self): pass

@pytest.mark.skip(reason="v7.0: 边缘情况待新测试覆盖")
class TestEdgeCases:
    def test_empty_text_classification(self): pass
    def test_empty_text_redaction(self): pass
    def test_none_memory_access(self): pass
    def test_unknown_privacy_level(self): pass
