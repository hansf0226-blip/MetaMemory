"""
测试异常处理模块 (API 同步 v7.0)
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils import (
    MetaMemoryError,
    DatabaseError,
    ValidationError,
)


class TestMetaMemoryError:
    """测试基础异常"""

    def test_base_error(self):
        """基础异常"""
        with pytest.raises(MetaMemoryError):
            raise MetaMemoryError("test error")

    def test_base_error_message(self):
        """异常消息"""
        try:
            raise MetaMemoryError("custom message")
        except MetaMemoryError as e:
            assert str(e) == "custom message"


class TestDatabaseError:
    """测试数据库异常"""

    def test_database_error_inherits(self):
        """继承关系"""
        assert issubclass(DatabaseError, MetaMemoryError)

    def test_database_error_raise(self):
        """抛出数据库异常"""
        with pytest.raises(DatabaseError):
            raise DatabaseError("db connection failed")


class TestValidationError:
    """测试验证异常"""

    def test_validation_error_inherits(self):
        """继承关系"""
        assert issubclass(ValidationError, MetaMemoryError)

    def test_validation_error_raise(self):
        """抛出验证异常"""
        with pytest.raises(ValidationError):
            raise ValidationError("invalid field")
