"""
测试公共工具模块 (v7.0 路径已同步)
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils import (
    ValidationError,
    ensure_directory,
    generate_id,
    get_logger,
    validate_dict,
    validate_float,
    validate_integer,
    validate_list,
    validate_required_fields,
    validate_string,
)


class TestGenerateId:
    def test_generates_string(self):
        result = generate_id()
        assert isinstance(result, str)
        assert len(result) == 12

    def test_with_seed(self):
        r1 = generate_id("hello")
        r2 = generate_id("hello")
        assert r1 == r2  # deterministic

    def test_without_seed_is_unique(self):
        r1 = generate_id()
        r2 = generate_id()
        assert r1 != r2


class TestValidation:
    def test_validate_dict(self):
        validate_dict({"a": 1})
        with pytest.raises(ValidationError):
            validate_dict("not a dict")

    def test_validate_string(self):
        validate_string("hello")
        with pytest.raises(ValidationError):
            validate_string(123)

    def test_validate_float(self):
        validate_float(3.14)
        with pytest.raises(ValidationError):
            validate_float("abc")

    def test_validate_integer(self):
        validate_integer(42)
        with pytest.raises(ValidationError):
            validate_integer(3.14)

    def test_validate_list(self):
        validate_list([1, 2])
        with pytest.raises(ValidationError):
            validate_list("abc")

    def test_validate_required_fields(self):
        validate_required_fields({"a": 1, "b": 2}, ["a"])
        with pytest.raises(ValidationError):
            validate_required_fields({"a": 1}, ["a", "b"])


class TestEnsureDirectory:
    def test_creates_dir(self):
        import tempfile, os
        d = tempfile.mkdtemp()
        test_dir = os.path.join(d, "test_subdir")
        ensure_directory(test_dir)
        assert os.path.isdir(test_dir)


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test")
        import logging
        assert isinstance(logger, logging.Logger)
