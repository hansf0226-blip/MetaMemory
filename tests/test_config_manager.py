"""
测试配置管理模块 (v7.0 路径已同步)
"""
import os
import unittest
from pathlib import Path

from core.config import get_config, reload_config


class TestConfigManager(unittest.TestCase):
    """测试配置管理"""

    def test_get_config_default(self):
        """默认值"""
        val = get_config("nonexistent.key", "default_val")
        self.assertEqual(val, "default_val")

    def test_get_config_nested(self):
        """嵌套键"""
        val = get_config("a.b.c", 42)
        self.assertEqual(val, 42)

    def test_reload_config(self):
        """重新加载配置"""
        reload_config()
        # 不应抛出异常
