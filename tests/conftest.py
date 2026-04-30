"""
测试配置和共享 Fixture
"""
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def temp_dir():
    """会话级临时目录"""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def temp_db_path(temp_dir):
    """临时数据库文件路径"""
    return temp_dir / "test_memory.db"


@pytest.fixture
def temp_vector_db_path(temp_dir):
    """临时向量数据库文件路径"""
    return temp_dir / "test_vector.db"


@pytest.fixture
def sample_memory_data():
    """示例记忆数据"""
    return {
        "content": "这是一条测试记忆内容，关于 AI 和机器学习的话题",
        "agent_id": "test_agent_001",
        "session_id": "test_session_001",
        "importance": 0.75,
        "metadata": {
            "source": "test",
            "category": "conversation",
            "tags": ["AI", "机器学习", "测试"]
        }
    }


@pytest.fixture
def sample_yao_encoding():
    """示例易经编码数据"""
    return {
        "text": "天行健，君子以自强不息",
        "hexagram_line": 1,
        "element": "fire"
    }


@pytest.fixture
def test_config():
    """测试配置"""
    return {
        "log_level": "DEBUG",
        "database": {
            "type": "sqlite",
            "echo": False
        },
        "memory": {
            "decay_rate": 0.01,
            "max_memories": 10000
        }
    }
