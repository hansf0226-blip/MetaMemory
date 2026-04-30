#!/usr/bin/env python3
"""
测试内存数据库模块
"""

import os
import tempfile
import unittest

from core.memory_db import memory_db


class TestMemoryDB(unittest.TestCase):
    """
    测试内存数据库模块
    """

    def setUp(self):
        """
        设置测试环境
        """
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        # 初始化内存数据库
        self.memory_db = memory_db
        self.memory_db.initialize(db_path=os.path.join(self.temp_dir, "test.db"))

    def tearDown(self):
        """
        清理测试环境
        """
        # 清理临时目录
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_create_memory(self):
        """
        测试创建记忆
        """
        # 创建记忆
        memory_data = {
            "agent_id": "test_agent",
            "content": "测试内容",
            "hexagram": "乾卦",
            "memory_type": "core",
            "hot": 0.5,
            "weight": 0.8,
            "priority": 1,
        }
        memory_id = self.memory_db.create_memory(memory_data)
        self.assertIsInstance(memory_id, int)

    def test_create_memories_batch(self):
        """
        测试批量创建记忆
        """
        # 批量创建记忆
        memories_data = [
            {
                "agent_id": "test_agent",
                "content": "测试内容1",
                "hexagram": "乾卦",
                "memory_type": "core",
                "hot": 0.5,
                "weight": 0.8,
                "priority": 1,
            },
            {
                "agent_id": "test_agent",
                "content": "测试内容2",
                "hexagram": "坤卦",
                "memory_type": "normal",
                "hot": 0.3,
                "weight": 0.6,
                "priority": 2,
            },
        ]
        memory_ids = self.memory_db.create_memories_batch(memories_data)
        self.assertIsInstance(memory_ids, list)
        self.assertEqual(len(memory_ids), 2)

    def test_get_memory(self):
        """
        测试获取记忆
        """
        # 创建记忆
        memory_data = {
            "agent_id": "test_agent",
            "content": "测试内容",
            "hexagram": "乾卦",
            "memory_type": "core",
            "hot": 0.5,
            "weight": 0.8,
            "priority": 1,
        }
        memory_id = self.memory_db.create_memory(memory_data)

        # 获取记忆
        memory = self.memory_db.get_memory(memory_id)
        self.assertIsInstance(memory, dict)
        self.assertEqual(memory["id"], memory_id)
        self.assertEqual(memory["agent_id"], "test_agent")
        self.assertEqual(memory["content"], "测试内容")

    def test_get_memories(self):
        """
        测试获取记忆列表
        """
        # 创建测试数据
        for i in range(3):
            memory_data = {
                "agent_id": "test_agent",
                "content": f"测试内容{i}",
                "hexagram": "乾卦",
                "memory_type": "core",
                "hot": 0.5,
                "weight": 0.8,
                "priority": 1,
            }
            self.memory_db.create_memory(memory_data)

        # 获取记忆列表
        memories = self.memory_db.get_memories(agent_id="test_agent", limit=2)
        self.assertIsInstance(memories, list)
        self.assertEqual(len(memories), 2)

    def test_update_memory(self):
        """
        测试更新记忆
        """
        # 创建记忆
        memory_data = {
            "agent_id": "test_agent",
            "content": "测试内容",
            "hexagram": "乾卦",
            "memory_type": "core",
            "hot": 0.5,
            "weight": 0.8,
            "priority": 1,
        }
        memory_id = self.memory_db.create_memory(memory_data)

        # 更新记忆
        update_data = {"content": "更新后的测试内容", "hot": 0.8, "weight": 0.9}
        updated_memory = self.memory_db.update_memory(memory_id, update_data)
        self.assertIsInstance(updated_memory, dict)
        self.assertEqual(updated_memory["content"], "更新后的测试内容")
        self.assertEqual(updated_memory["hot"], 0.8)
        self.assertEqual(updated_memory["weight"], 0.9)

    def test_delete_memory(self):
        """
        测试删除记忆
        """
        # 创建记忆
        memory_data = {
            "agent_id": "test_agent",
            "content": "测试内容",
            "hexagram": "乾卦",
            "memory_type": "core",
            "hot": 0.5,
            "weight": 0.8,
            "priority": 1,
        }
        memory_id = self.memory_db.create_memory(memory_data)

        # 删除记忆
        result = self.memory_db.delete_memory(memory_id)
        self.assertTrue(result)

        # 验证删除
        memory = self.memory_db.get_memory(memory_id)
        self.assertIsNone(memory)

    def test_count_memories(self):
        """
        测试统计记忆数量
        """
        # 初始数量
        count = self.memory_db.count_memories(agent_id="test_agent")
        self.assertEqual(count, 0)

        # 创建测试数据
        for i in range(3):
            memory_data = {
                "agent_id": "test_agent",
                "content": f"测试内容{i}",
                "hexagram": "乾卦",
                "memory_type": "core",
                "hot": 0.5,
                "weight": 0.8,
                "priority": 1,
            }
            self.memory_db.create_memory(memory_data)

        # 统计数量
        count = self.memory_db.count_memories(agent_id="test_agent")
        self.assertEqual(count, 3)

    def test_search_memories(self):
        """
        测试搜索记忆
        """
        # 创建测试数据
        self.memory_db.create_memory(
            {
                "agent_id": "test_agent",
                "content": "测试内容1",
                "hexagram": "乾卦",
                "memory_type": "core",
                "hot": 0.5,
                "weight": 0.8,
                "priority": 1,
            }
        )
        self.memory_db.create_memory(
            {
                "agent_id": "test_agent",
                "content": "测试内容2",
                "hexagram": "坤卦",
                "memory_type": "normal",
                "hot": 0.3,
                "weight": 0.6,
                "priority": 2,
            }
        )

        # 搜索记忆
        results = self.memory_db.search_memories(agent_id="test_agent", keyword="测试")
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)

    def test_get_memories_by_type(self):
        """
        测试按类型获取记忆
        """
        # 创建测试数据
        self.memory_db.create_memory(
            {
                "agent_id": "test_agent",
                "content": "核心记忆",
                "hexagram": "乾卦",
                "memory_type": "core",
                "hot": 0.5,
                "weight": 0.8,
                "priority": 1,
            }
        )
        self.memory_db.create_memory(
            {
                "agent_id": "test_agent",
                "content": "普通记忆",
                "hexagram": "坤卦",
                "memory_type": "normal",
                "hot": 0.3,
                "weight": 0.6,
                "priority": 2,
            }
        )

        # 按类型获取记忆
        core_memories = self.memory_db.get_memories_by_type(agent_id="test_agent", memory_type="core")
        self.assertIsInstance(core_memories, list)
        self.assertEqual(len(core_memories), 1)
        self.assertEqual(core_memories[0]["memory_type"], "core")


if __name__ == "__main__":
    unittest.main()
