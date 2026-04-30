#!/usr/bin/env python3
"""
测试 MySQL 存储模块
"""

import unittest

from core.mysql_store import mysql_store


class TestMySQLStore(unittest.TestCase):
    """
    测试 MySQL 存储模块
    """

    def setUp(self):
        """
        设置测试环境
        """
        # 初始化 MySQL 存储
        self.mysql_store = mysql_store

    def tearDown(self):
        """
        清理测试环境
        """
        # 清理测试数据
        from sqlalchemy import text

        with self.mysql_store.db.get_session() as session:
            session.execute(text("DELETE FROM yijing_memory WHERE agent_id = 'test_agent'"))
            session.execute(text("DELETE FROM yao_change_history WHERE agent_id = 'test_agent'"))
            session.commit()

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
        result = self.mysql_store.create_memory(memory_data)
        self.assertIsInstance(result, dict)
        self.assertIn("id", result)
        self.assertIn("memory_id", result)
        self.assertIsInstance(result["id"], int)
        self.assertIsInstance(result["memory_id"], str)

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
        memory_ids = self.mysql_store.create_memories_batch(memories_data)
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
        memory_id = self.mysql_store.create_memory(memory_data)

        # 获取记忆
        memory = self.mysql_store.get_memory(memory_id["id"], "test_agent")
        self.assertIsInstance(memory, dict)
        self.assertEqual(memory["id"], memory_id["id"])
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
            self.mysql_store.create_memory(memory_data)

        # 获取记忆列表
        memories = self.mysql_store.get_memories(agent_id="test_agent", limit=2)
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
        memory_id = self.mysql_store.create_memory(memory_data)

        # 更新记忆
        update_data = {"content": "更新后的测试内容", "hot": 0.8, "weight": 0.9}
        updated_memory = self.mysql_store.update_memory(memory_id["id"], "test_agent", update_data)
        self.assertIsInstance(updated_memory, dict)
        self.assertEqual(updated_memory["content"], "更新后的测试内容")
        self.assertEqual(updated_memory["hot"], 0.8)
        self.assertEqual(updated_memory["weight"], 0.9)

    def test_update_memory_hot(self):
        """
        测试更新记忆热度
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
        memory_id = self.mysql_store.create_memory(memory_data)

        # 更新记忆热度
        result = self.mysql_store.update_memory_hot(memory_id["id"], "test_agent", 0.8)
        self.assertTrue(result)

        # 验证更新
        memory = self.mysql_store.get_memory(memory_id["id"], "test_agent")
        self.assertGreaterEqual(memory["hot"], 0.8)  # get_memory boosts hot on read

    def test_update_memory_weight(self):
        """
        测试更新记忆权重
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
        memory_id = self.mysql_store.create_memory(memory_data)

        # 更新记忆权重
        result = self.mysql_store.update_memory_weight(memory_id["id"], "test_agent", 0.9)
        self.assertTrue(result)

        # 验证更新
        memory = self.mysql_store.get_memory(memory_id["id"], "test_agent")
        self.assertEqual(memory["weight"], 0.9)

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
        memory_id = self.mysql_store.create_memory(memory_data)

        # 删除记忆
        result = self.mysql_store.delete_memory(memory_id["id"], "test_agent")
        self.assertTrue(result)

        # 验证删除
        memory = self.mysql_store.get_memory(memory_id["id"], "test_agent")
        self.assertEqual(memory["status"], "deleted")

    def test_hard_delete_memory(self):
        """
        测试物理删除记忆
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
        memory_id = self.mysql_store.create_memory(memory_data)

        # 物理删除记忆
        result = self.mysql_store.hard_delete_memory(memory_id["id"], "test_agent")
        self.assertTrue(result)

        # 验证删除
        memory = self.mysql_store.get_memory(memory_id["id"], "test_agent")
        self.assertIsNone(memory)

    def test_record_yao_change(self):
        """
        测试记录爻变历史
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
        memory_id = self.mysql_store.create_memory(memory_data)

        # 记录爻变历史
        change_data = {
            "memory_id": memory_id["id"],
            "agent_id": "test_agent",
            "old_hexagram": "乾卦",
            "new_hexagram": "坤卦",
            "change_reason": "测试爻变",
        }
        change = self.mysql_store.record_yao_change(change_data)
        self.assertIsInstance(change, object)

    def test_get_yao_history(self):
        """
        测试获取爻变历史
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
        memory_id = self.mysql_store.create_memory(memory_data)

        # 记录爻变历史
        change_data = {
            "memory_id": memory_id["id"],
            "agent_id": "test_agent",
            "old_hexagram": "乾卦",
            "new_hexagram": "坤卦",
            "change_reason": "测试爻变",
        }
        self.mysql_store.record_yao_change(change_data)

        # 获取爻变历史
        history = self.mysql_store.get_yao_history(memory_id["id"], "test_agent")
        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 1)

    def test_get_agent_stats(self):
        """
        测试获取 Agent 统计信息
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
            self.mysql_store.create_memory(memory_data)

        # 获取统计信息
        stats = self.mysql_store.get_agent_stats("test_agent")
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats["total"], 3)

    def test_clean_expired_memories(self):
        """
        测试清理过期记忆
        """
        # 创建测试数据
        memory_data = {
            "agent_id": "test_agent",
            "content": "测试内容",
            "hexagram": "乾卦",
            "memory_type": "core",
            "hot": 0.5,
            "weight": 0.8,
            "priority": 1,
            "expire_time": "2020-01-01 00:00:00",
        }
        self.mysql_store.create_memory(memory_data)

        # 清理过期记忆
        count = self.mysql_store.clean_expired_memories("test_agent")
        self.assertIsInstance(count, int)


if __name__ == "__main__":
    unittest.main()
