#!/usr/bin/env python3
"""
测试向量存储模块
"""

import tempfile
import unittest

from core.vector_store import vector_store


class TestVectorStore(unittest.TestCase):
    """
    测试向量存储模块
    """

    def setUp(self):
        """
        设置测试环境
        """
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        # 初始化向量存储
        self.vector_store = vector_store
        self.vector_store.initialize(collection_name="test", embedding_dim=768, storage_dir=self.temp_dir)

    def tearDown(self):
        """
        清理测试环境
        """
        # 清理临时目录
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_add_embedding(self):
        """
        测试添加嵌入向量
        """
        # 测试添加单个嵌入向量
        vector_id = self.vector_store.add_embedding([0.1] * 768, {"key": "value"})
        self.assertIsInstance(vector_id, str)

        # 测试添加多个嵌入向量
        vectors = [[0.1] * 768, [0.2] * 768]
        metadatas = [{"key1": "value1"}, {"key2": "value2"}]
        vector_ids = self.vector_store.add_embeddings(vectors, metadatas)
        self.assertIsInstance(vector_ids, list)
        self.assertEqual(len(vector_ids), 2)

    def test_search_similar(self):
        """
        测试相似性搜索
        """
        # 添加测试数据
        for i in range(5):
            vector = [i * 0.1] * 768
            metadata = {"id": i, "value": f"test_{i}"}
            self.vector_store.add_embedding(vector, metadata)

        # 搜索相似向量
        query_vector = [0.2] * 768
        results = self.vector_store.search_similar(query_vector, top_k=3)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        # 检查结果是否按相似度排序
        self.assertLessEqual(results[0][2], results[1][2])
        self.assertLessEqual(results[1][2], results[2][2])

    def test_get_embedding(self):
        """
        测试获取嵌入向量
        """
        # 添加测试数据
        vector_id = self.vector_store.add_embedding([0.1] * 768, {"key": "value"})

        # 获取嵌入向量
        embedding, metadata = self.vector_store.get_embedding(vector_id)
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 768)
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata["key"], "value")

    def test_update_embedding(self):
        """
        测试更新嵌入向量
        """
        # 添加测试数据
        vector_id = self.vector_store.add_embedding([0.1] * 768, {"key": "value"})

        # 更新嵌入向量
        new_vector = [0.2] * 768
        new_metadata = {"key": "new_value"}
        self.vector_store.update_embedding(vector_id, new_vector, new_metadata)

        # 验证更新
        embedding, metadata = self.vector_store.get_embedding(vector_id)
        self.assertEqual(embedding, new_vector)
        self.assertEqual(metadata["key"], "new_value")

    def test_delete_embedding(self):
        """
        测试删除嵌入向量
        """
        # 添加测试数据
        vector_id = self.vector_store.add_embedding([0.1] * 768, {"key": "value"})

        # 删除嵌入向量
        self.vector_store.delete_embedding(vector_id)

        # 验证删除
        result = self.vector_store.get_embedding(vector_id)
        self.assertIsNone(result)

    def test_get_all_embeddings(self):
        """
        测试获取所有嵌入向量
        """
        # 先清空所有嵌入向量
        self.vector_store.clear_embeddings()

        # 添加测试数据
        for i in range(3):
            vector = [i * 0.1] * 768
            metadata = {"id": i}
            self.vector_store.add_embedding(vector, metadata)

        # 获取所有嵌入向量
        embeddings = self.vector_store.get_all_embeddings()
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), 3)

    def test_count_embeddings(self):
        """
        测试统计嵌入向量数量
        """
        # 初始数量
        count = self.vector_store.count_embeddings()
        self.assertEqual(count, 0)

        # 添加测试数据
        for i in range(3):
            vector = [i * 0.1] * 768
            metadata = {"id": i}
            self.vector_store.add_embedding(vector, metadata)

        # 统计数量
        count = self.vector_store.count_embeddings()
        self.assertEqual(count, 3)

    def test_clear_embeddings(self):
        """
        测试清空嵌入向量
        """
        # 添加测试数据
        for i in range(3):
            vector = [i * 0.1] * 768
            metadata = {"id": i}
            self.vector_store.add_embedding(vector, metadata)

        # 清空嵌入向量
        self.vector_store.clear_embeddings()

        # 验证清空
        count = self.vector_store.count_embeddings()
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
