"""
智能编码系统单元测试
"""

import unittest

from core.core_yijing import calculate_similarity, content_to_hexagram, get_gua_by_hex, wuxing_life_schedule


class TestCoreYijing(unittest.TestCase):
    """核心智能编码功能测试"""

    def test_content_to_hexagram(self):
        """测试内容转换为编码"""
        # 测试不同类型的内容
        content = "这是一个测试内容"
        memory_type = "chat"
        result = content_to_hexagram(content, memory_type)

        # 验证返回值格式
        self.assertIsInstance(result, tuple)
        # 检查返回值长度，可能包含 is_fallback
        self.assertIn(len(result), [4, 5])
        if len(result) == 5:
            hex_arr, bagua, wuxing, layer, is_fallback = result
            self.assertIsInstance(is_fallback, bool)
        else:
            hex_arr, bagua, wuxing, layer = result

        self.assertIsInstance(hex_arr, list)
        self.assertEqual(len(hex_arr), 6)
        self.assertIsInstance(bagua, str)
        self.assertIsInstance(wuxing, str)
        self.assertIsInstance(layer, str)

    def test_get_gua_by_hex(self):
        """测试根据编码获取卦"""
        # 测试乾卦
        qian_hex = [1, 1, 1, 1, 1, 1]
        gua = get_gua_by_hex(qian_hex)
        self.assertIsInstance(gua, dict)
        self.assertIn("name", gua)

        # 测试坤卦
        kun_hex = [0, 0, 0, 0, 0, 0]
        gua = get_gua_by_hex(kun_hex)
        self.assertIsInstance(gua, dict)
        self.assertIn("name", gua)

    def test_calculate_similarity(self):
        """测试编码相似度计算"""
        hex1 = [1, 1, 1, 1, 1, 1]
        hex2 = [1, 1, 1, 1, 1, 1]
        similarity = calculate_similarity(hex1, hex2)
        self.assertEqual(similarity, 1.0)

        hex3 = [1, 1, 1, 1, 1, 0]
        similarity = calculate_similarity(hex1, hex3)
        self.assertEqual(similarity, 5 / 6)

    def test_wuxing_life_schedule(self):
        """测试智能调度"""
        mem_dict = {"wuxing": "mu", "hot_score": 0.8}
        create_time = None  # 这里不需要实际时间
        result = wuxing_life_schedule(mem_dict, create_time)

        self.assertIsInstance(result, dict)
        self.assertIn("wuxing", result)
        self.assertIn("hot_score", result)
        self.assertEqual(result["wuxing"], "huo")  # 木生火
        self.assertLess(result["hot_score"], 0.8)  # 热度衰减


if __name__ == "__main__":
    unittest.main()
