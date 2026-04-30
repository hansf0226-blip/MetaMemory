"""
测试易经核心编码算法（匹配实际 API）
"""
import pytest
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.core_yijing import (
    content_to_hexagram,
    get_gua_by_hex,
    hexagram_to_vector,
    calculate_similarity,
    wuxing_life_schedule,
    gua_data_check,
    yao_bian_update,
)


class TestContentToHexagram:
    """测试内容转六爻卦象"""

    def test_basic_conversion(self):
        """基本转换"""
        result = content_to_hexagram("测试内容", "chat")
        assert isinstance(result, tuple)
        assert len(result) == 5  # (hex_arr, bagua, wuxing, layer, is_fallback)
        hex_arr, bagua, wuxing, layer, is_fallback = result
        assert isinstance(hex_arr, list)
        assert len(hex_arr) == 6
        assert all(bit in (0, 1) for bit in hex_arr)
        assert isinstance(bagua, str) and len(bagua) > 0
        assert isinstance(wuxing, str) and len(wuxing) > 0
        assert isinstance(layer, str) and len(layer) > 0
        assert isinstance(is_fallback, bool)

    def test_consistency(self):
        """相同文本结果一致"""
        text = "测试文本的一致性"
        result1 = content_to_hexagram(text, "chat")
        result2 = content_to_hexagram(text, "chat")
        assert result1 == result2

    def test_different_contents_produce_different(self):
        """不同内容产生不同卦象"""
        r1 = content_to_hexagram("第一个文本AAAA", "chat")
        r2 = content_to_hexagram("第二个文本BBBB完全不同", "chat")
        assert r1 != r2

    def test_empty_text(self):
        """空文本处理"""
        result = content_to_hexagram("", "chat")
        hex_arr, bagua, wuxing, layer, is_fallback = result
        assert len(hex_arr) == 6
        assert all(bit in (0, 1) for bit in hex_arr)

    def test_different_memory_types(self):
        """不同记忆类型"""
        r_chat = content_to_hexagram("测试", "chat")
        r_knowledge = content_to_hexagram("测试", "knowledge")
        # 不同类型可能有不同编码
        assert isinstance(r_chat, tuple) and isinstance(r_knowledge, tuple)


class TestGetGuaByHex:
    """测试卦象匹配"""

    def test_qian_gua(self):
        """乾卦 [1,1,1,1,1,1]"""
        result = get_gua_by_hex([1, 1, 1, 1, 1, 1])
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_kun_gua(self):
        """坤卦 [0,0,0,0,0,0]"""
        result = get_gua_by_hex([0, 0, 0, 0, 0, 0])
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_unknown_hexagram(self):
        """未知卦象返回空"""
        result = get_gua_by_hex([9, 9, 9, 9, 9, 9])
        assert result == {}


class TestCalculateSimilarity:
    """测试六爻相似度计算"""

    def test_identical(self):
        """完全相同的六爻"""
        hex1 = [1, 1, 1, 1, 1, 1]
        similarity = calculate_similarity(hex1, hex1)
        assert similarity == 1.0

    def test_completely_different(self):
        """完全不同的六爻"""
        hex1 = [1, 1, 1, 1, 1, 1]
        hex2 = [0, 0, 0, 0, 0, 0]
        similarity = calculate_similarity(hex1, hex2)
        assert similarity == 0.0

    def test_partial_match(self):
        """部分匹配"""
        hex1 = [1, 1, 1, 1, 1, 1]
        hex2 = [1, 1, 1, 1, 1, 0]
        similarity = calculate_similarity(hex1, hex2)
        assert similarity == 5 / 6


class TestWuxingLifeSchedule:
    """测试五行生命周期调度"""

    def test_mu_to_huo(self):
        """木 → 火（相生流转）"""
        mem = {"wuxing": "mu", "hot_score": 0.8}
        result = wuxing_life_schedule(mem, None)
        assert isinstance(result, dict)
        assert result["wuxing"] == "huo"
        assert result["hot_score"] < 0.8  # 热度衰减

    def test_returns_expected_keys(self):
        """返回包含 wuxing 和 hot_score"""
        mem = {"wuxing": "jin", "hot_score": 0.5}
        result = wuxing_life_schedule(mem, datetime.now())
        assert "wuxing" in result
        assert "hot_score" in result
        assert isinstance(result["wuxing"], str)


class TestGuaDataCheck:
    """测试卦象数据校验"""

    def test_valid_data(self):
        """有效数据通过"""
        assert gua_data_check([1, 0, 1, 0, 1, 0], "kan", "shui", "ren") is True

    def test_invalid_hex_length(self):
        """六爻长度不对"""
        with pytest.raises(ValueError):
            gua_data_check([1, 0, 1], "qian", "jin", "tien")

    def test_invalid_bit_value(self):
        """爻位值非法"""
        with pytest.raises(ValueError):
            gua_data_check([1, 0, 1, 0, 1, 2], "qian", "jin", "tien")

    def test_invalid_bagua(self):
        """八卦编码非法"""
        with pytest.raises(ValueError):
            gua_data_check([1, 1, 1, 1, 1, 1], "invalid", "jin", "tien")


class TestYaoBianUpdate:
    """测试爻变更新"""

    def test_flip_position(self):
        """翻转指定位"""
        result = yao_bian_update([1, 0, 1, 0, 1, 0], 2)
        assert result[2] == 0  # 原来是 1，翻转为 0

    def test_no_change_other_positions(self):
        """其他位不变"""
        original = [1, 0, 1, 0, 1, 0]
        result = yao_bian_update(original, 0)
        assert result[0] == 0
        assert result[1:] == original[1:]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
