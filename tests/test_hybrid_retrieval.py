#!/usr/bin/env python3
"""
Hybrid Retrieval 单元测试 — Phase 3B 纯语义检索路径
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.hybrid_retrieval import (
    parse_hexagram,
    hamming_distance,
    RetrievalResult,
    STRUCTURAL_WEIGHT,
    SEMANTIC_WEIGHT,
)


class TestParseHexagram:
    """六爻解析测试"""

    def test_parse_list(self):
        result = parse_hexagram([1, 0, 1, 0, 1, 0])
        assert result == [1, 0, 1, 0, 1, 0]

    def test_parse_string(self):
        result = parse_hexagram("[1, 0, 1, 0, 1, 0]")
        assert result == [1, 0, 1, 0, 1, 0]

    def test_parse_none(self):
        assert parse_hexagram(None) is None

    def test_parse_wrong_length(self):
        assert parse_hexagram([1, 0, 1]) is None

    def test_parse_invalid_string(self):
        assert parse_hexagram("not a hexagram") is None


class TestHammingDistance:
    """Hamming 距离测试（已弃用，仅供调试）"""

    def test_identical(self):
        assert hamming_distance([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]) == 0

    def test_all_different(self):
        assert hamming_distance([0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]) == 6

    def test_half_different(self):
        assert hamming_distance([0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1]) == 3


class TestOptionBWeights:
    """Phase 3B Option B: 六爻权重归零验证"""

    def test_structural_weight_zero(self):
        """核心断言：六爻结构权重必须为 0"""
        assert STRUCTURAL_WEIGHT == 0.0, (
            f"Phase 3B Option B 要求 STRUCTURAL_WEIGHT=0.0，"
            f"实际值={STRUCTURAL_WEIGHT}。六爻不参与检索评分！"
        )

    def test_semantic_weight_one(self):
        """核心断言：语义权重必须为 1.0"""
        assert SEMANTIC_WEIGHT == 1.0, (
            f"Phase 3B Option B 要求 SEMANTIC_WEIGHT=1.0，"
            f"实际值={SEMANTIC_WEIGHT}。检索评分 100% 语义向量！"
        )

    def test_weights_sum_to_one(self):
        """权重之和应为 1.0"""
        assert abs(STRUCTURAL_WEIGHT + SEMANTIC_WEIGHT - 1.0) < 1e-6


class TestRetrievalResult:
    """检索结果数据类测试"""

    def test_to_dict(self):
        result = RetrievalResult(
            memory_id="mem_001",
            agent_id="agent_001",
            hexagram=[1, 0, 1, 0, 1, 0],
            bagua_type="乾",
            sancai_layer="天",
            wuxing="金",
            content="test content",
            hot_score=0.8,
            semantic_similarity=0.95,
            rank=1,
        )
        d = result.to_dict()
        assert d["memory_id"] == "mem_001"
        assert d["bagua_type"] == "乾"
        assert d["scores"]["semantic_similarity"] == 0.95

    def test_hexagram_as_label_only(self):
        """Phase 3B: hexagram 是分类标签，不参与评分"""
        result = RetrievalResult(
            memory_id="mem_002",
            agent_id="agent_001",
            hexagram=[0, 0, 0, 0, 0, 0],
            bagua_type="坤",
            sancai_layer="地",
            wuxing="土",
            content="irrelevant hex content",
            hot_score=0.5,
            semantic_similarity=0.88,
            rank=1,
        )
        # hexagram 存在于结果中但 semantic_similarity 是唯一评分来源
        d = result.to_dict()
        assert "hexagram" in d
        assert d["scores"]["semantic_similarity"] == 0.88
        # hexagram 不影响 semantic_similarity
