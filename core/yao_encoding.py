#!/usr/bin/env python3
"""
📜 六爻统一编码系统 - Yao Encoding System

基于 IMA V2.0 规范的六爻统一编码实现：
- 六爻编码：6 维二进制向量 (0=阴，1=阳)
- 映射关系：初爻→地，二爻→人，三爻→天，四爻→八卦，五爻→五行，六爻→时位
- 支持编码/解码/校验/转换

记忆编码格式：
{
    "id": "gua-63-20260418-1234",
    "name": "乾卦·九五",
    "hexagram": [1,1,1,1,1,1],  // 六爻二进制向量
    "element": "金",              // 五行属性
    "layer": "天",                // 三才层
    "trigram": "乾",              // 八卦分类
    "position": "当位",           // 时位状态
    "weight": 0.95,               // 重要性权重
    "hot": 0.8,                   // 热度
    "timestamp": "2026-04-18",
    "content": "...",
    "relations": ["gua-12", "gua-35"],
    "source": "user_input",
    "expire_time": "2026-05-18"
}
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ========== 基础枚举 ==========


class Polarity(Enum):
    """爻的阴阳"""

    YANG = 1  # 阳爻 —
    YIN = 0  # 阴爻 - -


class Layer(Enum):
    """三才分层"""

    TIAN = "天"  # 天层 - 长期记忆
    REN = "人"  # 人层 - 中期记忆
    DI = "地"  # 地层 - 短期记忆


class Element(Enum):
    """五行"""

    MU = "木"  # 木 - 生发/生成期
    HUO = "火"  # 火 - 活跃/活跃期
    TU = "土"  # 土 - 稳定/稳定期
    JIN = "金"  # 金 - 精炼/精炼期
    SHUI = "水"  # 水 - 闭藏/闭藏期


class Trigram(Enum):
    """八卦"""

    QIAN = "乾"  # 天 - 核心记忆
    KUN = "坤"  # 地 - 知识库
    ZHEN = "震"  # 雷 - 事件记忆
    XUN = "巽"  # 风 - 工作记忆
    KAN = "坎"  # 水 - 隐记忆
    LI = "离"  # 火 - 显记忆
    GEN = "艮"  # 山 - 结果记忆
    DUI = "兑"  # 泽 - 社交记忆


class Position(Enum):
    """时位状态"""

    DANG_WEI = "当位"  # 当位 - 合理有效
    SHI_WEI = "失位"  # 失位 - 需要调整


# ========== 数据结构 ==========


@dataclass
class MemoryEncoding:
    """记忆编码数据结构"""

    id: str
    name: str
    hexagram: List[int]  # 6 位二进制向量
    element: str
    layer: str
    trigram: str
    position: str
    weight: float
    hot: float
    timestamp: str
    content: str
    relations: List[str] = field(default_factory=list)
    source: str = "user_input"
    expire_time: Optional[str] = None
    agent_id: Optional[str] = None  # 多租户支持

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryEncoding":
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "MemoryEncoding":
        data = json.loads(json_str)
        return cls.from_dict(data)


# ========== 六爻编码系统 ==========


class YaoEncoder:
    """
    六爻编码器

    六爻位置定义（从下到上）：
    - 初爻 (索引 0): 地/短期
    - 二爻 (索引 1): 人/中期
    - 三爻 (索引 2): 天/长期
    - 四爻 (索引 3): 八卦类型
    - 五爻 (索引 4): 五行状态
    - 六爻 (索引 5): 时位优先级
    """

    # 八卦对应的三爻二进制（下卦/上卦）
    TRIGRAM_BINARY = {
        Trigram.QIAN: [1, 1, 1],  # 乾 ☰
        Trigram.KUN: [0, 0, 0],  # 坤 ☷
        Trigram.ZHEN: [0, 0, 1],  # 震 ☳
        Trigram.XUN: [1, 1, 0],  # 巽 ☴
        Trigram.KAN: [0, 1, 0],  # 坎 ☵
        Trigram.LI: [1, 0, 1],  # 离 ☲
        Trigram.GEN: [1, 0, 0],  # 艮 ☶
        Trigram.DUI: [1, 1, 0],  # 兑 ☱
    }

    # 五行对应的二进制编码
    ELEMENT_BINARY = {
        Element.MU: [1, 0],  # 木 - 01
        Element.HUO: [1, 1],  # 火 - 11
        Element.TU: [0, 1],  # 土 - 10
        Element.JIN: [0, 0],  # 金 - 00
        Element.SHUI: [1, 0],  # 水 - 01 (与木相同，通过上下文区分)
    }

    # 六十四卦序号（用于快速索引）
    HEXAGRAM_NUMBERS = {
        "乾": 1,
        "坤": 2,
        "屯": 3,
        "蒙": 4,
        "需": 5,
        "讼": 6,
        "师": 7,
        "比": 8,
        "小畜": 9,
        "履": 10,
        "泰": 11,
        "否": 12,
        "同人": 13,
        "大有": 14,
        "谦": 15,
        "豫": 16,
        "随": 17,
        "蛊": 18,
        "临": 19,
        "观": 20,
        "噬嗑": 21,
        "贲": 22,
        "剥": 23,
        "复": 24,
        "无妄": 25,
        "大畜": 26,
        "颐": 27,
        "大过": 28,
        "坎": 29,
        "离": 30,
        "咸": 31,
        "恒": 32,
        "遁": 33,
        "大壮": 34,
        "晋": 35,
        "明夷": 36,
        "家人": 37,
        "睽": 38,
        "蹇": 39,
        "解": 40,
        "损": 41,
        "益": 42,
        "夬": 43,
        "姤": 44,
        "萃": 45,
        "升": 46,
        "困": 47,
        "井": 48,
        "革": 49,
        "鼎": 50,
        "震": 51,
        "艮": 52,
        "渐": 53,
        "归妹": 54,
        "丰": 55,
        "旅": 56,
        "巽": 57,
        "兑": 58,
        "涣": 59,
        "节": 60,
        "中孚": 61,
        "小过": 62,
        "既济": 63,
        "未济": 64,
    }

    def __init__(self):
        self._init_reverse_mappings()

    def _init_reverse_mappings(self):
        """初始化反向映射"""
        # 二进制→八卦
        self.BINARY_TRIGRAM = {tuple(v): k for k, v in self.TRIGRAM_BINARY.items()}

        # 二进制→五行
        self.BINARY_ELEMENT = {tuple(v): k for k, v in self.ELEMENT_BINARY.items()}

    def encode(
        self,
        content: str,
        trigram: Trigram,
        layer: Layer,
        element: Element,
        position: Position = Position.DANG_WEI,
        weight: float = 0.5,
        hot: float = 0.5,
        agent_id: Optional[str] = None,
        source: str = "user_input",
    ) -> MemoryEncoding:
        """
        将记忆内容编码为六爻向量

        Args:
            content: 记忆内容
            trigram: 八卦分类
            layer: 三才分层
            element: 五行状态
            position: 时位状态
            weight: 重要性权重 (0-1)
            hot: 热度 (0-1)
            agent_id: Agent ID（多租户）
            source: 记忆来源

        Returns:
            MemoryEncoding 对象
        """
        # 生成六爻向量
        hexagram = self._generate_hexagram(trigram, layer, element, position)

        # 生成唯一 ID
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_input = f"{content}{timestamp}{trigram.value}"
        hash_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        hexagram_num = self.HEXAGRAM_NUMBERS.get(trigram.value, 1)
        memory_id = f"gua-{hexagram_num}-{timestamp}-{hash_id}"

        # 生成卦名
        name = f"{trigram.value}卦·{self._get_yao_name(hexagram)}"

        # 计算过期时间（地层记忆需要）
        expire_time = None
        if layer == Layer.DI:
            expire_time = (datetime.now() + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")

        return MemoryEncoding(
            id=memory_id,
            name=name,
            hexagram=hexagram,
            element=element.value,
            layer=layer.value,
            trigram=trigram.value,
            position=position.value,
            weight=weight,
            hot=hot,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            content=content,
            relations=[],
            source=source,
            expire_time=expire_time,
            agent_id=agent_id,
        )

    def _generate_hexagram(self, trigram: Trigram, layer: Layer, element: Element, position: Position) -> List[int]:
        """
        生成六爻向量

        六爻结构：
        - 初爻 (0): 地层=0, 人层=1
        - 二爻 (1): 人层=0, 天层=1
        - 三爻 (2): 根据八卦的上爻
        - 四爻 (3): 根据八卦的中爻
        - 五爻 (4): 根据八卦的下爻
        - 六爻 (5): 当位=1, 失位=0
        """
        # 获取八卦的三爻
        tri_bits = self.TRIGRAM_BINARY[trigram]

        # 构建六爻
        hexagram = [0] * 6

        # 初爻：地层=0, 人层=1
        hexagram[0] = 0 if layer == Layer.DI else (1 if layer == Layer.REN else 0)

        # 二爻：人层=0, 天层=1
        hexagram[1] = 0 if layer == Layer.REN else (1 if layer == Layer.TIAN else 0)

        # 三爻~五爻：八卦的三爻
        hexagram[2] = tri_bits[2]  # 上爻
        hexagram[3] = tri_bits[1]  # 中爻
        hexagram[4] = tri_bits[0]  # 下爻

        # 六爻：当位=1, 失位=0
        hexagram[5] = 1 if position == Position.DANG_WEI else 0

        return hexagram

    def _get_yao_name(self, hexagram: List[int]) -> str:
        """获取爻位名称（如初九、六二等）"""
        # 从下往上数，阳爻用"九"，阴爻用"六"
        yao_names = ["初", "二", "三", "四", "五", "上"]
        yao_types = {1: "九", 0: "六"}

        # 找到最显著的爻（这里简化为最上面的阳爻或最下面的阴爻）
        for i in range(5, -1, -1):
            if hexagram[i] == 1:
                return f"{yao_names[i]}{yao_types[1]}"

        return f"{yao_names[0]}{yao_types[0]}"

    def decode(self, encoding: MemoryEncoding) -> Dict[str, Any]:
        """
        解码六爻向量，提取记忆特征

        Returns:
            包含所有记忆特征的字典
        """
        hexagram = encoding.hexagram

        # 解析三才层
        if hexagram[0] == 0 and hexagram[1] == 0:
            layer = Layer.DI
        elif hexagram[0] == 1 and hexagram[1] == 0:
            layer = Layer.REN
        else:
            layer = Layer.TIAN

        # 解析八卦
        tri_bits = hexagram[2:5]
        trigram = self.BINARY_TRIGRAM.get(tuple(tri_bits), Trigram.QIAN)

        # 解析时位
        position = Position.DANG_WEI if hexagram[5] == 1 else Position.SHI_WEI

        return {
            "layer": layer,
            "trigram": trigram,
            "position": position,
            "hexagram_number": self._hexagram_to_number(hexagram),
            "binary_string": "".join(map(str, hexagram)),
        }

    def _hexagram_to_number(self, hexagram: List[int]) -> int:
        """将六爻向量转换为十进制数（用于索引）"""
        binary_str = "".join(map(str, hexagram))
        return int(binary_str, 2) + 1

    def calculate_similarity(self, enc1: MemoryEncoding, enc2: MemoryEncoding) -> float:
        """
        计算两个记忆的卦象相似度

        使用汉明距离计算六爻向量的相似度
        """
        h1, h2 = enc1.hexagram, enc2.hexagram
        hamming_distance = sum(a != b for a, b in zip(h1, h2))
        similarity = 1.0 - (hamming_distance / 6.0)
        return similarity

    def yao_change(self, encoding: MemoryEncoding, change_degree: float) -> Tuple[MemoryEncoding, List[int]]:
        """
        爻变：根据变化程度翻转爻位

        Args:
            encoding: 原记忆编码
            change_degree: 变化程度 (0-1)

        Returns:
            (新记忆编码，变化的爻位列表)
        """
        changed_positions = []
        new_hexagram = encoding.hexagram.copy()

        if change_degree < 0.3:
            # 微小变化：不触发爻变
            pass
        elif change_degree < 0.7:
            # 中等变化：翻转 1-2 个爻位
            import random

            num_changes = random.randint(1, 2)
            positions = random.sample(range(6), num_changes)
            for pos in positions:
                new_hexagram[pos] = 1 - new_hexagram[pos]
                changed_positions.append(pos)
        else:
            # 重大变化：翻转 3+ 个爻位（卦变）
            import random

            num_changes = random.randint(3, 5)
            positions = random.sample(range(6), num_changes)
            for pos in positions:
                new_hexagram[pos] = 1 - new_hexagram[pos]
                changed_positions.append(pos)

        # 创建新记忆
        new_encoding = MemoryEncoding(
            id=encoding.id + "_v2",
            name=encoding.name + "_变",
            hexagram=new_hexagram,
            element=encoding.element,
            layer=encoding.layer,
            trigram=encoding.trigram,
            position=encoding.position,
            weight=min(encoding.weight + 0.1, 1.0),
            hot=min(encoding.hot + 0.2, 1.0),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            content=encoding.content,
            relations=encoding.relations + [encoding.id],
            source="yao_change",
            agent_id=encoding.agent_id,
        )

        return new_encoding, changed_positions


# ========== 工具函数 ==========


def create_default_encoder() -> YaoEncoder:
    """创建默认编码器实例"""
    return YaoEncoder()


def encode_memory(content: str, trigram: str, layer: str, element: str, **kwargs) -> MemoryEncoding:
    """
    快速编码记忆的便捷函数

    示例:
        encoding = encode_memory(
            content="用户咨询退货政策",
            trigram="乾",
            layer="天",
            element="金",
            weight=0.9
        )
    """
    encoder = YaoEncoder()

    # 字符串转枚举
    trigram_enum = Trigram(trigram)
    layer_enum = Layer(layer)
    element_enum = Element(element)
    position_enum = Position(kwargs.get("position", "当位"))

    return encoder.encode(
        content=content,
        trigram=trigram_enum,
        layer=layer_enum,
        element=element_enum,
        position=position_enum,
        weight=kwargs.get("weight", 0.5),
        hot=kwargs.get("hot", 0.5),
        agent_id=kwargs.get("agent_id"),
        source=kwargs.get("source", "user_input"),
    )


# ========== 命令行测试 ==========

if __name__ == "__main__":
    print("=" * 80)
    print("📜 六爻统一编码系统测试")
    print("=" * 80)

    encoder = YaoEncoder()

    # 测试 1：编码核心记忆
    print("\n【测试 1】编码核心记忆（乾卦·天层·金）")
    mem1 = encoder.encode(
        content="用户咨询某商品的退货政策，要求了解 7 天无理由退货的具体规则。",
        trigram=Trigram.QIAN,
        layer=Layer.TIAN,
        element=Element.JIN,
        weight=0.95,
        hot=0.8,
    )
    print(f"ID: {mem1.id}")
    print(f"卦名：{mem1.name}")
    print(f"六爻：{mem1.hexagram}")
    print(f"五行：{mem1.element}, 三才：{mem1.layer}, 八卦：{mem1.trigram}")
    print(f"权重：{mem1.weight}, 热度：{mem1.hot}")

    # 测试 2：编码工作记忆
    print("\n【测试 2】编码工作记忆（巽卦·地层·木）")
    mem2 = encoder.encode(
        content="临时对话上下文：用户询问价格",
        trigram=Trigram.XUN,
        layer=Layer.DI,
        element=Element.MU,
        weight=0.3,
        hot=0.9,
    )
    print(f"ID: {mem2.id}")
    print(f"卦名：{mem2.name}")
    print(f"六爻：{mem2.hexagram}")
    print(f"过期时间：{mem2.expire_time}")

    # 测试 3：解码
    print("\n【测试 3】解码六爻向量")
    decoded = encoder.decode(mem1)
    print(f"解析结果：{json.dumps({k: str(v) for k, v in decoded.items()}, ensure_ascii=False, indent=2)}")

    # 测试 4：相似度计算
    print("\n【测试 4】卦象相似度计算")
    similarity = encoder.calculate_similarity(mem1, mem2)
    print(f"mem1 与 mem2 的相似度：{similarity:.2%}")

    # 测试 5：爻变
    print("\n【测试 5】爻变测试")
    new_mem, changed = encoder.yao_change(mem1, change_degree=0.5)
    print(f"原六爻：{mem1.hexagram}")
    print(f"新六爻：{new_mem.hexagram}")
    print(f"变化的爻位：{changed}")
    print(f"新卦名：{new_mem.name}")

    # 测试 6：JSON 序列化
    print("\n【测试 6】JSON 序列化")
    json_str = mem1.to_json()
    print(f"JSON 长度：{len(json_str)} 字符")

    # 测试 7：JSON 反序列化
    print("\n【测试 7】JSON 反序列化")
    mem1_restored = MemoryEncoding.from_json(json_str)
    print(f"恢复后 ID: {mem1_restored.id}")
    print(f"恢复后卦名：{mem1_restored.name}")

    print("\n" + "=" * 80)
    print("✅ 所有测试完成！")
    print("=" * 80)
