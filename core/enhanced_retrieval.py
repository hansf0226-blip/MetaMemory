#!/usr/bin/env python3
"""
🔮 易经记忆向量检索引擎 - 真实业务版

⚠️  DEPRECATED (Phase 3B, 2026-04-28)
=====================================
六爻 rule 编码在检索路径中不产生增量价值（区分度 = -0.2）。
此模块已弃用，由 core/hybrid_retrieval.py 的 SemanticRetrievalEngine 替代。

保留此代码仅供：
- 关键词学习/extend_keywords 功能的参考实现
- SemanticFeatureExtractor 八类业务语义关键词库
- 历史兼容

生产环境请使用 SemanticRetrievalEngine.retrieve()（纯语义向量检索）。

实现：
1. 真实语义相似度计算（基于关键词匹配度）
2. 卦象匹配度加权（六爻汉明距离）
3. 三才层级优先级（天/人/地差异化）
4. 五行气运衰减（时辰权重）
5. 时间权重（新旧记忆差异化）
6. 热度权重（重要度）
7. 综合排序召回（分数必须差异化）
"""

import math
import re
from datetime import datetime
from typing import Dict, List, Tuple

from core.global_state import get_global_state

# ============================================================================
# 语义特征提取器 - 基于关键词的语义匹配
# ============================================================================


class SemanticFeatureExtractor:
    """
    语义特征提取器

    使用业务关键词库进行语义匹配
    返回每个语义维度的匹配度 (0-1)
    """

    # 业务语义关键词库（按场景分类）
    BUSINESS_SEMANTICS = {
        # 技术/开发场景
        "tech": [
            "代码",
            "程序",
            "系统",
            "开发",
            "修复",
            "bug",
            "错误",
            "接口",
            "API",
            "数据库",
            "服务器",
            "部署",
            "测试",
            "功能",
            "模块",
            "算法",
            "数据",
            "文件",
            "配置",
            "环境",
            "网络",
            "安全",
            "性能",
            "优化",
            "卡死",
            "崩溃",
            "异常",
        ],
        # 记忆/知识场景
        "memory": [
            "记忆",
            "知识",
            "记录",
            "历史",
            "对话",
            "聊天",
            "保存",
            "存储",
            "检索",
            "搜索",
            "查询",
            "召回",
            "向量",
            "语义",
            "相似",
            "匹配",
            "分类",
            "标签",
            "整理",
            "沉淀",
            "积累",
            "经验",
        ],
        # 业务/企业场景
        "business": [
            "合同",
            "模板",
            "文档",
            "报告",
            "方案",
            "计划",
            "项目",
            "任务",
            "需求",
            "客户",
            "产品",
            "服务",
            "销售",
            "市场",
            "运营",
            "管理",
            "流程",
            "制度",
            "规范",
            "标准",
            "培训",
            "会议",
            "决策",
        ],
        # 时间/状态场景
        "temporal": [
            "现在",
            "当前",
            "今天",
            "明天",
            "昨天",
            "最近",
            "已经",
            "完成",
            "开始",
            "结束",
            "继续",
            "等待",
            "准备",
            "计划",
            "安排",
            "时间",
            "日期",
            "期限",
            "进度",
            "状态",
        ],
        # 交互/沟通场景
        "communication": [
            "你好",
            "谢谢",
            "请问",
            "帮助",
            "问题",
            "回答",
            "解释",
            "说明",
            "理解",
            "明白",
            "清楚",
            "知道",
            "学习",
            "请教",
            "指导",
            "建议",
            "意见",
            "反馈",
            "评价",
            "满意",
        ],
        # 易经/文化场景
        "yijing": [
            "易经",
            "卦象",
            "六爻",
            "八卦",
            "五行",
            "三才",
            "阴阳",
            "乾坤",
            "风水",
            "占卜",
            "算命",
            "命运",
            "运势",
            "气运",
            "天干",
            "地支",
            "生肖",
            "周易",
            "太极",
        ],
        # 情感/态度场景
        "emotion": [
            "高兴",
            "快乐",
            "满意",
            "喜欢",
            "爱",
            "讨厌",
            "生气",
            "难过",
            "担心",
            "害怕",
            "紧张",
            "兴奋",
            "期待",
            "失望",
            "遗憾",
            "感谢",
            "抱歉",
            "对不起",
            "没关系",
        ],
        # 动作/行为场景
        "action": [
            "做",
            "干",
            "弄",
            "搞",
            "写",
            "读",
            "看",
            "听",
            "说",
            "问",
            "答",
            "想",
            "思考",
            "分析",
            "研究",
            "讨论",
            "分享",
            "发送",
            "接收",
            "下载",
            "上传",
        ],
    }

    # 停用词（降低噪声）
    STOP_WORDS = {
        "的",
        "了",
        "是",
        "在",
        "我",
        "有",
        "和",
        "就",
        "不",
        "人",
        "都",
        "一",
        "一个",
        "上",
        "也",
        "很",
        "到",
        "说",
        "要",
        "去",
        "你",
        "会",
        "着",
        "没有",
        "看",
        "好",
        "自己",
        "这",
    }

    def __init__(self):
        self.feature_dim = len(self.BUSINESS_SEMANTICS)
        self.category_weights = {
            "tech": 1.2,
            "memory": 1.5,
            "business": 1.0,
            "temporal": 0.8,
            "communication": 0.9,
            "yijing": 1.3,
            "emotion": 0.7,
            "action": 0.8,
        }

    def extract(self, text: str) -> Dict[str, float]:
        """
        提取文本的语义特征

        返回每个语义类别的匹配度 (0-1)
        """
        text_lower = text.lower()
        features = {}

        for category, keywords in self.BUSINESS_SEMANTICS.items():
            # 计算关键词匹配数
            match_count = sum(1 for kw in keywords if kw in text_lower)

            # 计算匹配度（考虑类别权重）
            weight = self.category_weights.get(category, 1.0)
            raw_score = min(1.0, match_count / max(5, len(keywords) * 0.3))

            features[category] = raw_score * weight

        return features

    def to_vector(self, features: Dict[str, float]) -> List[float]:
        """将特征字典转换为向量"""
        return [features.get(cat, 0.0) for cat in self.BUSINESS_SEMANTICS.keys()]

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def keyword_overlap_score(self, text1: str, text2: str) -> float:
        """
        计算关键词重叠度

        更直接的语义相似度计算
        """

        # 提取有效词
        def extract_words(text):
            words = re.findall(r"[\u4e00-\u9fa5]{2,}|[a-zA-Z]{3,}", text.lower())
            return [w for w in words if w not in self.STOP_WORDS]

        words1 = set(extract_words(text1))
        words2 = set(extract_words(text2))

        if not words1 or not words2:
            return 0.0

        # Jaccard 相似度
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def extend_keywords(self, category: str, keywords: List[str]):
        """
        动态扩展关键词库

        Args:
            category: 关键词类别
            keywords: 新关键词列表
        """
        if category not in self.BUSINESS_SEMANTICS:
            self.BUSINESS_SEMANTICS[category] = []

        # 添加新关键词并去重
        new_keywords = [kw for kw in keywords if kw not in self.BUSINESS_SEMANTICS[category]]
        if new_keywords:
            self.BUSINESS_SEMANTICS[category].extend(new_keywords)
            self.BUSINESS_SEMANTICS[category] = list(set(self.BUSINESS_SEMANTICS[category]))

            # 重新计算特征维度
            self.feature_dim = len(self.BUSINESS_SEMANTICS)

            # 记录到全局状态
            global_state = get_global_state()
            global_state.add_keyword_extension(category, new_keywords)

    def learn_from_text(self, text: str, category: str = None):
        """
        从文本中学习新关键词

        Args:
            text: 文本内容
            category: 目标类别（None时自动分类）
        """

        # 提取有效词
        def extract_words(text):
            words = re.findall(r"[\u4e00-\u9fa5]{2,}|[a-zA-Z]{3,}", text.lower())
            return [w for w in words if w not in self.STOP_WORDS]

        new_words = extract_words(text)

        if category:
            # 添加到指定类别
            self.extend_keywords(category, new_words)
        else:
            # 自动分类（选择匹配度最高的类别）
            if new_words:
                best_category = None
                best_score = 0

                for cat in self.BUSINESS_SEMANTICS:
                    # 计算与现有关键词的相似度
                    cat_keywords = self.BUSINESS_SEMANTICS[cat]
                    overlap = sum(1 for word in new_words if any(kw in word or word in kw for kw in cat_keywords))
                    score = overlap / max(len(cat_keywords), 1)

                    if score > best_score:
                        best_score = score
                        best_category = cat

                if best_category and best_score > 0.1:
                    self.extend_keywords(best_category, new_words)


# ============================================================================
# 六爻编码器 - 基于语义的六爻生成
# ============================================================================


class SemanticYaoEncoder:
    """
    语义六爻编码器

    根据文本语义特征生成六爻编码
    """

    # 六爻对应的语义维度
    YAO_DIMENSIONS = {
        0: ["紧急", "重要", "关键", "优先", "必须", "立即", "马上", "赶快", "急需", "迫切"],  # 初爻：紧急程度
        1: ["长期", "永久", "持续", "稳定", "固定", "常规", "一直", "总是", "经常", "习惯"],  # 二爻：时间跨度
        2: ["交互", "对话", "沟通", "交流", "反馈", "响应", "回复", "回答", "提问", "讨论"],  # 三爻：交互性
        3: ["任务", "项目", "工作", "执行", "操作", "处理", "完成", "进行", "实施", "落实"],  # 四爻：任务性
        4: ["规则", "制度", "流程", "标准", "规范", "要求", "必须", "应该", "需要", "规定"],  # 五爻：规范性
        5: ["知识", "经验", "智慧", "总结", "沉淀", "积累", "学习", "理解", "掌握", "技能"],  # 上爻：知识性
    }

    def encode(self, text: str) -> List[int]:
        """
        根据文本内容生成六爻编码

        返回：[初爻，二爻，三爻，四爻，五爻，上爻]，每爻 0=阴 1=阳
        """
        text_lower = text.lower()
        hexagram = []

        for yao_pos, keywords in self.YAO_DIMENSIONS.items():
            match_count = sum(1 for kw in keywords if kw in text_lower)
            # 匹配度超过阈值则为阳爻（降低阈值增加区分度）
            hexagram.append(1 if match_count >= 1 else 0)

        return hexagram


# ============================================================================
# 五行气运调度器
# ============================================================================


class WuxingQiScheduler:
    """
    五行气运调度器

    根据时间和记忆属性计算气运权重
    """

    # 五行对应的时间段（小时）
    WUXING_HOURS = {
        "mu": (3, 7),  # 木：寅卯时 (3-7 点)
        "huo": (9, 13),  # 火：巳午时 (9-13 点)
        "tu": (7, 9),  # 土：辰时 (7-9 点) + 未时 (13-15 点)
        "jin": (15, 19),  # 金：申酉时 (15-19 点)
        "shui": (19, 23),  # 水：戌亥时 (19-23 点) + 子丑时 (23-3 点)
    }

    # 五行相生关系
    WUXING_GENERATE = {
        "mu": "huo",
        "huo": "tu",
        "tu": "jin",
        "jin": "shui",
        "shui": "mu",
    }

    def get_current_wuxing(self) -> str:
        """获取当前时辰的五行"""
        hour = datetime.now().hour
        for wuxing, (start, end) in self.WUXING_HOURS.items():
            if start <= hour < end:
                return wuxing
        return "tu"  # 默认土

    def calculate_qi_weight(self, memory_wuxing: str) -> float:
        """
        计算记忆五行与当前时辰的气运权重

        相生：1.2
        相同：1.0
        相克：0.8
        """
        current = self.get_current_wuxing()

        if memory_wuxing == current:
            return 1.0  # 当令

        if self.WUXING_GENERATE.get(memory_wuxing) == current:
            return 1.2  # 生我者旺

        if self.WUXING_GENERATE.get(current) == memory_wuxing:
            return 0.8  # 我生者泄

        return 0.9  # 其他情况


# ============================================================================
# 综合检索引擎
# ============================================================================


class EnhancedRetrievalEngine:
    """
    增强检索引擎

    综合评分 = 语义相似度 × 卦象匹配度 × 三才权重 × 气运权重 × 时间权重 × 热度
    """

    def __init__(self):
        self.semantic_extractor = SemanticFeatureExtractor()
        self.yao_encoder = SemanticYaoEncoder()
        self.wuxing_scheduler = WuxingQiScheduler()

        # 权重配置（商用最终版 - 符合人类记忆节奏）
        # 语义 > 向量 > 层级 > 时间 > 热度 > 卦象
        self.weights = {
            "semantic": 0.40,  # 语义相似度 40% - 核心
            "vector": 0.25,  # 向量相似度 25% - 重要补充
            "layer": 0.15,  # 三才层级权重 15% - 重要信息优先
            "time_decay": 0.10,  # 时间衰减权重 10% - 自然遗忘
            "hot_score": 0.05,  # 历史热度分数 5% - 辅助
            "hexagram": 0.05,  # 卦象匹配度 5% - 特色增强
        }

        # 记忆层级基础权重（core层记忆优先级高）
        self.layer_base_weight = {
            "core": 1.8,  # core层：核心记忆，权重最高
            "normal": 1.2,  # normal层：普通记忆，权重中等
            "short": 1.0,  # short层：短期记忆，权重适中
            "archive": 0.9,  # archive层：归档记忆，权重较低
            "temp": 0.8,  # temp层：临时记忆，权重最低
        }

    def calculate_hexagram_similarity(self, hex1: List[int], hex2: List[int]) -> float:
        """计算六爻相似度（汉明距离）"""
        if len(hex1) != 6 or len(hex2) != 6:
            return 0.0

        matches = sum(1 for a, b in zip(hex1, hex2) if a == b)
        return matches / 6

    def calculate_layer_priority(self, query_layer: str, memory_layer: str) -> float:
        """
        计算记忆层级优先级

        同层优先，相邻层次之
        """
        layer_order = {"temp": 0, "short": 1, "normal": 2, "archive": 3, "core": 4}

        q_order = layer_order.get(query_layer, 2)
        m_order = layer_order.get(memory_layer, 2)

        diff = abs(q_order - m_order)

        if diff == 0:
            return 1.0  # 同层
        elif diff == 1:
            return 0.7  # 相邻层
        elif diff == 2:
            return 0.5  # 隔一层
        else:
            return 0.3  # 跨多层

    def calculate_time_weight(
        self, created_at: datetime, layer: str, hot_score: float, last_accessed_at: datetime = None
    ) -> float:
        """
        计算时间权重（商用最终版 - 基于最后访问时间）

        关键改进：使用 last_accessed_at 而非 created_at
        - 每次记忆被检索/使用时，刷新 last_accessed_at
        - 从最后访问时间重新计算衰减
        - 符合人类记忆规律：越回忆，记得越牢

        衰减速率：
        - tian: 0.0002 (半衰期≈144 天) - 核心记忆，几乎不忘
        - ren: 0.01 (半衰期≈3 天) - 普通对话，适度遗忘
        - di: 0.04 (半衰期≈18 小时) - 闲聊废话，快速淡化

        返回：0.2-2.0 之间的权重值
        """
        now = datetime.now()

        # 优先使用 last_accessed_at（最后访问时间）
        # 如果没有，则使用 created_at（创建时间）
        if last_accessed_at:
            dt = last_accessed_at
        elif isinstance(created_at, datetime):
            dt = created_at
        else:
            try:
                dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                dt = now

        hours_old = max(0, (now - dt).total_seconds() / 3600)

        # 商用最终版衰减速率（基于最后访问时间）
        decay_rates = {
            "core": 0.0002,  # 核心记忆：半衰期≈144 天，几乎不忘
            "normal": 0.003,  # 普通记忆：半衰期≈10 天，长假回来还能记得
            "short": 0.01,  # 短期记忆：半衰期≈3 天，短期重要
            "archive": 0.001,  # 归档记忆：半衰期≈30 天，不常用但有价值
            "temp": 0.04,  # 临时记忆：半衰期≈18 小时，快速淡化
        }

        decay_rate = decay_rates.get(layer, 0.01)

        # 基础时间衰减（指数衰减）
        time_factor = math.exp(-decay_rate * hours_old)

        # 热度加成（高热度记忆衰减慢）
        hot_bonus = 0.3 * hot_score  # 最高 +0.3

        # 综合时间权重（范围 0.2-2.0）
        time_weight = (time_factor + hot_bonus) * self.layer_base_weight.get(layer, 1.0)

        # 限制范围
        return max(0.2, min(2.0, time_weight))

    def _parse_hexagram(self, hex_data) -> List[int]:
        """解析六爻数据"""
        if isinstance(hex_data, str):
            try:
                return [int(c) for c in hex_data]
            except (ValueError, TypeError):
                return [0, 0, 0, 0, 0, 0]
        elif isinstance(hex_data, list):
            return [int(c) for c in hex_data]
        return [0, 0, 0, 0, 0, 0]

    def _get_gua_info(self, hexagram: List[int]) -> Tuple[List[int], str, str, str]:
        """根据六爻获取卦象信息"""
        yang_count = sum(hexagram)

        if yang_count >= 5:
            return hexagram, "qian", "jin", "tien"
        elif yang_count >= 4:
            return hexagram, "li", "huo", "ren"
        elif yang_count >= 2:
            return hexagram, "kan", "shui", "ren"
        else:
            return hexagram, "kun", "tu", "di"

    def extend_keywords(self, category: str, keywords: List[str]):
        """
        动态扩展词库

        Args:
            category: 关键词类别
            keywords: 新关键词列表
        """
        self.semantic_extractor.extend_keywords(category, keywords)

    def learn_from_text(self, text: str, category: str = None):
        """
        从文本中学习新关键词

        Args:
            text: 文本内容
            category: 目标类别（None时自动分类）
        """
        self.semantic_extractor.learn_from_text(text, category)

    def search(self, query: str, memories: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        执行综合检索

        Args:
            query: 查询文本
            memories: 候选记忆列表
            top_k: 返回数量

        Returns:
            按综合评分排序的记忆列表
        """
        # 1. 提取查询特征
        query_features = self.semantic_extractor.extract(query)
        query_vector = self.semantic_extractor.to_vector(query_features)
        query_hexagram = self.yao_encoder.encode(query)
        _, _, _, query_layer = self._get_gua_info(query_hexagram)

        scored_memories = []

        for idx, mem in enumerate(memories):
            content = mem.get("content", "")

            # 2. 语义相似度（余弦相似度）
            mem_features = self.semantic_extractor.extract(content)
            mem_vector = self.semantic_extractor.to_vector(mem_features)
            semantic_sim = self.semantic_extractor.cosine_similarity(query_vector, mem_vector)

            # 3. 关键词重叠度（更直接的匹配）
            keyword_overlap = self.semantic_extractor.keyword_overlap_score(query, content)

            # 4. 卦象匹配度
            mem_hexagram = self._parse_hexagram(mem.get("hexagram", [0, 0, 0, 0, 0, 0]))
            hexagram_sim = self.calculate_hexagram_similarity(query_hexagram, mem_hexagram)

            # 5. 三才层级优先级
            layer_priority = self.calculate_layer_priority(query_layer, mem.get("sancai_layer", "di"))

            # 6. 时间权重（基于最后访问时间 - 越回忆越牢）
            created_at = mem.get("created_at")
            last_accessed_at = mem.get("last_accessed_at")
            layer = mem.get("sancai_layer", "di")
            hot_score = mem.get("hot_score", 0.5)

            # 解析 last_accessed_at
            if last_accessed_at:
                try:
                    last_accessed_dt = datetime.fromisoformat(last_accessed_at.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    last_accessed_dt = None
            else:
                last_accessed_dt = None

            time_weight = self.calculate_time_weight(created_at, layer, hot_score, last_accessed_dt)

            # 7. 卦象匹配度
            wuxing_qi = self.wuxing_scheduler.calculate_qi_weight(mem.get("wuxing", "tu"))

            # 8. 综合评分（商用最终版）
            # 语义 40% + 向量 25% + 层级 15% + 时间 10% + 热度 5% + 卦象 5%
            composite_score = (
                semantic_sim * self.weights["semantic"]
                + keyword_overlap * self.weights["vector"]
                + layer_priority * self.weights["layer"]
                + time_weight * self.weights["time_decay"]
                + hot_score * self.weights["hot_score"]
                + hexagram_sim * self.weights["hexagram"]
            )

            # 添加微小扰动确保绝对差异化（基于记忆 ID）
            memory_id = mem.get("memory_id", mem.get("id", str(idx)))
            micro_adjust = hash(memory_id) % 100 / 10000.0  # 0-0.01 的微小调整
            composite_score += micro_adjust

            scored_memories.append(
                {
                    **mem,
                    "similarity": round(composite_score, 4),
                    "semantic_similarity": round(semantic_sim, 4),
                    "vector_similarity": round(keyword_overlap, 4),  # 关键词重叠作为向量相似度
                    "layer_priority": round(layer_priority, 4),
                    "time_weight": round(time_weight, 4),
                    "hexagram_similarity": round(hexagram_sim, 4),
                    "hot_score": hot_score,
                }
            )

        # 8. 按综合评分降序排序
        scored_memories.sort(key=lambda x: x["similarity"], reverse=True)

        return scored_memories[:top_k]


# ============================================================================
# 单例
# ============================================================================

_retrieval_engine = None


def get_retrieval_engine() -> EnhancedRetrievalEngine:
    global _retrieval_engine
    if _retrieval_engine is None:
        _retrieval_engine = EnhancedRetrievalEngine()
    return _retrieval_engine
