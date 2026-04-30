#!/usr/bin/env python3
"""
🔮 智能记忆层级分类器

功能：
- 自动判断记忆应存入 core/normal/short/archive/temp 哪一层
- 基于关键词 + 语义规则
- 符合人类记忆规律：重要不忘、日常记住、废话淡化
"""

import logging
import re
from typing import Dict, List, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# 分层规则关键词库
# ============================================================================

# core层关键词（核心永久记忆，144天）
CORE_KEYWORDS = {
    # 用户身份信息
    "identity": [
        "我叫",
        "姓名",
        "名字",
        "昵称",
        "称呼",
        "身份",
        "职位",
        "职业",
        "角色",
        "家人",
        "父母",
        "孩子",
        "爱人",
        "朋友",
        "同事",
        "老板",
        "客户",
    ],
    # 长期习惯偏好
    "preference": [
        "喜欢",
        "爱好",
        "习惯",
        "偏好",
        "厌恶",
        "讨厌",
        "禁忌",
        "不吃",
        "不喝",
        "经常",
        "总是",
        "一直",
        "从不",
        "很少",
        "通常",
        "一般",
    ],
    # 重要目标规划
    "goal": [
        "目标",
        "规划",
        "计划",
        "梦想",
        "愿景",
        "理想",
        "追求",
        "想要",
        "希望",
        "长期",
        "永久",
        "永远",
        "一生",
        "这辈子",
        "未来",
    ],
    # 系统规则指令
    "rule": [
        "规则",
        "制度",
        "标准",
        "规范",
        "流程",
        "机制",
        "原理",
        "架构",
        "核心",
        "指令",
        "要求",
        "必须",
        "一定",
        "禁止",
        "不要",
        "不能",
        "务必",
        "底层",
        "基础",
        "本质",
        "关键",
        "设计",
        "方法论",
        "价值观",
        "原则",
    ],
    # 关键信息
    "critical": [
        "账号",
        "密码",
        "电话",
        "邮箱",
        "地址",
        "联系方式",
        "身份证",
        "银行卡",
        "公司",
        "单位",
        "学校",
        "部门",
        "团队",
        "组织",
    ],
    # 明确记忆指令
    "remember": [
        "记住",
        "记得",
        "别忘了",
        "不要忘记",
        "一定记得",
        "永远记得",
        "帮我记",
        "记下来",
        "记录下",
        "保存",
        "收藏",
        "重要",
        "关键",
        "核心",
    ],
}

# normal层关键词（普通记忆，10天）
NORMAL_KEYWORDS = {
    # 具体事件
    "event": [
        "会议",
        "约会",
        "见面",
        "聚餐",
        "活动",
        "出差",
        "旅行",
        "考察",
        "参观",
        "今天",
        "明天",
        "后天",
        "昨天",
        "本周",
        "下周",
        "本月",
        "近期",
        "最近",
    ],
    # 工作项目
    "work": [
        "项目",
        "需求",
        "方案",
        "报告",
        "文档",
        "代码",
        "功能",
        "模块",
        "接口",
        "API",
        "系统",
        "平台",
        "产品",
        "服务",
        "客户",
        "合同",
        "订单",
    ],
    # 问题修复
    "issue": [
        "问题",
        "bug",
        "错误",
        "故障",
        "异常",
        "修复",
        "解决",
        "处理",
        "调试",
        "测试",
        "检查",
        "排查",
        "优化",
        "改进",
        "升级",
        "更新",
    ],
    # 有上下文价值
    "context": [
        "因为",
        "所以",
        "但是",
        "然而",
        "因此",
        "由于",
        "如果",
        "那么",
        "否则",
        "虽然",
        "尽管",
        "即使",
        "只要",
        "只有",
        "除非",
        "既然",
    ],
}

# short层关键词（短期记忆，3天）
SHORT_KEYWORDS = {
    # 临时任务待办
    "task": [
        "任务",
        "待办",
        "todo",
        "要做",
        "需要",
        "安排",
        "计划",
        "准备",
        "完成",
        "开始",
        "结束",
        "进度",
        "截止",
        "deadline",
        "提交",
        "交付",
    ],
    # 短期约定
    "agreement": ["约定", "答应", "承诺", "保证", "约好", "说好", "确定", "安排", "计划"],
    # 临时偏好
    "temp_preference": ["暂时", "临时", "短期", "近期", "这几天", "这段时间", "目前", "现在"],
}

# archive层关键词（归档记忆，30天）
ARCHIVE_KEYWORDS = {
    # 历史记录
    "history": ["历史", "过去", "以前", "曾经", "之前", "上次", "上回", "去年", "前年"],
    # 不常用但有价值
    "valuable": ["资料", "文档", "记录", "档案", "备份", "存档", "保存", "收藏"],
}

# temp层关键词（临时闲聊，18小时）
TEMP_KEYWORDS = {
    # 纯问候
    "greeting": [
        "你好",
        "您好",
        "嗨",
        "hello",
        "hi",
        "在吗",
        "在嘛",
        "干嘛呢",
        "吃了吗",
        "早上好",
        "中午好",
        "晚上好",
        "晚安",
        "拜拜",
        "再见",
    ],
    # 纯情绪
    "emotion": [
        "哈哈",
        "嘿嘿",
        "嘻嘻",
        "呵呵",
        "哇",
        "哦",
        "噢",
        "嗯",
        "呃",
        "哎",
        "啊",
        "呀",
        "嘛",
        "啦",
        "咯",
        "喽",
        "呗",
        "呐",
    ],
    # 敷衍回复
    "lazy": [
        "好的",
        "收到",
        "ok",
        "okay",
        "行",
        "可以",
        "没问题",
        "知道了",
        "了解",
        "明白",
        "清楚",
        "对的",
        "是的",
        "没错",
        "嗯嗯",
        "好的好的",
    ],
    # 无意义水聊
    "water": [
        "随便",
        "看看",
        "问问",
        "聊聊",
        "扯淡",
        "吹牛",
        "八卦",
        "无聊",
        "没事",
        "闲着",
        "打发时间",
        "消磨时间",
        "天气",
        "吃饭",
        "睡觉",
        "起床",
    ],
}

# ============================================================================
# 智能层级分类器
# ============================================================================


class MemoryLayerClassifier:
    """
    记忆层级智能分类器

    规则优先级：tian > ren > di
    """

    def __init__(self):
        self.core_keywords = CORE_KEYWORDS
        self.normal_keywords = NORMAL_KEYWORDS
        self.short_keywords = SHORT_KEYWORDS
        self.archive_keywords = ARCHIVE_KEYWORDS
        self.temp_keywords = TEMP_KEYWORDS

        # 特殊模式识别
        self.patterns = {
            "name": r"(我叫 | 名字是 | 称呼我 | 喊我)[\u4e00-\u9fa5]{2,6}",
            "contact": r"(\d{11}|\d{3,4}-\d{7,8}|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            "time_specific": r"(今天 | 明天 | 后天 | 昨天|本周 | 下周|本月 | 下月|近期 | 最近)",
        }

    def classify(self, content: str, memory_type: str = None) -> Tuple[str, Dict]:
        """
        分类记忆层级

        Args:
            content: 记忆内容
            memory_type: 记忆类型（可选）

        Returns:
            (layer, details)
            - layer: 'core' | 'normal' | 'short' | 'archive' | 'temp'
            - details: 分类依据详情
        """
        content_lower = content.lower()

        # 1. 检查core层规则（高权重关键词 1 个即可）
        core_matches = self._check_layer(content_lower, self.core_keywords)

        # 高权重分类：身份、偏好、目标、记忆指令
        high_weight_cats = {"identity", "preference", "goal", "remember"}
        has_high_weight = bool(set(core_matches["categories"]) & high_weight_cats)

        if core_matches["total"] >= 1 and (has_high_weight or core_matches["total"] >= 2):
            return "core", {
                "layer": "core",
                "reason": "匹配core层规则",
                "matched_categories": core_matches["categories"],
                "confidence": min(1.0, core_matches["total"] / 3),
            }

        # 2. 检查特殊模式（用户名、联系方式等）
        special_match = self._check_special_patterns(content)
        if special_match:
            return "core", {
                "layer": "core",
                "reason": special_match,
                "matched_categories": ["special"],
                "confidence": 1.0,
            }

        # 3. 检查记忆指令关键词
        if self._has_remember_instruction(content_lower):
            return "core", {
                "layer": "core",
                "reason": "包含明确记忆指令",
                "matched_categories": ["remember"],
                "confidence": 0.9,
            }

        # 4. 检查archive层规则（历史记录）
        archive_matches = self._check_layer(content_lower, self.archive_keywords)
        if archive_matches["total"] > 0:
            return "archive", {
                "layer": "archive",
                "reason": "匹配archive层规则",
                "matched_categories": archive_matches["categories"],
                "confidence": min(1.0, archive_matches["total"] / 2),
            }

        # 5. 检查normal层规则（工作、事件等）
        normal_matches = self._check_layer(content_lower, self.normal_keywords)
        if normal_matches["total"] > 0:
            return "normal", {
                "layer": "normal",
                "reason": "匹配normal层规则",
                "matched_categories": normal_matches["categories"],
                "confidence": min(1.0, normal_matches["total"] / 2),
            }

        # 6. 检查short层规则（临时任务、短期约定）
        short_matches = self._check_layer(content_lower, self.short_keywords)
        if short_matches["total"] > 0:
            return "short", {
                "layer": "short",
                "reason": "匹配short层规则",
                "matched_categories": short_matches["categories"],
                "confidence": min(1.0, short_matches["total"] / 2),
            }

        # 7. 检查temp层规则（纯闲聊）
        temp_matches = self._check_layer(content_lower, self.temp_keywords)
        if temp_matches["total"] >= 2:  # 2 个以上闲聊关键词 → temp层
            return "temp", {
                "layer": "temp",
                "reason": "纯闲聊内容",
                "matched_categories": temp_matches["categories"],
                "confidence": min(1.0, temp_matches["total"] / 3),
            }

        # 8. 默认规则：根据内容长度和 memory_type 判断
        return self._default_classification(content, memory_type)

    def _check_layer(self, content: str, keywords: Dict[str, List[str]]) -> Dict:
        """检查某一层级的关键词匹配"""
        matches = {
            "total": 0,
            "categories": [],
        }

        for category, words in keywords.items():
            matched_words = [w for w in words if w in content]
            if matched_words:
                matches["categories"].append(category)
                matches["total"] += len(matched_words)

        return matches

    def _check_special_patterns(self, content: str) -> str:
        """检查特殊模式（用户名、联系方式等）"""
        # 检查用户名
        if re.search(self.patterns["name"], content):
            return "包含用户姓名"

        # 检查联系方式
        if re.search(self.patterns["contact"], content):
            return "包含联系方式"

        return None

    def _has_remember_instruction(self, content: str) -> bool:
        """检查是否包含明确记忆指令（排除日常用法）"""
        # 需要前后文的"记住"才是记忆指令
        remember_patterns = [
            r"记住 (我的 | 这 | 这个 | 那 | 那个)",
            r"记得 (帮我 | 要 | 一定 | 千万)",
            r"别忘了",
            r"不要忘记",
            r"一定记得",
            r"永远记得",
            r"帮我记 (住 | 下来)",
            r"记下来",
            r"记录下",
            r"(这个 | 那 | 这) 很重要",
            r"(这个 | 那 | 这) 是关键",
        ]

        for pattern in remember_patterns:
            if re.search(pattern, content):
                return True

        return False

    def _default_classification(self, content: str, memory_type: str = None) -> Tuple[str, Dict]:
        """默认分类规则"""
        content_len = len(content)

        # 根据 memory_type 判断
        if memory_type:
            type_mapping = {
                "long_term": "core",
                "permanent": "core",
                "knowledge": "core",
                "medium_term": "normal",
                "task": "short",
                "project": "normal",
                "short_term": "short",
                "chat": "temp",
                "temp": "temp",
                "archive": "archive",
            }
            if memory_type in type_mapping:
                layer = type_mapping[memory_type]
                return layer, {
                    "layer": layer,
                    "reason": f"根据 memory_type: {memory_type}",
                    "matched_categories": ["default"],
                    "confidence": 0.7,
                }

        # 根据内容长度判断
        if content_len > 100:
            # 长内容更可能是重要信息
            return "normal", {
                "layer": "normal",
                "reason": "内容较长，可能是重要信息",
                "matched_categories": ["default"],
                "confidence": 0.6,
            }
        elif content_len < 10:
            # 超短内容很可能是闲聊
            return "temp", {
                "layer": "temp",
                "reason": "内容过短，可能是闲聊",
                "matched_categories": ["default"],
                "confidence": 0.5,
            }

        # 默认归入normal层（平衡策略）
        return "normal", {
            "layer": "normal",
            "reason": "默认分类",
            "matched_categories": ["default"],
            "confidence": 0.5,
        }


# ============================================================================
# 集成到易经编码流程
# ============================================================================


def classify_and_encode(content: str, memory_type: str = None) -> Tuple:
    """
    智能分类 + 卦象编码

    Returns:
        (hex_arr, bagua, wuxing, layer)
    """
    from core.core_yijing import content_to_hexagram_rule

    # 1. 智能分类层级
    classifier = MemoryLayerClassifier()
    layer, details = classifier.classify(content, memory_type)

    # 2. 获取卦象编码（直接调用底层规则，避免递归循环）
    hex_arr, bagua, wuxing, _, _ = content_to_hexagram_rule(content, memory_type or "chat")

    # 3. 使用智能分类的层级（覆盖原有层级）
    return hex_arr, bagua, wuxing, layer


# ============================================================================
# 单例
# ============================================================================

_classifier = None


def get_classifier() -> MemoryLayerClassifier:
    global _classifier
    if _classifier is None:
        _classifier = MemoryLayerClassifier()
    return _classifier


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🔮 智能记忆层级分类器 - 测试")
    logger.info("=" * 70)

    classifier = get_classifier()

    # 测试用例
    test_cases = [
        # core层测试
        ("我叫张三，以后叫我小张就行", "core"),
        ("我喜欢吃川菜，不喜欢吃辣", "core"),
        ("记住我的生日是 1990 年 1 月 1 日", "core"),
        ("系统架构设计文档 - 核心记忆", "core"),
        ("我的电话是 13800138000", "core"),
        # normal层测试
        ("今天下午 3 点开会讨论项目进度", "normal"),
        ("客户合同模板需要更新，请提供最新版本", "normal"),
        ("这个 bug 需要在本周内修复", "normal"),
        # short层测试
        ("明天记得提交周报", "short"),
        # temp层测试
        ("你好", "temp"),
        ("哈哈，好的", "temp"),
        ("吃了吗？随便聊聊", "temp"),
        ("嗯嗯，知道了", "temp"),
        # archive层测试
        ("去年的项目文档存档", "archive"),
        ("历史记录需要备份", "archive"),
    ]

    correct = 0
    total = len(test_cases)

    for content, expected in test_cases:
        layer, details = classifier.classify(content)
        status = "✅" if layer == expected else "❌"
        if layer == expected:
            correct += 1

        logger.info(f"\n{status} 内容：{content[:40]}...")
        logger.info(f"   预期：{expected} | 实际：{layer} | 置信度：{details['confidence']:.2f}")
        logger.info(f"   原因：{details['reason']}")

    logger.info("\n" + "=" * 70)
    logger.info(f"测试准确率：{correct}/{total} = {correct/total*100:.1f}%")
    logger.info("=" * 70)
