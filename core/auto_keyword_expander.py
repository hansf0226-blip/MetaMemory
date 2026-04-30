#!/usr/bin/env python3
"""
🔮 业务词库自动扩展引擎

功能：
1. 从对话内容自动识别行业关键词
2. 自动归类到 8 大业务场景
3. 自动更新 BUSINESS_SEMANTICS 词库
4. 开箱即用，无需人工配置

实现原理：
- 基于 TF-IDF 提取关键词
- 基于语义相似度归类到已有场景
- 新词自动入库，支持持久化存储
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# 日志管理
from logger import get_logger

logger = get_logger("auto_keyword_expander")

# 尝试导入 jieba，失败则使用简单分词
try:
    import jieba

    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("jieba 未安装，使用简单分词模式")

# ============================================================================
# 配置
# ============================================================================

# 词库存储路径
KEYWORD_DB_PATH = Path(__file__).parent.parent / "data" / "keyword_db.json"

# 中文停用词
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
    "那",
    "他",
    "她",
    "它",
    "们",
    "这个",
    "那个",
    "什么",
    "怎么",
    "可以",
    "没",
    "把",
    "被",
    "让",
    "给",
    "用",
    "以",
    "而",
    "但",
    "如果",
    "因为",
    "所以",
    "虽然",
    "但是",
    "然后",
    "现在",
    "已经",
    "开始",
    "进行",
    "通过",
}

# 初始业务场景词库（种子词）
INITIAL_SEEDS = {
    "tech": {
        "core": [
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
            "故障",
            "重启",
            "上线",
        ],
        "related": [
            "软件",
            "硬件",
            "电脑",
            "手机",
            "应用",
            "平台",
            "框架",
            "语言",
            "编程",
            "技术",
            "工程",
            "架构",
            "设计",
            "实现",
            "运行",
            "安装",
            "升级",
            "更新",
            "维护",
            "监控",
        ],
    },
    "memory": {
        "core": [
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
        "related": [
            "学习",
            "复习",
            "回顾",
            "总结",
            "笔记",
            "文档",
            "资料",
            "信息",
            "内容",
            "文本",
            "语音",
            "图片",
            "视频",
            "文件",
            "数据库",
            "云端",
            "本地",
            "备份",
            "同步",
            "共享",
        ],
    },
    "business": {
        "core": [
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
        ],
        "related": [
            "公司",
            "企业",
            "团队",
            "部门",
            "员工",
            "领导",
            "老板",
            "同事",
            "合作",
            "伙伴",
            "供应商",
            "渠道",
            "代理商",
            "经销商",
            "订单",
            "发票",
            "付款",
            "收款",
            "财务",
            "预算",
        ],
    },
    "temporal": {
        "core": [
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
        "related": [
            "之前",
            "之后",
            "以前",
            "以后",
            "刚才",
            "马上",
            "立刻",
            "尽快",
            "稍后",
            "待会",
            "周末",
            "周一",
            "月初",
            "月底",
            "年初",
            "年底",
            "季度",
            "年度",
            "周期",
            "频率",
        ],
    },
    "communication": {
        "core": [
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
        "related": [
            "你好",
            "您好",
            "再见",
            "拜拜",
            "欢迎",
            "恭喜",
            "抱歉",
            "对不起",
            "没关系",
            "不客气",
            "应该的",
            "没问题",
            "好的",
            "收到",
            "了解",
            "ok",
            "yes",
            "no",
            "maybe",
            "perhaps",
        ],
    },
    "yijing": {
        "core": [
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
            "爻",
        ],
        "related": [
            "变卦",
            "本卦",
            "互卦",
            "错卦",
            "综卦",
            "卦辞",
            "爻辞",
            "象传",
            "彖传",
            "系辞",
            "说卦",
            "序卦",
            "杂卦",
            "河图",
            "洛书",
            "先天",
            "后天",
            "体用",
            "生克",
        ],
    },
    "emotion": {
        "core": [
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
            "情绪",
        ],
        "related": [
            "开心",
            "愉快",
            "幸福",
            "满足",
            "欣慰",
            "激动",
            "感动",
            "温暖",
            "舒适",
            "放松",
            "平静",
            "安心",
            "踏实",
            "郁闷",
            "烦躁",
            "焦虑",
            "压力",
            "疲惫",
            "困倦",
            "无聊",
        ],
    },
    "action": {
        "core": [
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
        "related": [
            "创建",
            "删除",
            "修改",
            "更新",
            "查询",
            "导出",
            "导入",
            "复制",
            "粘贴",
            "剪切",
            "撤销",
            "重做",
            "保存",
            "打开",
            "关闭",
            "启动",
            "停止",
            "暂停",
            "恢复",
            "重置",
        ],
    },
}

# ============================================================================
# 关键词提取器
# ============================================================================


class KeywordExtractor:
    """
    关键词提取器

    使用 TF-IDF + 词性标注提取关键词
    """

    def __init__(self):
        self.idf_cache = {}
        self.loaded = False

    def _load_user_dict(self):
        """加载用户词典到 jieba"""
        if self.loaded:
            return

        # 加载业务种子词（仅当 jieba 可用时）
        if JIEBA_AVAILABLE:
            for category, seeds in INITIAL_SEEDS.items():
                for word_type, words in seeds.items():
                    for word in words:
                        jieba.add_word(word)

        self.loaded = True

    def extract_keywords(self, text: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        提取文本中的关键词

        Args:
            text: 输入文本
            top_k: 返回关键词数量

        Returns:
            [(keyword, score), ...] 按分数降序
        """
        self._load_user_dict()

        # 分词
        if JIEBA_AVAILABLE:
            import jieba  # 局部导入，避免在 jieba 未安装时出错

            words = jieba.lcut(text.lower())
        else:
            # 简单分词：按标点和空格分割
            words = re.findall(r"[\u4e00-\u9fa5]{2,}|[a-zA-Z]{3,}", text.lower())

        # 过滤停用词和非中文/英文词
        valid_words = []
        for word in words:
            word = word.strip()
            if not word or word in STOP_WORDS:
                continue
            # 保留 2 字以上中文词或 3 字母以上英文词
            if re.match(r"^[\u4e00-\u9fa5]{2,}$", word) or re.match(r"^[a-zA-Z]{3,}$", word):
                valid_words.append(word)

        # 计算词频
        word_freq = Counter(valid_words)

        # 计算 TF-IDF 分数
        total_words = len(valid_words)
        keyword_scores = []

        for word, freq in word_freq.most_common(top_k * 2):
            tf = freq / max(1, total_words)
            idf = self._get_idf(word)
            score = tf * idf

            # 长度加权（优先保留有意义的词）
            length_bonus = min(1.5, 1 + len(word) * 0.1)
            score *= length_bonus

            keyword_scores.append((word, score))

        # 按分数排序，取 top_k
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        return keyword_scores[:top_k]

    def _get_idf(self, word: str) -> float:
        """
        获取 IDF 值

        简化版：使用词长和常见度估算
        """
        if word in self.idf_cache:
            return self.idf_cache[word]

        # 词越长，IDF 越高（越罕见）
        base_idf = 1.0 + math.log(len(word) + 1)

        # 常见词 IDF 降低
        common_words = {"这个", "那个", "一些", "很多", "一些", "非常", "特别", "真的", "确实", "一定"}
        if word in common_words:
            base_idf *= 0.5

        self.idf_cache[word] = base_idf
        return base_idf


# ============================================================================
# 词库自动扩展器
# ============================================================================


class AutoKeywordExpander:
    """
    词库自动扩展器

    从对话中自动识别并归类新关键词
    """

    def __init__(self, db_path: Path = KEYWORD_DB_PATH):
        self.db_path = db_path
        self.extractor = KeywordExtractor()
        self.keyword_db = self._load_db()

        # 语义相似度计算（简化版：基于共现词）
        self.category_embeddings = self._build_category_embeddings()

    def _load_db(self) -> Dict:
        """加载词库数据库"""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    db = json.load(f)
                # 合并初始种子词
                for cat, seeds in INITIAL_SEEDS.items():
                    if cat not in db:
                        db[cat] = {"core": [], "related": [], "auto": []}
                    for key in ["core", "related", "auto"]:
                        if key not in db[cat]:
                            db[cat][key] = []
                        # 添加种子词（去重）
                        seed_words = set(seeds.get(key, []))
                        db[cat][key] = list(set(db[cat][key]) | seed_words)
                return db
            except Exception as e:
                logger.warning(f"加载词库失败：{e}，使用初始种子词")

        # 初始化数据库
        db = {}
        for cat, seeds in INITIAL_SEEDS.items():
            db[cat] = {
                "core": list(seeds.get("core", [])),
                "related": list(seeds.get("related", [])),
                "auto": [],  # 自动学习的新词
            }
        return db

    def _save_db(self):
        """保存词库数据库"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.keyword_db, f, ensure_ascii=False, indent=2)

    def _build_category_embeddings(self) -> Dict[str, Set[str]]:
        """为每个类别构建共现词集合（简化版语义表示）"""
        embeddings = {}
        for cat, seeds in self.keyword_db.items():
            all_words = set()
            for word_list in seeds.values():
                all_words.update(word_list)
            embeddings[cat] = all_words
        return embeddings

    def _calculate_category_similarity(self, word: str) -> Dict[str, float]:
        """
        计算词与各业务场景的相似度

        基于词形相似度和共现词重叠度
        """
        similarities = {}

        for cat, cat_words in self.category_embeddings.items():
            # 1. 字面重叠度
            overlap = sum(1 for cw in cat_words if word in cw or cw in word)
            overlap_score = overlap / max(1, len(cat_words))

            # 2. 部首/偏旁相似（简化：只看首字）
            radical_match = 0
            if word and cat_words:
                word_radical = word[0] if word else ""
                for cw in cat_words:
                    if cw and cw[0] == word_radical:
                        radical_match += 1
                radical_score = radical_match / max(1, len(cat_words))
            else:
                radical_score = 0

            # 3. 拼音首字母相似（简化）
            pinyin_match = 0
            # 这里简化处理，实际应该用 pypinyin 库

            # 综合相似度
            similarities[cat] = overlap_score * 0.6 + radical_score * 0.3 + 0.1

        return similarities

    def classify_keyword(self, word: str) -> Tuple[str, float]:
        """
        将新词分类到最匹配的业务场景

        Returns:
            (category, confidence)
        """
        similarities = self._calculate_category_similarity(word)

        best_cat = max(similarities, key=similarities.get)
        confidence = similarities[best_cat]

        return best_cat, confidence

    def process_text(self, text: str, auto_save: bool = True) -> Dict:
        """
        处理文本，自动提取并归类关键词

        Args:
            text: 输入文本
            auto_save: 是否自动保存到词库

        Returns:
            {
                'extracted': [(word, score), ...],
                'classified': {category: [(word, confidence), ...]},
                'new_words': [word, ...]
            }
        """
        # 1. 提取关键词
        keywords = self.extractor.extract_keywords(text, top_k=20)

        # 2. 分类并识别新词
        classified = defaultdict(list)
        new_words = []

        for word, score in keywords:
            category, confidence = self.classify_keyword(word)
            classified[category].append((word, confidence))

            # 检查是否为新词
            is_new = True
            for cat_words in self.keyword_db.values():
                for word_list in cat_words.values():
                    if word in word_list:
                        is_new = False
                        break

            if is_new and confidence > 0.1:  # 置信度阈值
                new_words.append((word, category, confidence))

        # 3. 自动保存新词
        if auto_save and new_words:
            for word, category, confidence in new_words:
                word_type = "core" if confidence > 0.5 else "auto"
                if word not in self.keyword_db[category][word_type]:
                    self.keyword_db[category][word_type].append(word)
            self._save_db()

        return {
            "extracted": keywords,
            "classified": dict(classified),
            "new_words": new_words,
        }

    def get_business_semantics(self) -> Dict[str, List[str]]:
        """
        获取完整的业务语义词库（用于检索引擎）

        Returns:
            {category: [keywords...], ...}
        """
        result = {}
        for cat, words_dict in self.keyword_db.items():
            all_words = []
            for word_list in words_dict.values():
                all_words.extend(word_list)
            result[cat] = list(set(all_words))
        return result

    def get_stats(self) -> Dict:
        """获取词库统计信息"""
        stats = {
            "total_categories": len(self.keyword_db),
            "total_words": 0,
            "auto_learned": 0,
            "by_category": {},
        }

        for cat, words_dict in self.keyword_db.items():
            cat_total = sum(len(words) for words in words_dict.values())
            cat_auto = len(words_dict.get("auto", []))

            stats["total_words"] += cat_total
            stats["auto_learned"] += cat_auto
            stats["by_category"][cat] = {
                "total": cat_total,
                "auto": cat_auto,
            }

        return stats


# ============================================================================
# 单例
# ============================================================================

_expander = None


def get_expander() -> AutoKeywordExpander:
    global _expander
    if _expander is None:
        _expander = AutoKeywordExpander()
    return _expander


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🔮 业务词库自动扩展引擎 - 测试")
    logger.info("=" * 70)

    expander = get_expander()

    # 测试文本
    test_texts = [
        "今天修复了一个严重的 bug，系统终于正常运行了",
        "客户合同模板需要更新，请提供最新版本",
        "这个项目的进度太慢了，需要加快开发速度",
        "易经卦象编码算法需要优化，提升检索准确率",
    ]

    for text in test_texts:
        logger.info(f"\n输入：{text}")
        logger.info("-" * 70)

        result = expander.process_text(text, auto_save=True)

        logger.info(f"提取关键词：{[w for w, s in result['extracted'][:5]]}")
        logger.info(f"新词学习：{result['new_words']}")

    # 统计信息
    logger.info("\n" + "=" * 70)
    logger.info("📊 词库统计")
    logger.info("=" * 70)
    stats = expander.get_stats()
    logger.info(f"业务场景数：{stats['total_categories']}")
    logger.info(f"总词数：{stats['total_words']}")
    logger.info(f"自动学习：{stats['auto_learned']}")

    for cat, cat_stats in stats["by_category"].items():
        logger.info(f"  {cat}: {cat_stats['total']} 词 (自动：{cat_stats['auto']})")

    logger.info("=" * 70)
