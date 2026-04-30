#!/usr/bin/env python3
"""
🔮 A/B 测试引擎

功能：
1. 对比三种检索方式的效果
   - 传统关键词检索
   - 纯向量检索
   - 易经特征增强检索
2. 输出准确率、召回率、响应速度对比
3. 用于商业演示和客户验证

测试场景：
- 语义相关性测试
- 时间敏感性测试
- 多轮对话上下文测试
- 行业术语识别测试
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# ============================================================================
# 测试数据集
# ============================================================================

# 标准测试查询集（带预期结果）
TEST_QUERIES = [
    {
        "id": "q001",
        "query": "系统 bug 修复",
        "expected_keywords": ["bug", "修复", "系统", "错误", "故障"],
        "expected_category": "tech",
        "description": "技术场景检索",
    },
    {
        "id": "q002",
        "query": "客户合同模板",
        "expected_keywords": ["合同", "模板", "客户", "文档"],
        "expected_category": "business",
        "description": "业务场景检索",
    },
    {
        "id": "q003",
        "query": "记忆保存和检索",
        "expected_keywords": ["记忆", "保存", "检索", "存储"],
        "expected_category": "memory",
        "description": "记忆功能检索",
    },
    {
        "id": "q004",
        "query": "项目进度安排",
        "expected_keywords": ["项目", "进度", "计划", "安排"],
        "expected_category": "business",
        "description": "项目管理检索",
    },
    {
        "id": "q005",
        "query": "易经卦象编码",
        "expected_keywords": ["易经", "卦象", "编码", "六爻"],
        "expected_category": "yijing",
        "description": "易经专业检索",
    },
]

# 标准测试记忆库
TEST_MEMORIES = [
    {
        "memory_id": "m001",
        "content": "今天修复了一个严重的 bug，系统终于正常运行了",
        "category": "tech",
        "keywords": ["bug", "修复", "系统", "运行"],
    },
    {
        "memory_id": "m002",
        "content": "客户合同模板需要更新，请提供最新版本",
        "category": "business",
        "keywords": ["合同", "模板", "客户", "版本"],
    },
    {
        "memory_id": "m003",
        "content": "记忆保存和检索功能已经测试通过",
        "category": "memory",
        "keywords": ["记忆", "保存", "检索", "测试"],
    },
    {
        "memory_id": "m004",
        "content": "项目进度安排需要调整，下周完成里程碑",
        "category": "business",
        "keywords": ["项目", "进度", "安排", "里程碑"],
    },
    {
        "memory_id": "m005",
        "content": "易经卦象编码算法需要优化，提升检索准确率",
        "category": "yijing",
        "keywords": ["易经", "卦象", "编码", "检索"],
    },
    {
        "memory_id": "m006",
        "content": "数据库配置错误导致服务无法启动",
        "category": "tech",
        "keywords": ["数据库", "配置", "错误", "服务"],
    },
    {
        "memory_id": "m007",
        "content": "销售报告需要包含季度数据和趋势分析",
        "category": "business",
        "keywords": ["销售", "报告", "数据", "分析"],
    },
    {
        "memory_id": "m008",
        "content": "对话历史记录已自动保存到知识库",
        "category": "memory",
        "keywords": ["对话", "历史", "保存", "知识库"],
    },
]

# ============================================================================
# 检索方法实现
# ============================================================================


class KeywordRetriever:
    """传统关键词检索"""

    def __init__(self, memories: List[Dict]):
        self.memories = memories

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """关键词匹配检索"""
        query_words = set(query.lower())

        scored = []
        for mem in self.memories:
            content_words = set(mem["content"].lower())
            # 简单重叠度
            overlap = len(query_words & content_words)
            score = overlap / max(1, len(query_words))

            scored.append(
                {
                    "memory_id": mem["memory_id"],
                    "content": mem["content"],
                    "score": score,
                    "method": "keyword",
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


class VectorRetriever:
    """纯向量检索（模拟）"""

    def __init__(self, memories: List[Dict]):
        self.memories = memories
        # 预计算向量（简化：使用关键词哈希）
        self.vectors = {}
        for mem in memories:
            self.vectors[mem["memory_id"]] = self._encode(mem["content"])

    def _encode(self, text: str) -> Dict[str, float]:
        """简化编码"""
        words = text.lower().split()
        return {w: 1.0 for w in words}

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """向量相似度检索"""
        query_vec = self._encode(query)

        scored = []
        for mem in self.memories:
            mem_vec = self.vectors[mem["memory_id"]]

            # 余弦相似度（简化）
            all_words = set(query_vec.keys()) | set(mem_vec.keys())
            dot = sum(query_vec.get(w, 0) * mem_vec.get(w, 0) for w in all_words)
            norm_q = len(query_vec) ** 0.5
            norm_m = len(mem_vec) ** 0.5

            sim = dot / (norm_q * norm_m) if norm_q > 0 and norm_m > 0 else 0

            scored.append(
                {
                    "memory_id": mem["memory_id"],
                    "content": mem["content"],
                    "score": sim,
                    "method": "vector",
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


class YijingHybridRetriever:
    """易经特征增强检索"""

    def __init__(self, memories: List[Dict]):
        self.memories = memories

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """混合检索：向量 + 易经特征"""
        # 这里简化实现，实际应该调用 milvus_retrieval
        from core.enhanced_retrieval import EnhancedRetrievalEngine

        engine = EnhancedRetrievalEngine()

        # 转换记忆格式
        memories_data = []
        for mem in self.memories:
            # 简化卦象编码
            from core.enhanced_retrieval import SemanticYaoEncoder

            encoder = SemanticYaoEncoder()
            hexagram = encoder.encode(mem["content"])

            memories_data.append(
                {
                    "memory_id": mem["memory_id"],
                    "content": mem["content"],
                    "hexagram": hexagram,
                    "bagua_type": "kun",
                    "wuxing": "tu",
                    "sancai_layer": "di",
                    "hot_score": 0.5,
                    "created_at": datetime.now().isoformat(),
                }
            )

        results = engine.search(query, memories_data, top_k)

        for r in results:
            r["method"] = "yijing_hybrid"

        return results


# ============================================================================
# A/B 测试引擎
# ============================================================================


class ABTestEngine:
    """
    A/B 测试引擎

    对比三种检索方法的效果
    """

    def __init__(self, memories: List[Dict] = None):
        self.memories = memories or TEST_MEMORIES
        self.results_history = []

    def run_single_test(self, query_def: Dict) -> Dict:
        """
        运行单个查询的对比测试

        Returns:
            {
                'query_id': str,
                'query': str,
                'results': {
                    'keyword': [...],
                    'vector': [...],
                    'yijing_hybrid': [...],
                },
                'metrics': {...},
                'latency': {...},
            }
        """
        query = query_def["query"]
        expected = query_def.get("expected_keywords", [])

        results = {}
        latencies = {}

        # 1. 关键词检索
        start = time.time()
        keyword_ret = KeywordRetriever(self.memories)
        results["keyword"] = keyword_ret.search(query, top_k=5)
        latencies["keyword"] = (time.time() - start) * 1000

        # 2. 向量检索
        start = time.time()
        vector_ret = VectorRetriever(self.memories)
        results["vector"] = vector_ret.search(query, top_k=5)
        latencies["vector"] = (time.time() - start) * 1000

        # 3. 易经混合检索
        start = time.time()
        yijing_ret = YijingHybridRetriever(self.memories)
        results["yijing_hybrid"] = yijing_ret.search(query, top_k=5)
        latencies["yijing_hybrid"] = (time.time() - start) * 1000

        # 计算指标
        metrics = {}
        for method, res in results.items():
            metrics[method] = self._calculate_metrics(res, expected)

        return {
            "query_id": query_def["id"],
            "query": query,
            "description": query_def.get("description", ""),
            "results": results,
            "metrics": metrics,
            "latency": latencies,
            "timestamp": datetime.now().isoformat(),
        }

    def _calculate_metrics(self, results: List[Dict], expected_keywords: List[str]) -> Dict:
        """
        计算检索指标
        """
        if not results:
            return {"precision": 0, "mrr": 0, "keyword_coverage": 0}

        # Precision@5
        relevant = sum(1 for r in results if r.get("score", 0) > 0.3)
        precision = relevant / len(results)

        # MRR
        mrr = 1.0 if results and results[0].get("score", 0) > 0.3 else 0.5

        # Keyword Coverage
        retrieved_keywords = set()
        for r in results:
            retrieved_keywords.update(r.get("content", "").lower())

        expected_set = set(k.lower() for k in expected_keywords)
        coverage = len(expected_set & retrieved_keywords) / max(1, len(expected_set))

        return {
            "precision": round(precision, 4),
            "mrr": round(mrr, 4),
            "keyword_coverage": round(coverage, 4),
        }

    def run_full_test(self) -> Dict:
        """
        运行完整测试集

        Returns:
            汇总报告
        """
        all_results = []

        for query_def in TEST_QUERIES:
            result = self.run_single_test(query_def)
            all_results.append(result)

        # 汇总指标
        summary = self._aggregate_results(all_results)

        report = {
            "test_date": datetime.now().isoformat(),
            "total_queries": len(TEST_QUERIES),
            "total_memories": len(self.memories),
            "results": all_results,
            "summary": summary,
        }

        self.results_history.append(report)
        return report

    def _aggregate_results(self, all_results: List[Dict]) -> Dict:
        """汇总所有测试结果"""
        methods = ["keyword", "vector", "yijing_hybrid"]

        summary = {}
        for method in methods:
            precisions = []
            mrrs = []
            coverages = []
            latencies = []

            for result in all_results:
                if method in result["metrics"]:
                    m = result["metrics"][method]
                    precisions.append(m.get("precision", 0))
                    mrrs.append(m.get("mrr", 0))
                    coverages.append(m.get("keyword_coverage", 0))

                if method in result["latency"]:
                    latencies.append(result["latency"][method])

            summary[method] = {
                "avg_precision": round(sum(precisions) / max(1, len(precisions)), 4),
                "avg_mrr": round(sum(mrrs) / max(1, len(mrrs)), 4),
                "avg_keyword_coverage": round(sum(coverages) / max(1, len(coverages)), 4),
                "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 4),
            }

        # 计算提升
        baseline = summary.get("keyword", {})
        summary["improvement"] = {
            "vector_vs_keyword": {
                "precision_gain": summary.get("vector", {}).get("avg_precision", 0) - baseline.get("avg_precision", 0),
                "latency_change": summary.get("vector", {}).get("avg_latency_ms", 0)
                - baseline.get("avg_latency_ms", 0),
            },
            "yijing_vs_keyword": {
                "precision_gain": summary.get("yijing_hybrid", {}).get("avg_precision", 0)
                - baseline.get("avg_precision", 0),
                "latency_change": summary.get("yijing_hybrid", {}).get("avg_latency_ms", 0)
                - baseline.get("avg_latency_ms", 0),
            },
        }

        return summary

    def export_report(self, output_path: str = None) -> str:
        """导出测试报告"""
        if not self.results_history:
            self.run_full_test()

        latest_report = self.results_history[-1]

        if output_path is None:
            output_path = Path(__file__).parent.parent / "data" / "ab_test_report.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(latest_report, f, ensure_ascii=False, indent=2)

        return str(output_path)

    def print_report(self):
        """打印测试报告"""
        if not self.results_history:
            self.run_full_test()

        report = self.results_history[-1]
        summary = report["summary"]

        print("=" * 90)
        print("🔮 易经记忆系统 - A/B 测试报告")
        print("=" * 90)
        print(f"测试时间：{report['test_date']}")
        print(f"查询数量：{report['total_queries']}")
        print(f"记忆数量：{report['total_memories']}")
        print()

        print("📊 方法对比")
        print("-" * 90)
        print(f"{'方法':<20} {'准确率':<12} {'MRR':<12} {'关键词覆盖':<12} {'延迟 (ms)':<12}")
        print("-" * 90)

        for method in ["keyword", "vector", "yijing_hybrid"]:
            m = summary.get(method, {})
            method_name = {
                "keyword": "关键词检索",
                "vector": "向量检索",
                "yijing_hybrid": "易经混合检索",
            }.get(method, method)

            print(
                f"{method_name:<20} {m.get('avg_precision', 0):<12.4f} {m.get('avg_mrr', 0):<12.4f} {m.get('avg_keyword_coverage', 0):<12.4f} {m.get('avg_latency_ms', 0):<12.4f}"
            )

        print()
        print("📈 效果提升")
        print("-" * 90)
        improvement = summary.get("improvement", {})

        v_gain = improvement.get("vector_vs_keyword", {}).get("precision_gain", 0)
        y_gain = improvement.get("yijing_vs_keyword", {}).get("precision_gain", 0)

        print(f"向量检索 vs 关键词：准确率 {'+' if v_gain >= 0 else ''}{v_gain:.4f}")
        print(f"易经混合 vs 关键词：准确率 {'+' if y_gain >= 0 else ''}{y_gain:.4f}")

        print()
        print("=" * 90)


# ============================================================================
# 单例
# ============================================================================

_test_engine = None


def get_test_engine(memories: List[Dict] = None) -> ABTestEngine:
    global _test_engine
    if _test_engine is None:
        _test_engine = ABTestEngine(memories)
    return _test_engine


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    engine = get_test_engine()
    engine.print_report()

    # 导出报告
    report_path = engine.export_report()
    print(f"\n📄 报告已导出：{report_path}")
