#!/usr/bin/env python3
"""
MetaMemory 综合性能基准测试 (Phase 3B)
=====================================
测试项：
  R1: Recall@K — 检索精度 vs ground truth
  R2: 编码延迟 — content_to_hexagram + embedding 生成耗时
  R3: 检索延迟 — vector / hamming 检索耗时（Phase 3B: 纯语义）
  R4: 吞吐量 — 批量创建/检索 ops/sec
  R5: Hash vs Semantic — fallback 与真实语义的精度对比
  R6: 缓存效果 — 命中率与加速比

Phase 3B 变更：六爻退化为分类标签，检索评分 100% 语义向量相似度。

用法：
  python -m core.benchmarks              # 快速模式（100条）
  python -m core.benchmarks --full       # 完整模式（1000条）
  python -m core.benchmarks --quick      # 极速模式（10条）
  python -m core.benchmarks --output report.json  # 输出 JSON
"""

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# 确保 core/ 在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.core_yijing import (
    calculate_similarity,
    content_to_hexagram,
    hexagram_to_vector,
)
from core.embedding_provider import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    STEmbeddingProvider,
    get_embedding_provider,
)


# ─── 测试数据集 ──────────────────────────────────────────────

# 语义相似对（共 20 组，用于 Recall@K 精度测试）
SEMANTIC_PAIRS = [
    # 同义改写
    ("猫在吃鱼", "猫咪正在享用鱼肉"),
    ("今天天气很好", "今日晴空万里"),
    ("我喜欢编程", "我对写代码充满热情"),
    ("机器学习很强大", "AI 技术能力惊人"),
    ("早餐吃了面包", "早上吃了个馒头"),
    # 近义关联
    ("Python 是一门编程语言", "Python 常用于数据科学"),
    ("数据库存储数据", "MySQL 是一种关系型数据库"),
    ("深度学习需要 GPU", "训练神经网络需要显卡"),
    ("网页前端用 React", "Vue 也是前端框架"),
    ("REST API 设计", "HTTP 接口开发规范"),
    # 直接相关
    ("用户登录系统", "输入账号密码验证身份"),
    ("文件上传功能", "将文件从本地上传到服务器"),
    ("缓存提升性能", "Redis 加速数据读取"),
    ("微服务架构", "将单体应用拆分成多个小服务"),
    ("容器化部署", "Docker 打包应用运行环境"),
    # 易混淆（硬负样本）
    ("苹果很好吃", "苹果公司发布新品"),
    ("Python 编程", "Python 是一种蛇"),
    ("内存不足", "童年记忆模糊"),
    ("云计算平台", "多云天气转凉"),
    ("模型训练完成", "模特走秀训练"),
]

# 长文本测试（用于延迟测试）
LONG_TEXTS = [
    "在人工智能领域，记忆系统是构建智能体的关键组件。一个优秀的记忆系统需要能够高效地存储、检索和管理信息，同时保持语义理解能力。传统的方法主要依赖向量数据库和语义嵌入，但往往缺乏结构化的生命周期管理。",
    "易经作为中国古代哲学经典，其数理体系中的阴阳二元论、八卦分类法和五行生克理论，为构建智能体记忆系统提供了独特的理论框架。六十四卦的编码方式天然适合二进制计算，五行流转对应记忆的生命周期演化。",
    "现代软件工程强调模块化设计、测试驱动开发和持续集成。一个高质量的项目应该有清晰的架构分层、完善的测试覆盖和自动化的部署流程。代码可读性和可维护性同样重要，好的命名胜过好的注释。",
]

# 批量测试数据
BULK_TEXTS = [
    f"这是第{i}条测试记忆数据，包含一些随机关键词和语义信息"
    for i in range(100)
]


# ─── 工具函数 ─────────────────────────────────────────────────

def format_duration(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f}ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f}µs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f}ms"
    return f"{seconds:.3f}s"


def time_it(func, *args, warmup=3, iterations=10, **kwargs):
    """微基准：预热 + 多次测量取最小"""
    # 预热
    for _ in range(warmup):
        func(*args, **kwargs)
    # 测量
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)
    return min(times), statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0


# ─── 基准测试项 ───────────────────────────────────────────────


def benchmark_encoding_latency(provider: EmbeddingProvider) -> dict:
    """R2: 编码延迟 — hexagram + embedding"""
    results = {}

    # 短文本
    t_min, t_avg, t_std = time_it(content_to_hexagram, "猫在吃鱼", "fact")
    results["hexagram_short_min"] = t_min
    results["hexagram_short_avg"] = t_avg

    # 长文本
    t_min, t_avg, t_std = time_it(content_to_hexagram, LONG_TEXTS[0], "fact")
    results["hexagram_long_min"] = t_min
    results["hexagram_long_avg"] = t_avg

    # embedding
    t_min, t_avg, t_std = time_it(provider.encode, "猫在吃鱼")
    results["embedding_short_min"] = t_min
    results["embedding_short_avg"] = t_avg

    return results


def benchmark_recall_at_k(provider: EmbeddingProvider, k_values: list = None) -> dict:
    """R1: Recall@K — 检索精度 vs ground truth"""
    if k_values is None:
        k_values = [1, 3, 5, 10]

    # 为所有文本生成向量
    all_texts = []
    ground_truth = {}  # query_idx -> [relevant_doc_indices]
    
    for i, (query, doc) in enumerate(SEMANTIC_PAIRS):
        q_idx = i * 2
        ground_truth[i] = [q_idx + 1]
        all_texts.extend([query, doc])

    vecs = [provider.encode(t) for t in all_texts]

    # 对每个 query 检索并计算 Recall@K
    results = {f"recall@{k}": 0.0 for k in k_values}
    n_queries = len(SEMANTIC_PAIRS)

    for i in range(n_queries):
        q_idx = i * 2
        query_vec = vecs[q_idx]

        # 相似度排序（排除自身）
        similarities = []
        for j, v in enumerate(vecs):
            if j == q_idx:
                continue
            # 余弦相似度
            dot = sum(a * b for a, b in zip(query_vec, v))
            mag_q = sum(a * a for a in query_vec) ** 0.5
            mag_v = sum(a * a for a in v) ** 0.5
            sim = dot / (mag_q * mag_v + 1e-8)
            similarities.append((j, sim))

        ranked = [j for j, _ in sorted(similarities, key=lambda x: -x[1])]

        for k in k_values:
            if ground_truth[i][0] in ranked[:k]:
                results[f"recall@{k}"] += 1

    for k in k_values:
        results[f"recall@{k}"] /= n_queries

    return results


def benchmark_retrieval_latency(provider: EmbeddingProvider, n_docs: int = 500) -> dict:
    """R3: 检索延迟 — 三种检索模式对比"""
    results = {}

    # 构建文档库
    docs = []
    doc_vecs = []
    for i in range(n_docs):
        text = BULK_TEXTS[i % len(BULK_TEXTS)]
        hex_tuple = content_to_hexagram(text, "fact")
        hex_arr = hex_tuple[0]  # hex_arr is first element
        vec = provider.encode(text)
        docs.append((hex_arr, vec))
        doc_vecs.append(vec)

    query_vec = provider.encode("检索测试查询")

    # Hamming 距离检索
    query_hex_tuple = content_to_hexagram("检索测试查询", "fact")
    query_hex_arr = query_hex_tuple[0]

    def hamming_search():
        distances = []
        for hex_arr, _ in docs:
            dist = sum(a != b for a, b in zip(query_hex_arr, hex_arr))
            distances.append(dist)
        return sorted(range(len(distances)), key=lambda i: distances[i])

    t_min, t_avg, t_std = time_it(hamming_search, warmup=5, iterations=20)
    results["hamming_5ms"] = t_min
    results["hamming_5avg"] = t_avg

    # 纯向量检索
    def vector_search():
        sims = []
        for v in doc_vecs:
            dot = sum(a * b for a, b in zip(query_vec, v))
            mag_q = sum(a * a for a in query_vec) ** 0.5
            mag_v = sum(a * a for a in v) ** 0.5
            sim = dot / (mag_q * mag_v + 1e-8)
            sims.append(sim)
        return sorted(range(len(sims)), key=lambda i: -sims[i])

    t_min, t_avg, t_std = time_it(vector_search, warmup=3, iterations=10)
    results["vector_5min"] = t_min
    results["vector_5avg"] = t_avg

    return results


def benchmark_throughput(provider: EmbeddingProvider, n: int = 200) -> dict:
    """R4: 吞吐量 — 批量创建/检索 ops/sec"""
    results = {}

    # 批量编码
    texts = BULK_TEXTS[:n]
    start = time.perf_counter()
    for t in texts:
        provider.encode(t)
    elapsed = time.perf_counter() - start
    results["embed_ops_per_sec"] = n / elapsed
    results["embed_total_sec"] = elapsed

    # 批量 hexagram
    start = time.perf_counter()
    for t in texts:
        content_to_hexagram(t, "fact")
    elapsed = time.perf_counter() - start
    results["hex_ops_per_sec"] = n / elapsed
    results["hex_total_sec"] = elapsed

    return results


def benchmark_hash_vs_semantic() -> dict:
    """R5: Hash fallback vs 真实语义 — 精度对比"""
    results = {}

    # Hash provider
    hash_provider = HashEmbeddingProvider(dim=384)

    # Semantic provider (if available)
    try:
        sem_provider = STEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        has_semantic = True
    except Exception:
        sem_provider = hash_provider
        has_semantic = False

    results["has_semantic"] = has_semantic

    # 测试：同义 vs 不同义的相似度分离度
    same_pairs = [
        ("猫在吃鱼", "猫咪正在享用鱼肉"),
        ("今天天气很好", "今日晴空万里"),
        ("我喜欢编程", "我对写代码充满热情"),
    ]
    diff_pairs = [
        ("猫在吃鱼", "今天天气很好"),
        ("我喜欢编程", "早餐吃了面包"),
        ("机器学习很强大", "苹果很好吃"),
    ]

    for name, prov in [("hash", hash_provider), ("semantic", sem_provider)]:
        same_sims = []
        for a, b in same_pairs:
            va, vb = prov.encode(a), prov.encode(b)
            dot = sum(x * y for x, y in zip(va, vb))
            mag_a = sum(x * x for x in va) ** 0.5
            mag_b = sum(x * x for x in vb) ** 0.5
            same_sims.append(dot / (mag_a * mag_b + 1e-8))

        diff_sims = []
        for a, b in diff_pairs:
            va, vb = prov.encode(a), prov.encode(b)
            dot = sum(x * y for x, y in zip(va, vb))
            mag_a = sum(x * x for x in va) ** 0.5
            mag_b = sum(x * x for x in vb) ** 0.5
            diff_sims.append(dot / (mag_a * mag_b + 1e-8))

        results[f"{name}_same_sim_avg"] = statistics.mean(same_sims) if same_sims else 0
        results[f"{name}_diff_sim_avg"] = statistics.mean(diff_sims) if diff_sims else 0
        # 分离度 = 同义平均 - 不同义平均（越高越好）
        results[f"{name}_separation"] = (
            results[f"{name}_same_sim_avg"] - results[f"{name}_diff_sim_avg"]
        )

    return results


def benchmark_cache_effect(provider: EmbeddingProvider) -> dict:
    """R6: 缓存效果 — 命中率与加速比"""
    results = {}

    # 先预热缓存
    for t in BULK_TEXTS[:50]:
        provider.encode(t)

    # 缓存命中延迟
    t_min, t_avg, t_std = time_it(provider.encode, BULK_TEXTS[0], warmup=1, iterations=100)
    results["cached_avg"] = t_avg
    results["cached_min"] = t_min

    # 缓存未命中延迟（新文本）
    t_min, t_avg, t_std = time_it(provider.encode, "一个全新的从未出现的文本字符串", warmup=1, iterations=20)
    results["uncached_avg"] = t_avg
    results["uncached_min"] = t_min

    if results["uncached_avg"] > 0:
        results["speedup"] = results["uncached_avg"] / max(results["cached_avg"], 1e-12)
    else:
        results["speedup"] = 1.0

    return results


# ─── 主运行器 ──────────────────────────────────────────────────


def run_all_benchmarks(mode: str = "quick") -> dict:
    """运行全部基准测试"""
    configs = {
        "quick": {"retrieval_docs": 100, "throughput_n": 50, "k_values": [1, 5]},
        "normal": {"retrieval_docs": 500, "throughput_n": 200, "k_values": [1, 3, 5, 10]},
        "full": {"retrieval_docs": 2000, "throughput_n": 1000, "k_values": [1, 3, 5, 10, 20]},
    }
    cfg = configs.get(mode, configs["quick"])
    report: dict[str, Any] = {
        "meta": {
            "version": "2.2.0",
            "mode": mode,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "environment": {},
        "results": {},
    }

    # 检测环境
    try:
        provider = get_embedding_provider()
        report["environment"]["provider_type"] = type(provider).__name__
    except Exception as e:
        provider = HashEmbeddingProvider(dim=384)
        report["environment"]["provider_type"] = "HashEmbeddingProvider (fallback)"
        report["environment"]["fallback_reason"] = str(e)

    report["environment"]["embedding_dim"] = len(provider.encode("test"))

    print("=" * 60)
    print(f"  MetaMemory Benchmark v2.2 ({mode} mode)")
    print(f"  Provider: {report['environment']['provider_type']}")
    print(f"  Dim: {report['environment']['embedding_dim']}")
    print("=" * 60)

    benchmarks = [
        ("R1: Recall@K", benchmark_recall_at_k, (provider,), {"k_values": cfg["k_values"]}),
        ("R2: Encoding Latency", benchmark_encoding_latency, (provider,), {}),
        ("R3: Retrieval Latency", benchmark_retrieval_latency, (provider,), {"n_docs": cfg["retrieval_docs"]}),
        ("R4: Throughput", benchmark_throughput, (provider,), {"n": cfg["throughput_n"]}),
        ("R5: Hash vs Semantic", benchmark_hash_vs_semantic, (), {}),
        ("R6: Cache Effect", benchmark_cache_effect, (provider,), {}),
    ]

    for name, func, args, kwargs in benchmarks:
        print(f"\n{'─' * 50}")
        print(f"  {name}")
        print(f"{'─' * 50}")
        try:
            result = func(*args, **kwargs)
            report["results"][name] = result
            _print_results(result)
        except Exception as e:
            print(f"  ❌ SKIPPED: {e}")
            report["results"][name] = {"error": str(e)}

    # 综合评分
    score = _calculate_score(report["results"])
    report["score"] = score
    print(f"\n{'=' * 60}")
    print(f"  📊 Benchmark Score: {score:.1f}/100")
    print(f"{'=' * 60}")

    return report


def _print_results(results: dict, indent: int = 2):
    """打印结果"""
    prefix = " " * indent
    for key, value in results.items():
        if key in ("has_semantic", "error"):
            print(f"{prefix}{key}: {value}")
        elif isinstance(value, float):
            # 相似度/分离度指标
            if "_sim_avg" in key or "_separation" in key:
                print(f"{prefix}{key}: {value:.4f}")
            # Recall 指标
            elif "recall" in key:
                print(f"{prefix}{key}: {value:.2%}")
            # 吞吐量
            elif "ops_per_sec" in key:
                print(f"{prefix}{key}: {value:.1f} ops/s")
            # 加速比
            elif "speedup" in key:
                print(f"{prefix}{key}: {value:.1f}x")
            # 总时间
            elif "total_sec" in key:
                print(f"{prefix}{key}: {value:.3f}s")
            # 延迟类
            elif "avg" in key or "min" in key or "latency" in key:
                print(f"{prefix}{key}: {format_duration(value)}")
            else:
                print(f"{prefix}{key}: {value:.4f}")


def _calculate_score(results: dict) -> float:
    """计算综合评分 (0-100)"""
    score = 0.0
    weights_total = 0

    weights = {
        "R1: Recall@K": 30,
        "R2: Encoding Latency": 10,
        "R3: Retrieval Latency": 15,
        "R4: Throughput": 15,
        "R5: Hash vs Semantic": 20,
        "R6: Cache Effect": 10,
    }

    for name, weight in weights.items():
        if name not in results or "error" in results[name]:
            continue
        r = results[name]
        weights_total += weight

        if name == "R1: Recall@K":
            # avg of recall@1 and recall@5
            r1 = r.get("recall@1", 0)
            r5 = r.get("recall@5", r.get("recall@3", 0))
            score += weight * (r1 * 0.6 + r5 * 0.4)

        elif name == "R2: Encoding Latency":
            # < 10ms = 100, > 100ms = 0
            lat = r.get("embedding_short_avg", 0.1)
            sub = max(0, 1 - (lat - 0.01) / 0.09) if lat > 0.01 else 1.0
            score += weight * sub

        elif name == "R3: Retrieval Latency":
            # < 1ms = 100, > 50ms = 0 (per 500 docs)
            lat = r.get("vector_5avg", 0.05)
            sub = max(0, 1 - (lat - 0.001) / 0.049) if lat > 0.001 else 1.0
            score += weight * sub

        elif name == "R4: Throughput":
            ops = r.get("embed_ops_per_sec", 0)
            # > 100 ops/s = 100, < 1 = 0
            sub = min(1.0, ops / 100)
            score += weight * sub

        elif name == "R5: Hash vs Semantic":
            # separation > 0.3 = 100, < 0 = 0
            sep = r.get("semantic_separation", r.get("hash_separation", 0))
            sub = min(1.0, sep / 0.3) if sep > 0 else 0
            score += weight * sub

        elif name == "R6: Cache Effect":
            sp = r.get("speedup", 1.0)
            # > 10x = 100, 1x = 0
            sub = min(1.0, (sp - 1) / 9)
            score += weight * sub

    return (score / weights_total * 100) if weights_total > 0 else 0


def main():
    """CLI 入口"""
    import argparse

    p = argparse.ArgumentParser(description="MetaMemory Benchmark Suite")
    p.add_argument("--quick", action="store_true", help="Quick mode (50 docs)")
    p.add_argument("--full", action="store_true", help="Full mode (2000 docs)")
    p.add_argument("--output", type=str, help="Output JSON file path")
    args = p.parse_args()

    mode = "normal"
    if args.quick:
        mode = "quick"
    elif args.full:
        mode = "full"

    report = run_all_benchmarks(mode)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n📄 Report saved to {args.output}")

    return report


if __name__ == "__main__":
    main()
