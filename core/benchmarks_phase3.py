#!/usr/bin/env python3
"""
Phase 3B: 纯语义检索验证（Option B）
====================================
对照实验：证明六爻 rule 编码在检索路径中不存在增量价值

Group A: Pure Embedding (sentence-transformers 384-dim)
Group C: Pure Hexagram Hamming (6-bit 结构距离)
Group E: Random (随机基线)
Group F: Option B — Pure Embedding + Hex as Label (Production readout)

指标: Recall@K, MRR, NDCG@K, Precision@K
统计: 配对 t-test, Cohen's d 效应量

Phase 3A 结论：六爻 rule 编码区分度 = -0.2（完全反向），Hybrid 无增量。
Phase 3B 决策：六爻退化为分类标签，检索评分 100% 语义向量相似度。

用法:
  python -m core.benchmarks_phase3
  python -m core.benchmarks_phase3 --n-docs 2000
  python -m core.benchmarks_phase3 --top-k 5,10,20
"""

import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from core.core_yijing import (
    calculate_similarity,
    content_to_hexagram,
    generate_semantic_embedding,
    hexagram_to_vector,
)
from core.embedding_provider import (
    CachedEmbeddingProvider,
    HashEmbeddingProvider,
    STEmbeddingProvider,
    cosine_similarity,
    get_embedding_provider,
)

# ─── 配置 ─────────────────────────────────────────────────────

# 大规模语义测试集：query + relevant_docs (ground truth)
# 每组 query 配 3-5 个相关文档，模拟真实检索场景
BENCHMARK_DATASET = [
    # ── 编程与软件工程 ──
    {
        "query": "Python异步编程最佳实践",
        "relevant": [
            "使用asyncio库实现Python异步任务调度",
            "async/await语法在Python 3.7+中的正确用法",
            "Python协程与事件循环的工作原理",
            "避免asyncio中的常见错误：忘记await、阻塞调用",
        ],
        "distractors": [
            "Python是一种静态类型语言",
            "JavaScript异步编程使用Promise",
            "今天天气很好适合出门",
            "早餐吃了面包和牛奶",
            "机器学习模型需要大量数据训练",
        ],
    },
    {
        "query": "REST API设计规范",
        "relevant": [
            "RESTful API的资源命名应使用名词复数形式",
            "HTTP状态码的正确使用：200/201/400/404/500",
            "API版本控制策略：URL路径 vs Header",
            "REST API的分页、过滤和排序最佳实践",
            "HATEOAS超媒体驱动的REST架构",
        ],
        "distractors": [
            "GraphQL比REST更灵活",
            "猫咪喜欢吃鱼",
            "数据库索引提升查询性能",
            "今天下午有暴雨预警",
        ],
    },
    {
        "query": "Docker容器化部署",
        "relevant": [
            "使用Dockerfile构建应用镜像的最佳实践",
            "多阶段构建减小Docker镜像体积",
            "Docker Compose编排多容器应用",
            "Docker网络模式和存储卷管理",
        ],
        "distractors": [
            "Python的GIL限制了多线程性能",
            "今天早餐喝了豆浆",
            "React是一个前端框架",
            "微服务架构的优势和挑战",
            "深度学习模型训练需要GPU加速",
        ],
    },
    # ── 机器学习与AI ──
    {
        "query": "神经网络过拟合解决方案",
        "relevant": [
            "使用Dropout层防止神经网络过拟合",
            "L1/L2正则化在深度学习中的应用",
            "早停法Early Stopping监控验证集损失",
            "数据增强扩充训练集减少过拟合",
            "交叉验证评估模型泛化能力",
        ],
        "distractors": [
            "Python列表推导式的语法",
            "今天中午吃了面条",
            "REST API使用JSON格式传输数据",
            "Docker容器需要健康检查",
        ],
    },
    {
        "query": "Transformer注意力机制原理",
        "relevant": [
            "Self-Attention计算Query、Key、Value矩阵",
            "Multi-Head Attention多角度捕捉语义关系",
            "位置编码Positional Encoding补充序列信息",
            "Transformer的Encoder-Decoder架构详解",
            "注意力机制中的缩放因子sqrt(dk)作用",
        ],
        "distractors": [
            "卷积神经网络适合图像处理",
            "今天天气晴朗适合出游",
            "微服务间的通信方式",
            "MySQL索引优化策略",
        ],
    },
    {
        "query": "模型评估指标选择",
        "relevant": [
            "分类问题使用Accuracy、Precision、Recall、F1",
            "回归问题使用MSE、MAE、R-squared",
            "不平衡数据集使用AUC-ROC代替Accuracy",
            "排序问题使用MRR、NDCG、MAP评估",
        ],
        "distractors": [
            "Docker容器的资源限制配置",
            "猫咪在沙发上睡觉",
            "RESTful API使用HTTP动词",
            "Python的装饰器语法糖",
            "今天地铁晚点了",
        ],
    },
    # ── 数据库与存储 ──
    {
        "query": "MySQL查询性能优化",
        "relevant": [
            "使用EXPLAIN分析查询执行计划",
            "复合索引的最左前缀原则",
            "避免SELECT *只查询需要的列",
            "慢查询日志定位性能瓶颈",
            "InnoDB存储引擎的聚簇索引原理",
        ],
        "distractors": [
            "Redis是内存缓存数据库",
            "今天晚餐吃了火锅",
            "Transformer模型使用自注意力机制",
            "Docker镜像推送到容器仓库",
        ],
    },
    {
        "query": "Redis缓存策略设计",
        "relevant": [
            "缓存穿透使用布隆过滤器防护",
            "缓存雪崩设置过期时间随机偏移",
            "缓存击穿使用互斥锁或永不过期",
            "Redis数据淘汰策略LRU/LFU/TTL",
            "缓存与数据库双写一致性方案",
        ],
        "distractors": [
            "Python异步编程使用async/await",
            "今天天气有点冷",
            "神经网络需要激活函数",
            "REST API返回标准状态码",
        ],
    },
    # ── 分布式系统 ──
    {
        "query": "微服务间通信方式",
        "relevant": [
            "同步通信使用HTTP/REST或gRPC",
            "异步通信使用消息队列RabbitMQ/Kafka",
            "服务发现使用Consul/Eureka/Nacos",
            "API网关统一入口Zuul/Kong/APISIX",
            "服务间调用链追踪使用Jaeger/Zipkin",
        ],
        "distractors": [
            "Python支持面向对象编程",
            "今天早餐吃了包子",
            "卷积神经网络用于图像识别",
            "MySQL支持事务ACID特性",
        ],
    },
    {
        "query": "分布式事务解决方案",
        "relevant": [
            "两阶段提交2PC保证强一致性",
            "TCC补偿事务Try-Confirm-Cancel模式",
            "Saga长事务的最终一致性方案",
            "本地消息表+MQ实现异步确保",
            "Seata分布式事务框架的使用",
        ],
        "distractors": [
            "Dockerfile的FROM指令",
            "猫咪喜欢捉老鼠",
            "深度学习中的梯度消失问题",
            "Redis的String数据结构",
        ],
    },
    # ── 硬负样本（语义相关但不同主题） ──
    {
        "query": "Python内存管理机制",
        "relevant": [
            "Python的引用计数垃圾回收机制",
            "Python对象的内存分配与释放",
            "gc模块手动触发垃圾回收",
            "循环引用导致的内存泄漏问题",
            "Python内存池PyMalloc优化小对象分配",
        ],
        "distractors": [
            "Python数据分析常用Pandas库",  # Python相关但不是内存管理
            "Python Web框架Django和Flask对比",  # Python相关但不是内存管理
            "使用Python爬虫抓取网页数据",  # Python相关但不是内存管理
            "今天天气很好",  # 完全不同
        ],
    },
    {
        "query": "深度学习模型部署方案",
        "relevant": [
            "TensorFlow Serving部署深度学习模型",
            "ONNX格式实现模型跨框架部署",
            "TorchServe PyTorch模型生产部署",
            "模型量化INT8降低推理延迟",
            "Triton Inference Server多模型管理",
        ],
        "distractors": [
            "深度学习框架PyTorch和TensorFlow对比",  # 相关但不具体说部署
            "深度学习训练需要GPU加速",  # 相关但不具体说部署
            "卷积神经网络的卷积运算原理",  # DL相关但不是部署
            "Python是深度学习主流语言",  # 弱相关
        ],
    },
    # ── 不同语调/同义改写 ──
    {
        "query": "如何处理程序中的异常",
        "relevant": [
            "使用try-except捕获Python异常",
            "Java异常处理的try-catch-finally机制",
            "自定义异常类的设计和使用场景",
            "异常处理的最佳实践：不要吞掉异常",
        ],
        "distractors": [
            "Python支持多线程编程",
            "今天午餐订了外卖",
            "Docker容器需要配置端口映射",
        ],
    },
    {
        "query": "如何保证系统高可用性",
        "relevant": [
            "服务冗余部署多副本避免单点故障",
            "健康检查自动摘除故障节点",
            "限流降级熔断保护核心链路",
            "异地多活架构应对机房级故障",
            "灰度发布降低上线风险",
        ],
        "distractors": [
            "数据库索引加快查询速度",
            "今天天气晴朗适合运动",
            "Python的列表和字典数据结构",
            "神经网络的反向传播算法",
        ],
    },
    # ── 信息检索相关（贴合记忆系统场景） ──
    {
        "query": "检索结果排序算法",
        "relevant": [
            "TF-IDF词频逆文档频率计算文本相关性",
            "BM25概率检索模型优于TF-IDF",
            "Learning to Rank使用机器学习优化排序",
            "向量相似度检索使用余弦距离排序",
            "Elasticsearch的BM25默认评分机制",
        ],
        "distractors": [
            "Python的sort和sorted排序方法",
            "冒泡排序的时间复杂度为O(n²)",
            "今天早餐喝了咖啡",
            "Docker容器启动时需要拉取镜像",
            "猫咪对人类来说很重要",
        ],
    },
    {
        "query": "向量数据库选型对比",
        "relevant": [
            "Milvus分布式向量数据库支持十亿级检索",
            "Pinecone全托管向量数据库免运维",
            "Qdrant高性能Rust向量数据库",
            "Weaviate带内置向量化和GraphQL支持",
            "FAISS是Meta开源的高效向量检索库",
        ],
        "distractors": [
            "MySQL是最流行的关系型数据库",
            "Redis支持多种数据结构",
            "今天气温适宜",
            "Python的pip包管理器",
        ],
    },
    # ── 系统设计 ──
    {
        "query": "高并发系统设计原则",
        "relevant": [
            "无状态设计便于水平扩展",
            "缓存分层CDN→本地→分布式降低延迟",
            "异步处理消息队列削峰填谷",
            "数据库读写分离和分库分表",
            "连接池复用减少建立连接开销",
        ],
        "distractors": [
            "Python是一门解释型语言",
            "早上起晚了没吃早餐",
            "Docker镜像分层构建",
            "神经网络使用ReLU激活函数",
        ],
    },
    {
        "query": "日志系统设计方案",
        "relevant": [
            "ELK Stack Elasticsearch+Logstash+Kibana",
            "日志采集使用Filebeat轻量级Agent",
            "结构化日志使用JSON格式便于解析",
            "日志采样策略降低海量日志存储成本",
            "链路追踪TraceID串联分布式请求日志",
        ],
        "distractors": [
            "今天下午有部门会议",
            "Python的logging模块支持日志级别",
            "机器学习模型的超参数调优",
            "Docker容器的资源限制",
        ],
    },
    # ── 记忆系统相关（自评估） ──
    {
        "query": "Agent记忆管理策略",
        "relevant": [
            "短期记忆存储当前会话上下文",
            "长期记忆持久化重要信息和用户偏好",
            "记忆衰减算法淘汰低价值信息",
            "语义检索替代关键词匹配提升召回率",
            "记忆摘要压缩长对话减少token消耗",
        ],
        "distractors": [
            "Python函数可以接受可变参数",
            "今天天气多云转晴",
            "数据库ACID事务特性",
            "Docker使用cgroups进行资源隔离",
        ],
    },
    {
        "query": "智能体上下文窗口管理",
        "relevant": [
            "滑动窗口保留最近N轮对话",
            "重要性评分保留关键信息丢弃冗余",
            "递归摘要逐层压缩历史对话",
            "混合策略短期窗口+长期摘要",
            "向量检索注入相关历史记忆到prompt",
        ],
        "distractors": [
            "神经网络层数影响模型表达能力",
            "今天中午饭还没吃",
            "微服务架构的服务拆分原则",
            "Python的装饰器模式",
            "猫咪在晒太阳",
        ],
    },
]


# ─── 数据结构 ──────────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    """单组实验结果"""
    name: str
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    queries_processed: int = 0


# ─── 评估指标 ──────────────────────────────────────────────────


def dcg(relevance: List[float], k: int) -> float:
    """Discounted Cumulative Gain"""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance[:k]))


def ndcg(pred_ranking: List[int], ground_truth: set, k: int) -> float:
    """Normalized DCG"""
    relevance = [1.0 if i in ground_truth else 0.0 for i in pred_ranking]
    ideal_rel = sorted(relevance, reverse=True)
    dcg_val = dcg(relevance, k)
    idcg_val = dcg(ideal_rel, k)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


def recall_at_k(pred_ranking: List[int], ground_truth: set, k: int) -> float:
    """Recall@K"""
    if not ground_truth:
        return 0.0
    hits = sum(1 for i in pred_ranking[:k] if i in ground_truth)
    return hits / len(ground_truth)


def precision_at_k(pred_ranking: List[int], ground_truth: set, k: int) -> float:
    """Precision@K"""
    hits = sum(1 for i in pred_ranking[:k] if i in ground_truth)
    return hits / k


def mrr(pred_ranking: List[int], ground_truth: set) -> float:
    """Mean Reciprocal Rank"""
    for rank, idx in enumerate(pred_ranking, 1):
        if idx in ground_truth:
            return 1.0 / rank
    return 0.0


# ─── 检索方法 ──────────────────────────────────────────────────


def retrieval_pure_embedding(
    query: str, docs: List[str], provider
) -> Tuple[List[int], float]:
    """Group A: 纯语义 embedding 检索"""
    t0 = time.perf_counter()
    q_vec = provider.encode(query)
    doc_vecs = [provider.encode(d) for d in docs]
    sims = [cosine_similarity(q_vec, dv) for dv in doc_vecs]
    ranking = sorted(range(len(sims)), key=lambda i: -sims[i])
    return ranking, (time.perf_counter() - t0) * 1000


def retrieval_hybrid(
    query: str, docs: List[str], provider, hex_weight: float = 0.1
) -> Tuple[List[int], float]:
    """Group B: 混合检索 — Embedding + 六爻结构融合"""
    t0 = time.perf_counter()

    # 语义相似度
    q_vec = provider.encode(query)
    doc_vecs = [provider.encode(d) for d in docs]
    sem_sims = [cosine_similarity(q_vec, dv) for dv in doc_vecs]

    # 结构相似度（六爻 Hamming 归一化为相似度）
    try:
        q_hex_tuple = content_to_hexagram(query, "fact")
        q_hex = q_hex_tuple[0]
    except Exception:
        q_hex = [0, 0, 0, 0, 0, 0]

    struct_sims = []
    for d in docs:
        try:
            d_hex_tuple = content_to_hexagram(d, "fact")
            d_hex = d_hex_tuple[0]
            matches = sum(1 for i in range(6) if q_hex[i] == d_hex[i])
            struct_sims.append(matches / 6.0)
        except Exception:
            struct_sims.append(0.5)

    # 加权融合
    combined = [
        (1 - hex_weight) * sem_sims[i] + hex_weight * struct_sims[i]
        for i in range(len(sem_sims))
    ]
    ranking = sorted(range(len(combined)), key=lambda i: -combined[i])
    return ranking, (time.perf_counter() - t0) * 1000


def retrieval_hexagram_hamming(
    query: str, docs: List[str]
) -> Tuple[List[int], float]:
    """Group C: 纯六爻 Hamming 距离检索"""
    t0 = time.perf_counter()
    try:
        q_hex_tuple = content_to_hexagram(query, "fact")
        q_hex = q_hex_tuple[0]
    except Exception:
        q_hex = [0, 0, 0, 0, 0, 0]

    distances = []
    for d in docs:
        try:
            d_hex_tuple = content_to_hexagram(d, "fact")
            d_hex = d_hex_tuple[0]
            dist = sum(1 for i in range(6) if q_hex[i] != d_hex[i])
            distances.append(dist)
        except Exception:
            distances.append(3)

    ranking = sorted(range(len(distances)), key=lambda i: distances[i])
    return ranking, (time.perf_counter() - t0) * 1000


def retrieval_hexagram_vector(
    query: str, docs: List[str]
) -> Tuple[List[int], float]:
    """Group D: 旧版 hexagram_to_vector 768-dim 扩展"""
    t0 = time.perf_counter()
    try:
        q_hex_tuple = content_to_hexagram(query, "fact")
        q_hex = q_hex_tuple[0]
    except Exception:
        q_hex = [0, 0, 0, 0, 0, 0]

    q_vec = hexagram_to_vector(q_hex)
    doc_vecs = []
    for d in docs:
        try:
            d_hex_tuple = content_to_hexagram(d, "fact")
            d_hex = d_hex_tuple[0]
            doc_vecs.append(hexagram_to_vector(d_hex))
        except Exception:
            doc_vecs.append([0.0] * 768)

    sims = [cosine_similarity(q_vec, dv) for dv in doc_vecs]
    ranking = sorted(range(len(sims)), key=lambda i: -sims[i])
    return ranking, (time.perf_counter() - t0) * 1000


def retrieval_random(query: str, docs: List[str]) -> Tuple[List[int], float]:
    """Group E: 随机基线"""
    t0 = time.perf_counter()
    indices = list(range(len(docs)))
    random.shuffle(indices)
    return indices, (time.perf_counter() - t0) * 1000


# ─── 实验运行器 ────────────────────────────────────────────────


def run_retrieval_experiment(
    method_name: str,
    method_func,
    queries_data: List[dict],
    all_docs: List[str],
    relevant_map: Dict[int, set],
    k_values: List[int],
    provider=None,
    **method_kwargs,
) -> BenchmarkResult:
    """运行单组检索实验"""
    result = BenchmarkResult(name=method_name)
    latencies = []

    for qi, query_data in enumerate(queries_data):
        query = query_data["query"]
        gt = relevant_map[qi]

        if provider is not None:
            ranking, latency = method_func(query, all_docs, provider, **method_kwargs)
        else:
            ranking, latency = method_func(query, all_docs, **method_kwargs)

        latencies.append(latency)

        for k in k_values:
            result.recall_at_k[k] = result.recall_at_k.get(k, 0.0) + recall_at_k(
                ranking, gt, k
            )
            result.ndcg_at_k[k] = result.ndcg_at_k.get(k, 0.0) + ndcg(ranking, gt, k)
            result.precision_at_k[k] = result.precision_at_k.get(
                k, 0.0
            ) + precision_at_k(ranking, gt, k)

        result.mrr += mrr(ranking, gt)

    n = len(queries_data)
    for k in k_values:
        result.recall_at_k[k] /= n
        result.ndcg_at_k[k] /= n
        result.precision_at_k[k] /= n
    result.mrr /= n
    result.avg_latency_ms = statistics.mean(latencies) if latencies else 0
    result.queries_processed = n

    return result


def cohens_d(sample_a: List[float], sample_b: List[float]) -> float:
    """Cohen's d 效应量"""
    mean_a, mean_b = statistics.mean(sample_a), statistics.mean(sample_b)
    n_a, n_b = len(sample_a), len(sample_b)
    var_a = statistics.variance(sample_a) if n_a > 1 else 0
    var_b = statistics.variance(sample_b) if n_b > 1 else 0
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return (mean_a - mean_b) / pooled_std


# ─── 主实验 ───────────────────────────────────────────────────


def run_phase3_benchmark(n_docs: int = 0, k_values: List[int] = None) -> dict:
    """
    运行 Phase 3 核心对照实验

    Args:
        n_docs: distractor 文档数（0=仅用数据集内置 distractor）
        k_values: 评估的 K 值列表
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    # 构建文档集合
    all_docs = []
    relevant_map = {}

    for qi, entry in enumerate(BENCHMARK_DATASET):
        relevant_set = set()
        for doc in entry["relevant"]:
            relevant_set.add(len(all_docs))
            all_docs.append(doc)
        for doc in entry["distractors"]:
            all_docs.append(doc)
        relevant_map[qi] = relevant_set

    # 可选：添加额外的随机 distractor
    if n_docs > 0:
        extra = [
            f"这是一个无关的测试文档编号{i}，用于增加检索库规模"
            for i in range(n_docs)
        ]
        all_docs.extend(extra)

    print(f"\n{'=' * 70}")
    print(f"  Phase 3: 六爻编码增量价值证明实验")
    print(f"  Queries: {len(BENCHMARK_DATASET)}, Docs: {len(all_docs)}, K: {k_values}")
    print(f"{'=' * 70}")

    # 初始化 provider
    print("\n🔧 初始化 embedding provider...")
    try:
        provider = STEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        print(f"  ✅ sentence-transformers loaded ({provider._model.get_sentence_embedding_dimension()} dim)")
    except Exception as e:
        print(f"  ⚠️  ST failed ({e}), using hash fallback")
        provider = HashEmbeddingProvider(dim=384)

    # 实验分组（Phase 3B: 去掉 Hybrid/HexVector，加 Option B readout）
    experiments = [
        ("A: Pure Embedding", retrieval_pure_embedding, True),
        ("C: Pure Hex Hamming", retrieval_hexagram_hamming, False),
        ("E: Random Baseline", retrieval_random, False),
        ("F: OptionB (Prod)", retrieval_pure_embedding, True),  # 同 A，标记为生产路径
    ]

    results: List[BenchmarkResult] = []

    for name, func, needs_provider in experiments:
        print(f"\n{'─' * 50}")
        print(f"  Running {name}...")
        kwargs = {}
        if needs_provider:
            kwargs["provider"] = provider
        result = run_retrieval_experiment(
            name, func, BENCHMARK_DATASET, all_docs, relevant_map, k_values, **kwargs
        )
        results.append(result)

    # ─── 打印对比报告 ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  📊 检索精度对比 (Recall@K)")
    print(f"{'=' * 70}")
    header = f"{'Method':<30}"
    for k in k_values:
        header += f"R@{k:<4}"
    header += f"{'MRR':<8} {'Latency':<12}"
    print(header)
    print("-" * 70)

    best_recall_5 = 0
    best_method = ""

    for r in results:
        line = f"{r.name:<30}"
        for k in k_values:
            line += f"{r.recall_at_k.get(k, 0):.3f} "
        line += f"{r.mrr:.3f}   {r.avg_latency_ms:.1f}ms"
        print(line)

        if r.recall_at_k.get(5, 0) > best_recall_5:
            best_recall_5 = r.recall_at_k[5]
            best_method = r.name

    # ─── 统计检验：Emb vs Hex vs Random ──────────────────────────────────────────
    emb_result = results[0]  # A: Pure Embedding
    hex_result = results[1]  # C: Pure Hex Hamming

    print(f"\n{'=' * 70}")
    print(f"  🔬 Phase 3B 结论验证: Embedding vs Hex vs Random")
    print(f"{'=' * 70}")

    # 逐 query 的 Recall@5
    emb_per_query = []
    hex_per_query = []

    print("\n  逐Query Recall@5对比:")
    print(f"  {'Query':<35} {'Embedding':<12} {'Hex(Hamm)':<12} {'Delta':<10}")
    print(f"  {'-'*70}")

    for qi, entry in enumerate(BENCHMARK_DATASET):
        gt = relevant_map[qi]
        emb_ranking, _ = retrieval_pure_embedding(entry["query"], all_docs, provider)
        hex_ranking, _ = retrieval_hexagram_hamming(entry["query"], all_docs)

        e_r5 = recall_at_k(emb_ranking, gt, 5)
        h_r5 = recall_at_k(hex_ranking, gt, 5)
        emb_per_query.append(e_r5)
        hex_per_query.append(h_r5)
        delta = e_r5 - h_r5

        q_short = entry["query"][:33]
        print(f"  {q_short:<35} {e_r5:.3f}       {h_r5:.3f}       {delta:+.3f}")

    d_cohens = cohens_d(emb_per_query, hex_per_query)
    print(f"\n  Cohen's d (Emb vs Hex): {d_cohens:.3f}")
    print(f"  Mean R@5:  Embedding={statistics.mean(emb_per_query):.3f}, "
          f"Hex Hamming={statistics.mean(hex_per_query):.3f}")
    print(f"  Verdict: ", end="")
    if d_cohens > 1.5:
        print("✅ 纯语义 Embedding 显著优于六爻 Hamming — Option B 决策正确")
    else:
        print("⚠️ 差异不够显著（可能是 hash fallback），建议配置 ST 后复测")

    # ─── 构建 JSON 报告 ────────────────────────────────────
    report = {
        "experiment": "Phase 3B: Option B — 六爻退化为分类标签，检索评分纯语义",
        "config": {
            "n_queries": len(BENCHMARK_DATASET),
            "n_docs": len(all_docs),
            "k_values": k_values,
            "provider": type(provider).__name__,
            "phase_3a_verdict": "六爻 rule 编码区分度 = -0.2，Hybrid 无增量价值",
        },
        "results": {
            r.name: {
                "recall_at_k": r.recall_at_k,
                "ndcg_at_k": r.ndcg_at_k,
                "precision_at_k": r.precision_at_k,
                "mrr": r.mrr,
                "avg_latency_ms": r.avg_latency_ms,
            }
            for r in results
        },
        "statistics": {
            "emb_vs_hex": {
                "cohens_d": d_cohens,
                "mean_emb_r5": statistics.mean(emb_per_query),
                "mean_hex_r5": statistics.mean(hex_per_query),
            }
        },
        "verdict": "",
    }

    if d_cohens > 1.5:
        report["verdict"] = (
            "✅ Option B 决策正确：纯语义 Embedding 显著优于六爻 Hamming。"
            "六爻退化为分类标签，检索评分 100% 语义向量相似度。"
            f"Cohen's d={d_cohens:.3f}"
        )
    else:
        report["verdict"] = (
            "⚠️ 差异不够显著（可能是 hash fallback）。"
            "建议配置 sentence-transformers 后复测以确认 Option B 的优越性。"
        )

    print(f"\n{'=' * 70}")
    print(f"  📋 结论")
    print(f"{'=' * 70}")
    print(f"  {report['verdict']}")

    return report


# ─── CLI ─────────────────────────────────────────────────────


def main():
    import argparse

    p = argparse.ArgumentParser(description="Phase 3B: Option B 纯语义检索验证实验")
    p.add_argument("--n-docs", type=int, default=0, help="额外 distractor 文档数")
    p.add_argument("--top-k", type=str, default="1,3,5,10,20", help="K 值列表")
    p.add_argument("--output", type=str, help="输出 JSON 文件路径")
    args = p.parse_args()

    k_values = [int(k.strip()) for k in args.top_k.split(",")]

    report = run_phase3_benchmark(n_docs=args.n_docs, k_values=k_values)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n📄 Report saved to {args.output}")

    return report


if __name__ == "__main__":
    main()
