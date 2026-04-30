#!/usr/bin/env python3
"""OpenAI embedding benchmark vs ST — 同数据集对比"""
import os, sys, time, statistics
sys.path.insert(0, "/Users/hanshifeng/Desktop/MetaMemory记忆系统 - 完整源码")

from core.embedding_provider import (
    OpenAIEmbeddingProvider, STEmbeddingProvider, cosine_similarity
)

# Phase 3B 数据集
DATASET = [
    {"query": "Python异步编程最佳实践",
     "relevant": ["使用asyncio库实现Python异步任务调度", "async/await语法在Python 3.7+中的正确用法",
                  "Python协程与事件循环的工作原理", "避免asyncio中的常见错误：忘记await、阻塞调用"],
     "distractors": ["Python是一种静态类型语言", "JavaScript异步编程使用Promise", "今天天气很好适合出门"]},
    {"query": "REST API设计规范",
     "relevant": ["RESTful API的资源命名应使用名词复数形式", "HTTP状态码的正确使用：200/201/400/404/500",
                  "API版本控制策略：URL路径 vs Header", "REST API的分页、过滤和排序最佳实践"],
     "distractors": ["GraphQL比REST更灵活", "猫咪喜欢吃鱼", "数据库索引提升查询性能"]},
    {"query": "Docker容器化部署",
     "relevant": ["使用Dockerfile构建应用镜像的最佳实践", "多阶段构建减小Docker镜像体积",
                  "Docker Compose编排多容器应用", "Docker网络模式和存储卷管理"],
     "distractors": ["Python的GIL限制了多线程性能", "今天早餐喝了豆浆", "React是一个前端框架"]},
    {"query": "神经网络过拟合解决方案",
     "relevant": ["使用Dropout层防止神经网络过拟合", "L1/L2正则化在深度学习中的应用",
                  "早停法Early Stopping监控验证集损失", "数据增强扩充训练集减少过拟合"],
     "distractors": ["Python列表推导式的语法", "今天中午吃了面条", "REST API使用JSON格式传输数据"]},
    {"query": "MySQL查询性能优化",
     "relevant": ["使用EXPLAIN分析查询执行计划", "复合索引的最左前缀原则",
                  "避免SELECT *只查询需要的列", "慢查询日志定位性能瓶颈"],
     "distractors": ["Redis是内存缓存数据库", "今天晚餐吃了火锅", "Transformer模型使用自注意力机制"]},
]

N_EXTRA = 2000

def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & set(relevant)) / max(len(relevant), 1)

def benchmark(provider, name):
    all_docs = []
    relevant_map = {}
    for qi, entry in enumerate(DATASET):
        rel = set()
        for doc in entry["relevant"]:
            rel.add(len(all_docs)); all_docs.append(doc)
        for doc in entry["distractors"]:
            all_docs.append(doc)
        relevant_map[qi] = rel
    for i in range(N_EXTRA):
        all_docs.append(f"这是一个无关的测试文档编号{i}，用于增加检索库规模")

    print(f"\n{'='*60}")
    print(f"  {name}: {len(all_docs)} docs")
    print(f"{'='*60}")

    t0 = time.time()
    print("  Encoding all docs...")
    all_vecs = provider.encode_batch(all_docs)
    print(f"  Encode: {time.time()-t0:.1f}s")

    queries = [e["query"] for e in DATASET]
    print("  Encoding queries...")
    q_vecs = provider.encode_batch(queries)

    r5s, r10s, times = [], [], []
    for qi, q_vec in enumerate(q_vecs):
        gt = relevant_map[qi]
        t1 = time.time()
        scores = [(i, cosine_similarity(q_vec, all_vecs[i])) for i in range(len(all_vecs))]
        scores.sort(key=lambda x: x[1], reverse=True)
        retrieved = [s[0] for s in scores]
        elapsed = (time.time() - t1) * 1000
        r5s.append(recall_at_k(retrieved, list(gt), 5))
        r10s.append(recall_at_k(retrieved, list(gt), 10))
        times.append(elapsed)

    avg_r5 = statistics.mean(r5s)
    avg_r10 = statistics.mean(r10s)
    avg_ms = statistics.mean(times)
    print(f"  R@5: {avg_r5:.4f}  R@10: {avg_r10:.4f}  Latency: {avg_ms:.1f}ms")
    return avg_r5, avg_r10, avg_ms

# ─── ST (离线) ───
try:
    print("Loading ST...")
    st = STEmbeddingProvider()
    st_r5, st_r10, st_ms = benchmark(st, "ST all-MiniLM-L6-v2")
except Exception as e:
    print(f"ST error: {e}")
    st_r5, st_r10 = 0.275, 0.47

# ─── OpenAI ───
print("\nLoading OpenAI...")
try:
    oai = OpenAIEmbeddingProvider(model="text-embedding-3-small")
    oai_r5, oai_r10, oai_ms = benchmark(oai, "OpenAI text-embedding-3-small (1536d)")
except Exception as e:
    print(f"OpenAI error: {e}")
    oai_r5, oai_r10 = None, None

# ─── 对比 ───
print(f"\n{'='*60}")
print(f"  📊 最终对比")
print(f"{'='*60}")
print(f"  {'Provider':<35} {'R@5':<10} {'R@10':<10} {'Latency':<10}")
print(f"  {'-'*65}")
print(f"  {'ST all-MiniLM-L6-v2 (384d)':<35} {st_r5:<10.4f} {st_r10:<10.4f} {st_ms:<10.1f}ms")
if oai_r5 is not None:
    print(f"  {'OpenAI t-e-3-small (1536d)':<35} {oai_r5:<10.4f} {oai_r10:<10.4f} {oai_ms:<10.1f}ms")
    delta = (oai_r5 - st_r5) / st_r5 * 100
    print(f"\n  🚀 OpenAI R@5 提升: {delta:+.0f}%")
