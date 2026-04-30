#!/usr/bin/env python3
"""
路线 B: MySQL 生产验证 — 规模化测试（2086 文档对齐 Phase 3B）
=================================================================
对比 Phase 3B 的纯 Python benchmark vs 真实 MySQL 存储 + embedding 检索。
21 条 relevant docs + 41 条 distractor docs (Phase 3B 数据) + 2000 条额外 distractor = ~2062 文档
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys, time, json, statistics, hashlib
from datetime import datetime

sys.path.insert(0, "/Users/hanshifeng/Desktop/MetaMemory记忆系统 - 完整源码")

from core.mysql_store import DatabaseManager, MySQLMemoryStore
from core.embedding_provider import (
    STEmbeddingProvider, HashEmbeddingProvider, cosine_similarity,
)
from core.core_yijing import content_to_hexagram

# ─── Phase 3B 完整数据集 ───
BENCHMARK_DATASET = [
    {"query": "Python异步编程最佳实践",
     "relevant": ["使用asyncio库实现Python异步任务调度", "async/await语法在Python 3.7+中的正确用法",
                  "Python协程与事件循环的工作原理", "避免asyncio中的常见错误：忘记await、阻塞调用"],
     "distractors": ["Python是一种静态类型语言", "JavaScript异步编程使用Promise",
                     "今天天气很好适合出门", "早餐吃了面包和牛奶", "机器学习模型需要大量数据训练"]},
    {"query": "REST API设计规范",
     "relevant": ["RESTful API的资源命名应使用名词复数形式", "HTTP状态码的正确使用：200/201/400/404/500",
                  "API版本控制策略：URL路径 vs Header", "REST API的分页、过滤和排序最佳实践",
                  "HATEOAS超媒体驱动的REST架构"],
     "distractors": ["GraphQL比REST更灵活", "猫咪喜欢吃鱼", "数据库索引提升查询性能", "今天下午有暴雨预警"]},
    {"query": "Docker容器化部署",
     "relevant": ["使用Dockerfile构建应用镜像的最佳实践", "多阶段构建减小Docker镜像体积",
                  "Docker Compose编排多容器应用", "Docker网络模式和存储卷管理"],
     "distractors": ["Python的GIL限制了多线程性能", "今天早餐喝了豆浆", "React是一个前端框架",
                     "微服务架构的优势和挑战", "深度学习模型训练需要GPU加速"]},
    {"query": "神经网络过拟合解决方案",
     "relevant": ["使用Dropout层防止神经网络过拟合", "L1/L2正则化在深度学习中的应用",
                  "早停法Early Stopping监控验证集损失", "数据增强扩充训练集减少过拟合",
                  "交叉验证评估模型泛化能力"],
     "distractors": ["Python列表推导式的语法", "今天中午吃了面条", "REST API使用JSON格式传输数据",
                     "Docker容器需要健康检查"]},
    {"query": "Transformer注意力机制原理",
     "relevant": ["Self-Attention计算Query、Key、Value矩阵", "Multi-Head Attention多角度捕捉语义关系",
                  "位置编码Positional Encoding补充序列信息", "Transformer的Encoder-Decoder架构详解",
                  "注意力机制中的缩放因子sqrt(dk)作用"],
     "distractors": ["卷积神经网络适合图像处理", "今天天气晴朗适合出游", "微服务间的通信方式", "MySQL索引优化策略"]},
    {"query": "模型评估指标选择",
     "relevant": ["分类问题使用Accuracy、Precision、Recall、F1", "回归问题使用MSE、MAE、R-squared",
                  "不平衡数据集使用AUC-ROC代替Accuracy", "排序问题使用MRR、NDCG、MAP评估"],
     "distractors": ["Docker容器的资源限制配置", "猫咪在沙发上睡觉", "RESTful API使用HTTP动词",
                     "Python的装饰器语法糖", "今天地铁晚点了"]},
    {"query": "MySQL查询性能优化",
     "relevant": ["使用EXPLAIN分析查询执行计划", "复合索引的最左前缀原则", "避免SELECT *只查询需要的列",
                  "慢查询日志定位性能瓶颈", "InnoDB存储引擎的聚簇索引原理"],
     "distractors": ["Redis是内存缓存数据库", "今天晚餐吃了火锅", "Transformer模型使用自注意力机制",
                     "Docker镜像推送到容器仓库"]},
    {"query": "Redis缓存策略设计",
     "relevant": ["缓存穿透使用布隆过滤器防护", "缓存雪崩设置过期时间随机偏移", "缓存击穿使用互斥锁或永不过期",
                  "Redis数据淘汰策略LRU/LFU/TTL", "缓存与数据库双写一致性方案"],
     "distractors": ["Python异步编程使用async/await", "今天天气有点冷", "神经网络需要激活函数", "REST API返回标准状态码"]},
    {"query": "微服务间通信方式",
     "relevant": ["同步通信使用HTTP/REST或gRPC", "异步通信使用消息队列RabbitMQ/Kafka",
                  "服务发现使用Consul/Eureka/Nacos", "API网关统一入口Zuul/Kong/APISIX",
                  "服务间调用链追踪使用Jaeger/Zipkin"],
     "distractors": ["Python支持面向对象编程", "今天早餐吃了包子", "卷积神经网络用于图像识别", "MySQL支持事务ACID特性"]},
    {"query": "分布式事务解决方案",
     "relevant": ["两阶段提交2PC保证强一致性", "TCC补偿事务Try-Confirm-Cancel模式",
                  "Saga长事务的最终一致性方案", "本地消息表+MQ实现异步确保", "Seata分布式事务框架的使用"],
     "distractors": ["Dockerfile的FROM指令", "猫咪喜欢捉老鼠", "深度学习中的梯度消失问题", "Redis的String数据结构"]},
]

N_EXTRA_DISTRACTORS = 2000

def recall_at_k(retrieved, relevant, k):
    s = set(retrieved[:k]) & set(relevant)
    return len(s) / max(len(relevant), 1)

def mrr(retrieved, relevant):
    s = set(relevant)
    for i, rid in enumerate(retrieved):
        if rid in s:
            return 1.0 / (i + 1)
    return 0.0

print("=" * 70)
print("  路线 B: MySQL 生产验证 — 规模化测试 (含 2000 distractors)")
print("=" * 70)
print(f"  开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ─── 1. DB + Embedding 初始化 ───
print("\n─── 1. 初始化 ───")
db_mgr = DatabaseManager(database_url="mysql+pymysql://root:@localhost:3306/memory_system")
db_mgr.create_tables()
store = MySQLMemoryStore(db_mgr)
print(f"  ✅ DB: {db_mgr.health_check()}")

provider = STEmbeddingProvider()
dim = provider.dim
print(f"  ✅ Embedding: ST(all-MiniLM-L6-v2), dim={dim}")

# ─── 2. 清表 + 构建文档列表 ───
print("\n─── 2. 建文档索引 (Phase 3B 数据 + 2000 distractors) ───")
with db_mgr.get_session() as session:
    from sqlalchemy import text
    session.execute(text("DELETE FROM yijing_memory"))
    session.execute(text("DELETE FROM yao_change_history"))
    session.execute(text("DELETE FROM taiji_core_config"))
    session.commit()
store._memory_cache.clear()

all_docs = []
relevant_map = {}
for qi, entry in enumerate(BENCHMARK_DATASET):
    rel_set = set()
    for doc in entry["relevant"]:
        rel_set.add(len(all_docs))
        all_docs.append((doc, qi, True))
    for doc in entry["distractors"]:
        all_docs.append((doc, qi, False))
    relevant_map[qi] = rel_set

# 额外 distractor docs (与 Phase 3B 一致的中文模板)
for i in range(N_EXTRA_DISTRACTORS):
    all_docs.append((f"这是一个无关的测试文档编号{i}，用于增加检索库规模", -1, False))

print(f"  总文档数: {len(all_docs)} (relevant: {sum(len(v) for v in relevant_map.values())}, "
      f"distractor: {len(all_docs) - sum(len(v) for v in relevant_map.values())})")

# ─── 3. 批量写入 MySQL ───
print("\n─── 3. 批量写入 MySQL ───")
batch_size = 100
total_written = 0
errors = 0
write_start = time.time()

# 预先批量 encode 所有文档
print("  预计算 embedding (batch)...")
emb_start = time.time()
all_contents = [d[0] for d in all_docs]
all_embeddings = []
for i in range(0, len(all_contents), batch_size):
    batch = all_contents[i:i+batch_size]
    embs = provider.encode_batch(batch)
    all_embeddings.extend(embs)
    if (i // batch_size) % 5 == 0:
        print(f"    encode {i+len(batch)}/{len(all_contents)}...")
emb_time = time.time() - emb_start
print(f"  ✅ Embedding 完成: {len(all_embeddings)} vectors, {emb_time:.1f}s")

print("  写入 MySQL...")
ins_start = time.time()
for idx, ((content, qi, is_rel), emb) in enumerate(zip(all_docs, all_embeddings)):
    hex_tuple = content_to_hexagram(content, "fact")
    hex_str = "".join(str(b) for b in hex_tuple[0]) if hex_tuple else "000000"

    mem_id = f"scale_{idx}"
    memory_data = {
        "memory_id": mem_id, "agent_id": "scale_agent",
        "content": content, "hexagram": hex_str,
        "bagua_type": "乾", "sancai_layer": "天", "wuxing": "金",
        "hot_score": 0.5, "position_status": "当位",
    }
    try:
        store.create_memory(memory_data)
        total_written += 1
    except Exception:
        errors += 1
        if errors <= 3:
            print(f"  ⚠️ 写入失败 [{mem_id}]")

    if total_written % 500 == 0:
        print(f"    已写入 {total_written}/{len(all_docs)}...")

ins_elapsed = time.time() - ins_start
total_elapsed = time.time() - write_start
print(f"  ✅ 写入 {total_written}/{len(all_docs)} 条 (错误: {errors})")
print(f"  写入耗时: {ins_elapsed:.1f}s, 总耗时(含encode): {total_elapsed:.1f}s")
print(f"  写入 QPS: {total_written/ins_elapsed:.1f}, 批次 QPS: {total_written/total_elapsed:.1f}")

# ─── 4. Embedding 检索 benchmark ───
print("\n─── 4. Embedding 检索 benchmark ───")
# 预编码所有 query
queries = [entry["query"] for entry in BENCHMARK_DATASET]
query_embs = [provider.encode(q) for q in queries]

retrieval_times = []
all_r5, all_r10, all_mrr = [], [], []

for qi, query_emb in enumerate(query_embs):
    gt = relevant_map[qi]

    r_start = time.time()
    scores = []
    for idx, doc_emb in enumerate(all_embeddings):
        scores.append((idx, cosine_similarity(query_emb, doc_emb)))

    scores.sort(key=lambda x: x[1], reverse=True)
    retrieved = [s[0] for s in scores]
    r_elapsed = (time.time() - r_start) * 1000

    retrieval_times.append(r_elapsed)
    all_r5.append(recall_at_k(retrieved, list(gt), 5))
    all_r10.append(recall_at_k(retrieved, list(gt), 10))
    all_mrr.append(mrr(retrieved, list(gt)))

avg_retrieval_ms = statistics.mean(retrieval_times)
avg_r5 = statistics.mean(all_r5)
avg_r10 = statistics.mean(all_r10)
avg_mrr = statistics.mean(all_mrr)

print(f"  检索延迟 (avg): {avg_retrieval_ms:.1f}ms")
print(f"  Recall@5: {avg_r5:.4f}")
print(f"  Recall@10: {avg_r10:.4f}")
print(f"  MRR: {avg_mrr:.4f}")

# ─── 5. 逐 Query 明细 ───
print(f"\n  {'Query':<30} {'R@5':<8} {'R@10':<8} {'MRR':<8} {'Lat(ms)':<10}")
print(f"  {'-'*65}")
for qi, q in enumerate(queries):
    print(f"  {q[:28]:<30} {all_r5[qi]:.4f}  {all_r10[qi]:.4f}  {all_mrr[qi]:.4f}  {retrieval_times[qi]:.1f}")

# ─── 6. 汇总 ───
print("\n" + "=" * 70)
print("  📊 路线 B vs 路线 A (Phase 3B) 最终对比")
print("=" * 70)

results = {
    "timestamp": datetime.now().isoformat(),
    "config": {
        "total_docs": len(all_docs),
        "n_relevant": sum(len(v) for v in relevant_map.values()),
        "n_distractors": len(all_docs) - sum(len(v) for v in relevant_map.values()),
        "embedding": "ST(all-MiniLM-L6-v2) 384d",
        "database": "MySQL 5.7.24 local",
    },
    "write": {
        "total_written": total_written, "errors": errors,
        "insert_sec": round(ins_elapsed, 1),
        "total_sec": round(total_elapsed, 1),
        "qps_insert": round(total_written/ins_elapsed, 1),
        "qps_total": round(total_written/total_elapsed, 1),
    },
    "retrieval": {
        "n_queries": len(queries),
        "avg_latency_ms": round(avg_retrieval_ms, 1),
        "recall_at_5": round(avg_r5, 4),
        "recall_at_10": round(avg_r10, 4),
        "mrr": round(avg_mrr, 4),
    },
    "comparison": {
        "phase3b_r5": 0.143, "phase3b_r10": 0.345, "phase3b_mrr": 0.298,
        "mysql_r5": round(avg_r5, 4), "mysql_r10": round(avg_r10, 4), "mysql_mrr": round(avg_mrr, 4),
    },
}

print(json.dumps(results, indent=2, ensure_ascii=False))

output_path = "/Users/hanshifeng/Desktop/route_b_mysql_report.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n📄 报告: {output_path}")
