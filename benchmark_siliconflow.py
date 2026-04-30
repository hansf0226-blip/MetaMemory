#!/usr/bin/env python3
"""SiliconFlow BGE vs ST 对比 — 同 Phase 3B 数据集"""
import os, time, statistics, sys
os.environ['OPENAI_API_KEY']='sk-tuufhabcjjdpfiztsfmwwaphfzchhrztnbxebvjemfloczje'
sys.path.insert(0, "/Users/hanshifeng/Desktop/MetaMemory记忆系统 - 完整源码")

from openai import OpenAI
from core.embedding_provider import STEmbeddingProvider, cosine_similarity

client = OpenAI(base_url='https://api.siliconflow.cn/v1')

DATASET = [
    ("Python异步编程最佳实践",
     ["使用asyncio库实现Python异步任务调度","async/await语法在Python 3.7+中的正确用法",
      "Python协程与事件循环的工作原理","避免asyncio中的常见错误：忘记await、阻塞调用"],
     ["Python是一种静态类型语言","JavaScript异步编程使用Promise","今天天气很好适合出门","早餐吃了面包和牛奶","机器学习模型需要大量数据训练"]),
    ("REST API设计规范",
     ["RESTful API的资源命名应使用名词复数形式","HTTP状态码的正确使用：200/201/400/404/500",
      "API版本控制策略：URL路径 vs Header","REST API的分页、过滤和排序最佳实践","HATEOAS超媒体驱动的REST架构"],
     ["GraphQL比REST更灵活","猫咪喜欢吃鱼","数据库索引提升查询性能","今天下午有暴雨预警"]),
    ("Docker容器化部署",
     ["使用Dockerfile构建应用镜像的最佳实践","多阶段构建减小Docker镜像体积",
      "Docker Compose编排多容器应用","Docker网络模式和存储卷管理"],
     ["Python的GIL限制了多线程性能","今天早餐喝了豆浆","React是一个前端框架","微服务架构的优势和挑战","深度学习模型训练需要GPU加速"]),
    ("神经网络过拟合解决方案",
     ["使用Dropout层防止神经网络过拟合","L1/L2正则化在深度学习中的应用",
      "早停法Early Stopping监控验证集损失","数据增强扩充训练集减少过拟合","交叉验证评估模型泛化能力"],
     ["Python列表推导式的语法","今天中午吃了面条","REST API使用JSON格式传输数据","Docker容器需要健康检查"]),
    ("Transformer注意力机制原理",
     ["Self-Attention计算Query、Key、Value矩阵","Multi-Head Attention多角度捕捉语义关系",
      "位置编码Positional Encoding补充序列信息","Transformer的Encoder-Decoder架构详解","注意力机制中的缩放因子sqrt(dk)作用"],
     ["卷积神经网络适合图像处理","今天天气晴朗适合出游","微服务间的通信方式","MySQL索引优化策略"]),
    ("模型评估指标选择",
     ["分类问题使用Accuracy、Precision、Recall、F1","回归问题使用MSE、MAE、R-squared",
      "不平衡数据集使用AUC-ROC代替Accuracy","排序问题使用MRR、NDCG、MAP评估"],
     ["Docker容器的资源限制配置","猫咪在沙发上睡觉","RESTful API使用HTTP动词","Python的装饰器语法糖","今天地铁晚点了"]),
    ("MySQL查询性能优化",
     ["使用EXPLAIN分析查询执行计划","复合索引的最左前缀原则","避免SELECT *只查询需要的列",
      "慢查询日志定位性能瓶颈","InnoDB存储引擎的聚簇索引原理"],
     ["Redis是内存缓存数据库","今天晚餐吃了火锅","Transformer模型使用自注意力机制","Docker镜像推送到容器仓库"]),
    ("Redis缓存策略设计",
     ["缓存穿透使用布隆过滤器防护","缓存雪崩设置过期时间随机偏移","缓存击穿使用互斥锁或永不过期",
      "Redis数据淘汰策略LRU/LFU/TTL","缓存与数据库双写一致性方案"],
     ["Python异步编程使用async/await","今天天气有点冷","神经网络需要激活函数","REST API返回标准状态码"]),
    ("微服务间通信方式",
     ["同步通信使用HTTP/REST或gRPC","异步通信使用消息队列RabbitMQ/Kafka",
      "服务发现使用Consul/Eureka/Nacos","API网关统一入口Zuul/Kong/APISIX","服务间调用链追踪使用Jaeger/Zipkin"],
     ["Python支持面向对象编程","今天早餐吃了包子","卷积神经网络用于图像识别","MySQL支持事务ACID特性"]),
    ("分布式事务解决方案",
     ["两阶段提交2PC保证强一致性","TCC补偿事务Try-Confirm-Cancel模式",
      "Saga长事务的最终一致性方案","本地消息表+MQ实现异步确保","Seata分布式事务框架的使用"],
     ["Dockerfile的FROM指令","猫咪喜欢捉老鼠","深度学习中的梯度消失问题","Redis的String数据结构"]),
]

N_EXTRA = 2000
BATCH = 32

def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & set(relevant)) / max(len(relevant), 1)

def encode_batch_chunked(model_name, texts):
    """硅基流动每批最多 32 条，分包编码"""
    all_vecs = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i+BATCH]
        resp = client.embeddings.create(model=model_name, input=chunk)
        all_vecs.extend([list(d.embedding) for d in resp.data])
    return all_vecs

def build_docs():
    all_docs = []; relevant_map = {}
    for qi, (q, rel, dist) in enumerate(DATASET):
        rset = set()
        for d in rel: rset.add(len(all_docs)); all_docs.append(d)
        for d in dist: all_docs.append(d)
        relevant_map[qi] = rset
    for i in range(N_EXTRA):
        all_docs.append(f"这是一个无关的测试文档编号{i}，用于增加检索库规模")
    return all_docs, relevant_map

all_docs, relevant_map = build_docs()
n_docs = len(all_docs)
n_rel = sum(len(v) for v in relevant_map.values())
print(f"Total docs: {n_docs} (relevant: {n_rel}, distractors: {n_docs-n_rel})")

# ─── BGE-large-zh ───
model = "BAAI/bge-large-zh-v1.5"
print(f"\n{'='*60}\n  {model} (1024d)\n{'='*60}")
t0 = time.time()
all_vecs = encode_batch_chunked(model, all_docs)
print(f"  Encode: {time.time()-t0:.1f}s ({(time.time()-t0)/n_docs*1000:.1f}ms/doc)")

queries = [q for q,_,_ in DATASET]
q_vecs = encode_batch_chunked(model, queries)

r5s, r10s, mrr_s, times = [], [], [], []
for qi, qv in enumerate(q_vecs):
    gt = list(relevant_map[qi])
    t1 = time.time()
    scores = [(i, cosine_similarity(qv, all_vecs[i])) for i in range(n_docs)]
    scores.sort(key=lambda x: x[1], reverse=True)
    retrieved = [s[0] for s in scores]
    elapsed = (time.time()-t1)*1000
    r5s.append(recall_at_k(retrieved, gt, 5))
    r10s.append(recall_at_k(retrieved, gt, 10))
    mrr_s.append(1.0/(retrieved.index(gt[0])+1) if gt[0] in retrieved[:100] else 0)
    times.append(elapsed)

bge_r5 = statistics.mean(r5s)
bge_r10 = statistics.mean(r10s)
bge_mrr = statistics.mean(mrr_s)
bge_ms = statistics.mean(times)
print(f"  R@5={bge_r5:.4f}  R@10={bge_r10:.4f}  MRR={bge_mrr:.4f}  Lat={bge_ms:.0f}ms")

# ─── ST baseline ───
print(f"\n{'='*60}\n  ST all-MiniLM-L6-v2 (384d) baseline\n{'='*60}")
st = STEmbeddingProvider()
t0 = time.time()
st_vecs = st.encode_batch(all_docs)
print(f"  Encode: {time.time()-t0:.1f}s")

st_qvecs = st.encode_batch(queries)
st_r5s, st_r10s, st_mrr_s, st_times = [], [], [], []
for qi, qv in enumerate(st_qvecs):
    gt = list(relevant_map[qi])
    t1 = time.time()
    scores = [(i, cosine_similarity(qv, st_vecs[i])) for i in range(n_docs)]
    scores.sort(key=lambda x: x[1], reverse=True)
    retrieved = [s[0] for s in scores]
    st_r5s.append(recall_at_k(retrieved, gt, 5))
    st_r10s.append(recall_at_k(retrieved, gt, 10))
    st_mrr_s.append(1.0/(retrieved.index(gt[0])+1) if gt[0] in retrieved[:100] else 0)
    st_times.append((time.time()-t1)*1000)

st_r5 = statistics.mean(st_r5s)
st_r10 = statistics.mean(st_r10s)
st_mrr = statistics.mean(st_mrr_s)
st_ms = statistics.mean(st_times)
print(f"  R@5={st_r5:.4f}  R@10={st_r10:.4f}  MRR={st_mrr:.4f}  Lat={st_ms:.0f}ms")

# ─── 对比 ───
print(f"\n{'='*60}")
print(f"  📊 BGE 1024d vs ST 384d")
print(f"{'='*60}")
print(f"  {'Metric':<15} {'ST 384d':<12} {'BGE 1024d':<12} {'提升':<10}")
print(f"  {'-'*50}")
print(f"  {'Recall@5':<15} {st_r5:<12.4f} {bge_r5:<12.4f} {((bge_r5/st_r5-1)*100):+.0f}%")
print(f"  {'Recall@10':<15} {st_r10:<12.4f} {bge_r10:<12.4f} {((bge_r10/st_r10-1)*100):+.0f}%")
print(f"  {'MRR':<15} {st_mrr:<12.4f} {bge_mrr:<12.4f} {((bge_mrr/st_mrr-1)*100) if st_mrr>0 else 'N/A':.0f}%")
print(f"  {'Latency':<15} {st_ms:<12.0f}ms {bge_ms:<12.0f}ms")
