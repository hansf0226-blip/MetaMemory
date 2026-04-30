# MetaMemory 系统架构设计 v2.2

> **文档版本**：2.2 | **最后更新**：2026-04-29 | **作者**：MetaMemory 团队

---

## 1. 概述

MetaMemory 是一个面向 AI Agent 的长期记忆系统，为多租户 AI 应用提供记忆的持久化存储、语义检索、生命周期管理和可视化监控能力。

### 1.1 设计目标

| 目标 | 指标 |
|------|------|
| **检索精度** | R@5 ≥ 0.46, R@10 ≥ 0.73（SiliconFlow BGE 1024d） |
| **检索延迟** | P50 < 200ms, P99 < 500ms |
| **多租户隔离** | Agent 级别数据隔离，查询不透传 |
| **零运维依赖** | SQLite 默认，零配置即可运行 |
| **水平扩展** | 可选 MySQL 替换 SQLite，支持读写分离 |

### 1.2 架构哲学

- **约定优于配置**：默认即最佳实践（BGE 1024d + 纯语义检索），开箱即用
- **渐进式复杂度**：SQLite 起步 → MySQL 扩展；本地 ST 兜底 → 云端 BGE 增强
- **配置驱动**：所有行为通过 `config.yaml` 控制，无需改代码
- **前端零构建**：单 HTML 文件，浏览器直接打开，无需 Node.js/npm

---

## 2. 多层架构

```
                                ┌──────────────────────────────────────┐
                                │           用户 / AI Agent             │
                                └──────────┬───────────────┬───────────┘
                                           │               │
                              HTTP/REST    │               │  file:// 直开
                                           │               │
┌──────────────────────────────────────────▼───────────────▼───────────┐
│                          接入层 (Access Layer)                        │
│                                                                       │
│   ┌─────────────────────┐              ┌─────────────────────────┐   │
│   │   Nginx / Caddy     │              │   frontend/index.html   │   │
│   │  (反向代理 + 静态)   │              │   (SPA · Linear Dark)   │   │
│   │                     │              │                         │   │
│   │  · SSL 终端         │              │  · Tailwind CSS         │   │
│   │  · 限流 (rate-limit)│              │  · ECharts 图表         │   │
│   │  · Gzip 压缩        │              │  · 5 页面 SPA           │   │
│   │  · CORS 白名单      │              │  · 紫金配色 UI          │   │
│   └──────────┬──────────┘              └─────────────┬───────────┘   │
│              │                                       │                │
│              │  /api/v1/*                            │  fetch API     │
└──────────────┼───────────────────────────────────────┼────────────────┘
               │                                       │
┌──────────────▼───────────────────────────────────────▼────────────────┐
│                       应用层 (Application Layer)                       │
│                                                                        │
│   ┌──────────────────────────────────────────────────────────────┐    │
│   │                   FastAPI (main_api.py)                       │    │
│   │                   Version: 2.2.0 · 端口: 8000                 │    │
│   │                                                               │    │
│   │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │    │
│   │   │ 记忆 CRUD │ │ 向量检索  │ │ 统计监控  │ │  配置管理     │   │    │
│   │   │          │ │          │ │          │ │              │   │    │
│   │   │ POST     │ │ GET      │ │ GET      │ │ GET/PUT      │   │    │
│   │   │ GET/PUT  │ │ /search  │ │ /stats   │ │ /config      │   │    │
│   │   │ DELETE   │ │          │ │ /metrics │ │              │   │    │
│   │   └──────────┘ └──────────┘ └──────────┘ └──────────────┘   │    │
│   │                                                               │    │
│   │   中间件层:                                                    │    │
│   │   · CORS 中间件 (白名单)                                       │    │
│   │   · Request ID 追踪 (X-Request-ID + X-Process-Time-Ms)        │    │
│   │   · 异常处理 (8 种异常类型 → HTTP 状态码映射)                   │    │
│   │   · APScheduler 定时任务 (自愈 + 衰减 + 清理)                   │    │
│   └──────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                        核心引擎层 (Core Engine)                          │
│                                                                         │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
│   │  Embedding 引擎   │  │   Storage 管理    │  │   检索 引擎       │    │
│   │                  │  │                  │  │                  │    │
│   │  embedding_      │  │  storage_        │  │  hybrid_         │    │
│   │  provider.py     │  │  manager.py      │  │  retrieval.py    │    │
│   │                  │  │                  │  │                  │    │
│   │  ┌────────────┐  │  │  ┌────────────┐  │  │  ┌────────────┐  │    │
│   │  │ SiliconFlow│  │  │  │  MemoryDB  │  │  │  │ Semantic   │  │    │
│   │  │ BGE 1024d  │  │  │  │  (SQLite)  │  │  │  │ Score 1.0  │  │    │
│   │  ├────────────┤  │  │  ├────────────┤  │  │  ├────────────┤  │    │
│   │  │ ST MiniLM  │  │  │  │ MySQLStore │  │  │  │ Structure  │  │    │
│   │  │ 384d (兜底) │  │  │  │ (可选)     │  │  │  │ Score 0.0  │  │    │
│   │  ├────────────┤  │  │  ├────────────┤  │  │  └────────────┘  │    │
│   │  │ OpenAI     │  │  │  │ VectorDB   │  │  │                  │    │
│   │  │ 1536d(付费)│  │  │  │ (FAISS)    │  │  │  分类过滤:       │    │
│   │  ├────────────┤  │  │  └────────────┘  │  │  bagua / wuxing  │    │
│   │  │ Hash SHA256│  │  │                  │  │  / sancai_layer  │    │
│   │  │ 256d(极兜底)│  │  │  · 记忆图谱      │  │                  │    │
│   │  └────────────┘  │  │  · 身份策略      │  │  结果排序:       │    │
│   │                  │  │  · 生命周期      │  │  similarity DESC │    │
│   │  缓存: LRU 5000  │  │  · 自愈调度      │  │  hot_score 加权  │    │
│   │  TTL: 1h         │  │                  │  │                  │    │
│   └──────────────────┘  └──────────────────┘  └──────────────────┘    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                         存储层 (Storage Layer)                           │
│                                                                         │
│   ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐   │
│   │    SQLite       │  │    FAISS       │  │    config.yaml          │   │
│   │                 │  │                │  │                         │   │
│   │  data/          │  │  In-Memory     │  │  · embedding.provider   │   │
│   │  memory_system  │  │  Index         │  │  · retrieval.top_k      │   │
│   │  .db            │  │                │  │  · database.type        │   │
│   │                 │  │  HNSW 索引     │  │  · yijing.encode_mode   │   │
│   │  表:             │  │  1024 维       │  │                         │   │
│   │  yijing_memory  │  │                │  │  .env                   │   │
│   │  agents         │  │  持久化:        │  │  · SILICONFLOW_API_KEY  │   │
│   │                 │  │  data/*.faiss  │  │  · OPENAI_API_KEY       │   │
│   │  索引:           │  │                │  │                         │   │
│   │  agent_id       │  │  检索:          │  │                         │   │
│   │  memory_id(UK)  │  │  cosine 距离    │  │                         │   │
│   │  bagua_type     │  │  top-k 返回     │  │                         │   │
│   │  wuxing         │  │                │  │                         │   │
│   └────────────────┘  └────────────────┘  └────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心数据流

### 3.1 写入路径（记忆创建）

```
Client (AI Agent)
  │
  │  POST /api/v1/memories
  │  {agent_id, content, bagua_type, wuxing, sancai_layer, ...}
  ▼
FastAPI Router → MemorySaveReq 校验
  │
  ▼
StorageManager.create_memory()
  │
  ├─► MemoryDB (SQLite)    → 写入 yijing_memory 表
  │     · agent_id, memory_id (UUID), content
  │     · bagua_type, wuxing, sancai_layer
  │     · hot_score, weight, created_at
  │
  ├─► EmbeddingProvider   → 对 content 生成 1024 维向量
  │     · SiliconFlow BGE API (默认)
  │     · 缓存命中直接返回（LRU, TTL 1h）
  │
  └─► VectorDB (FAISS)    → 插入向量 + metadata
        · 构建 HNSW 索引
        · 关联 memory_id / agent_id
```

### 3.2 检索路径（语义搜索）

```
Client (AI Agent)
  │
  │  GET /api/v1/memories/search?content=查询文本&top_k=10
  ▼
FastAPI Router → search_memories()
  │
  ▼
StorageManager.search_memories()
  │
  ├─► EmbeddingProvider → 查询向量化 (1024d)
  │
  ├─► VectorDB.search()  → FAISS 余弦相似度检索
  │     · 召回 top_k * 3 候选 (粗排)
  │
  ├─► MemoryDB.filter()  → 按 agent_id 过滤 (租户隔离)
  │
  ├─► HybridRetrievalEngine.rank()
  │     · 语义分数: cosine_similarity × 1.0
  │     · 结构分数: 0.0 (v2.2 纯语义，不使用六爻编码)
  │     · 热度加权: hot_score 正相关
  │     · 最终排序: 综合分数 DESC
  │
  └─► 返回 Top-K 结果
       · memory_id, content[:100], similarity, hot_score
```

### 3.3 配置热更新路径

```
Client (前端 / API)
  │
  │  PUT /api/v1/config  {embedding: {provider: "openai"}, retrieval: {top_k: 20}}
  ▼
FastAPI → update_system_config()
  │
  ├─► 读取当前 config.yaml
  ├─► deep_merge(现有配置, 请求数据)
  ├─► 写入 config.yaml (YAML dump)
  └─► 返回 {"restart_required": true}
```

---

## 4. 部署架构

### 4.1 单机部署（开发/小规模）

```
┌─────────────────────────────────────┐
│           单台服务器                  │
│                                     │
│   ┌──────────┐    ┌──────────────┐  │
│   │ Nginx    │    │ MetaMemory   │  │
│   │ :80/443  │───►│ :8000        │  │
│   │ (静态+API)│    │              │  │
│   └──────────┘    │ ├─ SQLite    │  │
│                   │ ├─ FAISS     │  │
│                   │ └─ Scheduler │  │
│                   └──────────────┘  │
│                                     │
│   适用: 记忆 < 10 万条，QPS < 100    │
└─────────────────────────────────────┘
```

### 4.2 生产部署（中大规模）

```
                         ┌──────────────┐
                         │   CDN / LB   │
                         └───┬────┬─────┘
                             │    │
              ┌──────────────▼┐  ┌▼──────────────┐
              │  Nginx #1     │  │  Nginx #2      │
              │  (静态 + 反向)  │  │  (静态 + 反向)  │
              └──────┬────────┘  └──────┬─────────┘
                     │                  │
         ┌───────────▼──────┐  ┌────────▼────────┐
         │ MetaMemory #1    │  │ MetaMemory #2   │
         │ :8000            │  │ :8000           │
         └───────┬──────────┘  └───────┬─────────┘
                 │                     │
         ┌───────▼─────────────────────▼─────────┐
         │            MySQL (主从)                │
         │            :3306                       │
         └───────────────────────────────────────┘

适用: 记忆 > 10 万条，QPS > 100，需高可用
```

### 4.3 Docker 部署

```bash
# docker-compose.yml
version: '3.8'
services:
  metamemory:
    image: metamemory:2.2
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./config.yaml:/app/config.yaml
    environment:
      - SILICONFLOW_API_KEY=${SILICONFLOW_API_KEY}
    restart: unless-stopped
```

---

## 5. 安全架构

### 5.1 多层安全防护

```
┌──────────────────────────────────────────────────┐
│                    安全层                         │
│                                                  │
│  传输层                                           │
│  ├─ TLS 1.3 (Nginx 终端)                         │
│  └─ HSTS 头                                      │
│                                                  │
│  应用层                                           │
│  ├─ CORS 白名单 (仅允许配置的域名)                  │
│  ├─ API Key 通过 .env 注入，不入仓                 │
│  ├─ config.yaml 以 ${VAR} 引用敏感信息             │
│  ├─ GET /config 自动脱敏 (api_key → ***)          │
│  └─ 请求体大小限制                                 │
│                                                  │
│  数据层                                           │
│  ├─ Agent 级数据隔离 (agent_id 强制过滤)           │
│  ├─ SQLite WAL 模式 (防并发写锁)                   │
│  └─ 数据库文件权限 0600                            │
│                                                  │
│  运维层                                           │
│  ├─ 结构化日志 (JSON 格式)                         │
│  ├─ Request ID 全链路追踪                          │
│  └─ X-Process-Time-Ms 性能监控头                   │
└──────────────────────────────────────────────────┘
```

### 5.2 CORS 白名单配置

```yaml
# config.yaml
cors:
  allowed_origins:
    - "http://localhost:3000"
    - "http://127.0.0.1:3002"
    - "https://your-domain.com"
    - "null"  # file:// 协议直接打开前端
```

### 5.3 异常类型 → HTTP 状态码映射

| 异常类 | HTTP 状态码 | 说明 |
|--------|:----------:|------|
| `ValidationError` | 422 | 请求参数校验失败 |
| `DatabaseError` | 500 | 数据库操作异常 |
| `NotFoundError` | 404 | 资源不存在 |
| `ConflictError` | 409 | 资源冲突（重复创建） |
| `AuthenticationError` | 401 | 认证失败 |
| `AuthorizationError` | 403 | 权限不足 |
| `RateLimitError` | 429 | 请求频率超限 |
| `ServiceUnavailableError` | 503 | 依赖服务不可用 |
| `TimeoutError` | 504 | 上游超时 |

---

## 6. 监控与可观测性

### 6.1 内置指标

| API | 指标 | 用途 |
|-----|------|------|
| `GET /api/v1/metrics` | memory_total, agent_count, avg_hot_score, vector_total, uptime_seconds | 业务监控大盘 |
| `GET /api/v1/stats` | memory_db + vector_db 详细信息 | 存储层健康检查 |
| `GET /health` | status: ok / version | 负载均衡健康检查 |

### 6.2 请求追踪

每个 HTTP 响应自动携带：
- `X-Request-ID`: 8 位 UUID 前缀，用于日志关联
- `X-Process-Time-Ms`: 处理耗时（毫秒），用于性能监控
- `X-API-Version`: v1，用于客户端兼容性判断

### 6.3 日志规范

```
# 格式
2026-04-29 10:30:00 - main_api - INFO - ✅ 记忆创建成功：mem_abc123

# 日志级别
DEBUG  - 详细调试信息（开发环境）
INFO   - 关键操作（记忆创建/删除，配置变更）
WARNING - 降级方案触发（Fallback 编码、调度器异常）
ERROR   - 功能异常（API 调用失败、数据库错误）
```

---

## 7. 性能特征

| 操作 | P50 延迟 | P99 延迟 | 备注 |
|------|:-------:|:-------:|------|
| 记忆创建 (POST) | 250ms | 600ms | 含 Embedding API 调用 |
| 向量检索 (GET) | 120ms | 350ms | FAISS HNSW 索引 |
| 列表查询 (GET) | 15ms | 40ms | SQLite 索引查询 |
| 配置读取 (GET) | 5ms | 10ms | 纯内存操作 |
| 健康检查 (GET) | 1ms | 3ms | 无依赖 |

**Embedding 引擎对比**：

| 引擎 | 维度 | 延迟 | 费用 | R@5 | R@10 |
|------|:----:|:----:|:----:|:---:|:----:|
| SiliconFlow BGE-zh | 1024 | 201ms | 免费 | 0.465 | 0.730 |
| ST all-MiniLM-L6-v2 | 384 | 82ms | 免费 | 0.275 | 0.470 |
| OpenAI text-3-small | 1536 | 350ms | 付费 | — | — |
| Hash (SHA-256) | 256 | <1ms | 免费 | ~0.00 | ~0.05 |

**推荐配置**：SiliconFlow BGE-zh（精度最高，免费） + 本地 ST（离线兜底）

---

## 8. 容量规划

| 规模 | 记忆数 | 推荐数据库 | 推荐 Embedding | 服务器配置 |
|------|:-----:|-----------|---------------|-----------|
| 开发/测试 | < 1 万 | SQLite | SiliconFlow BGE | 2C4G |
| 小规模生产 | 1-10 万 | SQLite | SiliconFlow BGE | 4C8G |
| 中规模生产 | 10-100 万 | MySQL | SiliconFlow BGE | 8C16G × 2 |
| 大规模生产 | > 100 万 | MySQL 集群 | 自建 Embedding 服务 | 16C32G × 3+ |

---

*本文档描述 MetaMemory v2.2 架构，与代码实现保持同步。如有疑问，请参考 `main_api.py` (1200 行) 和 `core/` 目录下的实现。*
