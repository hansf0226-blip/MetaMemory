# MetaMemory API 参考文档 v2.2

> **Base URL**：`http://localhost:8000` | **API 版本**：v1 | **协议**：REST/JSON
> **OpenAPI 文档**：启动服务后访问 `http://localhost:8000/docs` (Swagger UI) 或 `/redoc`

---

## 目录

1. [通用约定](#1-通用约定)
2. [系统接口](#2-系统接口)
3. [记忆 CRUD](#3-记忆-crud)
4. [向量检索](#4-向量检索)
5. [配置管理](#5-配置管理)
6. [Agent 管理](#6-agent-管理)
7. [多模态接口](#7-多模态接口)
8. [向量操作](#8-向量操作)
9. [A/B 测试](#9-ab-测试)
10. [词库管理](#10-词库管理)
11. [错误码参考](#11-错误码参考)

---

## 1. 通用约定

### 1.1 认证

当前版本（v2.2）API 为**开放接口**，未强制要求认证头。生产环境建议在前置 Nginx 层添加 API Key 或 JWT 验证。

未来版本将通过 `Authorization: Bearer <token>` 头进行认证（预留）。

### 1.2 请求格式

- Content-Type: `application/json`
- 编码: UTF-8
- 请求体大小限制: 默认 1MB（FastAPI 默认值）

### 1.3 响应格式

**成功响应**：
```json
{
  "code": 200,
  "data": { ... },
  "total": 100
}
```

**错误响应**：
```json
{
  "code": 4xx,
  "message": "人类可读的错误描述",
  "details": { "field": "具体错误字段" }
}
```

### 1.4 响应头

| 头名称 | 说明 | 示例 |
|--------|------|------|
| `X-Request-ID` | 请求唯一标识（8 位） | `a1b2c3d4` |
| `X-Process-Time-Ms` | 服务端处理耗时 | `123.4` |
| `X-API-Version` | API 版本 | `v1` |

### 1.5 分页

列表类接口统一使用 `page` + `page_size` 参数：

| 参数 | 类型 | 默认 | 最大 | 说明 |
|------|------|:---:|:---:|------|
| `page` | int | 1 | — | 页码（1-indexed） |
| `page_size` | int | 20 | 100 | 每页条数 |

---

## 2. 系统接口

### 2.1 健康检查

```http
GET /health
```

**用途**：负载均衡健康检查、服务存活探针。

**cURL 示例**：
```bash
curl http://localhost:8000/health
```

**响应**：
```json
{
  "code": 200,
  "status": "ok",
  "service": "yijing-agent-multi-tenant"
}
```

**Python 示例**：
```python
import requests
resp = requests.get("http://localhost:8000/health")
print(resp.json())  # {"code": 200, "status": "ok", ...}
```

---

### 2.2 系统统计

```http
GET /api/v1/stats
```

**用途**：获取系统整体统计信息（记忆数、Agent 数、各维度分布）。

**cURL 示例**：
```bash
curl http://localhost:8000/api/v1/stats
```

**响应**：
```json
{
  "code": 200,
  "data": {
    "memory_db": {
      "total_memories": 2090,
      "agent_count": 5,
      "avg_hot_score": 0.52
    },
    "vector_db": {
      "total_vectors": 2090,
      "index_type": "HNSW"
    }
  }
}
```

---

### 2.3 监控指标

```http
GET /api/v1/metrics
```

**用途**：Prometheus 兼容格式的监控指标，可用于告警系统。

**cURL 示例**：
```bash
curl http://localhost:8000/api/v1/metrics
```

**响应**：
```json
{
  "status": "ok",
  "uptime_seconds": 86400,
  "metrics": {
    "memory_total": 2090,
    "agent_count": 5,
    "avg_hot_score": 0.5234,
    "vector_total": 2090,
    "api_version": "v1"
  }
}
```

---

## 3. 记忆 CRUD

### 3.1 创建记忆

```http
POST /api/v1/memories
```

**请求体**：

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|:---:|------|------|
| `agent_id` | string | 是 | — | 租户标识 |
| `memory_id` | string | 否 | 自动生成 | 唯一记忆 ID（不传自动生成 UUID） |
| `content` | string | 是 | — | 记忆正文 |
| `bagua_type` | string | 否 | — | 分类标签（乾/坤/震/巽/坎/离/艮/兑） |
| `wuxing` | string | 否 | — | 属性标签（木/火/土/金/水） |
| `sancai_layer` | string | 否 | — | 层级标签（天/人/地） |
| `hexagram` | string | 否 | — | 六爻编码（6 位二进制，如 `[1,0,1,1,0,0]`） |
| `hot_score` | float | 否 | 0.5 | 热度分数 (0-1) |

**cURL 示例**：
```bash
curl -X POST http://localhost:8000/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent_001",
    "content": "用户偏好 Python 编程，喜欢简洁的函数式风格",
    "bagua_type": "乾",
    "wuxing": "金",
    "sancai_layer": "天",
    "hot_score": 0.8
  }'
```

**成功响应** (201):
```json
{
  "code": 200,
  "msg": "创建成功",
  "memory_id": "mem_abc123"
}
```

**错误响应** (409):
```json
{
  "code": 409,
  "message": "记忆已存在",
  "details": {"memory_id": "mem_abc123"}
}
```

**Python 示例**：
```python
import requests

data = {
    "agent_id": "agent_001",
    "content": "用户偏好 Python 编程",
    "bagua_type": "乾",
    "wuxing": "金",
    "hot_score": 0.8
}
resp = requests.post("http://localhost:8000/api/v1/memories", json=data)
print(resp.json())  # {"code": 200, "msg": "创建成功", "memory_id": "..."}
```

---

### 3.2 获取记忆列表

```http
GET /api/v1/memories
```

**查询参数**：

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|:---:|------|------|
| `agent_id` | string | 否 | `""` | 租户过滤（空=全部） |
| `page` | int | 否 | 1 | 页码 |
| `page_size` | int | 否 | 20 | 每页条数 |

**cURL 示例**：
```bash
# 获取第 1 页，每页 20 条
curl "http://localhost:8000/api/v1/memories?page=1&page_size=20"

# 按 Agent 过滤
curl "http://localhost:8000/api/v1/memories?agent_id=agent_001&page=1"
```

**响应**：
```json
{
  "code": 200,
  "total": 2090,
  "page": 1,
  "page_size": 20,
  "list": [
    {
      "memory_id": "mem_abc123",
      "agent_id": "agent_001",
      "content": "用户偏好 Python 编程...",
      "bagua_type": "乾",
      "wuxing": "金",
      "sancai_layer": "天",
      "hot_score": 0.8,
      "created_at": "2026-04-29T10:00:00"
    }
  ]
}
```

---

### 3.3 获取单条记忆

```http
GET /api/v1/memories/{memory_id}
```

**路径参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `memory_id` | string | 记忆唯一标识 |

**查询参数**：

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|:---:|------|------|
| `agent_id` | string | 否 | `""` | 不传自动搜索所有 Agent |

**cURL 示例**：
```bash
curl "http://localhost:8000/api/v1/memories/mem_abc123?agent_id=agent_001"
```

**成功响应**：
```json
{
  "code": 200,
  "data": {
    "memory_id": "mem_abc123",
    "agent_id": "agent_001",
    "content": "用户偏好 Python 编程...",
    "bagua_type": "乾",
    "wuxing": "金",
    "hot_score": 0.8,
    "created_at": "2026-04-29T10:00:00"
  }
}
```

**错误响应** (404):
```json
{"code": 404, "message": "记忆不存在", "details": {"memory_id": "mem_nonexist"}}
```

---

### 3.4 更新记忆

```http
PUT /api/v1/memories/{memory_id}
```

**请求体**：同创建接口（全部字段可选，仅更新传入字段）。

**cURL 示例**：
```bash
curl -X PUT http://localhost:8000/api/v1/memories/mem_abc123 \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent_001",
    "content": "用户偏好 Python 和 Rust 编程",
    "hot_score": 0.9
  }'
```

**响应**：
```json
{"code": 200, "msg": "更新成功"}
```

---

### 3.5 删除记忆

```http
DELETE /api/v1/memories/{memory_id}
```

**查询参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `agent_id` | string | 否 | 不传自动搜索所有 Agent |

**cURL 示例**：
```bash
curl -X DELETE "http://localhost:8000/api/v1/memories/mem_abc123?agent_id=agent_001"
```

**响应**：
```json
{"code": 200, "msg": "删除成功"}
```

---

### 3.6 记忆关联图谱

```http
GET /api/v1/memories/{memory_id}/relations
```

**查询参数**：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `agent_id` | string | `""` | Agent 过滤 |
| `top_k` | int | 10 | 返回关联数量 |

**cURL 示例**：
```bash
curl "http://localhost:8000/api/v1/memories/mem_abc123/relations?top_k=5"
```

---

### 3.7 记忆变更历史

```http
GET /api/v1/memories/{memory_id}/history
```

**cURL 示例**：
```bash
curl "http://localhost:8000/api/v1/memories/mem_abc123/history"
```

**响应**：
```json
{
  "code": 200,
  "data": {
    "memory_id": "mem_abc123",
    "create_time": "2026-04-29T10:00:00",
    "update_time": "2026-04-29T12:00:00",
    "history": []
  }
}
```

---

## 4. 向量检索

### 4.1 语义搜索

```http
GET /api/v1/memories/search
```

**查询参数**：

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|:---:|------|------|
| `content` | string | 是 | — | 查询文本 |
| `top_k` | int | 否 | 5 | 返回数量 (1-100) |
| `agent_id` | string | 否 | `""` | Agent 过滤（空=搜索全部） |

**cURL 示例**：
```bash
# 基础搜索
curl "http://localhost:8000/api/v1/memories/search?content=用户喜欢什么编程语言&top_k=5"

# 按 Agent 过滤
curl "http://localhost:8000/api/v1/memories/search?content=Python&agent_id=agent_001&top_k=10"
```

**响应**：
```json
{
  "code": 200,
  "data": [
    {
      "memory_id": "mem_abc123",
      "agent_id": "agent_001",
      "bagua_type": "乾",
      "wuxing": "金",
      "sancai_layer": "天",
      "content": "用户偏好 Python 编程...",
      "similarity": 0.4253,
      "hot_score": 0.8
    },
    {
      "memory_id": "mem_def456",
      "agent_id": "agent_001",
      "content": "上次讨论了 Python 异步编程...",
      "similarity": 0.3847,
      "hot_score": 0.6
    }
  ],
  "total": 2,
  "query": "用户喜欢什么编程语言"
}
```

**Python 示例**：
```python
import requests

resp = requests.get(
    "http://localhost:8000/api/v1/memories/search",
    params={"content": "Python 编程", "top_k": 5}
)
data = resp.json()
for item in data["data"]:
    print(f"相似度: {item['similarity']:.2%} | {item['content'][:50]}")
```

---

### 4.2 编码接口

```http
POST /api/v1/memory/encode/hexagram
```

**用途**：将文本内容转换为分类标签（bagua_type / wuxing / sancai_layer）。

**请求体**：
```json
{
  "content": "用户偏好 Python 编程",
  "memory_type": "chat"
}
```

**cURL 示例**：
```bash
curl -X POST http://localhost:8000/api/v1/memory/encode/hexagram \
  -H "Content-Type: application/json" \
  -d '{"content": "用户偏好 Python 编程", "memory_type": "chat"}'
```

**响应**：
```json
{
  "code": 200,
  "hexagram": [1, 1, 1, 0, 1, 0],
  "bagua_type": "乾",
  "wuxing": "金",
  "sancai_layer": "天",
  "is_fallback": false
}
```

---

## 5. 配置管理

### 5.1 获取配置

```http
GET /api/v1/config
```

**cURL 示例**：
```bash
curl http://localhost:8000/api/v1/config
```

**响应**（敏感信息已脱敏）：
```json
{
  "code": 200,
  "data": {
    "version": "2.2.0",
    "server": {"host": "0.0.0.0", "port": 8000},
    "embedding": {
      "provider": "siliconflow",
      "siliconflow": {"api_key": "***", "model": "BAAI/bge-large-zh-v1.5"}
    },
    "retrieval": {"top_k": 10, "min_similarity": 0.3}
  }
}
```

### 5.2 更新配置

```http
PUT /api/v1/config
```

**用途**：深度合并到 `config.yaml`，更新后需重启服务。

**请求体**（仅传需要修改的字段）：
```json
{
  "retrieval": {
    "top_k": 20,
    "min_similarity": 0.5
  },
  "embedding": {
    "provider": "openai"
  }
}
```

**cURL 示例**：
```bash
curl -X PUT http://localhost:8000/api/v1/config \
  -H "Content-Type: application/json" \
  -d '{"retrieval": {"top_k": 20}}'
```

**响应**：
```json
{"code": 200, "msg": "配置已更新", "restart_required": true}
```

---

## 6. Agent 管理

### 6.1 注册 Agent

```http
POST /api/v1/agents
```

**请求体**：
```json
{
  "agent_id": "customer_support_01",
  "name": "客服助手"
}
```

**cURL 示例**：
```bash
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "customer_support_01", "name": "客服助手"}'
```

**响应**：
```json
{"code": 200, "msg": "创建成功", "agent_id": "customer_support_01"}
```

### 6.2 Agent 列表

```http
GET /api/v1/agents
```

**cURL 示例**：
```bash
curl http://localhost:8000/api/v1/agents
```

**响应**：
```json
{
  "code": 200,
  "list": [
    {"agent_id": "agent_001"},
    {"agent_id": "customer_support_01"}
  ]
}
```

### 6.3 Agent 详情

```http
GET /api/v1/agents/{agent_id}/info
```

**cURL 示例**：
```bash
curl http://localhost:8000/api/v1/agents/agent_001/info
```

---

## 7. 多模态接口

### 7.1 语音转文字

```http
POST /api/v1/multimodal/speech-to-text
Content-Type: multipart/form-data
```

**参数**：`audio_file` (file, 必填)

**cURL 示例**：
```bash
curl -X POST http://localhost:8000/api/v1/multimodal/speech-to-text \
  -F "audio_file=@recording.wav"
```

**响应**：
```json
{"code": 200, "data": {"text": "你好，今天天气怎么样？"}}
```

### 7.2 图像处理

```http
POST /api/v1/multimodal/process-image
Content-Type: multipart/form-data
```

**参数**：`image_file` (file, 必填)

### 7.3 生成多模态嵌入

```http
POST /api/v1/multimodal/generate-embedding
```

**请求体**：
```json
{"text": "示例文本", "image_path": "/path/to/image.jpg"}
```

### 7.4 内容分析

```http
POST /api/v1/multimodal/analyze-content
```

### 7.5 多模态服务状态

```http
GET /api/v1/multimodal/status
```

---

## 8. 向量操作

### 8.1 向量插入

```http
POST /api/v1/vector/insert
```

**查询参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `memory_id` | string | 记忆 ID |
| `content` | string | 文字内容 |
| `agent_id` | string | Agent ID (默认 "default") |

**cURL 示例**：
```bash
curl -X POST "http://localhost:8000/api/v1/vector/insert?memory_id=vec_001&content=测试内容&agent_id=default"
```

### 8.2 向量搜索

```http
GET /api/v1/vector/search
```

**查询参数**：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `query` | string | 必填 | 查询文本 |
| `top_k` | int | 5 | 返回数量 |
| `agent_id` | string | `"default"` | Agent 过滤 |

---

## 9. A/B 测试

### 9.1 运行测试

```http
GET /api/v1/ab-test/run
```

### 9.2 获取报告

```http
GET /api/v1/ab-test/report
```

---

## 10. 词库管理

### 10.1 学习关键词

```http
POST /api/v1/keywords/learn
```

**查询参数**：`text` (string, 必填)

### 10.2 关键词统计

```http
GET /api/v1/keywords/stats
```

---

## 11. 错误码参考

| HTTP 状态码 | 错误类型 | 说明 | 常见原因 |
|:---------:|------|------|---------|
| 200 | — | 成功 | — |
| 400 | `ValidationError` | 请求参数校验失败 | 必填字段缺失、格式错误 |
| 401 | `AuthenticationError` | 认证失败 | API Key 无效或缺失 |
| 403 | `AuthorizationError` | 权限不足 | Agent 无权访问该资源 |
| 404 | `NotFoundError` | 资源不存在 | memory_id/agent_id 不存在 |
| 409 | `ConflictError` | 资源冲突 | memory_id 已存在 |
| 422 | `ValidationError` | 语义校验失败 | content 为空、hot_score 超出范围 |
| 429 | `RateLimitError` | 请求频率超限 | 超过限流阈值 |
| 500 | `DatabaseError` | 内部服务器错误 | 数据库连接失败、SQL 错误 |
| 503 | `ServiceUnavailableError` | 依赖服务不可用 | Embedding API 不可达 |
| 504 | `TimeoutError` | 上游超时 | Embedding API 响应超时 |

**错误响应示例**：
```json
{
  "code": 404,
  "message": "记忆不存在",
  "details": {
    "memory_id": "mem_nonexist",
    "agent_id": "agent_001"
  }
}
```

---

## 附录：SDK 快速集成

### Python

```python
import requests

BASE = "http://localhost:8000/api/v1"

class MetaMemoryClient:
    def __init__(self, base_url=BASE):
        self.base = base_url
        self.session = requests.Session()

    def search(self, content: str, agent_id: str = "", top_k: int = 5) -> list:
        resp = self.session.get(
            f"{self.base}/memories/search",
            params={"content": content, "agent_id": agent_id, "top_k": top_k}
        )
        return resp.json()["data"]

    def create_memory(self, agent_id: str, content: str, **kw) -> dict:
        data = {"agent_id": agent_id, "content": content, **kw}
        resp = self.session.post(f"{self.base}/memories", json=data)
        return resp.json()

    def stats(self) -> dict:
        return self.session.get(f"{self.base}/stats").json()

# 使用
client = MetaMemoryClient()
results = client.search("用户偏好")
for r in results:
    print(f"[{r['similarity']:.2f}] {r['content'][:60]}")
```

### cURL 速查

```bash
# 健康检查
curl localhost:8000/health

# 搜索
curl "localhost:8000/api/v1/memories/search?content=查询内容&top_k=5"

# 创建
curl -X POST localhost:8000/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"agt","content":"记忆内容"}'

# 列表
curl "localhost:8000/api/v1/memories?page=1&page_size=20"

# 统计
curl localhost:8000/api/v1/stats

# 配置
curl localhost:8000/api/v1/config
```

---

*本文档基于 `main_api.py` (1200 行) 的实际路由定义生成，与代码保持同步。完整交互式文档请访问 `http://localhost:8000/docs`。*
