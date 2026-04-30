# 🚀 MetaMemory 部署指南 v2.3

> 从零到生产 — 完整部署流程

---

## 📋 前置要求

| 软件 | 版本 | 必需 | 用途 |
|------|------|:--:|------|
| Python | 3.11+ | ✅ | 运行环境 |
| pip | 23.0+ | ✅ | 包管理 |
| MySQL | 8.0+ | ⚠️ | 生产扩展（默认 SQLite） |
| Docker | 20.0+ | ⚠️ | 容器化部署 |

---

## 🎯 方式一：直接运行（开发/小规模，2 分钟）

```bash
# 1. 进入项目
cd MetaMemory记忆系统 - 完整源码

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 SILICONFLOW_API_KEY

# 4. 启动（v2.3 起自动加载 .env）
bash start.sh
# 或 python main_api.py
# → Uvicorn running on http://0.0.0.0:8000

# 5. 验证
curl http://localhost:8000/health
# → {"code":200,"status":"ok","service":"yijing-agent-multi-tenant"}
```

---

## 🛠️ 方式二：手动部署（生产推荐，10 分钟）

### Step 1: Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### Step 2: 配置 Embedding

**推荐：SiliconFlow BGE（免费，精度最高）**

```bash
# .env
SILICONFLOW_API_KEY=sk-xxxxxxxxxxxx
```

**备选：本地 ST 模型（离线，无需 API Key）**

```yaml
# config.yaml
embedding:
  provider: "sentence_transformer"
```

### Step 3: 数据库选择

**选项 A — SQLite（默认，零配置）**

无需额外操作，自动使用 `data/memory_system.db`

**选项 B — MySQL（生产推荐，>10 万记忆）**

```bash
mysql -u root -p -e "CREATE DATABASE memory_system CHARACTER SET utf8mb4"
```

编辑 `.env`：
```bash
MYSQL_PASSWORD=your_password
```

编辑 `config.yaml`：
```yaml
database:
  type: "mysql"
```

### Step 4: systemd 服务

```ini
# /etc/systemd/system/metamemory.service
[Unit]
Description=MetaMemory AI Agent 长期记忆系统
After=network.target

[Service]
Type=simple
User=metamemory
WorkingDirectory=/opt/metamemory
Environment=PATH=/opt/metamemory/.venv/bin:/usr/bin
EnvironmentFile=/opt/metamemory/.env
ExecStart=/opt/metamemory/.venv/bin/python main_api.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now metamemory
```

### Step 5: Nginx 反向代理

```nginx
server {
    listen 80;
    server_name metamemory.example.com;

    root /opt/metamemory/frontend;
    index index.html;

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 30s;
    }

    location /health {
        proxy_pass http://127.0.0.1:8000;
    }
}
```

---

## 🐳 方式三：Docker 部署

```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "main_api.py"]
```

```bash
# 构建并启动
docker build -t metamemory:2.2 .
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -e SILICONFLOW_API_KEY=$SILICONFLOW_API_KEY \
  --name metamemory \
  metamemory:2.2
```

---

## ⚙️ Embedding Provider 选择

| Provider | 维度 | 延迟 | 费用 | R@5 | 适用场景 |
|----------|:---:|:---:|:---:|:---:|------|
| **SiliconFlow BGE** | 1024 | 201ms | 免费 | 0.465 | **默认推荐** |
| ST MiniLM (本地) | 384 | 82ms | 免费 | 0.275 | 离线环境 |
| OpenAI text-3-small | 1536 | 350ms | 付费 | — | 已有 OpenAI Key |
| Hash SHA-256 | 256 | <1ms | 免费 | ~0.00 | 调试/兜底 |

```yaml
# config.yaml
embedding:
  provider: "siliconflow"          # 默认
  # provider: "sentence_transformer"  # 离线切换
```

---

## 🔍 生产验证

```bash
# 健康检查
curl http://localhost:8000/health

# API 文档
open http://localhost:8000/docs

# 创建测试记忆
curl -X POST http://localhost:8000/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test","hexagram":"000000","bagua_type":"乾","wuxing":"金","sancai_layer":"unknown","content":"部署验证测试","hot_score":0.8}'

# 检索测试
curl "http://localhost:8000/api/v1/memories/search?content=部署&top_k=3"

# 系统统计
curl http://localhost:8000/api/v1/stats
```

---

## 📊 性能基准 (v2.3)

| 指标 | 值 | 条件 |
|------|:---:|------|
| 记忆创建延迟 | ~250ms | SiliconFlow BGE, 含 Embedding API 调用 |
| 检索延迟 P50 | ~120ms | FAISS HNSW, 2090 条 |
| 检索 R@5 | 0.465 | SiliconFlow BGE 1024d |
| 内存占用 | ~500MB | 含 ST 模型（使用 BGE 时不加载） |

---

## 🛡️ 安全配置

1. **API Key 保护**：放在 `.env` 文件，不入 Git
2. **config.yaml**：以 `${VAR}` 引用敏感值，`GET /config` 自动脱敏
3. **CORS 白名单**：`config.yaml` → `cors.allowed_origins` 限制域名
4. **日志级别**：生产环境设为 `INFO` 或 `WARNING`

---

## 🐛 常见问题

| 问题 | 解决 |
|------|------|
| `No module named 'fastapi'` | `pip install -r requirements.txt` |
| `端口 8000 已被占用` | `lsof -i :8000` 找进程或改 `config.yaml` 中端口 |
| `SILICONFLOW_API_KEY not set` | 编辑 `.env` 填入 Key |
| 检索无结果 | 降低 `retrieval.min_similarity` 到 0.1，检查 Embedding 引擎 |
| SQLite `database is locked` | 启用 WAL 模式或切换到 MySQL |
| 前端无法连接 API | 检查 CORS 白名单和后端是否运行 |

---

*详细部署配置（Nginx/systemd/Docker）见 [docs/用户文档.md](docs/用户文档.md#3-生产部署指南)*
