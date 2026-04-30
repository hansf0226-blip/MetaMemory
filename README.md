# 📦 MetaMemory — AI Agent 长期记忆系统

> 为 AI Agent 提供长期记忆存储、语义检索与可视化监控能力

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Evaluation: 5.00/5](https://img.shields.io/badge/evaluation-5.00%2F5-brightgreen.svg)](EVALUATION_REPORT_v10.md)

---

## ✨ 核心特性

- 🔍 **语义检索** — SiliconFlow BGE-large-zh (1024d) 纯语义向量检索，R@5 = 0.465
- 🔐 **多租户隔离** — Agent 级别数据隔离，查询不透传
- 📊 **可视化仪表盘** — Linear Dark 风格前端，ECharts 图表，5 页面 SPA
- ⚙️ **配置驱动** — 所有行为通过 `config.yaml` 控制，支持运行时热更新
- 🏷️ **分类标签** — 自动分类/属性/层级标签，前端筛选过滤
- 🔄 **生命周期管理** — APScheduler 定时任务（自愈 + 衰减 + 过期清理）
- 📈 **监控指标** — 内置 Prometheus 兼容指标 + 全链路 Request ID 追踪
- 🐳 **零依赖部署** — SQLite 默认，单文件前端，`python main_api.py` 即可运行

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- 磁盘 2 GB（含 .venv）

### 2. 安装

```bash
cd MetaMemory记忆系统 - 完整源码
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入:
#   SILICONFLOW_API_KEY=***
```

### 4. 启动

```bash
# 方式一：一键启动（推荐，自动加载 .env）
bash start.sh

# 方式二：直接启动（v2.3 起自动加载 .env）
python main_api.py
# → Uvicorn running on http://0.0.0.0:8000
```

### 5. 打开前端

直接双击 `frontend/index.html`，或在浏览器打开 `http://localhost:8000`

---

## 📖 文档导航

| 文档 | 说明 |
|------|------|
| [架构设计](docs/architecture.md) | 多层架构图、数据流、安全、容量规划 |
| [技术参考](docs/技术文档.md) | 技术栈、数据模型、Embedding 引擎、检索算法 |
| [用户手册](docs/用户文档.md) | 快速上手、Nginx/systemd/Docker 部署、排错 FAQ |
| [API 文档](docs/API文档.md) | 28 个端点的完整参考（cURL + Python 示例） |
| [模块目录](docs/模块目录.md) | 43 个核心模块清单、分层架构、调用关系 |
| [适配器](adapters/) | LangChain / Hermes / 通用适配器，一行接入 |
| [评估报告](EVALUATION_REPORT_v10.md) | 综合评分 5.00/5，检索精度 Benchmark |
| [部署指南](DEPLOYMENT.md) | 生产环境部署（Docker/手动/安全配置） |

---

## 🏗️ 系统架构

```
┌──────────────────────────────────────────┐
│            前端 SPA (708 行)              │
│     frontend/index.html                  │
│     Tailwind CSS + ECharts · Linear Dark │
│     仪表盘 / 记忆列表 / 检索 / 配置       │
└──────────────────┬───────────────────────┘
                   │ REST API
┌──────────────────▼───────────────────────┐
│         FastAPI (main_api.py, 1200 行)    │
│     记忆 CRUD / 向量检索 / 统计监控        │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│           核心引擎 (core/, 43 文件)        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │Embedding │ │ Storage  │ │ Retrieval│ │
│  │BGE 1024d │ │ Manager  │ │ Engine   │ │
│  └──────────┘ └──────────┘ └──────────┘ │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│          存储层                           │
│  SQLite (默认) / MySQL (可选)            │
│  FAISS 向量索引 · config.yaml 配置        │
└──────────────────────────────────────────┘
```

详细架构请参考 [docs/architecture.md](docs/architecture.md)。

---

## 🛠️ 技术栈

| 类别 | 技术 | 版本 |
|------|------|:---:|
| **后端框架** | FastAPI | 0.104 |
| **数据库** | SQLite (默认) / MySQL (可选) | — |
| **向量引擎** | FAISS (HNSW 索引) | — |
| **Embedding** | SiliconFlow BAAI/bge-large-zh-v1.5 (1024d) | — |
| **ORM** | SQLAlchemy | 2.0 |
| **前端** | 纯 HTML + Tailwind CSS + ECharts | — |
| **配置** | YAML + dotenv (敏感信息分离) | — |
| **调度** | APScheduler | 3.10 |

---

## 📊 评估成绩

| 指标 | 数值 | 评级 |
|------|:----:|:---:|
| 检索 R@5 | 0.465 | 🟢 |
| 检索 R@10 | 0.730 | 🟢 |
| MRR | 0.251 | 🟡 |
| **综合评分** | **5.00/5** | ⭐⭐⭐⭐⭐ |

> 完整报告：[EVALUATION_REPORT_v10.md](EVALUATION_REPORT_v10.md)

---

## 📁 项目结构

```
MetaMemory/
├── main_api.py              # FastAPI 应用入口 (1200 行)
├── config.yaml              # 系统配置
├── .env                     # 敏感信息（不入 Git）
├── requirements.txt         # Python 依赖
├── core/                    # 核心引擎 (43 模块)
│   ├── embedding_provider.py   # 多引擎 Embedding (441 行)
│   ├── storage_manager.py      # 统一存储管理 (1069 行)
│   ├── hybrid_retrieval.py     # 检索 + 排序 (338 行)
│   ├── models.py               # ORM 数据模型 (368 行)
│   └── ...
├── frontend/
│   └── index.html           # 单文件 SPA 前端 (708 行)
├── docs/                    # 文档
├── data/                    # 运行时数据 (SQLite + FAISS)
├── tests/                   # 测试 (17 文件)
└── modules/core/tooling/    # 工具模块
```

---

## 📄 许可证

MIT License

---

**🌟 MetaMemory — 让 AI Agent 拥有长期记忆**
