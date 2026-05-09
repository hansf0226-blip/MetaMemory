# MetaMemory v2.2 — 多阶段构建
# 用法: docker build -t metamemory . && docker run -p 8000:8000 metamemory

# ========== Stage 1: Builder ==========
FROM python:3.11-slim AS builder

WORKDIR /build

# 系统依赖（编译用）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖到临时目录
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --target=/build/deps \
    numpy pyyaml sentence-transformers

# ========== Stage 2: Runtime ==========
FROM python:3.11-slim

LABEL org.opencontainers.image.title="MetaMemory"
LABEL org.opencontainers.image.description="AI Agent Memory System with Structural Encoding"
LABEL org.opencontainers.image.version="2.3.0"

WORKDIR /app

# 运行时系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 复制预编译依赖
COPY --from=builder /build/deps /usr/local/lib/python3.11/site-packages/

# 复制应用代码
COPY core/ ./core/
COPY data/ ./data/
COPY scripts/ ./scripts/

# 复制轻量级依赖
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --no-deps -e .

# 创建运行时目录
RUN mkdir -p /app/logs /app/data

# 环境变量（可在运行时覆盖）
ENV PYTHONUNBUFFERED=1
ENV METAMEMORY_STORAGE=sqlite
ENV METAMEMORY_EMBEDDING=sentence-transformers
ENV METAMEMORY_EMBEDDING_MODEL=all-MiniLM-L6-v2

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 默认运行 benchmark（可被 docker run 覆盖为 uvicorn）
ENTRYPOINT ["python", "-m", "core.benchmarks"]
