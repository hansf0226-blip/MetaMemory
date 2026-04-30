#!/bin/bash
# ============================================
# MetaMemory v2.3 一键启动
# ============================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 MetaMemory AI Agent 记忆系统 v2.3"
echo "========================================"

# 检查 Python
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "❌ 未找到 $PYTHON，请安装 Python 3.10+"
    exit 1
fi
echo "✅ Python: $($PYTHON --version)"

# 自动加载 .env
if [ -f ".env" ]; then
    echo "✅ 加载 .env 配置"
    set -a
    source .env
    set +a
else
    echo "⚠️  未找到 .env，请从 config.yaml.example 复制并配置 API Key"
fi

# 检查端口占用
PORT="${PORT:-8000}"
if lsof -ti :"$PORT" &>/dev/null; then
    echo "⚠️  端口 $PORT 已占用，尝试关闭旧进程..."
    kill $(lsof -ti :"$PORT") 2>/dev/null || true
    sleep 1
fi

# 启动
echo "🔧 启动 FastAPI 服务 (端口 $PORT)..."
$PYTHON main_api.py

echo "🛑 服务已停止"
