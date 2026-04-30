#!/bin/bash
# ============================================
# MetaMemory v2.3 初始化脚本
# 首次部署时运行一次即可
# ============================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔧 MetaMemory v2.3 — 初始化"
echo "================================"

# 1. 检查 Python
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "❌ 未找到 $PYTHON，请安装 Python 3.10+"
    exit 1
fi
echo "✅ Python: $($PYTHON --version)"

# 2. 安装依赖
if [ -f "requirements.txt" ]; then
    echo ""
    echo "📦 安装 Python 依赖..."
    $PYTHON -m pip install -r requirements.txt -q 2>&1 | tail -1
    echo "✅ 依赖安装完成"
fi

# 3. 初始化数据目录
echo ""
echo "📁 初始化数据目录..."
mkdir -p data

# 初始化空向量库（1024 维）
$PYTHON << 'PYEOF'
import sqlite3, os

db_path = os.path.join('data', 'vector_store.db')
conn = sqlite3.connect(db_path)
conn.execute('''
    CREATE TABLE IF NOT EXISTS vectors (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        embedding TEXT NOT NULL,
        metadata TEXT DEFAULT '{}',
        tags TEXT DEFAULT '[]',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.execute('''
    CREATE TABLE IF NOT EXISTS knowledge_base (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        embedding TEXT NOT NULL,
        metadata TEXT DEFAULT '{}',
        category TEXT DEFAULT 'general',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()
conn.close()
print('✅ 向量数据库初始化完成')
PYEOF

# 4. 配置检查
echo ""
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "📝 已创建 .env（从 .env.example 复制）"
        echo "   ⚠️  请编辑 .env 填入 SILICONFLOW_API_KEY"
    else
        echo "⚠️  缺少 .env.example，请手动创建 .env"
    fi
else
    echo "✅ .env 已存在"
fi

echo ""
echo "================================"
echo "✅ 初始化完成！"
echo ""
echo "下一步："
echo "  1. 编辑 .env 填入 SILICONFLOW_API_KEY"
echo "  2. bash start.sh"
echo "  3. 浏览器打开 http://localhost:8000"
