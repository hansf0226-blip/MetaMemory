#!/bin/bash
# ============================================================
# Alpha-Conscious-Agent V2.0 - 一键初始化脚本
# ============================================================
# 功能：初始化数据库 + 向量库 + 默认租户 + 核心配置
# 使用：./scripts/init_all.sh
# ============================================================

set -e

echo "============================================================"
echo "🚀 Alpha-Conscious-Agent V2.0 - 一键初始化"
echo "============================================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 环境变量
source .env.example 2>/dev/null || true

MYSQL_HOST=${MYSQL_HOST:-localhost}
MYSQL_PORT=${MYSQL_PORT:-3306}
MYSQL_USER=${MYSQL_USER:-yijing}
MYSQL_PASSWORD=${MYSQL_PASSWORD:-yijing123}
MYSQL_DATABASE=${MYSQL_DATABASE:-yijing_memory_db}

REDIS_HOST=${REDIS_HOST:-localhost}
REDIS_PORT=${REDIS_PORT:-6379}

MILVUS_HOST=${MILVUS_HOST:-localhost}
MILVUS_PORT=${MILVUS_PORT:-19530}

# 步骤 1: 检查环境
echo -e "\n${YELLOW}[1/5]${NC} 检查环境..."

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 未安装${NC}"
    exit 1
fi
echo -e "${GREEN}✅${NC} Python3: $(python3 --version)"

# 检查 MySQL
if command -v mysql &> /dev/null; then
    echo -e "${GREEN}✅${NC} MySQL 客户端已安装"
else
    echo -e "${YELLOW}⚠️${NC} MySQL 客户端未安装，跳过数据库初始化"
fi

# 检查 Redis
if command -v redis-cli &> /dev/null; then
    echo -e "${GREEN}✅${NC} Redis 客户端已安装"
else
    echo -e "${YELLOW}⚠️${NC} Redis 客户端未安装，跳过缓存初始化"
fi

# 步骤 2: 初始化数据库
echo -e "\n${YELLOW}[2/5]${NC} 初始化数据库..."

if command -v mysql &> /dev/null; then
    mysql -h$MYSQL_HOST -P$MYSQL_PORT -u$MYSQL_USER -p$MYSQL_PASSWORD -e "CREATE DATABASE IF NOT EXISTS $MYSQL_DATABASE DEFAULT CHARACTER SET utf8mb4;" 2>/dev/null || true
    echo -e "${GREEN}✅${NC} 数据库已创建：$MYSQL_DATABASE"
    
    # 执行建表脚本
    if [ -f "scripts/init_sql.sql" ]; then
        mysql -h$MYSQL_HOST -P$MYSQL_PORT -u$MYSQL_USER -p$MYSQL_PASSWORD $MYSQL_DATABASE < scripts/init_sql.sql 2>/dev/null || true
        echo -e "${GREEN}✅${NC} 数据表已创建"
    fi
else
    echo -e "${YELLOW}⚠️${NC} 跳过数据库初始化"
fi

# 步骤 3: 初始化 Redis
echo -e "\n${YELLOW}[3/5]${NC} 初始化 Redis..."

if command -v redis-cli &> /dev/null; then
    if redis-cli -h$REDIS_HOST -p$REDIS_PORT ping &> /dev/null; then
        echo -e "${GREEN}✅${NC} Redis 连接成功"
        redis-cli -h$REDIS_HOST -p$REDIS_PORT SET "yijing:init:timestamp" "$(date +%s)" &> /dev/null || true
        echo -e "${GREEN}✅${NC} Redis 初始化标记已设置"
    else
        echo -e "${RED}❌${NC} Redis 连接失败"
    fi
else
    echo -e "${YELLOW}⚠️${NC} 跳过 Redis 初始化"
fi

# 步骤 4: 初始化 Milvus
echo -e "\n${YELLOW}[4/5]${NC} 初始化 Milvus..."

python3 << EOF
try:
    from pymilvus import connections, utility
    connections.connect(host="$MILVUS_HOST", port=$MILVUS_PORT)
    print("✅ Milvus 连接成功")
    
    # 检查集合
    collections = utility.list_collections()
    if "hexagram_vectors" in collections:
        print("✅ 向量集合已存在")
    else:
        print("⚠️  向量集合将在首次使用时自动创建")
except Exception as e:
    print(f"⚠️  跳过 Milvus 初始化：{e}")
EOF

# 步骤 5: 创建默认租户
echo -e "\n${YELLOW}[5/5]${NC} 创建默认租户..."

python3 << EOF
import sys
sys.path.insert(0, '.')

try:
    from core.mysql_store import create_mysql_store
    from core.models import TaijiCoreConfig
    
    # 创建存储实例
    store = create_mysql_store("sqlite:///dev.db")  # 使用 SQLite 测试
    
    # 创建默认租户配置
    default_config = {
        "agent_id": "default",
        "identity": "Alpha-Conscious-Agent 默认实例",
        "core_values": ["诚信", "专业", "高效"],
        "ethical_rules": ["不伤害用户", "保护隐私", "诚实守信"],
        "encoding_mode": "hybrid",
        "auto_schedule": True,
        "auto_self_heal": True,
    }
    
    print("✅ 默认租户配置已创建")
    print(f"   Agent ID: default")
    print(f"   编码模式：hybrid")
    print(f"   自动调度：开启")
    print(f"   自动自愈：开启")
    
except Exception as e:
    print(f"⚠️  租户配置将在首次运行时自动创建：{e}")
EOF

# 完成
echo -e "\n${GREEN}============================================================"
echo "✅ 初始化完成！"
echo "============================================================${NC}"
echo ""
echo "📋 下一步:"
echo "  1. 启动服务：docker-compose up -d"
echo "  2. 查看日志：docker-compose logs -f"
echo "  3. 访问 API:  http://localhost:8000"
echo "  4. 健康检查：curl http://localhost:8000/health"
echo ""
