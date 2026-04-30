#!/bin/bash
# ============================================================
# Alpha-Conscious-Agent V2.0 - 数据备份脚本
# ============================================================
# 功能：备份 MySQL + Redis + Milvus 数据
# 使用：./scripts/backup.sh [备份目录]
# ============================================================

set -e

BACKUP_DIR=${1:-"./backups/$(date +%Y%m%d_%H%M%S)"}

echo "============================================================"
echo "💾 Alpha-Conscious-Agent V2.0 - 数据备份"
echo "============================================================"
echo "备份目录：$BACKUP_DIR"

# 创建备份目录
mkdir -p "$BACKUP_DIR"

# 备份 MySQL
echo -e "\n[1/3] 备份 MySQL..."
if command -v mysqldump &> /dev/null; then
    mysqldump -h${MYSQL_HOST:-localhost} -P${MYSQL_PORT:-3306} \
        -u${MYSQL_USER:-yijing} -p${MYSQL_PASSWORD:-yijing123} \
        ${MYSQL_DATABASE:-yijing_memory_db} > "$BACKUP_DIR/mysql_backup.sql"
    echo "✅ MySQL 备份完成"
else
    echo "⚠️  mysqldump 未安装，跳过"
fi

# 备份 Redis
echo -e "\n[2/3] 备份 Redis..."
if command -v redis-cli &> /dev/null; then
    redis-cli -h${REDIS_HOST:-localhost} -p${REDIS_PORT:-6379} BGSAVE
    sleep 2
    cp /var/lib/redis/dump.rdb "$BACKUP_DIR/redis_backup.rdb" 2>/dev/null || true
    echo "✅ Redis 备份完成"
else
    echo "⚠️  redis-cli 未安装，跳过"
fi

# 打包
echo -e "\n[3/3] 打包备份文件..."
cd "$BACKUP_DIR/.."
tar -czf "$(basename $BACKUP_DIR).tar.gz" "$(basename $BACKUP_DIR)"
cd - > /dev/null

echo -e "\n✅ 备份完成！"
echo "📦 备份文件：$BACKUP_DIR.tar.gz"
echo "💡 恢复命令：./scripts/restore.sh $BACKUP_DIR.tar.gz"
