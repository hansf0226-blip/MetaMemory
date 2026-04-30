#!/bin/bash

# 易经 Agent 记忆系统 - 数据库备份脚本

DB_NAME="yijing_memory_db"
DB_USER="root"
DB_PWD="${DB_PWD:-123456}"
BACKUP_DIR="./backup"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p "$BACKUP_DIR"

# 备份数据库
mysqldump -u"$DB_USER" -p"$DB_PWD" "$DB_NAME" > "$BACKUP_DIR/yijing_db_${DATE}.sql"

if [ $? -eq 0 ]; then
    echo "✅ 数据库备份完成：backup/yijing_db_${DATE}.sql"
    
    # 压缩备份
    gzip "$BACKUP_DIR/yijing_db_${DATE}.sql"
    echo "✅ 压缩完成：backup/yijing_db_${DATE}.sql.gz"
    
    # 清理 7 天前的备份
    find "$BACKUP_DIR" -name "yijing_db_*.sql.gz" -mtime +7 -delete
    echo "✅ 已清理 7 天前的旧备份"
else
    echo "❌ 备份失败"
    exit 1
fi
