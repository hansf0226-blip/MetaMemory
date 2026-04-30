-- 多租户/多 Agent 隔离升级 SQL
USE yijing_memory_db;

-- 记忆主表增加 agent_id 字段（如果不存在）
ALTER TABLE yijing_memory
ADD COLUMN IF NOT EXISTS agent_id varchar(64) NOT NULL DEFAULT 'default_agent' COMMENT '多 Agent 隔离 ID',
ADD INDEX IF NOT EXISTS idx_agent_id (agent_id);

-- 创建多租户 Agent 管理表（如果不存在）
CREATE TABLE IF NOT EXISTS agent_tenant (
  id int NOT NULL AUTO_INCREMENT,
  agent_id varchar(64) NOT NULL,
  agent_name varchar(128) NOT NULL,
  status tinyint(1) NOT NULL DEFAULT 1,
  remark varchar(255) DEFAULT '',
  create_time datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_time datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_agent_id (agent_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 插入默认 Agent（如果不存在）
INSERT IGNORE INTO agent_tenant (agent_id, agent_name, remark) 
VALUES ('default_agent', '默认主 Agent', '旧版历史数据租户');
