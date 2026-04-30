-- API Key 认证系统 - 数据库表
-- 用于存储和管理 API Key

USE yijing_memory_db;

-- API Key 表
CREATE TABLE IF NOT EXISTS api_keys (
  id INT NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  key_hash VARCHAR(64) NOT NULL COMMENT 'API Key 哈希值',
  key_prefix VARCHAR(8) NOT NULL COMMENT 'Key 前缀（用于显示）',
  agent_id VARCHAR(64) NOT NULL COMMENT '关联的 Agent ID',
  name VARCHAR(128) DEFAULT '' COMMENT 'Key 名称/描述',
  status TINYINT(1) NOT NULL DEFAULT 1 COMMENT '状态（1 启用 0 禁用）',
  rate_limit INT NOT NULL DEFAULT 1000 COMMENT '速率限制（次/小时）',
  expires_at DATETIME DEFAULT NULL COMMENT '过期时间',
  last_used_at DATETIME DEFAULT NULL COMMENT '最后使用时间',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_key_hash (key_hash),
  KEY idx_agent_id (agent_id),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='API Key 表';

-- API Key 使用记录表
CREATE TABLE IF NOT EXISTS api_key_usage (
  id BIGINT NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  key_id INT NOT NULL COMMENT 'API Key ID',
  endpoint VARCHAR(255) NOT NULL COMMENT '请求端点',
  method VARCHAR(10) NOT NULL COMMENT '请求方法',
  status_code INT NOT NULL DEFAULT 200 COMMENT '响应状态码',
  response_time INT NOT NULL DEFAULT 0 COMMENT '响应时间（ms）',
  ip_address VARCHAR(45) DEFAULT '' COMMENT '请求 IP',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (id),
  KEY idx_key_id (key_id),
  KEY idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='API Key 使用记录表';

-- 插入默认 API Key（用于测试）
-- Key: sk_test_1234567890abcdef
-- 哈希值使用 SHA256 计算
INSERT INTO api_keys (key_hash, key_prefix, agent_id, name, status, rate_limit)
VALUES (
  SHA2('sk_test_1234567890abcdef', 256),
  'sk_test_',
  'default_agent',
  '测试 Key',
  1,
  1000
);

-- 查看已创建的表
SHOW TABLES LIKE 'api_keys';
SHOW TABLES LIKE 'api_key_usage';
