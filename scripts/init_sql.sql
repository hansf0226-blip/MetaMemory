CREATE DATABASE IF NOT EXISTS yijing_memory_db DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE yijing_memory_db;

-- 记忆主表
CREATE TABLE yijing_memory (
  id bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  memory_id varchar(64) NOT NULL COMMENT '记忆唯一 ID',
  agent_id varchar(64) NOT NULL DEFAULT 'default_agent' COMMENT '多 Agent 隔离 ID',
  hexagram varchar(32) NOT NULL COMMENT '六爻数组',
  bagua_type varchar(16) NOT NULL COMMENT '八卦编码',
  sancai_layer varchar(16) NOT NULL COMMENT '三才层级',
  wuxing varchar(16) NOT NULL COMMENT '五行属性',
  position_status varchar(16) DEFAULT '当位' COMMENT '时位状态',
  weight decimal(3,2) NOT NULL DEFAULT '0.00' COMMENT '权重',
  hot_score decimal(3,2) NOT NULL DEFAULT '0.00' COMMENT '热度分数',
  is_core tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否核心记忆',
  content longtext COMMENT '记忆内容',
  related_memory_ids varchar(512) DEFAULT '' COMMENT '关联记忆 ID 列表',
  create_time datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  update_time datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_memory_id (memory_id),
  KEY idx_agent_id (agent_id),
  KEY idx_bagua (bagua_type),
  KEY idx_layer (sancai_layer)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='易经记忆主表';

-- 太极核心配置表
CREATE TABLE taiji_core_config (
  id int NOT NULL AUTO_INCREMENT,
  agent_id varchar(64) NOT NULL COMMENT 'Agent ID',
  core_values text COMMENT '核心价值观',
  ethic_rules text COMMENT '伦理规则',
  write_auth varchar(32) DEFAULT 'admin_only' COMMENT '写入权限',
  create_time datetime DEFAULT CURRENT_TIMESTAMP,
  update_time datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_agent_id (agent_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='太极核心配置表';

-- 多租户 Agent 管理表
CREATE TABLE agent_tenant (
  id int NOT NULL AUTO_INCREMENT,
  agent_id varchar(64) NOT NULL COMMENT 'Agent ID',
  agent_name varchar(128) NOT NULL COMMENT 'Agent 名称',
  status tinyint(1) NOT NULL DEFAULT 1 COMMENT '1 启用 0 禁用',
  remark varchar(255) DEFAULT '' COMMENT '备注',
  create_time datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_time datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_agent_id (agent_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='多租户 Agent 管理表';

-- 插入默认 Agent
INSERT INTO agent_tenant (agent_id, agent_name, remark) VALUES 
('default_agent', '默认主 Agent', '旧版历史数据租户');

-- 插入默认太极配置
INSERT INTO taiji_core_config (agent_id, core_values, ethic_rules) VALUES
('default_agent', '理性、好奇、友善', '不伤害、诚实、尊重隐私');
