#!/usr/bin/env python3
"""
📊 数据模型层 - SQLAlchemy ORM Models

IMA V2.0 规范 - 多租户数据模型
包含：记忆主表 + 太极核心配置表 + 租户管理表
"""

try:
    from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Index, Integer, String, Text
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.sql import func
except ImportError:
    JSON = Boolean = Column = DateTime = Float = Index = Integer = String = Text = None
    declarative_base = lambda: type("Base", (), {})
    func = None

Base = declarative_base()


# ========== 多租户记忆主表 ==========


class YijingMemory(Base):
    """
    易经记忆主表

    支持多租户隔离 (agent_id)
    包含六爻/八卦/五行/三才完整字段
    """

    __tablename__ = "yijing_memory"

    # 主键
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 多租户隔离
    agent_id = Column(String(64), nullable=False, index=True, comment="Agent/租户 ID")

    # 记忆唯一 ID
    memory_id = Column(String(64), unique=True, nullable=False, comment="记忆唯一 ID")

    # 六爻编码 (6 位二进制)
    hexagram = Column(String(6), nullable=False, comment="六爻向量 (如:111111)")

    # 八卦分类
    bagua_type = Column(String(10), nullable=False, index=True, comment="八卦 (乾/坤/震/巽/坎/离/艮/兑)")

    # 六十四卦序号
    hexagram_num = Column(Integer, nullable=True, comment="六十四卦序号 (1-64)")

    # 三才分层
    sancai_layer = Column(String(10), nullable=False, index=True, comment="三才 (天/人/地)")

    # 五行状态
    wuxing = Column(String(10), nullable=False, index=True, comment="五行 (木/火/土/金/水)")

    # 时位状态
    position_status = Column(String(10), default="当位", comment="时位 (当位/失位)")

    # 记忆内容
    content = Column(Text, nullable=False, comment="记忆内容")

    # 权重与热度
    weight = Column(Float, default=0.5, comment="重要性权重 (0-1)")
    hot_score = Column(Float, default=0.5, comment="热度 (0-1)")

    # 冗余度 (用于五行调度)
    redundancy = Column(Float, default=0.0, comment="冗余度 (0-1)")

    # 生命周期
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="更新时间")
    last_accessed_at = Column(DateTime, nullable=True, comment="最后访问时间（检索/使用）")
    access_count = Column(Integer, default=0, nullable=False, comment="访问次数计数器（用于热度排序）")
    expire_time = Column(DateTime, nullable=True, comment="过期时间")

    # 爻变历史 (JSON 存储)
    yao_history = Column(JSON, default=list, comment="爻变历史记录")

    # 关联记忆
    relations = Column(JSON, default=list, comment="关联记忆 ID 列表")

    # 元数据
    source = Column(String(50), default="user_input", comment="来源")
    category = Column(String(50), nullable=True, comment="分类")
    tags = Column(JSON, default=list, comment="标签")

    # 状态
    status = Column(String(20), default="active", comment="状态 (active/archived/deleted)")

    # 索引
    __table_args__ = (
        Index("idx_agent_layer", "agent_id", "sancai_layer"),
        Index("idx_agent_element", "agent_id", "wuxing"),
        Index("idx_agent_trigram", "agent_id", "bagua_type"),
        Index("idx_created_at", "created_at"),
        Index("idx_expire_time", "expire_time"),
        Index("idx_agent_status", "agent_id", "status"),
        Index("idx_agent_hot", "agent_id", "hot_score"),
        Index("idx_agent_status_created", "agent_id", "status", "created_at"),
    )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "hexagram": self.hexagram,
            "bagua_type": self.bagua_type,
            "hexagram_num": self.hexagram_num,
            "sancai_layer": self.sancai_layer,
            "wuxing": self.wuxing,
            "position_status": self.position_status,
            "content": self.content,
            "weight": self.weight,
            "hot_score": self.hot_score,
            "redundancy": self.redundancy,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "expire_time": self.expire_time.isoformat() if self.expire_time else None,
            "yao_history": self.yao_history or [],
            "relations": self.relations or [],
            "source": self.source,
            "category": self.category,
            "tags": self.tags or [],
            "status": self.status,
        }

    def __repr__(self):
        return f"<YijingMemory(id={self.id}, agent_id={self.agent_id}, bagua_type={self.bagua_type})>"


# ========== 太极核心配置表 ==========


class TaijiCoreConfig(Base):
    """
    太极核心配置表

    存储 Agent 的核心身份/价值观/规则
    只读 + 高权限更新
    """

    __tablename__ = "taiji_core_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(64), nullable=False, unique=True, index=True, comment="Agent ID")

    # 核心身份
    identity = Column(Text, nullable=False, comment="核心身份描述")
    core_values = Column(JSON, default=list, comment="核心价值观列表")
    ethical_rules = Column(JSON, default=list, comment="伦理规则列表")

    # 配置
    encoding_mode = Column(String(20), default="hybrid", comment="编码模式 (symbolic/hybrid)")
    auto_schedule = Column(Boolean, default=True, comment="自动五行调度")
    auto_self_heal = Column(Boolean, default=True, comment="自动自愈")

    # 阈值配置
    hot_threshold_activate = Column(Float, default=0.8, comment="活跃热度阈值")
    hot_threshold_hide = Column(Float, default=0.2, comment="闭藏热度阈值")
    age_threshold_forget = Column(Integer, default=365, comment="遗忘年龄阈值 (天)")

    # 状态
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "identity": self.identity,
            "core_values": self.core_values or [],
            "ethical_rules": self.ethical_rules or [],
            "encoding_mode": self.encoding_mode,
            "auto_schedule": self.auto_schedule,
            "auto_self_heal": self.auto_self_heal,
            "hot_threshold_activate": self.hot_threshold_activate,
            "hot_threshold_hide": self.hot_threshold_hide,
            "age_threshold_forget": self.age_threshold_forget,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self):
        return f"<TaijiCoreConfig(agent_id={self.agent_id})>"


# ========== 租户管理表 ==========


class AgentTenant(Base):
    """
    租户管理表

    多租户系统的租户信息
    """

    __tablename__ = "agent_tenants"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(64), nullable=False, unique=True, index=True, comment="Agent ID")

    # 租户信息
    name = Column(String(100), nullable=False, comment="租户名称")
    description = Column(Text, nullable=True, comment="描述")

    # 配额限制
    max_memories = Column(Integer, default=10000, comment="最大记忆数")
    max_storage_mb = Column(Integer, default=1024, comment="最大存储 (MB)")

    # 状态
    status = Column(String(20), default="active", comment="状态 (active/suspended/deleted)")

    # 时间
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    last_active_at = Column(DateTime, nullable=True, comment="最后活跃时间")

    # API Key (用于鉴权)
    api_key = Column(String(64), nullable=True, unique=True, comment="API 密钥")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "max_memories": self.max_memories,
            "max_storage_mb": self.max_storage_mb,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_active_at": self.last_active_at.isoformat() if self.last_active_at else None,
        }

    def __repr__(self):
        return f"<AgentTenant(agent_id={self.agent_id}, name={self.name})>"


# ========== 爻变历史表 ==========


class YaoChangeHistory(Base):
    """
    爻变历史表

    记录每次记忆的爻变/卦变历史
    """

    __tablename__ = "yao_change_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    memory_id = Column(Integer, nullable=False, index=True, comment="记忆 ID")
    agent_id = Column(String(64), nullable=False, index=True, comment="Agent ID")

    # 变化信息
    old_hexagram = Column(String(6), comment="原六爻")
    new_hexagram = Column(String(6), comment="新六爻")
    changed_positions = Column(JSON, default=list, comment="变化的爻位 (索引列表)")

    # 变化类型
    change_type = Column(String(20), comment="变化类型 (yao_change/gua_change/self_heal)")
    action = Column(String(50), comment="调度动作")
    reason = Column(Text, comment="变化原因")

    # 状态变化
    old_state = Column(String(20), comment="原状态")
    new_state = Column(String(20), comment="新状态")

    # 权重/热度变化
    weight_change = Column(Float, default=0.0)
    hot_change = Column(Float, default=0.0)

    # 时间
    created_at = Column(DateTime, server_default=func.now())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "memory_id": self.memory_id,
            "agent_id": self.agent_id,
            "old_hexagram": self.old_hexagram,
            "new_hexagram": self.new_hexagram,
            "changed_positions": self.changed_positions or [],
            "change_type": self.change_type,
            "action": self.action,
            "reason": self.reason,
            "old_state": self.old_state,
            "new_state": self.new_state,
            "weight_change": self.weight_change,
            "hot_change": self.hot_change,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f"<YaoChangeHistory(memory_id={self.memory_id}, change_type={self.change_type})>"


# ========== 辅助函数 ==========


def create_all_tables(engine):
    """创建所有表"""
    Base.metadata.create_all(engine)
    print("✅ 数据库表已创建")


def drop_all_tables(engine):
    """删除所有表"""
    Base.metadata.drop_all(engine)
    print("⚠️ 数据库表已删除")


# ========== 初始化脚本 ==========

if __name__ == "__main__":
    from sqlalchemy import create_engine

    # 测试连接
    engine = create_engine("sqlite:///test_models.db", echo=True)

    print("=" * 80)
    print("📊 数据模型测试")
    print("=" * 80)

    # 创建表
    create_all_tables(engine)

    # 测试插入
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=engine)
    session = Session()

    # 创建测试租户
    tenant = AgentTenant(agent_id="test-agent-001", name="测试租户", description="用于测试的租户")
    session.add(tenant)

    # 创建测试记忆
    memory = YijingMemory(
        agent_id="test-agent-001",
        memory_id="mem-test-001",
        hexagram="111111",
        bagua_type="qian",
        hexagram_num=1,
        sancai_layer="tien",
        wuxing="jin",
        content="测试记忆内容",
        weight=0.9,
        hot_score=0.8,
    )
    session.add(memory)

    session.commit()

    # 查询测试
    memories = session.query(YijingMemory).filter_by(agent_id="test-agent-001").all()
    print(f"\n✅ 查询到 {len(memories)} 条记忆")
    for m in memories:
        print(f"  - ID:{m.id}, 卦:{m.bagua_type}, 内容:{m.content[:20]}...")

    session.close()

    print("\n" + "=" * 80)
    print("✅ 数据模型测试完成！")
    print("=" * 80)
