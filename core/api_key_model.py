#!/usr/bin/env python3
"""
API Key 数据模型
"""

from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class APIKey(Base):
    """
    API Key 表

    用于存储和管理 API Key
    """

    __tablename__ = "api_keys"

    # 主键
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Key 哈希值
    key_hash = Column(String(64), unique=True, nullable=False, comment="API Key 哈希值")

    # Key 前缀
    key_prefix = Column(String(8), nullable=False, comment="Key 前缀")

    # 关联 Agent
    agent_id = Column(String(64), nullable=False, index=True, comment="关联的 Agent ID")

    # 名称
    name = Column(String(128), default="", comment="Key 名称/描述")

    # 状态
    status = Column(Integer, default=1, comment="状态（1 启用 0 禁用）")

    # 速率限制
    rate_limit = Column(Integer, default=1000, comment="速率限制（次/小时）")

    # 过期时间
    expires_at = Column(DateTime, nullable=True, comment="过期时间")

    # 最后使用时间
    last_used_at = Column(DateTime, nullable=True, comment="最后使用时间")

    # 创建时间
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")

    # 更新时间
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="更新时间")
