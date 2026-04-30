"""MetaMemory API Schemas"""
from pydantic import BaseModel, Field
from typing import Optional


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "2.0.0"


class MemorySaveReq(BaseModel):
    memory_id: Optional[str] = None
    agent_id: str = ""
    hexagram: Optional[str] = None
    bagua_type: Optional[str] = None
    sancai_layer: Optional[str] = None
    wuxing: Optional[str] = None
    content: str = ""
    hot_score: float = 0.5


class AgentTenantCreateReq(BaseModel):
    agent_id: str
    name: str = ""


class EncodeHexagramReq(BaseModel):
    content: str
    agent_id: Optional[str] = ""
