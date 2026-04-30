import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

# 自动加载 .env 环境变量（SiliconFlow API Key 等）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 手动加载 .env（没有 python-dotenv 时）
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        for _line in _env_path.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ[_k.strip()] = _v.strip()

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# 新增：A/B 测试和词库扩展
from fastapi.responses import FileResponse, JSONResponse

from core.ab_test_engine import get_test_engine
from core.auto_keyword_expander import get_expander
from core.core_yijing import content_to_hexagram
from core.multimodal import (
    analyze_multimodal_content,
    generate_multimodal_embedding,
    get_multimodal_processor,
    process_image,
    speech_to_text,
)

# 新模块
from core.storage_manager import get_storage_manager
from logger import logger
from modules.core.tooling.common_utils import (
    AuthenticationError,
    AuthorizationError,
    BaseError,
    ConflictError,
    DatabaseError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    TimeoutError,
    ValidationError,
)

# 配置管理
from modules.core.tooling.config_manager import get_config
from scheduler_task import shutdown_scheduler, start_scheduler
from schemas import AgentTenantCreateReq, EncodeHexagramReq, HealthResponse, MemorySaveReq

# 主动记忆引擎
from core.memory_auto import get_auto_memory

# 创建 FastAPI 应用
app = FastAPI(
    title="MetaMemory - AI Agent 长期记忆系统",
    description="MetaMemory: 为 AI Agent 提供长期记忆存储与检索能力的多租户系统",
    version="2.0.0",
)

# 全局跨域配置
# 从配置管理模块获取允许的域名列表
# 支持通过环境变量覆盖：CORS_ALLOWED_ORIGINS='["http://example.com", "https://example.com"]'
ALLOWED_ORIGINS = get_config(
    "cors.allowed_origins", [
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:3002", "http://127.0.0.1:3002",
        "http://localhost:8080",
        "null"  # file:// 协议直接打开 HTML
    ]
)

# 确保返回的是列表
if not isinstance(ALLOWED_ORIGINS, list):
    ALLOWED_ORIGINS = [ALLOWED_ORIGINS]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # 生产环境限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求追踪中间件 - 每个请求生成唯一ID + 记录处理时间
@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    import time
    import uuid

    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{(time.time()-start)*1000:.1f}"
    return response


# API版本头
@app.middleware("http")
async def add_api_version_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-API-Version"] = "v1"
    return response


# 异常处理中间件
@app.exception_handler(ValidationError)
def handle_validation_error(request, exc):
    """处理验证异常"""
    logger.error(f"验证错误: {exc.message}")
    return JSONResponse(status_code=422, content={"code": exc.code, "message": exc.message, "details": exc.details})


@app.exception_handler(DatabaseError)
def handle_database_error(request, exc):
    """处理数据库异常"""
    logger.error(f"数据库错误: {exc.message}")
    return JSONResponse(status_code=500, content={"code": exc.code, "message": exc.message, "details": exc.details})


@app.exception_handler(NotFoundError)
def handle_not_found_error(request, exc):
    """处理资源不存在异常"""
    logger.error(f"资源不存在: {exc.message}")
    return JSONResponse(status_code=404, content={"code": exc.code, "message": exc.message, "details": exc.details})


@app.exception_handler(ConflictError)
def handle_conflict_error(request, exc):
    """处理资源冲突异常"""
    logger.error(f"资源冲突: {exc.message}")
    return JSONResponse(status_code=409, content={"code": exc.code, "message": exc.message, "details": exc.details})


@app.exception_handler(AuthenticationError)
def handle_authentication_error(request, exc):
    """处理认证异常"""
    logger.error(f"认证错误: {exc.message}")
    return JSONResponse(status_code=401, content={"code": exc.code, "message": exc.message, "details": exc.details})


@app.exception_handler(AuthorizationError)
def handle_authorization_error(request, exc):
    """处理授权异常"""
    logger.error(f"授权错误: {exc.message}")
    return JSONResponse(status_code=403, content={"code": exc.code, "message": exc.message, "details": exc.details})


@app.exception_handler(RateLimitError)
def handle_rate_limit_error(request, exc):
    """处理速率限制异常"""
    logger.error(f"速率限制: {exc.message}")
    return JSONResponse(status_code=429, content={"code": exc.code, "message": exc.message, "details": exc.details})


@app.exception_handler(ServiceUnavailableError)
def handle_service_unavailable_error(request, exc):
    """处理服务不可用异常"""
    logger.error(f"服务不可用: {exc.message}")
    return JSONResponse(status_code=503, content={"code": exc.code, "message": exc.message, "details": exc.details})


@app.exception_handler(TimeoutError)
def handle_timeout_error(request, exc):
    """处理超时异常"""
    logger.error(f"超时错误: {exc.message}")
    return JSONResponse(status_code=504, content={"code": exc.code, "message": exc.message, "details": exc.details})


@app.exception_handler(Exception)
def handle_generic_error(request, exc):
    """处理通用异常"""
    logger.error(f"未知错误: {str(exc)}")
    return JSONResponse(
        status_code=500, content={"code": 500, "message": "内部服务器错误", "details": {"error": str(exc)}}
    )


# 启动事件
@app.on_event("startup")
async def startup_event():
    """
    服务启动事件

    初始化存储管理器 + 启动 APScheduler 定时任务
    （自愈调度 + 智能衰减 + 过期清理）
    """
    global startup_time
    startup_time = time.time()
    logger.info("🚀 MetaMemory AI Agent 记忆系统启动中...")
    # 启用调度器（SQLite 并发问题已通过 try/except 兜底）
    try:
        start_scheduler()
        logger.info("✅ 调度器启动成功")
    except Exception as e:
        logger.warning(f"调度器启动失败（不影响主服务）：{e}")

    # 初始化存储管理器
    global storage_manager
    storage_manager = get_storage_manager()
    logger.info("✅ 存储管理器初始化完成")

    logger.info("✅ 系统启动完成")


# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """
    服务关闭事件

    取消所有 APScheduler 定时任务，释放资源
    """
    logger.info("🛑 系统关闭中...")
    shutdown_scheduler()


# ============================================================================
# 1. 健康检查接口
# ============================================================================
@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查 - 运维监控使用

    返回服务健康状态，包含服务名称、状态码和基本信息

    Returns:
        dict: {"code": 200, "status": "ok", "service": "yijing-agent-multi-tenant"}"""
    return {"code": 200, "status": "ok", "service": "yijing-agent-multi-tenant"}


# ============================================================================
# 2. 卦象编码接口
# ============================================================================
@app.post("/api/v1/memory/encode/hexagram", tags=["记忆"])
async def encode_hexagram(req: EncodeHexagramReq):
    """
    将内容转换为编码

    使用智能编码算法将文本内容转换为编码、类型、属性和层次

    Args:
        req: 编码请求（包含内容和记忆类型）

    Returns:
        dict: 编码结果，包含编码、类型、属性、层次和是否使用了降级方案

    Raises:
        HTTPException: 编码失败时抛出 400 错误
    """
    try:
        result = content_to_hexagram(req.content, req.memory_type)
        # 处理不同长度的返回值
        if len(result) == 5:
            hex_arr, bagua, wuxing, layer, is_fallback = result
        else:
            hex_arr, bagua, wuxing, layer = result
            is_fallback = False

        # 如果使用了降级方案，记录日志
        if is_fallback:
            logger.warning(f"编码使用了降级方案：{req.content[:50]}...")

        return {
            "code": 200,
            "hexagram": hex_arr,
            "bagua_type": bagua,
            "wuxing": wuxing,
            "sancai_layer": layer,
            "is_fallback": is_fallback,
        }
    except Exception as e:
        logger.error(f"编码失败：{str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# 3. 创建记忆
# ============================================================================
@app.post("/api/v1/memories", tags=["记忆"])
async def create_memory(req: MemorySaveReq):
    """
    创建新记忆

    将记忆写入 MySQL 关系数据库和向量数据库

    Args:
        req: 记忆创建请求（包含卦象、五行、三才等信息）

    Returns:
        dict: 创建结果 {"code": 200, "msg": "创建成功", "memory_id": "..."}

    Raises:
        HTTPException: 记忆 ID 已存在时抛出 400 错误
    """
    try:
        # 准备记忆数据
        memory_data = {
            "agent_id": req.agent_id,
            "memory_id": req.memory_id,
            "hexagram": req.hexagram,
            "bagua_type": req.bagua_type,
            "sancai_layer": req.sancai_layer,
            "wuxing": req.wuxing,
            "content": req.content,
            "hot_score": req.hot_score,
        }

        # 使用存储管理器创建记忆
        memory_id = storage_manager.create_memory(memory_data)

        logger.info(f"✅ 记忆创建成功：{memory_id}")

        return {"code": 200, "msg": "创建成功", "memory_id": memory_id}
    except (HTTPException, BaseError):
        raise
    except Exception as e:
        logger.error(f"创建记忆失败：{str(e)}")
        raise DatabaseError(f"创建记忆失败：{str(e)}", code=500, details={"error": str(e)})


# ============================================================================
# 4. 获取记忆列表
# ============================================================================
@app.get("/api/v1/memories", tags=["记忆"])
async def get_memories(agent_id: str = "", page: int = 1, page_size: int = 20):
    """
    获取记忆列表（分页）

    根据Agent ID获取记忆列表，支持分页查询

    Args:
        agent_id: Agent ID（不传则返回所有记忆）
        page: 页码，默认为1
        page_size: 每页大小，默认为20

    Returns:
        dict: 分页结果，包含总数量、页码、每页大小和记忆列表

    Raises:
        HTTPException: 查询失败时抛出 500 错误
    """
    try:
        # 使用存储管理器获取记忆列表
        memories = storage_manager.get_memories(agent_id=agent_id, limit=page_size, offset=(page - 1) * page_size)

        # 获取总数量
        total = storage_manager.count_memories(agent_id=agent_id)

        return {"code": 200, "total": total, "page": page, "page_size": page_size, "list": memories}
    except (HTTPException, BaseError):
        raise
    except Exception as e:
        logger.error(f"获取记忆列表失败：{str(e)}")
        raise DatabaseError(f"获取记忆列表失败：{str(e)}", code=500, details={"error": str(e)})


# ============================================================================
# 5. 向量检索
# ============================================================================
@app.get("/api/v1/memories/search", tags=["记忆"])
async def search_memories(
    content: str,
    top_k: int = 5,
    agent_id: str = "",
):
    """
    基于卦象相似度的向量检索

    根据查询内容和卦象相似度检索相关记忆

    Args:
        content: 查询内容
        top_k: 返回结果数量，默认为5
        agent_id: Agent ID（不传则搜索所有）

    Returns:
        dict: 检索结果，包含匹配的记忆列表和相似度信息

    Raises:
        HTTPException: 检索失败时抛出 500 错误
    """
    try:
        # 生成查询向量的语义嵌入
        from core.core_yijing import generate_semantic_embedding
        query_vector = generate_semantic_embedding(content)

        # 使用存储管理器搜索记忆（带语义向量）
        results = storage_manager.search_memories(
            agent_id=agent_id,
            keyword=content,
            limit=top_k,
            query_vector=query_vector,
        )

        # 格式化返回结果
        formatted_results = []
        for r in results:
            formatted_results.append(
                {
                    "memory_id": r.get("memory_id"),
                    "agent_id": agent_id,
                    "bagua_type": r.get("bagua_type"),
                    "wuxing": r.get("wuxing"),
                    "sancai_layer": r.get("sancai_layer"),
                    "content": (
                        r.get("content", "")[:100] + "..." if len(r.get("content", "")) > 100 else r.get("content", "")
                    ),
                    "similarity": r.get("similarity", 0.0),
                    "hot_score": r.get("hot_score", 0.0),
                }
            )

        return {
            "code": 200,
            "data": formatted_results,
            "total": len(results),
            "query": content,
        }
    except Exception as e:
        logger.error(f"向量检索失败：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/memories/{memory_id}/relations", tags=["记忆"])
async def get_memory_relations(
    memory_id: str = Path(description="记忆ID"),
    agent_id: str = "",
    top_k: int = 10,
):
    """
    查询指定记忆的关联记忆

    基于记忆图谱关系（天然卦对/同卦象/同层级/同五行）返回关联记忆列表
    """
    try:
        results = storage_manager.get_related_memories(
            memory_id=memory_id,
            agent_id=agent_id,
            top_k=top_k,
        )
        return {
            "code": 200,
            "data": results,
            "total": len(results),
            "memory_id": memory_id,
        }
    except Exception as e:
        logger.error(f"关联查询失败：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 9. 系统统计
# ============================================================================
@app.get("/api/v1/memories/{memory_id}", tags=["记忆"])
async def get_memory(memory_id: str, agent_id: str = ""):
    """
    获取记忆详情

    根据记忆ID获取详细信息

    Args:
        memory_id: 记忆ID
        agent_id: Agent ID（可选，不提供时自动搜索）

    Returns:
        dict: 记忆详情，包含卦象、五行、三才等信息

    Raises:
        HTTPException: 记忆不存在时抛出 404 错误
        HTTPException: 查询失败时抛出 500 错误
    """
    try:
        # 使用存储管理器获取记忆详情
        if agent_id:
            memory = storage_manager.get_memory(memory_id, agent_id)
        else:
            # 搜索所有 agent 找这条记忆
            all_agents = storage_manager.get_all_agent_ids()
            memory = None
            for aid in all_agents:
                m = storage_manager.get_memory(memory_id, aid)
                if m:
                    memory = m
                    break

        if not memory:
            raise NotFoundError("记忆不存在", code=404, details={"memory_id": memory_id})

        return {"code": 200, "data": memory}
    except (HTTPException, BaseError):
        raise
    except Exception as e:
        logger.error(f"获取记忆详情失败：{str(e)}")
        raise DatabaseError(f"获取记忆详情失败：{str(e)}", code=500, details={"error": str(e)})


# ============================================================================
# 6. 更新记忆
# ============================================================================
@app.put("/api/v1/memories/{memory_id}", tags=["记忆"])
async def update_memory(memory_id: str, req: MemorySaveReq):
    """
    更新记忆

    根据记忆ID更新记忆信息

    Args:
        memory_id: 记忆ID
        req: 记忆更新请求（包含卦象、五行、三才等信息）

    Returns:
        dict: 更新结果 {"code": 200, "msg": "更新成功"}

    Raises:
        HTTPException: 记忆不存在时抛出 404 错误
        HTTPException: 更新失败时抛出 500 错误
    """
    try:
        # 准备更新数据
        update_data = {
            "hexagram": req.hexagram,
            "bagua_type": req.bagua_type,
            "sancai_layer": req.sancai_layer,
            "wuxing": req.wuxing,
            "content": req.content,
            "hot_score": req.hot_score,
        }

        # 使用存储管理器更新记忆
        updated_memory = storage_manager.update_memory(memory_id, req.agent_id, update_data)

        if not updated_memory:
            raise NotFoundError("记忆不存在", code=404, details={"memory_id": memory_id})

        logger.info(f"✅ 记忆更新成功：{memory_id}")

        return {"code": 200, "msg": "更新成功"}
    except (HTTPException, BaseError):
        raise
    except Exception as e:
        logger.error(f"更新记忆失败：{str(e)}")
        raise DatabaseError(f"更新记忆失败：{str(e)}", code=500, details={"error": str(e)})


# ============================================================================
# 7. 删除记忆
# ============================================================================
@app.delete("/api/v1/memories/{memory_id}", tags=["记忆"])
async def delete_memory(memory_id: str, agent_id: str = ""):
    """
    删除记忆

    根据记忆ID删除记忆（支持跨Agent搜索）

    Args:
        memory_id: 记忆ID
        agent_id: Agent ID（可选）

    Returns:
        dict: 删除结果 {"code": 200, "msg": "删除成功"}

    Raises:
        HTTPException: 记忆不存在时抛出 404 错误
        HTTPException: 删除失败时抛出 500 错误
    """
    try:
        # 如果没有指定 agent_id，尝试查找
        if not agent_id:
            all_agents = storage_manager.get_all_agent_ids()
            for aid in all_agents:
                m = storage_manager.get_memory(memory_id, aid)
                if m:
                    agent_id = aid
                    break

        # 使用存储管理器删除记忆
        deleted = storage_manager.delete_memory(memory_id, agent_id)

        if not deleted:
            raise NotFoundError("记忆不存在", code=404, details={"memory_id": memory_id})

        logger.info(f"✅ 记忆删除成功：{memory_id}")

        return {"code": 200, "msg": "删除成功"}
    except (HTTPException, BaseError):
        raise
    except Exception as e:
        logger.error(f"删除记忆失败：{str(e)}")
        raise DatabaseError(f"删除记忆失败：{str(e)}", code=500, details={"error": str(e)})


# ============================================================================
# 7. 监控指标
# ============================================================================
@app.get("/api/v1/metrics", tags=["监控"])
async def get_metrics(request: Request):
    """
    系统监控指标（Prometheus 兼容格式）

    返回当前系统的实时运行指标，供监控告警系统拉取。

    Returns:
        - memory_total: 总记忆数
        - agent_count: agent 数量
        - avg_hot_score: 平均热度
        - storage_backends: 各后端类型统计
        - cache_size: 缓存条目数
        - uptime_seconds: 服务运行时长（秒）

    Raises:
        500: 服务器内部错误
    """
    try:
        import time

        now = time.time()
        uptime = int(now - startup_time)

        stats = storage_manager.get_system_stats()
        mem_stats = stats.get("memory_db", {})
        vec_stats = stats.get("vector_db", {})

        return {
            "status": "ok",
            "uptime_seconds": uptime,
            "metrics": {
                "memory_total": mem_stats.get("total_memories", 0),
                "agent_count": mem_stats.get("agent_count", 0),
                "avg_hot_score": round(mem_stats.get("avg_hot_score", 0), 4),
                "vector_total": vec_stats.get("total_vectors", 0),
                "api_version": "v1",
            },
        }
    except Exception as e:
        logger.error(f"获取监控指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取监控指标失败: {str(e)}")


# ============================================================================
# 8. 向量检索
# ============================================================================
@app.get("/api/v1/stats", tags=["系统"])
async def get_stats():
    """
    获取系统统计数据

    获取系统的记忆总数、Agent总数和五行分布等统计信息

    Returns:
        dict: 系统统计数据，包含记忆数量、Agent数量和五行分布

    Raises:
        HTTPException: 查询失败时抛出 500 错误
    """
    try:
        # 使用存储管理器获取统计数据
        stats = storage_manager.get_system_stats()

        return {"code": 200, "data": stats}
    except Exception as e:
        logger.error(f"获取统计失败：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 10. 记忆历史
# ============================================================================
@app.get("/api/v1/memories/{memory_id}/history", tags=["记忆"])
async def get_memory_history(memory_id: str, agent_id: str = ""):
    """获取记忆变更历史

    返回指定记忆的完整访问和修改记录，包括创建时间、最后访问时间等

    Args:
        memory_id: 记忆ID
        agent_id: Agent ID（可选）

    Returns:
        dict: 包含历史记录列表，每条记录含时间戳和操作类型

    Raises:
        HTTPException: 记忆不存在时抛出 404"""
    try:
        # 使用存储管理器获取记忆详情
        if agent_id:
            memory = storage_manager.get_memory(memory_id, agent_id)
        else:
            all_agents = storage_manager.get_all_agent_ids()
            memory = None
            for aid in all_agents:
                m = storage_manager.get_memory(memory_id, aid)
                if m:
                    memory = m
                    break

        if not memory:
            raise NotFoundError("记忆不存在", code=404, details={"memory_id": memory_id})

        # 返回基本信息（实际项目中应有单独的历史表）
        return {
            "code": 200,
            "data": {
                "memory_id": memory.get("memory_id"),
                "create_time": memory.get("created_at"),
                "update_time": memory.get("updated_at"),
                "history": [],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取历史失败：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== 多模态相关接口 ==========


@app.post("/api/v1/multimodal/speech-to-text", tags=["多模态"])
async def speech_to_text_api(audio_file: UploadFile = File(...)):
    """
    语音转文字

    将音频文件转换为文本

    Args:
        audio_file: 音频文件

    Returns:
        dict: 转换结果，包含文本内容

    Raises:
        HTTPException: 转换失败时抛出 400 错误
    """
    try:
        temp_path = f"/tmp/yijing_multimodal_{uuid.uuid4().hex}_{audio_file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await audio_file.read())

        try:
            text = speech_to_text(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if not text:
            raise HTTPException(status_code=400, detail="语音转文字失败")

        return {"code": 200, "data": {"text": text}}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"语音转文字失败：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/multimodal/process-image", tags=["多模态"])
async def process_image_api(image_file: UploadFile = File(...)):
    """
    图像处理

    分析图像并返回图像信息

    Args:
        image_file: 图像文件

    Returns:
        dict: 图像处理结果

    Raises:
        HTTPException: 处理失败时抛出 400 错误
    """
    try:
        temp_path = f"/tmp/yijing_multimodal_{uuid.uuid4().hex}_{image_file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await image_file.read())

        try:
            result = process_image(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if not result:
            raise HTTPException(status_code=400, detail="图像处理失败")

        return {"code": 200, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"图像处理失败：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/multimodal/generate-embedding", tags=["多模态"])
async def generate_embedding_api(data: dict):
    """
    生成多模态嵌入

    为文本、图像或音频生成多模态嵌入

    Args:
        data: 多模态数据，包含 text、image_path 或 audio_path

    Returns:
        dict: 生成的嵌入向量

    Raises:
        HTTPException: 生成失败时抛出 400 错误
    """
    try:
        # 调用多模态嵌入生成
        embedding = generate_multimodal_embedding(data)

        if not embedding:
            raise HTTPException(status_code=400, detail="生成多模态嵌入失败")

        return {"code": 200, "data": {"embedding": embedding}}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成多模态嵌入失败：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/multimodal/analyze-content", tags=["多模态"])
async def analyze_content_api(data: dict):
    """
    分析多模态内容

    分析文本、图像或音频内容

    Args:
        data: 多模态数据，包含 text、image_path 或 audio_path

    Returns:
        dict: 分析结果

    Raises:
        HTTPException: 分析失败时抛出 400 错误
    """
    try:
        # 调用多模态内容分析
        analysis = analyze_multimodal_content(data)

        if not analysis:
            raise HTTPException(status_code=400, detail="分析多模态内容失败")

        return {"code": 200, "data": analysis}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分析多模态内容失败：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/multimodal/status", tags=["多模态"])
async def multimodal_status_api():
    """
    获取多模态处理器状态

    返回多模态处理器的功能可用性

    Returns:
        dict: 功能可用性状态
    """
    try:
        processor = get_multimodal_processor()
        status = processor.is_available()

        return {"code": 200, "data": {"status": status}}
    except Exception as e:
        logger.error(f"获取多模态状态失败：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 11. 创建 Agent
# ============================================================================
@app.post("/api/v1/agents", tags=["Agent 管理"])
async def create_agent(req: AgentTenantCreateReq):
    """创建新的 Agent 多租户实例

    在系统中注册新的 Agent 租户，分配唯一标识符

    Args:
        req: AgentTenantCreateReq，包含租户名称和配置

    Returns:
        dict: 创建结果，包含 agent_id"""
    try:
        # 准备Agent数据
        agent_data = {"agent_id": req.agent_id, "name": req.agent_name, "description": req.remark, "status": "active"}

        # 使用存储管理器创建Agent
        created_agent = storage_manager.create_agent(agent_data)

        if not created_agent:
            raise HTTPException(status_code=400, detail="Agent 已存在")

        logger.info(f"✅ Agent 创建成功：{req.agent_id}")

        return {"code": 200, "msg": "创建成功", "agent_id": req.agent_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建 Agent 失败：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 12. Agent 列表
# ============================================================================
@app.get("/api/v1/agents", tags=["Agent 管理"])
async def get_agents():
    """获取所有已注册的 Agent 租户列表

    从数据库查询所有 Agent 的基本信息（agent_id 列表）

    Args:
        无

    Returns:
        dict: {"code": 200, "list": [{"agent_id": "..."}]}"""
    try:
        # 使用存储管理器获取Agent列表
        agent_ids = storage_manager.get_all_agent_ids()

        return {"code": 200, "list": [{"agent_id": aid} for aid in agent_ids]}
    except Exception as e:
        logger.error(f"获取 Agent 列表失败：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 额外接口：获取 Agent 信息
# ============================================================================
@app.get("/api/v1/agents/{agent_id}/info", tags=["Agent 管理"])
async def get_agent_info(agent_id: str):
    """获取指定 Agent 的详细信息

    根据 agent_id 查询该租户的配置、状态和统计信息

    Args:
        agent_id: 租户唯一标识

    Returns:
        dict: Agent 详细信息"""
    try:
        # 使用存储管理器获取Agent信息
        agent_info = storage_manager.get_agent_info(agent_id)

        if not agent_info:
            raise HTTPException(status_code=404, detail="Agent 不存在")

        return {"code": 200, "data": agent_info}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取 Agent 信息失败：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 启动命令


# ============================================================================
# A/B 测试接口
# ============================================================================
@app.get("/api/v1/ab-test/run", tags=["测试"])
async def run_ab_test():
    """运行 A/B 测试对比实验

    执行分流实验，收集实验组/对照组数据

    Returns:
        dict: 实验运行状态和初步结果"""
    try:
        test_engine = get_test_engine()
        report = test_engine.run_full_test()

        return {
            "code": 200,
            "data": report["summary"],
            "report_path": test_engine.export_report(),
        }
    except Exception as e:
        logger.error(f"A/B 测试失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ab-test/report", tags=["测试"])
async def get_ab_test_report():
    """获取 A/B 测试报告

    返回已完成的 A/B 测试的详细对比数据，包括转化率、置信度等

    Returns:
        dict: 测试报告，包含统计显著性分析"""
    try:
        test_engine = get_test_engine()
        report_path = test_engine.export_report()

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        return {
            "code": 200,
            "data": report,
        }
    except Exception as e:
        logger.error(f"获取报告失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 词库管理接口
# ============================================================================
@app.post("/api/v1/keywords/learn", tags=["词库"])
async def learn_keywords(text: str):
    """从文本中自动学习新关键词

    对输入文本进行 NLP 处理，提取并保存新的关键词到词库

    Args:
        text: 待学习的文本内容

    Returns:
        dict: 学习结果，包含新增关键词数量"""
    try:
        expander = get_expander()
        result = expander.process_text(text, auto_save=True)

        return {
            "code": 200,
            "data": {
                "extracted": result["extracted"][:10],
                "new_words": result["new_words"],
                "stats": expander.get_stats(),
            },
        }
    except Exception as e:
        logger.error(f"学习失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/keywords/stats", tags=["词库"])
async def get_keyword_stats():
    """获取词库统计信息

    返回系统中已学习的关键词统计，包括总数、学习时间等

    Returns:
        dict: 包含关键词总数、各类词频统计等信息"""
    try:
        expander = get_expander()
        stats = expander.get_stats()

        return {
            "code": 200,
            "data": stats,
        }
    except Exception as e:
        logger.error(f"获取统计失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 向量检索测试接口
# ============================================================================
@app.post("/api/v1/vector/insert", tags=["向量"])
async def insert_vector(memory_id: str, content: str, agent_id: str = "default"):
    """插入向量到向量数据库

    将高维向量连同其元数据（agent_id/memory_id）写入向量数据库，支持后续相似度检索

    Args:
        data: 包含 vector (list[float])、agent_id、memory_id 的字典

    Returns:
        dict: 插入结果确认

    Raises:
        HTTPException: 向量维度不匹配时抛出 400"""
    try:
        storage_manager = get_storage_manager()

        from core.core_yijing import content_to_hexagram

        hex_arr, bagua, wuxing, layer = content_to_hexagram(content, "chat")

        memory = {
            "memory_id": memory_id,
            "content": content,
            "agent_id": agent_id,
            "hexagram": hex_arr,
            "bagua_type": bagua,
            "wuxing": wuxing,
            "sancai_layer": layer,
            "hot_score": 0.5,
            "created_at": datetime.now().isoformat(),
        }

        # 使用storage_manager来插入记忆
        storage_manager.create_memory(memory)

        return {
            "code": 200,
            "success": True,
        }
    except Exception as e:
        logger.error(f"插入失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/vector/search", tags=["向量"])
async def search_vector(query: str, top_k: int = 5, agent_id: str = "default"):
    """向量相似度检索

    在向量数据库中检索与给定向量最相似的记忆，支持按 agent_id 过滤

    Args:
        query: 查询向量 (list[float])
        top_k: 返回数量（默认5）
        agent_id: Agent ID（可选）

    Returns:
        dict: 包含相似记忆列表及相似度分数"""
    try:
        storage_manager = get_storage_manager()
        # 使用storage_manager来搜索记忆
        results = storage_manager.search_memories(agent_id=agent_id, keyword=query, limit=top_k)

        return {
            "code": 200,
            "data": results,
        }
    except Exception as e:
        logger.error(f"检索失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 配置管理 API
# ============================================================================
@app.get("/api/v1/config", tags=["系统"])
async def get_system_config():
    """获取系统配置（API Key 脱敏）"""
    import yaml
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # 脱敏
    def mask_sensitive(obj, keys=("api_key", "password")):
        if isinstance(obj, dict):
            return {k: "***" if k in keys else mask_sensitive(v, keys) for k, v in obj.items()}
        if isinstance(obj, list):
            return [mask_sensitive(v, keys) for v in obj]
        return obj
    return {"code": 200, "data": mask_sensitive(config)}

@app.put("/api/v1/config", tags=["系统"])
async def update_system_config(config_data: dict):
    """更新系统配置（合并写入 config.yaml）"""
    import yaml
    config_path = Path(__file__).parent / "config.yaml"
    # 读取现有配置
    with open(config_path, "r") as f:
        current = yaml.safe_load(f) or {}
    # 深度合并
    def deep_merge(base, override):
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                deep_merge(base[k], v)
            else:
                base[k] = v
    deep_merge(current, config_data)
    # 写回
    with open(config_path, "w") as f:
        yaml.dump(current, f, allow_unicode=True, default_flow_style=False)
    logger.info(f"配置已更新: {list(config_data.keys())}")
    return {"code": 200, "msg": "配置已更新", "restart_required": True}


# ============================================================================
# 主动记忆 API (v2.3)
# ============================================================================
@app.post("/api/v1/memory/auto", tags=["主动记忆"])
async def auto_memory_process(data: dict):
    """
    一站式主动记忆处理

    输入对话文本，自动:
    1. LLM 提取关键记忆
    2. 自动存储到记忆库
    3. 加载历史上下文

    Args:
        data: {"conversation": "对话文本", "agent_id": "agent_001"}

    Returns:
        提取的记忆、存储结果、历史上下文
    """
    try:
        conversation = data.get("conversation", "")
        agent_id = data.get("agent_id", "default")

        if not conversation.strip():
            raise ValidationError("对话内容不能为空", code=400)

        engine = get_auto_memory()
        result = engine.process_conversation(conversation, agent_id)

        return {
            "code": 200,
            "data": result,
        }
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"主动记忆处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/memory/context", tags=["主动记忆"])
async def load_context(query: str, agent_id: str = "default", top_k: int = 10):
    """
    加载历史上下文

    根据当前查询检索相关历史记忆，返回可直接注入 system prompt 的文本。

    Args:
        query: 当前查询/对话文本
        agent_id: 租户 ID
        top_k: 返回条数

    Returns:
        相关记忆列表 + 格式化文本
    """
    try:
        engine = get_auto_memory()
        context = engine.load_context(query, agent_id)

        return {
            "code": 200,
            "data": context,
        }
    except Exception as e:
        logger.error(f"上下文加载失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 前端页面服务
# ============================================================================
@app.get("/", include_in_schema=False)
async def serve_frontend():
    """提供前端页面"""
    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    raise HTTPException(status_code=404, detail="前端页面未找到")


@app.get("/demo", include_in_schema=False)
async def serve_demo():
    """演示页面"""
    return await serve_frontend()


# 全局变量

# 存储管理器实例
storage_manager = None

if __name__ == "__main__":
    import uvicorn

    host = get_config("server.host", "0.0.0.0")
    port = get_config("server.port", 8000)
    uvicorn.run(app, host=host, port=port)
