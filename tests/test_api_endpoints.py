"""
测试 FastAPI 端点
使用 TestClient 进行 API 集成测试
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fastapi.testclient import TestClient
    from main_api import app

    client = TestClient(app)
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI test client not available"
)


class TestHealthEndpoint:
    """测试健康检查端点"""

    def test_health_check(self):
        """基本健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_health_check_version(self):
        """健康检查返回版本"""
        response = client.get("/health")
        data = response.json()
        assert "version" in data


class TestMemoryEndpoints:
    """测试记忆 CRUD 端点"""

    def test_create_memory(self):
        """创建记忆"""
        memory_data = {
            "content": "这是一条测试记忆内容",
            "agent_id": "test_agent",
            "session_id": "test_session",
            "metadata": {"source": "test"}
        }
        response = client.post("/api/v1/memories", json=memory_data)
        assert response.status_code in (200, 201)
        data = response.json()
        assert "id" in data or "memory_id" in data

    def test_get_memory(self):
        """获取单个记忆"""
        # 先创建
        create_response = client.post(
            "/api/v1/memories",
            json={"content": "测试获取", "agent_id": "a1", "session_id": "s1"}
        )
        assert create_response.status_code in (200, 201)
        memory_id = create_response.json().get("id") or create_response.json().get("memory_id")
        
        # 再获取
        response = client.get(f"/api/v1/memories/{memory_id}")
        assert response.status_code == 200
        assert "content" in response.json()

    def test_list_memories(self):
        """列出记忆"""
        response = client.get("/api/v1/memories?agent_id=test_agent")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_update_memory(self):
        """更新记忆"""
        # 先创建
        create_response = client.post(
            "/api/v1/memories",
            json={"content": "待更新", "agent_id": "a1", "session_id": "s1"}
        )
        memory_id = create_response.json().get("id") or create_response.json().get("memory_id")
        
        # 再更新
        update_response = client.put(
            f"/api/v1/memories/{memory_id}",
            json={"content": "已更新内容"}
        )
        assert update_response.status_code in (200, 201)

    def test_delete_memory(self):
        """删除记忆"""
        # 先创建
        create_response = client.post(
            "/api/v1/memories",
            json={"content": "待删除", "agent_id": "a1", "session_id": "s1"}
        )
        memory_id = create_response.json().get("id") or create_response.json().get("memory_id")
        
        # 再删除
        delete_response = client.delete(f"/api/v1/memories/{memory_id}")
        assert delete_response.status_code in (200, 204)

    def test_get_nonexistent_memory(self):
        """获取不存在的记忆"""
        response = client.get("/api/v1/memories/nonexistent_id_12345")
        assert response.status_code == 404


class TestSearchEndpoint:
    """测试搜索端点"""

    def test_basic_search(self):
        """基本搜索"""
        response = client.get("/api/v1/search?q=测试查询&agent_id=test_agent")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_search_with_limit(self):
        """带数量限制的搜索"""
        response = client.get("/api/v1/search?q=测试&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 5

    def test_search_with_threshold(self):
        """带阈值的搜索"""
        response = client.get("/api/v1/search?q=测试&threshold=0.5")
        assert response.status_code == 200

    def test_empty_search_query(self):
        """空搜索查询"""
        response = client.get("/api/v1/search?q=")
        assert response.status_code in (200, 400)  # 正常返回或400都合理


class TestYijingEndpoints:
    """测试易经相关端点"""

    def test_encode_text(self):
        """文本编码"""
        response = client.post(
            "/api/v1/yijing/encode",
            json={"text": "天行健君子以自强不息"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "hexagram" in data or "binary" in data

    def test_get_hexagram_info(self):
        """获取卦象信息"""
        response = client.get("/api/v1/yijing/hexagram/1")
        assert response.status_code == 200
        data = response.json()
        assert "number" in data
        assert "name" in data


class TestMetricsEndpoint:
    """测试监控指标端点"""

    def test_get_metrics(self):
        """获取系统指标"""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "memory_total" in data
        assert "agent_count" in data
        assert "uptime_seconds" in data


class TestAgentEndpoints:
    """测试 Agent 相关端点"""

    def test_create_agent(self):
        """创建 Agent"""
        agent_data = {
            "name": "test_agent_001",
            "description": "测试用 Agent"
        }
        response = client.post("/api/v1/agents", json=agent_data)
        assert response.status_code in (200, 201)

    def test_list_agents(self):
        """列出所有 Agent"""
        response = client.get("/api/v1/agents")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestBatchOperations:
    """测试批量操作"""

    def test_batch_create(self):
        """批量创建记忆"""
        memories = [
            {"content": f"批量记忆{i}", "agent_id": "batch_test"}
            for i in range(5)
        ]
        response = client.post("/api/v1/memories/batch", json={"memories": memories})
        assert response.status_code in (200, 201)
        data = response.json()
        assert "created" in data or "results" in data


class TestValidation:
    """测试请求验证"""

    def test_invalid_memory_data(self):
        """无效的记忆数据"""
        response = client.post(
            "/api/v1/memories",
            json={"invalid_field": "value"}  # 缺少必要字段
        )
        # 422 验证错误 或 400 错误都合理
        assert response.status_code in (400, 422)


class TestCORS:
    """测试 CORS 配置"""

    def test_cors_headers(self):
        """CORS 响应头"""
        response = client.options(
            "/api/v1/memories",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            }
        )
        assert "access-control-allow-origin" in response.headers


class TestResponseHeaders:
    """测试响应头"""

    def test_request_id_header(self):
        """请求 ID 响应头"""
        response = client.get("/health")
        assert "x-request-id" in response.headers

    def test_process_time_header(self):
        """处理时间响应头"""
        response = client.get("/health")
        assert "x-process-time-ms" in response.headers

    def test_api_version_header(self):
        """API 版本响应头"""
        response = client.get("/health")
        assert "x-api-version" in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
