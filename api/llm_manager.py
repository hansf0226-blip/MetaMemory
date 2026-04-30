#!/usr/bin/env python3
"""
🤖 大模型管理器 - 完整实现
"""

import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """LLM 异常基类"""


class LLMAuthenticationError(LLMError):
    """认证错误"""


class LLMRateLimitError(LLMError):
    """限流错误"""


class LLMTimeoutError(LLMError):
    """超时错误"""


@dataclass
class ChatMessage:
    """聊天消息"""

    role: str
    content: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class LLMResponse:
    """LLM 响应"""

    content: str
    model: str
    provider: str
    tokens: Dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0})
    latency: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: Optional[Dict] = None


class LLMBackend(ABC):
    """LLM 后端抽象基类"""

    @abstractmethod
    def chat_completion(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """聊天补全"""
        pass

    @abstractmethod
    def chat_completion_stream(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """流式聊天补全"""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        pass


class DeepSeekBackend(LLMBackend):
    """DeepSeek API 后端"""

    BASE_URL = "https://api.deepseek.com/v1"
    DEFAULT_MODEL = "deepseek-chat"
    AVAILABLE_MODELS = ["deepseek-chat", "deepseek-coder"]

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        use_cache: bool = True,
    ):
        if not api_key:
            raise LLMAuthenticationError("DeepSeek API Key 不能为空")

        self.api_key = api_key
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_cache = use_cache

        self._client = httpx.Client(timeout=timeout)
        self._cache: Dict[str, Tuple[LLMResponse, float]] = {}
        self._cache_lock = threading.Lock()

        logger.info(f"✅ DeepSeek 后端已初始化: {self.base_url}")

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _build_cache_key(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> str:
        data = {
            "messages": [m.to_dict() for m in messages],
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        content = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> Optional[LLMResponse]:
        if not self.use_cache:
            return None
        with self._cache_lock:
            item = self._cache.get(cache_key)
            if item:
                response, expire_at = item
                if expire_at > time.time():
                    return response
                else:
                    del self._cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, response: LLMResponse, ttl: int = 3600):
        if not self.use_cache:
            return
        with self._cache_lock:
            self._cache[cache_key] = (response, time.time() + ttl)
            if len(self._cache) > 1000:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

    def chat_completion(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        cache_ttl: int = 3600,
        **kwargs
    ) -> LLMResponse:
        model = model or self.DEFAULT_MODEL
        start_time = time.time()

        if temperature == 0 and self.use_cache:
            cache_key = self._build_cache_key(messages, model, temperature, max_tokens)
            cached = self._get_cached(cache_key)
            if cached:
                logger.debug(f"🔄 DeepSeek 缓存命中")
                return cached

        last_error = None
        for attempt in range(self.max_retries):
            try:
                url = f"{self.base_url}/chat/completions"
                payload = {
                    "model": model,
                    "messages": [m.to_dict() for m in messages],
                    "temperature": temperature,
                    "stream": False,
                }
                if max_tokens:
                    payload["max_tokens"] = max_tokens

                response = self._client.post(url, headers=self._get_headers(), json=payload)

                if response.status_code == 401:
                    raise LLMAuthenticationError("DeepSeek API Key 无效或已过期")
                elif response.status_code == 429:
                    raise LLMRateLimitError("DeepSeek 限流，请稍后再试")
                elif response.status_code >= 500:
                    raise LLMError(f"DeepSeek 服务器错误: {response.status_code}")

                response.raise_for_status()
                data = response.json()

                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})

                result = LLMResponse(
                    content=content,
                    model=model,
                    provider="deepseek",
                    tokens={
                        "prompt": usage.get("prompt_tokens", 0),
                        "completion": usage.get("completion_tokens", 0),
                        "total": usage.get("total_tokens", 0),
                    },
                    latency=time.time() - start_time,
                    raw_response=data,
                )

                if temperature == 0 and self.use_cache:
                    self._set_cache(cache_key, result, cache_ttl)

                logger.debug(f"✅ DeepSeek 调用完成: {result.latency:.2f}s")
                return result

            except httpx.TimeoutException:
                last_error = LLMTimeoutError(f"请求超时")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                last_error = e
                break

        raise last_error or LLMError("未知错误")

    def chat_completion_stream(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        model = model or self.DEFAULT_MODEL

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        with self._client.stream("POST", url, headers=self._get_headers(), json=payload) as response:
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue

    def get_available_models(self) -> List[str]:
        try:
            url = f"{self.base_url}/models"
            response = self._client.get(url, headers=self._get_headers())
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.warning(f"⚠️ 获取 DeepSeek 模型列表失败: {e}")
            return self.AVAILABLE_MODELS

    def health_check(self) -> Dict[str, Any]:
        try:
            start = time.time()
            models = self.get_available_models()
            latency = (time.time() - start) * 1000
            return {
                "status": "healthy",
                "provider": "deepseek",
                "latency_ms": round(latency, 2),
                "model_count": len(models),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "deepseek",
                "error": str(e),
            }


class LLMManager:
    """大模型管理器"""

    def __init__(self, config_path: Optional[str] = None):
        self._backends: Dict[str, LLMBackend] = {}
        self._default_provider: str = "deepseek"
        self._lock = threading.Lock()
        self._load_config(config_path)

    def _load_config(self, config_path: Optional[str]):
        """加载配置"""
        try:
            import yaml

            if config_path:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
            else:
                config = {}

            # 配置 DeepSeek
            deepseek_config = config.get("deepseek", {})
            api_key = deepseek_config.get("api_key", "")
            if api_key:
                self._backends["deepseek"] = DeepSeekBackend(api_key=api_key)
                logger.info("✅ DeepSeek 后端已注册")

            # 设置默认 provider
            self._default_provider = config.get("default_provider", "deepseek")

        except Exception as e:
            logger.warning(f"⚠️ LLM 管理器加载配置失败: {e}")

    def register_backend(self, name: str, backend: LLMBackend):
        """注册后端"""
        with self._lock:
            self._backends[name] = backend
            logger.info(f"✅ 已注册 LLM 后端: {name}")

    def get_backend(self, provider: Optional[str] = None) -> LLMBackend:
        """获取后端"""
        provider = provider or self._default_provider
        with self._lock:
            if provider not in self._backends:
                raise LLMError(f"未找到 LLM 后端: {provider}")
            return self._backends[provider]

    def chat(
        self,
        messages: List[ChatMessage],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """快捷聊天接口"""
        backend = self.get_backend(provider)
        return backend.chat_completion(messages, model, temperature, max_tokens, **kwargs)

    def simple_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """简单聊天接口"""
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        response = self.chat(messages, provider, model, temperature)
        return response.content

    def health_check_all(self) -> Dict[str, Any]:
        """检查所有后端健康状态"""
        results = {}
        with self._lock:
            for name, backend in self._backends.items():
                results[name] = backend.health_check()
        return results
