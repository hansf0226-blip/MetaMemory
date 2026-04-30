"""MetaMemory 配置管理"""
import os
import json

_config_cache = {}

def get_config(key: str, default=None):
    """从环境变量或默认值获取配置"""
    env_key = key.upper().replace('.', '_')
    env_val = os.environ.get(env_key)
    if env_val:
        try:
            return json.loads(env_val)
        except (json.JSONDecodeError, TypeError):
            return env_val
    return default

def set_config(key: str, value):
    _config_cache[key] = value
