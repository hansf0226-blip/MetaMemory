"""
Minimal configuration — reads config.yaml, no threads, no auto-create, no monitors.
Replaces the 677-line modules.core.tooling.config_manager entirely.
"""
import os
from pathlib import Path
from typing import Any

import yaml

_config: dict = {}
_loaded: bool = False


def _load() -> dict:
    global _config, _loaded
    if _loaded:
        return _config

    search_paths = [
        Path.cwd() / "config.yaml",
        Path(__file__).parent.parent / "config.yaml",
        Path.home() / ".metamemory" / "config.yaml",
    ]
    for p in search_paths:
        if p.exists():
            with open(p) as f:
                _config = yaml.safe_load(f) or {}
            _loaded = True
            return _config

    _loaded = True
    return {}


def get_config(key: str, default: Any = None) -> Any:
    """Get config value by dot-separated key, with env var override."""
    env_key = key.upper().replace(".", "_")
    env_val = os.environ.get(env_key)
    if env_val is not None:
        return env_val

    cfg = _load()
    parts = key.split(".")
    for part in parts:
        if isinstance(cfg, dict) and part in cfg:
            cfg = cfg[part]
        else:
            return default
    return cfg


def reload_config():
    global _loaded
    _loaded = False
    _config.clear()
