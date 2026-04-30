"""
Standalone utilities — replaces all modules.core.tooling.common_utils imports.
No external dependencies beyond stdlib.
"""
import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


# ===== Logging =====
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ===== Exceptions =====
class MetaMemoryError(Exception):
    """Base exception."""


class DatabaseError(MetaMemoryError):
    """Database errors."""


class ValidationError(MetaMemoryError):
    """Validation errors."""


# ===== Validation =====
def validate_dict(data, name="data"):
    if not isinstance(data, dict):
        raise ValidationError(f"{name} must be a dict")


def validate_list(data, name="data", min_length=0):
    if not isinstance(data, list):
        raise ValidationError(f"{name} must be a list")
    if len(data) < min_length:
        raise ValidationError(f"{name} must have at least {min_length} items")


def validate_string(value, name="value", min_length=0):
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string")
    if len(value) < min_length:
        raise ValidationError(f"{name} must be at least {min_length} chars")


def validate_float(value, name="value", min_value=None, max_value=None):
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number")
    if min_value is not None and value < min_value:
        raise ValidationError(f"{name} must be >= {min_value}")
    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} must be <= {max_value}")


def validate_integer(value, name="value", min_value=None, max_value=None):
    if not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer")
    if min_value is not None and value < min_value:
        raise ValidationError(f"{name} must be >= {min_value}")
    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} must be <= {max_value}")


def validate_required_fields(data: dict, fields: List[str]):
    for f in fields:
        if f not in data:
            raise ValidationError(f"Missing required field: {f}")


# ===== ID generation =====
def generate_id(seed: str = "") -> str:
    if seed:
        return hashlib.md5(seed.encode()).hexdigest()[:12]
    return str(uuid.uuid4())[:12]


# ===== File utils =====
def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)
