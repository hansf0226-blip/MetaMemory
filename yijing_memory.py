#!/usr/bin/env python3
"""易经记忆系统主入口 (v7.0)

使用方法：
    from yijing_memory import YijingMemorySystem
"""

import logging

from core import (
    ab_test_engine,
    auto_keyword_expander,
    core_yijing,
    embedding,
    enhanced_retrieval,
    five_elements_scheduler,
    i_ching_geometry,
    i_ching_v2_system,
    intelligent_decay,
    learning,
    memory,
    memory_db,
    memory_layer_classifier,
    memory_permeation,
    mind_state,
    models,
    mysql_store,
    rag_engine,
    storage_manager,
    vector_store,
    yao_encoding,
)


class YijingMemorySystem:
    """完整的易经记忆系统"""

    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def version(self):
        return "7.0"
