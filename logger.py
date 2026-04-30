"""MetaMemory Logger - 轻量日志模块"""
import logging

def get_logger(name: str = "metamemory"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S'))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger()
