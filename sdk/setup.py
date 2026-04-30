#!/usr/bin/env python3
"""
易经 Agent 记忆系统 - Python SDK 安装包
"""

from pathlib import Path

from setuptools import find_packages, setup

# 读取 README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="yijing-agent-memory",
    version="2.0.0",
    author="Yijing Agent Memory Contributors",
    author_email="your.email@example.com",
    description="易经 Agent 记忆系统 Python SDK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/yijing-agent-memory",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "requests>=2.31.0",
    ],
    keywords="yijing agent memory ai易经 记忆 智能体",
    project_urls={
        "Documentation": "https://github.com/YOUR_USERNAME/yijing-agent-memory/blob/main/API.md",
        "Source": "https://github.com/YOUR_USERNAME/yijing-agent-memory",
        "Tracker": "https://github.com/YOUR_USERNAME/yijing-agent-memory/issues",
    },
)
