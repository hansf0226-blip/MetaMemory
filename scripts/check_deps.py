#!/usr/bin/env python3
"""
检查 MetaMemory 依赖安装情况
用法: python scripts/check_deps.py
"""
import sys
import os
from importlib.util import find_spec

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 依赖检查列表
DEPS = [
    # 核心依赖
    ("fastapi", "Web 框架", "core", True),
    ("uvicorn", "ASGI 服务器", "core", True),
    ("sqlalchemy", "ORM 数据库", "core", True),
    ("pydantic", "数据验证", "core", True),
    ("numpy", "数值计算", "core", True),
    ("apscheduler", "任务调度", "core", True),
    ("jieba", "中文分词", "core", True),
    ("yaml", "YAML 配置", "core", True),
    ("dotenv", "环境变量", "core", True),
    ("jwt", "JWT 认证", "core", True),
    
    # 数据库扩展
    ("pymysql", "MySQL 驱动", "db", False),
    ("redis", "Redis 缓存", "db", False),
    ("pymongo", "MongoDB 驱动", "db", False),
    ("py2neo", "Neo4j 知识图谱", "db", False),
    
    # ML/多模态
    ("torch", "PyTorch 深度学习", "ml", False),
    ("transformers", "HuggingFace 模型", "ml", False),
    ("cv2", "OpenCV 图像处理", "ml", False),
    ("whisper", "语音识别", "ml", False),
    ("faiss", "向量检索加速", "ml", False),
    ("spacy", "NLP 处理", "ml", False),
    
    # 数据科学
    ("pandas", "数据分析", "data", False),
    ("sklearn", "机器学习", "data", False),
    
    # 开发依赖
    ("pytest", "测试框架", "dev", False),
    ("black", "代码格式化", "dev", False),
    ("isort", "导入排序", "dev", False),
    ("flake8", "代码检查", "dev", False),
    ("mypy", "类型检查", "dev", False),
]

CATEGORIES = {
    "core": ("🎯 核心依赖", "\033[92m"),
    "db": ("🗄️  数据库扩展", "\033[94m"),
    "ml": ("🧠 ML/多模态", "\033[95m"),
    "data": ("📈 数据科学", "\033[96m"),
    "dev": ("🔧 开发依赖", "\033[93m"),
}


def check_dependency(module_name, display_name, category, required):
    """检查单个依赖是否安装"""
    installed = find_spec(module_name) is not None
    return {
        "module": module_name,
        "name": display_name,
        "category": category,
        "required": required,
        "installed": installed,
    }


def main():
    print("\n" + "=" * 70)
    print("🧠 MetaMemory 依赖检查工具")
    print("=" * 70)
    
    # 检查 Python 版本
    version = sys.version_info
    py_ok = (3, 9) <= (version.major, version.minor) < (3, 12)
    color = "\033[92m" if py_ok else "\033[91m"
    reset = "\033[0m"
    print(f"\n🐍 Python 版本: {color}{version.major}.{version.minor}.{version.micro}{reset}")
    if not py_ok:
        print("   ⚠️  建议使用 Python 3.9 - 3.11")
    
    # 检查所有依赖
    results = [check_dependency(*dep) for dep in DEPS]
    
    # 按分类显示
    current_category = None
    for result in results:
        cat = result["category"]
        if cat != current_category:
            current_category = cat
            cat_name, cat_color = CATEGORIES[cat]
            print(f"\n{cat_color}{cat_name}{reset}")
            print("-" * 70)
        
        status_color = "\033[92m" if result["installed"] else "\033[91m"
        status = "✅" if result["installed"] else "❌"
        req_mark = "*" if result["required"] else " "
        print(f"  {status} {result['name']:<25} {status_color}{result['module']:<20}{reset}")
    
    # 统计
    print("\n" + "=" * 70)
    print("📊 统计结果")
    print("=" * 70)
    
    required_installed = sum(1 for r in results if r["required"] and r["installed"])
    required_total = sum(1 for r in results if r["required"])
    optional_installed = sum(1 for r in results if not r["required"] and r["installed"])
    optional_total = sum(1 for r in results if not r["required"])
    
    print(f"\n  核心依赖:  {required_installed}/{required_total} 已安装")
    print(f"  可选依赖:  {optional_installed}/{optional_total} 已安装")
    
    all_core_ok = required_installed == required_total
    if all_core_ok:
        print(f"\n✅ \033[92m全部核心依赖已安装，可以正常运行！\033[0m")
    else:
        print(f"\n❌ \033[91m部分核心依赖缺失，请运行: python install.py\033[0m")
    
    print("\n💡 提示: 带 * 标记的是必需依赖")
    print("=" * 70 + "\n")
    
    return 0 if all_core_ok else 1


if __name__ == "__main__":
    sys.exit(main())
