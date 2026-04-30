#!/usr/bin/env python3
"""
MetaMemory 一键安装脚本
用法:
    python install.py              # 安装核心依赖
    python install.py --dev        # 安装开发依赖
    python install.py --ml         # 安装 ML/多模态依赖
    python install.py --all        # 安装全部依赖
"""
import argparse
import subprocess
import sys
from pathlib import Path

REQUIREMENTS_FILES = {
    "core": "requirements.txt",
    "dev": "requirements-dev.txt",
    "ml": "requirements-ml.txt",
}


def run_command(cmd, cwd=None):
    """运行 shell 命令"""
    print(f"$ {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 命令执行失败: {e}", file=sys.stderr)
        return False


def install_requirements(filepath):
    """安装 requirements 文件"""
    if not Path(filepath).exists():
        print(f"⚠️  文件不存在: {filepath}")
        return False
    print(f"\n📦 安装依赖: {filepath}")
    print("=" * 60)
    return run_command(f"{sys.executable} -m pip install -r {filepath}")


def check_python_version():
    """检查 Python 版本"""
    version = sys.version_info
    print(f"🐍 Python 版本: {version.major}.{version.minor}.{version.micro}")
    
    if not (3, 9) <= (version.major, version.minor) < (3, 12):
        print("❌ 需要 Python 3.9 - 3.11", file=sys.stderr)
        return False
    return True


def upgrade_pip():
    """升级 pip 到最新版本"""
    print("\n⬆️  升级 pip...")
    return run_command(f"{sys.executable} -m pip install --upgrade pip")


def main():
    parser = argparse.ArgumentParser(
        description="MetaMemory 一键安装脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python install.py              # 只安装核心依赖
  python install.py --dev        # 核心 + 开发依赖
  python install.py --ml         # 核心 + ML/多模态依赖
  python install.py --all        # 全部依赖
        """,
    )
    parser.add_argument("--dev", action="store_true", help="安装开发依赖")
    parser.add_argument("--ml", action="store_true", help="安装 ML/多模态依赖")
    parser.add_argument("--all", action="store_true", help="安装全部可选依赖")
    parser.add_argument("--upgrade-pip", action="store_true", help="先升级 pip")
    parser.add_argument("--no-upgrade-pip", action="store_true", help="不升级 pip")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 MetaMemory 安装工具")
    print("=" * 60)
    
    # 检查 Python 版本
    if not check_python_version():
        sys.exit(1)
    
    # 升级 pip
    if args.upgrade_pip or (not args.no_upgrade_pip):
        if not upgrade_pip():
            print("⚠️  pip 升级失败，继续安装...")
    
    # 确定要安装的文件
    files_to_install = ["core"]
    if args.dev or args.all:
        files_to_install.append("dev")
    if args.ml or args.all:
        files_to_install.append("ml")
    
    print(f"\n📋 将要安装: {', '.join(files_to_install)}")
    
    # 安装依赖
    success = True
    for name in files_to_install:
        if not install_requirements(REQUIREMENTS_FILES[name]):
            success = False
            print(f"❌ {REQUIREMENTS_FILES[name]} 安装失败", file=sys.stderr)
    
    # 检查版本
    print("\n" + "=" * 60)
    print("✅ 安装完成！验证安装...")
    print("=" * 60)
    
    try:
        import fastapi
        print(f"✅ FastAPI: {fastapi.__version__}")
    except ImportError:
        print("❌ FastAPI: 未安装")
        success = False
    
    try:
        import sqlalchemy
        print(f"✅ SQLAlchemy: {sqlalchemy.__version__}")
    except ImportError:
        print("❌ SQLAlchemy: 未安装")
        success = False
    
    try:
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
    except ImportError:
        print("❌ NumPy: 未安装")
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 全部依赖安装成功！")
        print("\n下一步:")
        print("  1. 复制 .env.example 为 .env 并配置")
        print("  2. 运行: python main_api.py")
        print("  3. 访问: http://localhost:8000/docs")
    else:
        print("⚠️  部分依赖安装失败，请检查上面的错误信息")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
