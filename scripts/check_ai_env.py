import os

import requests
from config import LLM_API_URL, LOCAL_MODEL_PATH


def check_ollama_service():
    """检查 Ollama 服务"""
    try:
        url = LLM_API_URL.replace("/api/generate", "/api/tags")
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            print("✅ Ollama 服务正常，易经编码模型可用")
            return True
    except Exception as e:
        print(f"❌ Ollama 连接失败：{str(e)}")
    return False


def check_local_model_file():
    """检查本地模型文件"""
    if os.path.exists(LOCAL_MODEL_PATH):
        print("✅ 本地微调易经小模型文件存在")
        return True
    else:
        print("❌ 本地小模型不存在，请执行 train_yijing_model.py 完成微调")
        return False


if __name__ == "__main__":
    print("===== 易经 AI 取象环境检测 =====")
    check_ollama_service()
    check_local_model_file()
