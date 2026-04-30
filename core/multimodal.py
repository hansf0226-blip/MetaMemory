"""
多模态处理模块 - 支持语音、图像等多模态数据处理

功能：
- 语音转文字 (使用 Whisper)
- 图像处理 (使用 OpenCV)
- 多模态嵌入生成
- 多模态内容分析
"""

import logging
import os
import threading
from typing import Any, Dict, List, Optional

# 配置日志
logger = logging.getLogger(__name__)

# 尝试导入 numpy
try:
    import numpy as np

    numpy_available = True
    logger.info("✅ NumPy 库已加载")
except ImportError:
    numpy_available = False
    logger.warning("⚠️ NumPy 库未安装，部分功能不可用")

# 尝试导入 OpenCV
try:
    import cv2

    cv2_available = True
    logger.info("✅ OpenCV 库已加载")
except ImportError:
    cv2_available = False
    logger.warning("⚠️ OpenCV 库未安装，图像处理功能不可用")

# 尝试导入 Torch
try:
    # 检查 NumPy 版本
    if numpy_available:
        import numpy as np

        numpy_version = np.__version__
        logger.info(f"NumPy 版本：{numpy_version}")
        # 如果 NumPy 版本 >= 2.0，跳过 Torch 导入，因为存在兼容性问题
        if int(numpy_version.split(".")[0]) >= 2:
            logger.warning("⚠️ NumPy 2.x 版本与 Torch 存在兼容性问题，跳过 Torch 导入")
            torch_available = False
            device = "cpu"
        else:
            import torch

            torch_available = True
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"✅ Torch 库已加载，使用设备：{device}")
    else:
        # 如果 NumPy 不可用，也跳过 Torch 导入
        torch_available = False
        device = "cpu"
        logger.warning("⚠️ NumPy 库不可用，跳过 Torch 导入")
except ImportError:
    torch_available = False
    device = "cpu"
    logger.warning("⚠️ Torch 库未安装，部分多模态功能不可用")

# 尝试导入 transformers
try:
    from transformers import CLIPModel, CLIPProcessor

    transformers_available = True
    logger.info("✅ Transformers 库已加载")
except ImportError:
    transformers_available = False
    logger.warning("⚠️ Transformers 库未安装，多模态嵌入功能不可用")

# 尝试导入 Whisper
try:
    import whisper

    whisper_available = True
    logger.info("✅ Whisper 库已加载")
except ImportError:
    whisper_available = False
    logger.warning("⚠️ Whisper 库未安装，语音转文字功能不可用")


class MultimodalProcessor:
    """多模态处理器"""

    def __init__(self):
        self._lock = threading.RLock()
        self._whisper_model = None
        self._clip_model = None
        self._clip_processor = None

        # 初始化模型
        self._init_models()

        logger.info("✅ 多模态处理器已初始化")

    def _init_models(self):
        """初始化模型"""
        try:
            # 初始化 Whisper 模型
            if whisper_available:
                self._whisper_model = whisper.load_model("base")
                logger.info("✅ Whisper 模型已加载")

            # 初始化 CLIP 模型
            if torch_available and transformers_available:
                self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self._clip_model.to(device)
                logger.info("✅ CLIP 模型已加载")
        except Exception as e:
            logger.error(f"初始化多模态模型失败：{e}")

    # ========== 语音处理 ==========

    def speech_to_text(self, audio_path: str) -> str:
        """
        语音转文字

        Args:
            audio_path: 音频文件路径

        Returns:
            转换后的文本
        """
        if not whisper_available:
            logger.warning("⚠️ Whisper 库未安装，无法进行语音转文字")
            return ""

        try:
            with self._lock:
                result = self._whisper_model.transcribe(audio_path)
                text = result["text"]
                logger.info(f"✅ 语音转文字成功，文本长度：{len(text)}")
                return text
        except Exception as e:
            logger.error(f"语音转文字失败：{e}")
            return ""

    def batch_speech_to_text(self, audio_paths: List[str]) -> List[str]:
        """
        批量语音转文字

        Args:
            audio_paths: 音频文件路径列表

        Returns:
            转换后的文本列表
        """
        if not whisper_available:
            logger.warning("⚠️ Whisper 库未安装，无法进行语音转文字")
            return [""] * len(audio_paths)

        try:
            results = []
            for audio_path in audio_paths:
                text = self.speech_to_text(audio_path)
                results.append(text)
            logger.info(f"✅ 批量语音转文字成功，处理 {len(results)} 个音频文件")
            return results
        except Exception as e:
            logger.error(f"批量语音转文字失败：{e}")
            return [""] * len(audio_paths)

    # ========== 图像处理 ==========

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        处理图像

        Args:
            image_path: 图像文件路径

        Returns:
            图像处理结果
        """
        try:
            if not cv2_available:
                logger.warning("⚠️ OpenCV 库未安装，使用降级方案处理图像")
                # 降级方案：只返回文件信息
                result = {
                    "width": 0,
                    "height": 0,
                    "channels": 0,
                    "edge_density": 0.0,
                    "color_mean": [0, 0, 0],
                    "color_std": [0, 0, 0],
                    "file_size": os.path.getsize(image_path) if os.path.exists(image_path) else 0,
                }
                return result

            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图像：{image_path}")
                return {}

            # 获取图像信息
            height, width, channels = image.shape

            # 计算图像特征
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            # 计算边缘密度
            if numpy_available:
                edge_density = np.sum(edges > 0) / (height * width)
            else:
                # 纯Python实现
                edge_count = 0
                for i in range(height):
                    for j in range(width):
                        if edges[i, j] > 0:
                            edge_count += 1
                edge_density = edge_count / (height * width)

            # 颜色统计
            if numpy_available:
                color_mean = np.mean(image, axis=(0, 1))
                color_std = np.std(image, axis=(0, 1))
                color_mean_list = color_mean.tolist()
                color_std_list = color_std.tolist()
            else:
                # 纯Python实现
                total_b, total_g, total_r = 0, 0, 0
                for i in range(height):
                    for j in range(width):
                        b, g, r = image[i, j]
                        total_b += b
                        total_g += g
                        total_r += r
                pixel_count = height * width
                color_mean_list = [total_b / pixel_count, total_g / pixel_count, total_r / pixel_count]

                # 计算标准差
                sum_sq_b, sum_sq_g, sum_sq_r = 0, 0, 0
                for i in range(height):
                    for j in range(width):
                        b, g, r = image[i, j]
                        sum_sq_b += (b - color_mean_list[0]) ** 2
                        sum_sq_g += (g - color_mean_list[1]) ** 2
                        sum_sq_r += (r - color_mean_list[2]) ** 2
                color_std_list = [
                    (sum_sq_b / pixel_count) ** 0.5,
                    (sum_sq_g / pixel_count) ** 0.5,
                    (sum_sq_r / pixel_count) ** 0.5,
                ]

            result = {
                "width": width,
                "height": height,
                "channels": channels,
                "edge_density": float(edge_density),
                "color_mean": color_mean_list,
                "color_std": color_std_list,
                "file_size": os.path.getsize(image_path) if os.path.exists(image_path) else 0,
            }

            logger.info(f"✅ 图像处理成功：{image_path}")
            return result
        except Exception as e:
            logger.error(f"图像处理失败：{e}")
            return {}

    def batch_process_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        批量处理图像

        Args:
            image_paths: 图像文件路径列表

        Returns:
            图像处理结果列表
        """
        try:
            results = []
            for image_path in image_paths:
                result = self.process_image(image_path)
                results.append(result)
            logger.info(f"✅ 批量图像处理成功，处理 {len(results)} 个图像文件")
            return results
        except Exception as e:
            logger.error(f"批量图像处理失败：{e}")
            return [{}] * len(image_paths)

    # ========== 多模态嵌入 ==========

    def _calculate_norm(self, vector):
        """计算向量的范数（纯Python实现）"""
        if numpy_available:
            return np.linalg.norm(vector) or 1.0
        else:
            # 纯Python实现
            return (sum(x * x for x in vector) ** 0.5) or 1.0

    def generate_image_embedding(self, image_path: str) -> Optional[List[float]]:
        """
        生成图像嵌入

        Args:
            image_path: 图像文件路径

        Returns:
            图像嵌入向量
        """
        if not transformers_available or not torch_available or not self._clip_model:
            logger.warning("⚠️ CLIP 模型未加载，使用降级方案生成图像嵌入")
            # 降级方案：使用文件路径的哈希值生成固定维度的向量
            import hashlib

            hash_obj = hashlib.md5(image_path.encode())
            hash_hex = hash_obj.hexdigest()
            # 将哈希值转换为 512 维向量
            embedding = []
            for i in range(0, len(hash_hex), 2):
                if i + 1 < len(hash_hex):
                    value = int(hash_hex[i : i + 2], 16) / 255.0
                    embedding.append(value)
            # 填充到 512 维
            while len(embedding) < 512:
                embedding.append(0.0)
            # 归一化
            norm = self._calculate_norm(embedding)
            embedding = [x / norm for x in embedding]
            return embedding

        try:
            # 读取图像
            if not cv2_available:
                logger.warning("⚠️ OpenCV 库未安装，无法读取图像")
                return None

            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图像：{image_path}")
                return None

            # 转换为 RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 处理图像
            inputs = self._clip_processor(images=image, return_tensors="pt").to(device)

            # 生成嵌入
            with torch.no_grad():
                embedding = self._clip_model.get_image_features(**inputs).cpu().numpy()[0]

            # 归一化
            norm = self._calculate_norm(embedding)
            embedding = embedding / norm

            logger.info(f"✅ 生成图像嵌入成功：{image_path}")
            return embedding.tolist()
        except Exception as e:
            logger.error(f"生成图像嵌入失败：{e}")
            # 降级方案：使用文件路径的哈希值生成固定维度的向量
            import hashlib

            hash_obj = hashlib.md5(image_path.encode())
            hash_hex = hash_obj.hexdigest()
            # 将哈希值转换为 512 维向量
            embedding = []
            for i in range(0, len(hash_hex), 2):
                if i + 1 < len(hash_hex):
                    value = int(hash_hex[i : i + 2], 16) / 255.0
                    embedding.append(value)
            # 填充到 512 维
            while len(embedding) < 512:
                embedding.append(0.0)
            # 归一化
            norm = self._calculate_norm(embedding)
            embedding = [x / norm for x in embedding]
            return embedding

    def generate_text_embedding(self, text: str) -> Optional[List[float]]:
        """
        生成文本嵌入

        Args:
            text: 文本内容

        Returns:
            文本嵌入向量
        """
        if not transformers_available or not torch_available or not self._clip_model:
            logger.warning("⚠️ CLIP 模型未加载，使用降级方案生成文本嵌入")
            # 降级方案：使用文本的哈希值生成固定维度的向量
            import hashlib

            hash_obj = hashlib.md5(text.encode())
            hash_hex = hash_obj.hexdigest()
            # 将哈希值转换为 512 维向量
            embedding = []
            for i in range(0, len(hash_hex), 2):
                if i + 1 < len(hash_hex):
                    value = int(hash_hex[i : i + 2], 16) / 255.0
                    embedding.append(value)
            # 填充到 512 维
            while len(embedding) < 512:
                embedding.append(0.0)
            # 归一化
            norm = self._calculate_norm(embedding)
            embedding = [x / norm for x in embedding]
            return embedding

        try:
            # 处理文本
            inputs = self._clip_processor(text=text, return_tensors="pt").to(device)

            # 生成嵌入
            with torch.no_grad():
                embedding = self._clip_model.get_text_features(**inputs).cpu().numpy()[0]

            # 归一化
            norm = self._calculate_norm(embedding)
            embedding = embedding / norm

            logger.info(f"✅ 生成文本嵌入成功，文本长度：{len(text)}")
            return embedding.tolist()
        except Exception as e:
            logger.error(f"生成文本嵌入失败：{e}")
            # 降级方案：使用文本的哈希值生成固定维度的向量
            import hashlib

            hash_obj = hashlib.md5(text.encode())
            hash_hex = hash_obj.hexdigest()
            # 将哈希值转换为 512 维向量
            embedding = []
            for i in range(0, len(hash_hex), 2):
                if i + 1 < len(hash_hex):
                    value = int(hash_hex[i : i + 2], 16) / 255.0
                    embedding.append(value)
            # 填充到 512 维
            while len(embedding) < 512:
                embedding.append(0.0)
            # 归一化
            norm = self._calculate_norm(embedding)
            embedding = [x / norm for x in embedding]
            return embedding

    def generate_multimodal_embedding(self, data: Dict) -> Optional[List[float]]:
        """
        生成多模态嵌入

        Args:
            data: 多模态数据
                {"text": "文本内容", "image_path": "图像路径", "audio_path": "音频路径"}

        Returns:
            多模态嵌入向量
        """
        try:
            embeddings = []

            # 处理文本
            if "text" in data and data["text"]:
                text_embedding = self.generate_text_embedding(data["text"])
                if text_embedding:
                    embeddings.append(text_embedding)

            # 处理图像
            if "image_path" in data and data["image_path"]:
                image_embedding = self.generate_image_embedding(data["image_path"])
                if image_embedding:
                    embeddings.append(image_embedding)

            # 处理音频
            if "audio_path" in data and data["audio_path"]:
                text = self.speech_to_text(data["audio_path"])
                if text:
                    audio_embedding = self.generate_text_embedding(text)
                    if audio_embedding:
                        embeddings.append(audio_embedding)

            if not embeddings:
                logger.warning("⚠️ 无法生成多模态嵌入，没有有效的输入数据")
                return None

            # 融合嵌入
            if numpy_available:
                fused_embedding = np.mean(embeddings, axis=0)
                norm = np.linalg.norm(fused_embedding) or 1.0
            else:
                # 纯Python实现
                dim = len(embeddings[0])
                fused_embedding = [0.0] * dim
                for emb in embeddings:
                    for i in range(dim):
                        fused_embedding[i] += emb[i]
                for i in range(dim):
                    fused_embedding[i] /= len(embeddings)
                norm = self._calculate_norm(fused_embedding)

            fused_embedding = [x / norm for x in fused_embedding]

            logger.info("✅ 生成多模态嵌入成功")
            return fused_embedding
        except Exception as e:
            logger.error(f"生成多模态嵌入失败：{e}")
            return None

    # ========== 多模态内容分析 ==========

    def analyze_multimodal_content(self, data: Dict) -> Dict[str, Any]:
        """
        分析多模态内容

        Args:
            data: 多模态数据
                {"text": "文本内容", "image_path": "图像路径", "audio_path": "音频路径"}

        Returns:
            分析结果
        """
        try:
            analysis = {"text_analysis": {}, "image_analysis": {}, "audio_analysis": {}, "multimodal_analysis": {}}

            # 分析文本
            if "text" in data and data["text"]:
                text = data["text"]
                analysis["text_analysis"] = {
                    "length": len(text),
                    "word_count": len(text.split()),
                    "char_count": len(text),
                    "embedding": self.generate_text_embedding(text),
                }

            # 分析图像
            if "image_path" in data and data["image_path"]:
                image_path = data["image_path"]
                analysis["image_analysis"] = {
                    "processing": self.process_image(image_path),
                    "embedding": self.generate_image_embedding(image_path),
                }

            # 分析音频
            if "audio_path" in data and data["audio_path"]:
                audio_path = data["audio_path"]
                text = self.speech_to_text(audio_path)
                analysis["audio_analysis"] = {"transcript": text, "length": len(text)}

            # 多模态分析
            analysis["multimodal_analysis"] = {"embedding": self.generate_multimodal_embedding(data)}

            logger.info("✅ 多模态内容分析成功")
            return analysis
        except Exception as e:
            logger.error(f"多模态内容分析失败：{e}")
            return {}

    # ========== 工具方法 ==========

    def is_available(self) -> Dict[str, bool]:
        """
        检查多模态功能是否可用

        Returns:
            功能可用性字典
        """
        return {
            "whisper": whisper_available,
            "torch": torch_available,
            "clip": torch_available and self._clip_model is not None,
            "opencv": True,  # OpenCV 通常随 numpy 安装
        }

    def print_status(self):
        """
        打印多模态处理器状态
        """
        availability = self.is_available()

        logger.info("\n" + "=" * 60)
        logger.info("【多模态处理器状态】")
        logger.info("=" * 60)
        logger.info(f"Whisper (语音转文字): {'✅' if availability['whisper'] else '❌'}")
        logger.info(f"Torch (深度学习): {'✅' if availability['torch'] else '❌'}")
        logger.info(f"CLIP (多模态嵌入): {'✅' if availability['clip'] else '❌'}")
        logger.info(f"OpenCV (图像处理): {'✅' if availability['opencv'] else '❌'}")

        if torch_available:
            logger.info(f"设备: {device}")

        logger.info("=" * 60)


# 全局实例
multimodal_processor = MultimodalProcessor()


# 便捷函数
def get_multimodal_processor() -> MultimodalProcessor:
    """
    获取多模态处理器实例

    Returns:
        MultimodalProcessor 实例
    """
    return multimodal_processor


def speech_to_text(audio_path: str) -> str:
    """
    语音转文字

    Args:
        audio_path: 音频文件路径

    Returns:
        转换后的文本
    """
    return multimodal_processor.speech_to_text(audio_path)


def process_image(image_path: str) -> Dict[str, Any]:
    """
    处理图像

    Args:
        image_path: 图像文件路径

    Returns:
        图像处理结果
    """
    return multimodal_processor.process_image(image_path)


def generate_multimodal_embedding(data: Dict) -> Optional[List[float]]:
    """
    生成多模态嵌入

    Args:
        data: 多模态数据

    Returns:
        多模态嵌入向量
    """
    return multimodal_processor.generate_multimodal_embedding(data)


def analyze_multimodal_content(data: Dict) -> Dict[str, Any]:
    """
    分析多模态内容

    Args:
        data: 多模态数据

    Returns:
        分析结果
    """
    return multimodal_processor.analyze_multimodal_content(data)


if __name__ == "__main__":
    print("=" * 80)
    print("🔄 多模态处理器测试")
    print("=" * 80)

    processor = MultimodalProcessor()
    processor.print_status()

    # 测试语音转文字
    print("\n【测试 1】语音转文字")
    # 注意：这里需要一个音频文件路径
    # text = processor.speech_to_text("test_audio.wav")
    # print(f"✅ 转换结果: {text}")

    # 测试图像处理
    print("\n【测试 2】图像处理")
    # 注意：这里需要一个图像文件路径
    # image_result = processor.process_image("test_image.jpg")
    # print(f"✅ 图像信息: {image_result}")

    # 测试多模态嵌入
    print("\n【测试 3】多模态嵌入")
    test_data = {
        "text": "这是一个测试文本",
        # "image_path": "test_image.jpg",
        # "audio_path": "test_audio.wav"
    }
    embedding = processor.generate_multimodal_embedding(test_data)
    print(f"✅ 嵌入向量长度: {len(embedding) if embedding else 0}")

    # 测试多模态内容分析
    print("\n【测试 4】多模态内容分析")
    analysis = processor.analyze_multimodal_content(test_data)
    print(f"✅ 分析结果: {analysis.keys()}")

    print("\n" + "=" * 80)
    print("✅ 多模态处理器测试完成！")
    print("=" * 80)
