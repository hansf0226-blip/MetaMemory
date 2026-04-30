import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List

import requests

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# 配置获取（优先 yaml，fallback 硬编码默认值）
from core.config import get_config as _get_config
YIJING_ENCODE_MODE = _get_config("yijing.encode_mode", "rule")
LLM_API_URL = _get_config("llm.api_url", "http://127.0.0.1:11434/api/generate")
LLM_TIMEOUT = _get_config("llm.timeout", 5)
LOCAL_MODEL_PATH = _get_config("llm.local_model_path", "./models/qwen-0.5b-yijing")
LLM_MODEL_NAME = _get_config("llm.model_name", "yijing-encode")

# 线程锁
_lock = threading.RLock()

# LLM 调用缓存
llm_cache = {}  # 格式: {cache_key: (result, expiry_time)}
CACHE_MAX_SIZE = 1000  # 缓存最大大小
CACHE_EXPIRY_TIME = 3600  # 缓存过期时间（秒），默认1小时
CACHE_CLEANUP_INTERVAL = 300  # 缓存清理间隔（秒），默认5分钟


# 启动缓存清理线程
def start_cache_cleanup_thread():
    """启动缓存清理线程"""

    def cleanup_task():
        while True:
            try:
                with _lock:
                    current_time = datetime.now().timestamp()
                    expired_keys = [key for key, (_, expiry) in llm_cache.items() if current_time > expiry]
                    if expired_keys:
                        for key in expired_keys:
                            del llm_cache[key]
                        logger.info(f"✅ 缓存清理完成，删除了 {len(expired_keys)} 个过期项")
            except Exception as e:
                logger.error(f"缓存清理线程错误：{str(e)}")
            time.sleep(CACHE_CLEANUP_INTERVAL)

    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()
    logger.info("✅ 缓存清理线程已启动")


# 启动缓存清理线程
start_cache_cleanup_thread()

# 加载 64 卦库
_data_dir = Path(__file__).parent.parent / "data"
_gua_path = _data_dir / "64_gua_full.json"
if not _gua_path.exists():
    # Fallback: try relative to cwd
    _gua_path = Path("data/64_gua_full.json")
with open(_gua_path, "r", encoding="utf-8") as f:
    GUA_LIB = json.load(f)

# 创建卦象哈希表，加速匹配
gua_hex_map = {}
for gua in GUA_LIB:
    hex_tuple = tuple(gua["hex"])
    gua_hex_map[hex_tuple] = gua

# 全局枚举强校验
BAGUA_ALL = {"qian", "kun", "zhen", "xun", "kan", "li", "gen", "dui"}
WUXING_ALL = {"jin", "tu", "mu", "huo", "shui"}
SANCAI_ALL = {"tien", "ren", "di"}


def gua_data_check(hex_arr: List[int], bagua: str, wuxing: str, layer: str) -> bool:
    """
    六爻/八卦/五行/三才强校验

    确保卦象数据符合易经规范，防止脏数据写入

    Args:
        hex_arr: 六爻数组，6 位 0/1 数组（0=阴，1=阳）
        bagua: 八卦类型（qian/kun/zhen/xun/kan/li/gen/dui）
        wuxing: 五行属性（jin/tu/mu/huo/shui）
        layer: 三才层级（tien/ren/di）

    Returns:
        bool: 校验通过返回 True

    Raises:
        ValueError: 数据格式错误时抛出
    """
    if len(hex_arr) != 6:
        raise ValueError("六爻必须为 6 位 0/1 数组")
    if not all(x in [0, 1] for x in hex_arr):
        raise ValueError("爻位只能是 0 或 1")
    if bagua not in BAGUA_ALL:
        raise ValueError(f"八卦编码非法：{bagua}")
    if wuxing not in WUXING_ALL:
        raise ValueError(f"五行编码非法：{wuxing}")
    if layer not in SANCAI_ALL:
        raise ValueError(f"三才层级非法：{layer}")
    return True


def content_to_hexagram_rule(content: str, memory_type: str) -> tuple:
    """
    优化版兜底规则编码

    Args:
        content: 记忆内容
        memory_type: 记忆类型

    Returns:
        tuple: (hex_arr, bagua, wuxing, layer, is_fallback)
            - hex_arr: 六爻数组 [0,1,0,1,1,0]
            - bagua: 八卦类型
            - wuxing: 五行属性
            - layer: 三才层级
            - is_fallback: 是否使用了降级方案
    """
    with _lock:
        try:
            content_str = str(content)
            content_len = len(content_str)

            # 1. 文本长度维度（更细粒度）
            if content_len < 10:
                c1 = 0  # 短文本
            elif content_len < 50:
                c1 = 1  # 中等长度
            else:
                c1 = 2  # 长文本

            # 2. 时效性维度
            time_keywords = ["实时", "当前", "现在", "今天", "刚刚", "最新"]
            c2 = 1 if any(keyword in content_str for keyword in time_keywords) else 0

            # 3. 记忆类型维度
            if memory_type in ["chat", "task"]:
                c3 = 1
            elif memory_type in ["knowledge", "rule"]:
                c3 = 2
            else:
                c3 = 0

            # 4. 目的性维度
            purpose_keywords = ["需求", "目标", "想要", "需要", "希望", "计划"]
            c4 = 1 if any(keyword in content_str for keyword in purpose_keywords) else 0

            # 5. 知识性维度
            knowledge_keywords = ["规则", "常识", "知识", "原理", "方法", "技巧"]
            c5 = 1 if any(keyword in content_str for keyword in knowledge_keywords) else 0

            # 6. 持久性维度
            permanent_keywords = ["永久", "底线", "原则", "核心", "重要", "关键"]
            c6 = 1 if any(keyword in content_str for keyword in permanent_keywords) else 0

            # 将多值特征映射到二进制
            hex_arr = [
                1 if c1 >= 1 else 0,  # 长度 >= 10
                c2,  # 时效性
                1 if c3 >= 1 else 0,  # 记忆类型
                c4,  # 目的性
                c5,  # 知识性
                c6,  # 持久性
            ]

            # 生成辅助特征用于增强区分度
            # 基于内容特征生成更丰富的编码
            if "问题" in content_str or "？" in content_str:
                hex_arr[3] = 1  # 增强目的性
            if "答案" in content_str or "解决方案" in content_str:
                hex_arr[4] = 1  # 增强知识性
            if "紧急" in content_str or "重要" in content_str:
                hex_arr[5] = 1  # 增强持久性

            gua_info = get_gua_by_hex(hex_arr)
            bagua = gua_info.get("bagua_type", "xun")
            wuxing = gua_info.get("wuxing", "mu")
            layer = gua_info.get("sancai_layer", "di")

            return hex_arr, bagua, wuxing, layer, False
        except Exception as e:
            logger.error(f"兜底规则编码失败：{str(e)}")
            # 返回默认值，并标记为降级
            return [0, 1, 0, 1, 1, 0], "kun", "tu", "ren", True


def content_to_hexagram_llm(content: str, memory_type: str) -> tuple:
    """
    优化版 LLM 精准取象

    使用大语言模型进行语义理解，将内容转换为卦象编码
    支持 Ollama/Minimax/其他兼容 API

    Args:
        content: 记忆内容
        memory_type: 记忆类型（chat/task/temp/knowledge 等）

    Returns:
        tuple: (hex_arr, bagua, wuxing, layer, is_fallback)
            - hex_arr: 六爻数组 [0,1,0,1,1,0]
            - bagua: 八卦类型
            - wuxing: 五行属性
            - layer: 三才层级
            - is_fallback: 是否使用了降级方案

    Note:
        如果 LLM 调用失败，自动降级到规则编码
    """
    with _lock:
        # 生成缓存键
        cache_key = f"{content[:200]}_{memory_type}"

        # 清理过期缓存
        current_time = datetime.now().timestamp()
        expired_keys = [key for key, (_, expiry) in llm_cache.items() if current_time > expiry]
        for key in expired_keys:
            del llm_cache[key]

        # 检查缓存
        if cache_key in llm_cache:
            result, expiry = llm_cache[cache_key]
            # 检查是否过期
            if current_time <= expiry:
                logger.info("使用 LLM 缓存结果")
                # 确保返回值包含 is_fallback
                if len(result) == 4:
                    return result + (False,)
                return result
            else:
                # 缓存过期，删除
                del llm_cache[cache_key]

        # 优化的 prompt 模板
        prompt = """【角色】你是专业的易经取象编码引擎，负责将内容语义准确映射到易经卦象系统。

【任务】
基于输入内容的语义特征，生成对应的易经编码，包括：
1. 六爻数组（6位二进制，0=阴，1=阳）
2. 八卦类型（qian/kun/zhen/xun/kan/li/gen/dui）
3. 五行属性（jin/tu/mu/huo/shui）
4. 三才层级（tien/ren/di）

【语义分类指南】
- 天层（tien）：规则、常识、永久原则、核心价值观
- 人层（ren）：交互、任务、需求、对话、情感
- 地层（di）：短期、临时、碎片化、具体事件

【六爻编码规则】
- 初爻（第1位）：内容长度（0=短，1=长）
- 二爻（第2位）：时效性（0=非实时，1=实时）
- 三爻（第3位）：重要性（0=普通，1=重要）
- 四爻（第4位）：目的性（0=无明确目的，1=有明确目的）
- 五爻（第5位）：知识性（0=非知识，1=知识）
- 上爻（第6位）：持久性（0=临时，1=永久）

【输入】
content: {content}
memory_type: {memory_type}

【输出格式】
严格按照以下 JSON 格式输出，不要添加任何额外内容：
{{
  "hexagram": [0, 1, 0, 1, 1, 0],
  "bagua_type": "",
  "wuxing": "",
  "sancai_layer": ""
}}

【示例】
输入：
content: "用户需要一个能够自动生成报表的功能"
memory_type: "task"

输出：
{{
  "hexagram": [1, 0, 1, 1, 0, 0],
  "bagua_type": "xun",
  "wuxing": "mu",
  "sancai_layer": "ren"
}}"""

        try:
            # 检测是否为 Minimax API
            if "minimax" in LLM_API_URL.lower():
                # Minimax API 格式
                res = requests.post(
                    LLM_API_URL,
                    headers={
                        "Authorization": f"Bearer {os.getenv('MINIMAX_API_KEY', '')}",
                        "Content-Type": "application/json",
                    },
                    json={"model": LLM_MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "stream": False},
                    timeout=LLM_TIMEOUT,
                )
                resp_data = res.json()
                resp_text = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            else:
                # Ollama API 格式
                res = requests.post(
                    LLM_API_URL, json={"model": LLM_MODEL_NAME, "prompt": prompt, "stream": False}, timeout=LLM_TIMEOUT
                )
                resp_text = res.json().get("response", "").strip()

            # 容错处理：提取 JSON 部分
            import re

            json_match = re.search(r"\{[\s\S]*\}", resp_text)
            if json_match:
                resp_text = json_match.group(0)

            data = json.loads(resp_text)

            # 提取数据
            hex_arr = data.get("hexagram", [0, 0, 0, 0, 0, 0])
            bagua = data.get("bagua_type", "xun")
            wuxing = data.get("wuxing", "mu")
            layer = data.get("sancai_layer", "ren")

            # 数据校验和修复
            try:
                # 验证六爻数组
                if not isinstance(hex_arr, list) or len(hex_arr) != 6:
                    hex_arr = [0, 0, 0, 0, 0, 0]
                # 确保所有值都是 0 或 1
                hex_arr = [1 if x else 0 for x in hex_arr]

                # 验证八卦类型
                if bagua not in BAGUA_ALL:
                    bagua = "xun"

                # 验证五行属性
                if wuxing not in WUXING_ALL:
                    wuxing = "mu"

                # 验证三才层级
                if layer not in SANCAI_ALL:
                    layer = "ren"

                # 最终校验
                gua_data_check(hex_arr, bagua, wuxing, layer)
            except Exception as e:
                logger.warning(f"LLM 结果校验失败，使用默认值：{str(e)}")
                # 使用默认值
                hex_arr = [0, 0, 0, 0, 0, 0]
                bagua = "xun"
                wuxing = "mu"
                layer = "ren"

            # 缓存结果，带过期时间
            result = (hex_arr, bagua, wuxing, layer, False)
            expiry_time = datetime.now().timestamp() + CACHE_EXPIRY_TIME
            llm_cache[cache_key] = (result, expiry_time)

            # 清理缓存，保持大小限制
            if len(llm_cache) > CACHE_MAX_SIZE:
                # 删除最旧的缓存项
                oldest_key = min(llm_cache.items(), key=lambda x: x[1][1])[0]
                del llm_cache[oldest_key]

            return result

        except Exception as e:
            logger.error(f"LLM 取象失败：{str(e)}")
            # 降级到规则编码，并标记为降级
            rule_result = content_to_hexagram_rule(content, memory_type)
            if len(rule_result) == 4:
                return rule_result + (True,)
            return rule_result


def content_to_hexagram(content: str, memory_type: str) -> tuple:
    """
    统一入口 - 集成智能层级分类

    Args:
        content: 记忆内容
        memory_type: 记忆类型

    Returns:
        tuple: (hex_arr, bagua, wuxing, layer, is_fallback)
            - hex_arr: 六爻数组 [0,1,0,1,1,0]
            - bagua: 八卦类型
            - wuxing: 五行属性
            - layer: 三才层级
            - is_fallback: 是否使用了降级方案
    """
    with _lock:
        try:
            # 使用智能分类器判断层级
            from core.memory_layer_classifier import classify_and_encode

            result = classify_and_encode(content, memory_type)
            # 确保返回值包含 is_fallback
            if len(result) == 4:
                return result + (False,)
            return result
        except Exception as e:
            logger.error(f"智能分类编码失败：{str(e)}")
            # 降级到兜底规则编码，并标记为降级
            rule_result = content_to_hexagram_rule(content, memory_type)
            if len(rule_result) == 4:
                return rule_result + (True,)
            return rule_result


def get_gua_by_hex(hex_list: List[int]) -> Dict[str, Any]:
    """卦匹配"""
    with _lock:
        hex_tuple = tuple(hex_list)
        return gua_hex_map.get(hex_tuple, {})


def hexagram_to_vector(hex_list: List[int]) -> List[float]:
    """向量转换 - 生成语义向量"""
    with _lock:
        # 基础转换：将六爻转换为6维向量
        basic_vector = [float(x) for x in hex_list]

        # 扩展到768维，使其与向量存储系统兼容
        # 方法1：重复基础向量
        extended_vector = basic_vector * (768 // 6)

        # 方法2：添加一些基于卦象的特征
        # 计算卦象的一些统计特征
        sum_hex = sum(hex_list)
        avg_hex = sum_hex / 6.0
        max_hex = max(hex_list)
        min_hex = min(hex_list)

        # 将特征添加到向量中
        features = [sum_hex, avg_hex, max_hex, min_hex, sum_hex % 2, avg_hex * 2]
        for i, feature in enumerate(features):
            if i < len(extended_vector):
                extended_vector[i] = feature

        # 确保向量长度为768
        while len(extended_vector) < 768:
            extended_vector.append(0.0)

        return extended_vector[:768]


def calculate_similarity(hex1: List[int], hex2: List[int]) -> float:
    """卦象相似度计算"""
    with _lock:
        if len(hex1) != 6 or len(hex2) != 6:
            return 0.0
        matches = sum(1 for i in range(6) if hex1[i] == hex2[i])
        return matches / 6.0


def generate_semantic_embedding(content: str, hex_list: List[int] = None) -> List[float]:
    """
    生成语义嵌入向量（Phase 3B 纯语义）

    使用 embedding_provider 生成真正的语义向量（sentence-transformer / OpenAI），
    fallback 到 hash 嵌入。六爻特征不再融合到检索向量中。

    Args:
        content: 记忆内容
        hex_list: 六爻数组（已弃用，保留参数兼容）

    Returns:
        List[float]: 语义嵌入向量（384-dim 或 768-dim）
    """
    with _lock:
        try:
            from core.embedding_provider import get_embedding_provider

            provider = get_embedding_provider()
            embedding = provider.encode(content)

            # Phase 3B: 六爻特征不再融合到检索向量
            # 六爻仅作为分类标签（bagua_type/sancai_layer/wuxing），
            # 不参与检索评分。接缝处无行为变更。

            return embedding
        except Exception as e:
            logger.error(f"生成语义嵌入失败：{str(e)}")
            # fallback 到 hexagram_to_vector
            if hex_list:
                return hexagram_to_vector(hex_list)
            # 最终 fallback
            import hashlib
            hash_bytes = hashlib.sha256(content.encode()).digest()
            embedding = []
            for i in range(768):
                byte_idx = i % len(hash_bytes)
                value = (hash_bytes[byte_idx] - 128) / 128.0
                embedding.append(value)
            return embedding


def wuxing_life_schedule(mem_dict: dict, create_time: datetime) -> dict:
    """
    五行调度 - 能量流转

    根据五行相生规则（木→火→土→金→水）自动流转能量，
    同时热度分数随时间衰减

    Args:
        mem_dict: 记忆数据字典 {"wuxing": "mu", "hot_score": 0.8}
        create_time: 创建时间

    Returns:
        dict: 更新后的五行和热度分数
            - wuxing: 新的五行属性
            - hot_score: 衰减后的热度分数
    """
    with _lock:
        wuxing = mem_dict.get("wuxing", "mu")
        hot_score = mem_dict.get("hot_score", 0.0)

        # 相生流转：木生火，火生土，土生金，金生水，水生木
        wuxing_cycle = {"mu": "huo", "huo": "tu", "tu": "jin", "jin": "shui", "shui": "mu"}
        new_wuxing = wuxing_cycle.get(wuxing, wuxing)

        # 热度衰减（每日 -0.01）
        new_hot_score = max(0.0, hot_score - 0.01)

        return {"wuxing": new_wuxing, "hot_score": round(new_hot_score, 2)}


def position_check_auto(
    bagua_type: str, sancai_layer: str, user_intent: bool = False, memory_id: str = None
) -> Dict[str, Any]:
    """
    自愈校验 - 时位检查

    根据易经"当位"理论，检查八卦与三才层级是否匹配。
    不匹配时提供修正建议，同时尊重用户意图。

    八卦时位规则：
    - 乾/坤 → 天层（统领全局）
    - 震/巽/坎/离 → 人层（人事活动）
    - 艮/兑 → 地层（基础稳固）

    Args:
        bagua_type: 八卦类型
        sancai_layer: 当前三才层级
        user_intent: 用户是否明确指定了层级（默认 False）
        memory_id: 记忆ID，用于日志记录（可选）

    Returns:
        dict: 校验结果
            - status: "normal" (正常) 或 "suggest" (建议修正) 或 "fix" (需修正)
            - standard_layer: 标准层级（仅 status 不为 "normal" 时返回）
            - confidence: 修正建议的置信度（0-1）
            - reason: 修正原因
    """
    with _lock:
        # 八卦与时位对应关系
        position_map = {
            "qian": "tien",
            "kun": "tien",  # 乾坤 → 天
            "zhen": "ren",
            "xun": "ren",  # 震巽 → 人
            "kan": "ren",
            "li": "ren",  # 坎离 → 人
            "gen": "di",
            "dui": "di",  # 艮兑 → 地
        }

        expected_layer = position_map.get(bagua_type, "ren")

        if sancai_layer == expected_layer:
            # 匹配正常
            logger.info(f"✅ 时位校验正常：八卦 {bagua_type} 对应层级 {sancai_layer}")
            return {"status": "normal", "confidence": 1.0, "reason": "八卦与层级匹配正常"}
        else:
            # 计算置信度
            # 基础置信度
            base_confidence = 0.9

            # 如果用户明确指定了层级，降低置信度
            if user_intent:
                confidence = max(0.5, base_confidence - 0.3)
                status = "suggest"
                reason = f"用户指定了层级 {sancai_layer}，但根据八卦 {bagua_type} 建议使用层级 {expected_layer}"
            else:
                confidence = base_confidence
                status = "fix"
                reason = f"八卦 {bagua_type} 建议使用层级 {expected_layer}，当前层级 {sancai_layer} 不匹配"

            # 记录日志
            memory_info = f"(记忆ID: {memory_id})" if memory_id else ""
            logger.warning(
                f"⚠️  时位校验建议修正 {memory_info}：八卦 {bagua_type} 当前层级 {sancai_layer}，建议层级 {expected_layer}，置信度: {confidence:.2f}"
            )

            return {
                "status": status,
                "standard_layer": expected_layer,
                "confidence": confidence,
                "reason": reason,
                "original_layer": sancai_layer,
                "bagua_type": bagua_type,
            }


def apply_position_fix(memory_data: Dict, user_intent: bool = False) -> Dict:
    """
    应用时位校验结果

    根据时位校验的结果，智能处理层级不匹配的情况，同时尊重用户意图。

    Args:
        memory_data: 记忆数据字典，包含 bagua_type 和 sancai_layer
        user_intent: 用户是否明确指定了层级（默认 False）

    Returns:
        dict: 更新后的记忆数据字典，包含修正信息
    """
    with _lock:
        bagua_type = memory_data.get("bagua_type")
        sancai_layer = memory_data.get("sancai_layer")
        memory_id = memory_data.get("id")

        if not bagua_type or not sancai_layer:
            logger.warning("⚠️  时位校验缺少必要数据：bagua_type 或 sancai_layer")
            return memory_data

        # 执行时位校验
        check_result = position_check_auto(bagua_type, sancai_layer, user_intent, memory_id)

        # 根据校验结果处理
        if check_result["status"] == "normal":
            # 无需修正
            memory_data["position_check"] = check_result
            return memory_data
        elif check_result["status"] == "suggest":
            # 建议修正，但尊重用户意图
            logger.info(f"ℹ️  时位校验建议修正，但尊重用户意图，保持当前层级 {sancai_layer}")
            memory_data["position_check"] = check_result
            memory_data["position_fixed"] = False
            return memory_data
        elif check_result["status"] == "fix":
            # 需要修正，自动修正层级
            old_layer = sancai_layer
            new_layer = check_result["standard_layer"]

            # 更新记忆数据
            memory_data["sancai_layer"] = new_layer
            memory_data["position_check"] = check_result
            memory_data["position_fixed"] = True
            memory_data["position_fixed_at"] = datetime.now().isoformat()
            memory_data["original_sancai_layer"] = old_layer

            # 记录修正日志
            memory_info = f"(记忆ID: {memory_id})" if memory_id else ""
            logger.info(f"🔧  时位校验自动修正 {memory_info}：八卦 {bagua_type} 层级从 {old_layer} 修正为 {new_layer}")

            return memory_data
        else:
            # 其他情况，保持不变
            memory_data["position_check"] = check_result
            return memory_data


def yao_bian_update(hex_arr: List[int], position: int) -> List[int]:
    """爻变更新"""
    with _lock:
        if position < 0 or position >= 6:
            raise ValueError("爻位必须在 0-5 范围内")

        new_hex = hex_arr.copy()
        new_hex[position] = 1 - new_hex[position]
        return new_hex


def track_yao_bian_history(hex_arr: List[int], position: int, history: List) -> List:
    """爻变历史追踪"""
    with _lock:
        new_hex = yao_bian_update(hex_arr, position)
        history.append(
            {"timestamp": datetime.now().isoformat(), "before": hex_arr.copy(), "after": new_hex, "position": position}
        )
        return new_hex
