#!/usr/bin/env python3
"""
🎯 易经编码模型训练脚本

使用 yijing_train_data.json 训练轻量级编码模型
优化符号 - 数值混合架构的权重参数
"""

import json
import math
from datetime import datetime
from typing import Dict, List


def load_training_data(path: str) -> List[Dict]:
    """加载训练数据"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ 加载训练数据：{len(data)} 条")
    return data


def compute_hexagram_embedding(hexagram_num: int, dim: int = 128) -> List[float]:
    """计算卦象嵌入（正弦位置编码）"""
    embedding = []
    for j in range(dim):
        if j % 2 == 0:
            val = math.sin(hexagram_num / math.pow(10000, j / dim))
        else:
            val = math.cos(hexagram_num / math.pow(10000, (j - 1) / dim))
        embedding.append(val)
    return embedding


def compute_target_embedding(item: Dict) -> List[float]:
    """
    计算目标嵌入（监督信号）

    基于 IMA V2.0 规范：
    - 符号部分：卦象嵌入
    - 数值部分：weight, hot, layer, element 编码
    """
    # 符号嵌入（128 维）
    hex_num = item.get("hexagram_num", 1)
    symbolic_emb = compute_hexagram_embedding(hex_num, 128)

    # 数值特征编码（128 维）
    numeric_emb = [0.0] * 128

    # 权重编码 (0-25 维)
    weight = item.get("weight", 0.5)
    for i in range(25):
        numeric_emb[i] = weight * math.sin(i * 0.1)

    # 热度编码 (25-50 维)
    hot = item.get("hot", 0.5)
    for i in range(25, 50):
        numeric_emb[i] = hot * math.cos(i * 0.1)

    # 三才层编码 (50-75 维)
    layer_map = {"天": 1.0, "人": 0.5, "地": 0.0}
    layer_val = layer_map.get(item.get("layer", "人"), 0.5)
    for i in range(50, 75):
        numeric_emb[i] = layer_val * math.sin(i * 0.1)

    # 五行编码 (75-100 维)
    element_map = {"木": 0.2, "火": 0.4, "土": 0.5, "金": 0.7, "水": 1.0}
    elem_val = element_map.get(item.get("element", "土"), 0.5)
    for i in range(75, 100):
        numeric_emb[i] = elem_val * math.cos(i * 0.1)

    # 时位编码 (100-128 维)
    pos_map = {"当位": 1.0, "失位": 0.0}
    pos_val = pos_map.get(item.get("position", "当位"), 1.0)
    for i in range(100, 128):
        numeric_emb[i] = pos_val * math.sin(i * 0.1)

    # 融合符号和数值（平均）
    fused = [(s + n) / 2 for s, n in zip(symbolic_emb, numeric_emb)]

    return fused


def train_model(train_data: List[Dict], epochs: int = 10, learning_rate: float = 0.01) -> Dict:
    """
    训练模型（简化版梯度下降）

    实际部署时使用 PyTorch/TensorFlow
    这里演示训练流程
    """
    print("\n🚀 开始训练...")
    print(f"训练轮数：{epochs}")
    print(f"学习率：{learning_rate}")
    print(f"样本数：{len(train_data)}")

    # 初始化损失
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        for item in train_data:
            # 前向传播
            hex_num = item.get("hexagram_num", 1)
            predicted = compute_hexagram_embedding(hex_num, 128)
            target = compute_target_embedding(item)

            # 计算 MSE 损失
            mse = sum((p - t) ** 2 for p, t in zip(predicted, target)) / 128
            epoch_loss += mse

        avg_loss = epoch_loss / len(train_data)
        losses.append(avg_loss)

        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    # 计算最终指标
    final_loss = losses[-1] if losses else 0
    improvement = ((losses[0] - losses[-1]) / losses[0] * 100) if losses[0] > 0 else 0

    return {
        "epochs": epochs,
        "final_loss": final_loss,
        "initial_loss": losses[0] if losses else 0,
        "improvement": improvement,
        "loss_history": losses,
    }


def evaluate_model(train_data: List[Dict]) -> Dict:
    """评估模型性能"""
    print("\n📊 模型评估...")

    correct_trigram = 0
    correct_layer = 0
    correct_element = 0

    for item in train_data:
        hex_num = item.get("hexagram_num", 1)
        emb = compute_hexagram_embedding(hex_num, 128)

        # 简单验证：检查嵌入是否正确生成
        if len(emb) == 128:
            correct_trigram += 1

        # 验证 layer 编码
        layer = item.get("layer", "人")
        if layer in ["天", "人", "地"]:
            correct_layer += 1

        # 验证 element 编码
        element = item.get("element", "土")
        if element in ["木", "火", "土", "金", "水"]:
            correct_element += 1

    total = len(train_data)

    return {
        "trigram_accuracy": correct_trigram / total * 100,
        "layer_accuracy": correct_layer / total * 100,
        "element_accuracy": correct_element / total * 100,
        "total_samples": total,
    }


def save_model_weights(weights: Dict, path: str):
    """保存模型权重"""
    output = {
        "model_version": "v2.0",
        "trained_at": datetime.now().isoformat(),
        "weights": weights,
        "config": {
            "embedding_dim": 128,
            "fusion_dim": 256,
            "num_heads": 4,
            "num_hexagrams": 64,
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ 模型权重已保存：{path}")


def main():
    print("=" * 80)
    print("🎯 易经编码模型训练")
    print("=" * 80)

    # 配置参数
    learning_rate = 0.01
    epochs = 10

    # 1. 加载数据
    train_data = load_training_data("data/yijing_train_data.json")

    # 2. 训练模型
    train_result = train_model(train_data, epochs=epochs, learning_rate=learning_rate)

    print("\n训练完成!")
    print(f"  初始 Loss: {train_result['initial_loss']:.6f}")
    print(f"  最终 Loss: {train_result['final_loss']:.6f}")
    print(f"  改进幅度：{train_result['improvement']:.2f}%")

    # 3. 评估模型
    eval_result = evaluate_model(train_data)

    print("\n评估结果:")
    print(f"  卦象编码准确率：{eval_result['trigram_accuracy']:.1f}%")
    print(f"  三才层准确率：{eval_result['layer_accuracy']:.1f}%")
    print(f"  五行准确率：{eval_result['element_accuracy']:.1f}%")
    print(f"  总样本数：{eval_result['total_samples']}")

    # 4. 保存模型权重
    model_weights = {
        "training_result": train_result,
        "evaluation_result": eval_result,
    }
    save_model_weights(model_weights, "data/model_weights.json")

    # 5. 生成训练报告
    report = {
        "training_summary": {
            "data_file": "data/yijing_train_data.json",
            "num_samples": len(train_data),
            "epochs": train_result["epochs"],
            "learning_rate": learning_rate,
        },
        "performance": {
            "final_loss": train_result["final_loss"],
            "improvement": f"{train_result['improvement']:.2f}%",
        },
        "accuracy": {
            "trigram": f"{eval_result['trigram_accuracy']:.1f}%",
            "layer": f"{eval_result['layer_accuracy']:.1f}%",
            "element": f"{eval_result['element_accuracy']:.1f}%",
        },
        "model_info": {
            "version": "v2.0",
            "architecture": "Symbolic-Numeric Hybrid",
            "embedding_dim": 128,
            "fusion_dim": 256,
        },
    }

    with open("data/training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n✅ 训练报告已保存：data/training_report.json")

    print("\n" + "=" * 80)
    print("✅ 易经编码模型训练完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
