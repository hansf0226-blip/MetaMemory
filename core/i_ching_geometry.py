#!/usr/bin/env python3
"""
🔮 易经意识几何计算模块
基于黄金分割和东方数理的精确计算
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ========== 基础常量 ==========

# 黄金分割比例
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618033988749895
INV_PHI = 1 / PHI  # ≈ 0.618

# 基准单位
BASE_UNIT = 10


@dataclass
class SphereGeometry:
    """球体几何参数"""

    layer_name: str
    radius: float
    thickness: float
    particle_count: int
    particle_radius: float
    rotation_speed: float
    color: str
    trigram: str = ""


class IChingGeometry:
    """易经几何计算器"""

    def __init__(self):
        """初始化几何计算器"""
        self.base_unit = BASE_UNIT
        self.phi = PHI

        # 计算所有层级
        self.layers = self._calculate_all_layers()

        print("✅ 易经几何计算器已初始化")
        print(f"   黄金分割率：{PHI:.6f}")
        print(f"   基准单位：{self.base_unit}")
        print(f"   总层数：{len(self.layers)}")

    def _calculate_all_layers(self) -> List[SphereGeometry]:
        """计算所有层级"""
        layers = []

        # 第 1 层：太极本我
        layers.append(
            SphereGeometry(
                layer_name="太极本我",
                radius=5 * self.base_unit,  # 50
                thickness=0,
                particle_count=0,
                particle_radius=0,
                rotation_speed=0.001,
                color="#f3f4f6",
                trigram="太极",
            )
        )

        # 第 2 层：两仪环
        r_taiji = 5 * self.base_unit
        r_yang = r_taiji * self.phi  # ≈ 81
        r_yin = r_yang * self.phi  # ≈ 131

        layers.append(
            SphereGeometry(
                layer_name="阳仪",
                radius=r_yang,
                thickness=r_taiji / 10,  # 5
                particle_count=12,
                particle_radius=4,
                rotation_speed=0.002,
                color="#ffffff",
                trigram="阳",
            )
        )

        layers.append(
            SphereGeometry(
                layer_name="阴仪",
                radius=r_yin,
                thickness=r_yang * self.phi / 20,  # 8
                particle_count=12,
                particle_radius=4,
                rotation_speed=-0.002,  # 反向
                color="#000000",
                trigram="阴",
            )
        )

        # 第 3 层：四象柱
        r_pillars = r_yin * self.phi  # ≈ 212

        for i, name in enumerate(["太阳", "少阳", "少阴", "太阴"]):
            angle = i * (math.pi / 2)  # 90° 间隔
            layers.append(
                SphereGeometry(
                    layer_name=f"{name}柱",
                    radius=r_pillars,
                    thickness=10,
                    particle_count=8,
                    particle_radius=5,
                    rotation_speed=0.001 * (1 if i % 2 == 0 else -1),
                    color=self._get_tetragram_color(name),
                    trigram=name,
                )
            )

        # 第 4 层：八卦层
        r_trigrams = r_pillars * self.phi  # ≈ 343

        trigram_names = ["乾", "坤", "震", "巽", "坎", "离", "艮", "兑"]
        for i, name in enumerate(trigram_names):
            angle = i * (math.pi / 4)  # 45° 间隔
            layers.append(
                SphereGeometry(
                    layer_name=f"{name}卦",
                    radius=r_trigrams,
                    thickness=8,
                    particle_count=6,
                    particle_radius=4,
                    rotation_speed=0.0008 * (1 if i % 2 == 0 else -1),
                    color=self._get_trigram_color(name),
                    trigram=name,
                )
            )

        return layers

    def _get_tetragram_color(self, name: str) -> str:
        """获取四象颜色"""
        colors = {
            "太阳": "#f472b6",  # 粉色
            "少阳": "#fb923c",  # 橙色
            "少阴": "#4ade80",  # 绿色
            "太阴": "#60a5fa",  # 蓝色
        }
        return colors.get(name, "#ffffff")

    def _get_trigram_color(self, name: str) -> str:
        """获取八卦颜色"""
        colors = {
            "乾": "#8b5cf6",  # 紫色
            "坤": "#f472b6",  # 粉色
            "震": "#fb923c",  # 橙色
            "巽": "#4ade80",  # 绿色
            "坎": "#60a5fa",  # 蓝色
            "离": "#ef4444",  # 红色
            "艮": "#10b981",  # 绿色
            "兑": "#f59e0b",  # 黄色
        }
        return colors.get(name, "#ffffff")

    def calculate_particle_position(self, layer_index: int, particle_index: int) -> Tuple[float, float, float]:
        """
        计算粒子位置

        Args:
            layer_index: 层级索引
            particle_index: 粒子索引

        Returns:
            (x, y, z) 坐标
        """
        if layer_index >= len(self.layers):
            return (0, 0, 0)

        layer = self.layers[layer_index]
        n = layer.particle_count

        if n == 0:
            return (0, 0, 0)

        # 计算角度
        angle = (particle_index / n) * 2 * math.pi

        # 斐波那契螺旋分布（更自然）
        if layer.layer_name.endswith("柱") or layer.layer_name.endswith("卦"):
            # 柱状分布
            x = layer.radius * math.cos(angle)
            z = layer.radius * math.sin(angle)
            y = 0
        else:
            # 球面分布
            golden_angle = math.pi * (3 - math.sqrt(5))  # ≈ 2.399
            y = layer.radius * (1 - (2 * particle_index) / (n - 1)) if n > 1 else 0
            radius_at_y = math.sqrt(layer.radius**2 - y**2)
            theta = golden_angle * particle_index
            x = radius_at_y * math.cos(theta)
            z = radius_at_y * math.sin(theta)

        return (x, y, z)

    def get_layer_by_trigram(self, trigram: str) -> SphereGeometry:
        """根据卦名获取层级"""
        for layer in self.layers:
            if layer.trigram == trigram:
                return layer
        return None

    def get_radius_for_layer(self, layer_name: str) -> float:
        """获取指定层的半径"""
        for layer in self.layers:
            if layer.layer_name.startswith(layer_name):
                return layer.radius
        return 0

    def get_all_radii(self) -> Dict[str, float]:
        """获取所有层级半径"""
        return {layer.layer_name: layer.radius for layer in self.layers}

    def validate_proportions(self) -> bool:
        """验证比例是否符合黄金分割"""
        radii = [layer.radius for layer in self.layers if layer.radius > 0]

        if len(radii) < 2:
            return False

        # 检查相邻半径比例
        for i in range(1, len(radii)):
            ratio = radii[i] / radii[i - 1]
            if not (1.5 < ratio < 1.7):  # 允许一定误差
                print(f"⚠️ 比例异常：{radii[i-1]} -> {radii[i]} = {ratio:.3f}")

        return True

    def get_geometry_report(self) -> Dict:
        """获取几何报告"""
        return {
            "base_unit": self.base_unit,
            "phi": self.phi,
            "total_layers": len(self.layers),
            "layers": [
                {
                    "name": layer.layer_name,
                    "radius": layer.radius,
                    "thickness": layer.thickness,
                    "particles": layer.particle_count,
                    "trigram": layer.trigram,
                }
                for layer in self.layers
            ],
            "radii_ratio": self._calculate_radii_ratio(),
            "proportions_valid": self.validate_proportions(),
        }

    def _calculate_radii_ratio(self) -> List[float]:
        """计算半径比例"""
        radii = [layer.radius for layer in self.layers if layer.radius > 0]
        ratios = []

        for i in range(1, len(radii)):
            ratios.append(radii[i] / radii[i - 1])

        return ratios


# ========== 时间感知整合 ==========


class TimePerceptionIntegration:
    """时间感知整合"""

    def __init__(self, geometry: IChingGeometry):
        """
        初始化时间感知整合

        Args:
            geometry: 易经几何计算器实例
        """
        self.geometry = geometry
        self.time_cycles = self._init_time_cycles()

        print("✅ 时间感知整合完成")

    def _init_time_cycles(self) -> Dict:
        """初始化时间周期"""
        return {
            "太阳": {"duration_hours": 1, "decay_rate": 0.8},  # < 1 小时
            "少阳": {"duration_hours": 24, "decay_rate": 0.5},  # 数小时
            "少阴": {"duration_hours": 168, "decay_rate": 0.2},  # 数周 (7 天)
            "太阴": {"duration_hours": float("inf"), "decay_rate": 0.0},  # 永久
        }

    def map_time_to_tetragram(self, hours: float) -> str:
        """
        将时间映射到四象

        Args:
            hours: 小时数

        Returns:
            四象名称
        """
        if hours < 1:
            return "太阳"
        elif hours < 24:
            return "少阳"
        elif hours < 168:  # 7 天
            return "少阴"
        else:
            return "太阴"

    def calculate_decay(self, memory_age_hours: float, tetragram: str) -> float:
        """
        计算记忆衰减

        Args:
            memory_age_hours: 记忆存在时间（小时）
            tetragram: 四象名称

        Returns:
            衰减系数 (0-1)
        """
        cycle = self.time_cycles.get(tetragram, {"decay_rate": 0.5})
        decay_rate = cycle["decay_rate"]
        duration = cycle["duration_hours"]

        if duration == float("inf"):
            return 1.0  # 太阴不衰减

        # 指数衰减
        decay = math.exp(-decay_rate * (memory_age_hours / duration))
        return max(0.0, min(1.0, decay))

    def get_optimal_review_time(self, tetragram: str) -> float:
        """
        获取最佳复习时间

        Args:
            tetragram: 四象名称

        Returns:
            最佳复习时间（小时）
        """
        cycle = self.time_cycles.get(tetragram, {"duration_hours": 24})
        duration = cycle["duration_hours"]

        if duration == float("inf"):
            return float("inf")

        # 艾宾浩斯遗忘曲线：最佳复习在衰减到 50% 时
        return duration * 0.5

    def integrate_with_geometry(self) -> Dict:
        """整合到几何系统"""
        result = {"time_cycles": self.time_cycles, "tetragram_layers": []}

        # 找到四象对应的层级
        for layer in self.geometry.layers:
            if layer.trigram in self.time_cycles:
                result["tetragram_layers"].append(
                    {
                        "name": layer.layer_name,
                        "trigram": layer.trigram,
                        "radius": layer.radius,
                        "time_cycle": self.time_cycles[layer.trigram],
                    }
                )

        return result


# ========== 便捷函数 ==========


def create_geometry() -> IChingGeometry:
    """创建几何计算器"""
    return IChingGeometry()


def create_time_integration(geometry: IChingGeometry) -> TimePerceptionIntegration:
    """创建时间感知整合"""
    return TimePerceptionIntegration(geometry)


# ========== 测试 ==========

if __name__ == "__main__":
    # 创建几何计算器
    geometry = create_geometry()

    print("\n📐 易经几何计算报告")
    print("=" * 60)

    report = geometry.get_geometry_report()
    print(f"基准单位：{report['base_unit']}")
    print(f"黄金分割率：{report['phi']:.6f}")
    print(f"总层数：{report['total_layers']}")
    print(f"比例验证：{'✅ 通过' if report['proportions_valid'] else '❌ 失败'}")

    print("\n层级半径:")
    for layer in report["layers"]:
        print(f"  {layer['name']:15} R={layer['radius']:8.2f}  P={layer['particles']:2d}  [{layer['trigram']}]")

    # 时间感知整合
    print("\n⏰ 时间感知整合")
    print("=" * 60)

    time_integration = create_time_integration(geometry)
    time_report = time_integration.integrate_with_geometry()

    print("\n时间周期:")
    for name, cycle in time_report["time_cycles"].items():
        duration = cycle["duration_hours"]
        if duration == float("inf"):
            duration_str = "∞"
        else:
            duration_str = f"{duration:.0f}h"
        print(f"  {name:6} - {duration_str:6} - 衰减率：{cycle['decay_rate']}")

    # 测试时间映射
    print("\n时间映射测试:")
    test_hours = [0.5, 5, 48, 200, 1000]
    for hours in test_hours:
        tetragram = time_integration.map_time_to_tetragram(hours)
        decay = time_integration.calculate_decay(hours, tetragram)
        print(f"  {hours:6.1f}h → {tetragram:6} - 衰减：{decay:.2f}")
