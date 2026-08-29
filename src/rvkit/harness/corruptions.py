#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""corruptions.py - 把好照片「故意做坏」的 7 种方式登记表。

背景：模型没见过恶劣条件的图。我们没法凭空变出真夜景，但可以把好图调暗、加噪、
模糊——只是像素变难看了，物体还在原地，所以标签一个字都不用改（这一点由
tests/test_corrupt_bbox.py 逐字节把关）。

7 种坏 → 实现方式（albumentations 1.4.24 实测验证，fog 除外）：
    low_light   → RandomBrightnessContrast   整体调暗        夜里、隧道
    motion_blur → MotionBlur                 直线运动拖影    行车抖动
    gauss_noise → GaussNoise                 撒高斯噪点      相机感光差
    fog         → 手写大气散射模型            贴合物理的雾    雾天
    rain        → RandomRain                 雨丝+天色压暗   雨天
    downscale   → Downscale                  缩小再放大变糊  低画质相机、远处小目标
    jpeg        → ImageCompression           压缩出方块      网络传输压缩

为什么以 albumentations 为主（用户拍板）：它是图像增广的事实标准，调好的视觉
效果（尤其雨）比手搓自然；版本钉死在 pyproject `albumentations>=1.4.24,<1.5`
（1.4.24 的参数名如 std_range/quality_range 在 2.0 大版本会改名，升级前必须
先过 tests/ 这套测试）。
唯一的例外是 fog：库里的 RandomFog 是"随机位置画径向雾团"，不分远近、看着像
白斑（用户实测否决），所以改用手写的大气散射模型——I = J·t + A·(1−t)，透射率
t 随距离下降，行车画面顶部远、底部近，天际线融进白色、近处路面保持清晰
（Foggy-Cityscapes 数据集造雾就是同款思路）。

可复现（关键机制）：每种坏 = 一个「名字 → apply_fn(图, 参数, seed)」，seed 由
generate_corrupt.py 用「图名+坏名」哈希派生 → 库算子的随机（MotionBlur 的方向、
GaussNoise 的噪声、RandomRain 的雨丝位置）和手写雾的浓淡抖动都只走这个 seed，
8 个线程谁先谁后都不影响结果。tests/test_corruption_is_deterministic 对 7 种坏
逐一验证「同图同种子 → 逐像素相等」。

图像格式：本模块对外收发 BGR uint8（cv2 约定）；喂 albumentations 前转成
它按设计假设的 RGB，取回再转回来——免得哪个彩色相关算子悄悄用错通道序。
"""

from __future__ import annotations

import hashlib

import albumentations as A    # 重依赖（约 1s 导入）；rvkit 的 CLI 路径不 import 本模块
import cv2
import numpy as np

# ---- 三档强度参数（轻 s1 / 中 s2 / 重 s3；generate 默认用中档 s2）--------------

# 每个键的取值：三元组 = 三档各自的参数（按档位取用）；元组里再套元组 = 该算子
# 的"范围型"参数，钉成上下界相同 → 参数本身不随机，随机交给 seed。
PARAMS = {
    # 等比压暗：亮度系数 0.75 / 0.55 / 0.38 ↔ brightness_limit = 系数-1。
    # 注意 brightness_by_max 必须 False（库默认 True 是"按最大值做加法"，会黑成剪影）
    "low_light":   {"brightness_limit": ((-0.25, -0.25), (-0.45, -0.45), (-0.62, -0.62))},
    # 拖影核边长（奇数）；方向 0~360° 随机 → 由 seed 决定，可复现
    "motion_blur": {"blur_limit": (7, 13, 19)},
    # 高斯噪声 σ：8 / 15 / 25（0~255 亮度域）。1.4.24 的 std_range 是 [0,1] 比例，
    # 实测标定 σ = std_range × 255 / √3（平坦图探针验证，15.0 一分不差），
    # 故 std_range = σ×√3/255，写成 (下限, 上限) 对、钉成同值；per_channel=True =
    # 三通道各撒各的（彩色噪点更像真传感器）
    "gauss_noise": {"std_range": tuple((round(s * 3 ** 0.5 / 255.0, 4),) * 2
                                       for s in (8.0, 15.0, 25.0))},
    # 雾（物理散射模型）：t = 透射率，从顶部（远景，雾浓）线性增到底部（近处，清晰）；
    # A = 大气光亮度。s2 = 用户选定的口径（fog_shootout 第 2 行"远处薄雾"，
    # 标定：亮度 156.9 / 清晰度 367，原图 132.6 / 594）；s1 更淡、s3 更浓。
    "fog":         {"t_top": (0.65, 0.55, 0.40),
                    "t_bottom": (0.95, 0.90, 0.82),
                    "A": 240.0},
    # 雨天（s2 = 用户选定的口径：default 雨型 + blur3，雨丝真实又不糊主体）
    "rain":        {"slant_range": ((-10, -5), (-15, -5), (-25, -15)),   # 统一风向略右斜
                    "drop_length": (10, 15, 25),                        # 雨丝长度（像素）
                    "drop_width": (1, 1, 2),                            # 雨丝粗细
                    "blur_value": (3, 3, 5),                            # 全图雨幕模糊（越小主体越保细节）
                    "brightness_coefficient": (0.90, 0.85, 0.70),       # 雨天天色压暗
                    "rain_type": ("default", "default", "heavy")},      # 雨密度档
    # 缩小倍数：1/2、1/3、1/4；缩小用面积平均、放大用线性插值（= "变糊"的来源）
    "downscale":   {"scale_range": ((0.5, 0.5), (1 / 3, 1 / 3), (0.25, 0.25))},
    # JPEG 质量，越小越出方块
    "jpeg":        {"quality_range": ((60, 60), (35, 35), (15, 15))},
}

# 登记表顺序 = 计划书表格顺序 = 生成文件夹的顺序
CORRUPTION_NAMES = ["low_light", "motion_blur", "gauss_noise",
                    "fog", "rain", "downscale", "jpeg"]

# RandomRain 的雨滴颜色：中性灰 → 对 BGR/RGB 通道序不敏感
RAIN_DROP_COLOR = (200, 200, 200)


# ---- 工具 -------------------------------------------------------------------

def name_seed(*parts):
    """由任意字符串算出一个稳定整数 seed（同输入必同输出，与线程顺序无关）。"""
    digest = hashlib.md5("/".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)          # 取前 8 位十六进制 → 32 位整数


def _resolve(params: dict, severity: int) -> dict:
    """把 PARAMS[name] 里的三元组按档位（1/2/3）取值；标量原样保留。"""
    return {key: (value[severity - 1] if isinstance(value, tuple) else value)
            for key, value in params.items()}


# ---- 算子实现：统一签名 apply_fn(rgb, params, seed) → rgb --------------------

def _compose_op(builder):
    """把「albumentations 算子工厂」包成统一的 apply_fn：随机全部走 Compose(seed)。"""

    def _apply(rgb, p, seed):
        return A.Compose([builder(p)], seed=seed)(image=rgb)["image"]

    return _apply


def _apply_scattering_fog(rgb, p, seed):
    """物理散射雾：I = J·t + A·(1−t)，透射率 t 沿画面高度从 t_top（远，雾浓）
    线性增到 t_bottom（近，清晰）。雾浓度每张图带 ±0.04 抖动，随机源 = 本图本坏法
    的 seed（与其他算子的可复现机制完全一致）。"""
    rng = np.random.default_rng(seed)
    t_top = float(np.clip(p["t_top"] + rng.uniform(-0.04, 0.04), 0.05, 1.0))
    t_bottom = float(np.clip(p["t_bottom"] + rng.uniform(-0.04, 0.04), 0.05, 1.0))
    t = np.linspace(t_top, t_bottom, rgb.shape[0], dtype=np.float32)[:, None, None]
    out = rgb.astype(np.float32) * t + p["A"] * (1.0 - t)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---- 六个 albumentations 算子工厂（只负责"按参数造算子"）----------------------

def _build_low_light(p):
    return A.RandomBrightnessContrast(brightness_limit=p["brightness_limit"],
                                      contrast_limit=(0, 0),
                                      brightness_by_max=False, p=1.0)


def _build_motion_blur(p):
    return A.MotionBlur(blur_limit=p["blur_limit"], p=1.0)


def _build_gauss_noise(p):
    return A.GaussNoise(std_range=p["std_range"], per_channel=True, p=1.0)


def _build_rain(p):
    return A.RandomRain(slant_range=list(p["slant_range"]),
                        drop_length=p["drop_length"], drop_width=p["drop_width"],
                        drop_color=RAIN_DROP_COLOR, blur_value=p["blur_value"],
                        brightness_coefficient=p["brightness_coefficient"],
                        rain_type=p["rain_type"], p=1.0)


def _build_downscale(p):
    return A.Downscale(scale_range=p["scale_range"],
                       interpolation_pair={"downscale": cv2.INTER_AREA,
                                           "upscale": cv2.INTER_LINEAR}, p=1.0)


def _build_jpeg(p):
    return A.ImageCompression(quality_range=p["quality_range"],
                              compression_type="jpeg", p=1.0)


# 名字 → 统一签名的 apply_fn（fog 是唯一不走库的）
OPS = {
    "low_light": _compose_op(_build_low_light),
    "motion_blur": _compose_op(_build_motion_blur),
    "gauss_noise": _compose_op(_build_gauss_noise),
    "fog": _apply_scattering_fog,
    "rain": _compose_op(_build_rain),
    "downscale": _compose_op(_build_downscale),
    "jpeg": _compose_op(_build_jpeg),
}


# ---- 对外主入口 -------------------------------------------------------------

def apply_corruption(img, name, severity=2, rng=None):
    """对一张 BGR uint8 图施加指定坏法，返回同形状的新图（不修改原图）。

    severity：1=轻 / 2=中（默认）/ 3=重；rng：numpy 随机源——generate_corrupt
    传入"按图名播种"的 rng，这里再从它派生一个整数种子交给具体算子。
    """
    if name not in OPS:
        raise KeyError(f"未知做坏方式：{name}（可选：{CORRUPTION_NAMES}）")
    if rng is None:                      # 不传 rng 就用无种子随机（临时预览用）
        rng = np.random.default_rng()

    seed = int(rng.integers(0, 2**31 - 1))   # 本图本坏法的专属种子（可复现的根）
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # albumentations 按 RGB 设计；手写算子不受影响
    out = OPS[name](rgb, _resolve(PARAMS[name], severity), seed)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)  # 回到 cv2 的 BGR 世界
