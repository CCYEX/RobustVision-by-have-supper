#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""corruptions.py - 把好照片「故意做坏」的 7 种方式登记表（D6）。

背景：模型没见过恶劣条件的图。我们没法凭空变出真夜景，但可以把好图调暗、加噪、
模糊——只是像素变难看了，物体还在原地，所以标签一个字都不用改（这一点由
tests/test_corrupt_bbox.py 逐字节把关）。

7 种坏（名字 → 做法 → 模拟什么）：
    low_light   整体调暗                 夜里、隧道
    motion_blur 斜向拖影卷积             行车抖动
    gauss_noise 撒高斯噪点               相机感光差
    fog         加一层半透明白雾         雾天
    rain        叠统一风向的半透明雨丝   雨天
    downscale   先缩小再放大回去（变糊） 低画质相机、远处小目标
    jpeg        低质量压缩再解码         网络传输压缩

关键设计（为什么手写 numpy/cv2 而不用现成的 albumentations）：
    ① 可复现：每种坏都接收一个 numpy 随机源 rng，generate_corrupt.py 用
       「图名+坏名」哈希出 seed → 无论 8 个线程谁先谁后，同一张图永远得到
       同一种坏法（计划书铁律：seed 42 精神，结果可复现）；
    ② 抗版本漂移：第三方库改个参数名就全线报错，手写算子只依赖 cv2/numpy；
    ③ 三档强度集中登记：所有"魔法数字"都在下面的 PARAMS 表里，想调只改这一处。

图像格式约定：cv2 读进来的 BGR uint8 数组（形状 H×W×3，值 0~255）。
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

# ---- 三档强度参数（轻 s1 / 中 s2 / 重 s3；generate 默认用中档 s2）--------------

# 每个键的取值：三元组 = 三档各自的参数；标量 = 各档共用
PARAMS = {
    "low_light":   {"factor": (0.75, 0.55, 0.38)},          # 亮度乘数，越小越黑
    "motion_blur": {"kernel": (7, 13, 19)},                  # 拖影核边长（奇数）
    "gauss_noise": {"sigma": (8.0, 15.0, 25.0)},             # 噪点标准差（0~255 亮度域）
    "fog":         {"alpha": (0.25, 0.45, 0.65)},            # 白雾浓度（越大越白）
    "rain":        {"n_streaks": (300, 600, 1000)},          # 雨丝根数
    "downscale":   {"factor": (2, 3, 4)},                    # 缩小倍数，越大越糊
    "jpeg":        {"quality": (60, 35, 15)},                # JPEG 质量，越小越出方块
}
# rain 的公共参数（不随档位变）：雨丝长度范围（像素）
RAIN_LEN_MIN, RAIN_LEN_MAX = 15, 40

# 登记表顺序 = 计划书表格顺序 = 生成文件夹的顺序
CORRUPTION_NAMES = ["low_light", "motion_blur", "gauss_noise",
                    "fog", "rain", "downscale", "jpeg"]

# 雾的浓淡每张图允许的小随机浮动（±0.05），让雾不至于像批量贴的白纸
FOG_ALPHA_JITTER = 0.05
# 雨丝的混合比例：雨丝颜色占 0.7（半透明），底下像素占 0.3
RAIN_BLEND = 0.7


# ---- 工具 -------------------------------------------------------------------

def name_seed(*parts):
    """由任意字符串算出一个稳定整数 seed（同输入必同输出，与线程顺序无关）。"""
    digest = hashlib.md5("/".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)          # 取前 8 位十六进制 → 32 位整数


def _resolve(params: dict, severity: int) -> dict:
    """把 PARAMS[name] 里的三元组按档位（1/2/3）取值，标量原样保留。"""
    out = {}
    for key, value in params.items():
        out[key] = value[severity - 1] if isinstance(value, tuple) else value
    return out


# ---- 7 种做坏算子（签名统一：img, params, rng → img）--------------------------

def _low_light(img, p, rng):
    """整体调暗：每个像素乘一个 <1 的系数（夜间/隧道进光少）。"""
    return np.clip(img.astype(np.float32) * p["factor"], 0, 255).astype(np.uint8)


def _motion_blur(img, p, rng):
    """运动拖影：用一条 45° 斜线当卷积核做平均，亮暗沿斜向抹开。"""
    k = p["kernel"]
    kernel = np.zeros((k, k), np.float32)
    for i in range(k):                   # 主对角线全 1 → 沿 45° 方向求平均
        kernel[i, i] = 1.0
    kernel /= k                          # 归一化：整体亮度不变，只糊不暗
    return cv2.filter2D(img, -1, kernel)


def _gauss_noise(img, p, rng):
    """撒高斯噪点：每个像素加一个 N(0, sigma) 的随机数（暗光下感光差）。"""
    noise = rng.normal(0.0, p["sigma"], img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _fog(img, p, rng):
    """加一层白雾：图像与纯白按 alpha 混合，越浓越看不清（雾天散射）。"""
    alpha = np.clip(p["alpha"] + rng.uniform(-FOG_ALPHA_JITTER, FOG_ALPHA_JITTER),
                    0.1, 0.8)            # 每张图浓淡略不同，更像真的雾
    out = img.astype(np.float32) * (1.0 - alpha) + 255.0 * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _rain(img, p, rng):
    """叠雨丝：先在透明图层上画 N 根同风向的细线，再半透明地盖到图上。"""
    h, w = img.shape[:2]
    streaks = np.zeros_like(img)         # 雨丝图层（初始全透明黑）
    for _ in range(p["n_streaks"]):
        x1 = int(rng.integers(0, w))
        y1 = int(rng.integers(0, h))
        length = int(rng.integers(RAIN_LEN_MIN, RAIN_LEN_MAX))
        x2, y2 = x1 + int(length * 0.3), y1 + length   # 统一风向：竖直略向右斜
        cv2.line(streaks, (x1, y1), (x2, y2), (235, 240, 245), 1)  # 淡蓝白
    mask = streaks.any(axis=2)           # 只有画过线的像素参与混合
    out = img.copy()
    out[mask] = ((1 - RAIN_BLEND) * img[mask] + RAIN_BLEND * streaks[mask]).astype(np.uint8)
    return out


def _downscale(img, p, rng):
    """先缩小再放大回去：小尺寸下细节永久丢失，放大回来就是"糊"（低画质相机）。"""
    h, w = img.shape[:2]
    f = p["factor"]
    small = cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA)  # 缩小：面积平均
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)         # 放大：线性插值


def _jpeg(img, p, rng):
    """低质量 JPEG 压缩再解码：8×8 块效应和振铃纹（网络传输压缩的痕迹）。"""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), p["quality"]])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)   # 编码→解码一个来回，压缩痕迹留在像素里


# 名字 → 算子（apply_corruption 按这个名字查表调用）
CORRUPTIONS = {
    "low_light": _low_light, "motion_blur": _motion_blur, "gauss_noise": _gauss_noise,
    "fog": _fog, "rain": _rain, "downscale": _downscale, "jpeg": _jpeg,
}


# ---- 对外主入口 -------------------------------------------------------------

def apply_corruption(img, name, severity=2, rng=None):
    """对一张 BGR uint8 图施加指定坏法，返回同形状的新图（不修改原图）。

    severity：1=轻 / 2=中（默认）/ 3=重；rng：numpy 随机源，传"按图名播种"的
    rng 即可复现（见 name_seed）。算子内部都不改原图 → 线程安全。
    """
    if name not in CORRUPTIONS:
        raise KeyError(f"未知做坏方式：{name}（可选：{CORRUPTION_NAMES}）")
    if rng is None:                      # 不传 rng 就用无种子随机（测试/预览方便）
        rng = np.random.default_rng()
    return CORRUPTIONS[name](img, _resolve(PARAMS[name], severity), rng)
