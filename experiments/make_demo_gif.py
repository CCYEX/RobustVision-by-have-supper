#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_demo_gif.py - 生成 README 演示动图（同一帧左右分屏：M0 vs M2）。

每帧 = 同一张图分别用 M0 / M2 推理（conf≥0.30 画框），左右拼接，
顶部加条件标签；PIL 合成 GIF（每帧约 2.5 秒）。输出 demo/demo.gif。

用法：python experiments/make_demo_gif.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "demo" / "demo.gif"
CONF = 0.30
W_HALF = 640            # 每半边缩放后的宽（1280→640，总宽 1280，README 友好）

# 每个条件取清单里的第一张图（seed 固定的切分 → 演示帧可复现）
FRAMES = [
    ("clean_day 白天晴", "conditions/clean_day.txt"),
    ("night 夜间", "conditions/night.txt"),
    ("rain 雨天", "conditions/rain.txt"),
    ("snow 雪天", "conditions/snow.txt"),
]


def draw(model, img_path, tag):
    """单边：推理 + 画框 + 角标（模型名 + 检测数）。返回 BGR 图。"""
    r = model.predict(str(img_path), conf=CONF, imgsz=640, device=0,
                      verbose=False)[0]
    im = r.plot()                                     # ultralytics 自带画框（BGR ndarray）
    n = len(r.boxes)
    cv2.rectangle(im, (0, 0), (250, 34), (24, 24, 24), -1)
    cv2.putText(im, f"{tag}: {n} dets", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return im


def main():
    from ultralytics import YOLO
    m0, m2 = YOLO(str(REPO / "runs/m0/weights/best.pt")), YOLO(str(REPO / "runs/m2/weights/best.pt"))
    pil_frames = []
    for label, list_rel in FRAMES:
        name = (DATA / "splits" / list_rel).read_text(encoding="utf-8").splitlines()[0].strip()
        img_path = DATA / "val" / "images" / name
        left, right = draw(m0, img_path, "M0"), draw(m2, img_path, "M2")
        h = 360
        both = np.hstack([cv2.resize(left, (W_HALF, h)),
                          cv2.resize(right, (W_HALF, h))])
        pil = Image.fromarray(cv2.cvtColor(both, cv2.COLOR_BGR2RGB))
        banner = Image.new("RGB", (both.shape[1], 30), (16, 16, 16))
        ImageDraw.Draw(banner).text((8, 8), f"{label}   |   left: M0 (clear-day only)   right: M2 (+real night/rain)",
                                    fill=(240, 240, 240))
        pil_frames.append(Image.vstack if False else Image.new("RGB", (both.shape[1], h + 30)))
        combo = pil_frames[-1]
        combo.paste(banner, (0, 0))
        combo.paste(pil, (0, 30))
        print(f"帧完成：{label}（{name}）")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(OUT, save_all=True, append_images=pil_frames[1:],
                       duration=2600, loop=0)
    print(f"完成：{OUT}")


if __name__ == "__main__":
    main()
