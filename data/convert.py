#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""convert.py - 把 BDD100K 官方检测标注（Scalabel 总表 JSON）转换为 YOLO 格式标签。

输入（train / val 两边格式完全一致，共用同一条解析路径）：
    data/train/annotations/bdd100k_labels_images_train.json   (~1.45 GB，70,000 张图)
    data/val/annotations/bdd100k_labels_images_val.json       (~208 MB，10,000 张图)

输出（每个有标注记录的图对应一个 .txt）：
    data/yolo/labels/train/<图名>.txt
    data/yolo/labels/val/<图名>.txt
    每行格式：<类别id> <cx> <cy> <w> <h>   （坐标归一化到 [0,1]，保留 6 位小数）

过滤规则（两条缺一不可）：
    1. category 在 10 类白名单 CLASSES 内（自动剔除 lane / drivable area 等多任务类别）
    2. 标注带 box2d 字段（只做 poly2d 的 lane / drivable area 自然被剔除）

其它约定：
    - 框先裁剪到 1280x720 画面内，裁剪后宽高 <= 0 的框丢弃
    - 无任何有效框的图仍输出空 .txt（保证“有 txt 即有效图”的约定）
    - JSON 中不存在的图（train 有 137 张）自然没有 txt，抽样时只认“有 txt 的图”即可避开
    - 程序内所有随机操作统一使用 random.Random(42)，保证可复现
    - train 的 JSON 约 1.45GB，json.load 需一两分钟属正常现象
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

# ---- 常量定义 ---------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parent          # 本文件位于 data/ 下
IMG_W, IMG_H = 1280, 720                              # BDD100K 图像分辨率

# 10 个检测类白名单（顺序即 YOLO 类别 id，0~9）
CLASSES = [
    "person", "rider", "car", "truck", "bus",
    "train", "motor", "bike", "traffic light", "traffic sign",
]
CLS_ID = {name: i for i, name in enumerate(CLASSES)}

# 待转换的标注总表（train / val 各一份，结构一致）
ANNOTATIONS = {
    "train": DATA_ROOT / "train" / "annotations" / "bdd100k_labels_images_train.json",
    "val":   DATA_ROOT / "val"   / "annotations" / "bdd100k_labels_images_val.json",
}
OUT_DIR = DATA_ROOT / "yolo" / "labels"

# 所有随机操作统一使用该 rng（seed 固定 = 结果可复现；当前脚本暂无随机分支，保留给后续抽样）
rng = random.Random(42)


# ---- 核心转换逻辑 -----------------------------------------------------------

def clip_to_image(x1, y1, x2, y2):
    """把框裁剪到 1280x720 画面内（容忍越界 / 负坐标输入）。"""
    return (max(0.0, x1), max(0.0, y1), min(IMG_W, x2), min(IMG_H, y2))


def to_yolo_line(category, box):
    """把一条【已通过白名单 + box2d 过滤】的标注转成 YOLO 行。

    先裁剪再归一化：cx/cy 为中心点，w/h 为宽高，均除以分辨率并保留 6 位小数。
    裁剪后宽高 <= 0（含完全在画面外）返回 None。
    """
    x1, y1, x2, y2 = clip_to_image(box["x1"], box["y1"], box["x2"], box["y2"])
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None

    cx = (x1 + x2) / 2 / IMG_W
    cy = (y1 + y2) / 2 / IMG_H
    return f"{CLS_ID[category]} {cx:.6f} {cy:.6f} {w / IMG_W:.6f} {h / IMG_H:.6f}"


def process_image(record, stats):
    """处理一张图的标注，返回 (图名, [YOLO 行])，并更新保留/丢弃统计。

    stats 形如 {"dropped": Counter}，dropped 的键为类别名。
    """
    name = record["name"]
    lines = []
    for label in record.get("labels", []):
        cat = label.get("category")
        box = label.get("box2d")

        if cat not in CLS_ID:          # 非白名单（lane / drivable area 等）
            stats["dropped"][cat] += 1
            continue
        if box is None:                # 白名单但缺 box2d（多任务对象常见）
            stats["dropped"][cat] += 1
            continue

        line = to_yolo_line(cat, box)
        if line is None:               # 裁剪后无面积
            stats["dropped"][cat] += 1
            continue
        lines.append(line)
    return name, lines


def convert_split(split, in_path, stats):
    """转换一个切分（train / val）：读取总表 -> 逐图转换 -> 写 txt。"""
    print(f"[{split}] 加载 {in_path.name} …（大文件需一两分钟）")
    with in_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"[{split}] 共 {len(records)} 张图，开始转换 …")

    out_split = OUT_DIR / split
    out_split.mkdir(parents=True, exist_ok=True)

    for i, record in enumerate(records, 1):
        name, lines = process_image(record, stats)
        stats["images"] += 1
        for line in lines:                          # 保留框按类别 id 计数
            stats["kept"][int(line.split()[0])] += 1

        # 有框写行，无框写空文件（保持“有 txt 即有效图”约定）
        text = "\n".join(lines) + ("\n" if lines else "")
        (out_split / Path(name).with_suffix(".txt")).write_text(text, encoding="utf-8")

        if i % 10000 == 0:                          # 每 1 万张报一次进度
            print(f"  [{split}] {i}/{len(records)} …")

    print(f"[{split}] 完成：{len(records)} 张图 -> {out_split}")


def print_stats(stats):
    """打印每个类别的保留 / 丢弃框数。"""
    print("\n===== 每类框统计（保留 / 丢弃）=====")
    extra = sorted(set(stats["dropped"]) - set(CLASSES))   # 白名单外的类别（如 lane）
    for cat in [*CLASSES, *extra]:
        kept = stats["kept"][CLS_ID[cat]] if cat in CLS_ID else 0
        print(f"  {cat:<14} 保留 {kept:>7}  | 丢弃 {stats['dropped'][cat]:>7}")
    print(f"  共处理 {stats['images']} 张图")


def main():
    stats = {
        "images": 0,          # 已处理图像数（= JSON 中有记录、产生了 txt 的图）
        "kept": Counter(),    # cls_id -> 保留框数
        "dropped": Counter(), # 类别名 -> 丢弃框数
    }
    for split, in_path in ANNOTATIONS.items():
        convert_split(split, in_path, stats)
    print_stats(stats)


if __name__ == "__main__":
    main()