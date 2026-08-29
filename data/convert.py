#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""convert.py - 把 BDD100K 官方检测标注（Scalabel 总表 JSON）转换为 YOLO 格式标签。

两种口径（--mode 选择，写到不同目录、互不覆盖）：
    bdd10     10 类原始口径 —— 我们自己训练的 M0/M1/M2 用这套。
              输出 data/<split>/labels/            （如 data/val/labels/）
    bdd8coco  8 类 COCO 对齐口径 —— 给 COCO 现成模型（yolo11s.pt）演示/评测用。
              处理：rider 并入 person；traffic sign 丢弃；
              关键点：类别编号改成 COCO 官方编号（person=0、car=2、traffic light=9…）。
              Ultralytics 对分是"编号对编号"认类的，编号不对齐会把 car 当 truck 打分。
              输出 data/<split>8/labels/           （如 data/val8/labels/）

为什么 8 类版的目录是 val8/ 而不是 val/labels8/：
    Ultralytics 找标签的规则是死的——把图片路径里最后一段 "images" 替换成 "labels"。
    所以每种口径必须各自有一对平级目录：<split>/images ↔ <split>/labels。
    8 类版的图片目录 val8/images 由 harness/datasets.py 按需用"硬链接"填（不占额外磁盘）。

输入（train / val 两边格式完全一致，共用同一条解析路径）：
    data/train/annotations/bdd100k_labels_images_train.json   (~1.45 GB，70,000 张图)
    data/val/annotations/bdd100k_labels_images_val.json       (~208 MB，10,000 张图)

输出（每个有标注记录的图对应一个 .txt）：
    每行格式：<类别id> <cx> <cy> <w> <h>   （坐标归一化到 [0,1]，保留 6 位小数）

过滤规则（两条缺一不可）：
    1. category 在当前口径的类别映射表内（lane / drivable area 永远剔除；
       bdd8coco 口径下 traffic sign 额外剔除）
    2. 标注带 box2d 字段（只做 poly2d 的 lane / drivable area 自然被剔除）

其它约定：
    - 框先裁剪到 1280x720 画面内，裁剪后宽高 <= 0 的框丢弃
    - 无任何有效框的图仍输出空 .txt（保证“有 txt 即有效图”的约定）
    - JSON 中不存在的图（train 有 137 张）自然没有 txt，抽样时只认“有 txt 的图”即可避开
    - 程序内所有随机操作统一使用 random.Random(42)，保证可复现
    - train 的 JSON 约 1.45GB，json.load 需一两分钟属正常现象

运行：
    python data/convert.py                  # bdd10 口径（自己的模型用，先跑这个）
    python data/convert.py --mode bdd8coco  # 8 类 COCO 对齐口径（现成模型演示用）
"""

from __future__ import annotations

import argparse                     # 标准库命令行参数解析：--mode bdd8coco 就靠它
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

# bdd8coco 口径：BDD 类别名 -> COCO 官方类别 id。
# COCO 编号速查：0 person / 1 bicycle(=bike) / 2 car / 3 motorcycle(=motor) /
#               5 bus / 6 train / 7 truck / 9 traffic light
# rider 在 COCO 里没有对应类，并入 person（都是"人"）；traffic sign 不在表内 = 丢弃。
BDD_TO_COCO_ID = {
    "person": 0, "rider": 0,
    "bike": 1, "car": 2, "motor": 3, "bus": 5,
    "train": 6, "truck": 7, "traffic light": 9,
}

# 两种口径的登记表：cls_map = 类别名 -> 输出 id；dir_suffix = 输出目录名的尾巴。
#   bdd10    -> data/val/labels    （无尾巴）
#   bdd8coco -> data/val8/labels   （尾巴 "8"）
MODES = {
    "bdd10":    {"cls_map": CLS_ID,         "dir_suffix": ""},
    "bdd8coco": {"cls_map": BDD_TO_COCO_ID, "dir_suffix": "8"},
}
# bdd8coco 的 id -> 展示名（打印统计用；rider 已并入 person）
COCO_ID_NAMES = {0: "person(+rider)", 1: "bike", 2: "car", 3: "motor",
                 5: "bus", 6: "train", 7: "truck", 9: "traffic light"}

# 待转换的标注总表（train / val 各一份，结构一致）
ANNOTATIONS = {
    "train": DATA_ROOT / "train" / "annotations" / "bdd100k_labels_images_train.json",
    "val":   DATA_ROOT / "val"   / "annotations" / "bdd100k_labels_images_val.json",
}
# 标签根目录：每种口径写到 <OUT_DIR>/<split><dir_suffix>/labels/，与对应 images/ 平级
# （Ultralytics 约定：把图片路径里的 images 换成 labels 找同名 txt）
OUT_DIR = DATA_ROOT

# 所有随机操作统一使用该 rng（seed 固定 = 结果可复现；当前脚本暂无随机分支，保留给后续抽样）
rng = random.Random(42)


# ---- 核心转换逻辑 -----------------------------------------------------------

def clip_to_image(x1, y1, x2, y2):
    """把框裁剪到 1280x720 画面内（容忍越界 / 负坐标输入）。"""
    return (max(0.0, x1), max(0.0, y1), min(IMG_W, x2), min(IMG_H, y2))


def to_yolo_line(cls_id, box):
    """把一条【已通过类别映射 + box2d 过滤】的标注转成 YOLO 行。

    cls_id 是当前口径下算好的类别编号；box 是 {"x1","y1","x2","y2"} 像素坐标。
    先裁剪再归一化：cx/cy 为中心点，w/h 为宽高，均除以分辨率并保留 6 位小数。
    裁剪后宽高 <= 0（含完全在画面外）返回 None。
    """
    x1, y1, x2, y2 = clip_to_image(box["x1"], box["y1"], box["x2"], box["y2"])
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None

    cx = (x1 + x2) / 2 / IMG_W
    cy = (y1 + y2) / 2 / IMG_H
    return f"{cls_id} {cx:.6f} {cy:.6f} {w / IMG_W:.6f} {h / IMG_H:.6f}"


def process_image(record, stats, cls_map):
    """处理一张图的标注，返回 (图名, [YOLO 行])，并更新保留/丢弃统计。

    cls_map 是当前口径的 {类别名: 输出 id} 映射表（bdd10 传 CLS_ID，bdd8coco 传
    BDD_TO_COCO_ID）。stats 形如 {"dropped": Counter}，dropped 的键为原始类别名。
    """
    name = record["name"]
    lines = []
    for label in record.get("labels", []):
        cat = label.get("category")
        box = label.get("box2d")

        if cat not in cls_map:         # 当前口径不收的类（lane / drivable / traffic sign(仅8类)）
            stats["dropped"][cat] += 1
            continue
        if box is None:                # 在映射表内但缺 box2d（多任务对象常见）
            stats["dropped"][cat] += 1
            continue

        line = to_yolo_line(cls_map[cat], box)   # id 由映射表给出（两种口径各自正确）
        if line is None:               # 裁剪后无面积
            stats["dropped"][cat] += 1
            continue
        lines.append(line)
    return name, lines


def convert_split(split, in_path, stats, out_dir, cls_map):
    """转换一个切分（train / val）：读取总表 -> 逐图转换 -> 写 txt。

    out_dir 和 cls_map 都显式传入、不读全局状态：测试可以指到临时目录、
    指定口径，完全不碰真实数据。
    """
    print(f"[{split}] 加载 {in_path.name} …（大文件需一两分钟）")
    with in_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"[{split}] 共 {len(records)} 张图，开始转换 …")

    out_split = Path(out_dir)
    out_split.mkdir(parents=True, exist_ok=True)

    for i, record in enumerate(records, 1):
        name, lines = process_image(record, stats, cls_map)
        stats["images"] += 1
        for line in lines:                          # 保留框按类别 id 计数
            stats["kept"][int(line.split()[0])] += 1

        # 有框写行，无框写空文件（保持“有 txt 即有效图”约定）
        text = "\n".join(lines) + ("\n" if lines else "")
        (out_split / Path(name).with_suffix(".txt")).write_text(text, encoding="utf-8")

        if i % 10000 == 0:                          # 每 1 万张报一次进度
            print(f"  [{split}] {i}/{len(records)} …")

    print(f"[{split}] 完成：{len(records)} 张图 -> {out_split}")


def print_stats(stats, mode):
    """打印当前口径下每个类别的保留 / 丢弃框数。"""
    cls_map = MODES[mode]["cls_map"]
    id_names = CLS_ID if mode == "bdd10" else COCO_ID_NAMES   # id -> 展示名

    print(f"\n===== 每类框统计（{mode} 口径，保留 / 丢弃）=====")
    extra = sorted(set(stats["dropped"]) - set(cls_map))   # 映射表外的类别（如 lane）
    for cls_id in sorted(id_names):                        # 按 id 从小到大打印，顺序稳定
        kept = stats["kept"][cls_id]
        # 该 id 对应的原始类别若丢弃过（如 rider 被并入、或缺 box2d），丢弃数并到展示行
        dropped = sum(n for cat, n in stats["dropped"].items() if cls_map.get(cat) == cls_id)
        print(f"  {id_names[cls_id]:<16} 保留 {kept:>7}  | 丢弃 {dropped:>7}")
    for cat in extra:                                      # 白名单外的类单独列在最后
        print(f"  {cat:<16} 保留 {0:>7}  | 丢弃 {stats['dropped'][cat]:>7}")
    print(f"  共处理 {stats['images']} 张图")


def main():
    # argparse：解析命令行参数。--mode 二选一，默认 bdd10（自己的模型用的口径）。
    parser = argparse.ArgumentParser(description="BDD100K 标注 -> YOLO 标签（双口径）")
    parser.add_argument("--mode", choices=[*MODES], default="bdd10",
                        help="bdd10=10 类原始口径；bdd8coco=8 类 COCO 对齐口径")
    _mode = parser.parse_args().mode
    print(f"口径：{_mode}（标签将写入 data/<split>{MODES[_mode]['dir_suffix']}/labels/）")

    stats = {
        "images": 0,          # 已处理图像数（= JSON 中有记录、产生了 txt 的图）
        "kept": Counter(),    # cls_id -> 保留框数
        "dropped": Counter(), # 原始类别名 -> 丢弃框数
    }
    suffix = MODES[_mode]["dir_suffix"]
    for split, in_path in ANNOTATIONS.items():
        out_dir = OUT_DIR / f"{split}{suffix}" / "labels"
        convert_split(split, in_path, stats, out_dir, MODES[_mode]["cls_map"])
    print_stats(stats, _mode)


if __name__ == "__main__":
    main()
