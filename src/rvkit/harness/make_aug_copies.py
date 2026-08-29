#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_aug_copies.py - 从训练名单抽一部分，每张随机做一种坏（D9 等待期的 CPU 活）。

两种模式：
    corrupt（默认）：抽 ratio 比例的图，每张随机做一种坏 → 训练用增强副本
    copy           ：不做坏，整份名单的图+标签原样拷贝（拼 M2 练习盘用）

用法（D9 卡片原样，训练挂显卡时另开终端跑）：
    python -m rvkit.harness.make_aug_copies --list data/splits/train_clear_all.txt \
        --ratio 0.3 --out train_m1/                       # 3.4K 张坏图 → M1 练习盘
    python -m rvkit.harness.make_aug_copies --list data/splits/m2_real_adverse.txt \
        --ratio 1.0 --mode copy --out train_m2_seed/      # 4000 张真夜景雨天 → M2 种子盘

产出 <out>/images/*.jpg + <out>/labels/*.txt（10 类标签原样复制——做坏不移动物体）。
可复现：抽哪几张、每张做哪种坏，全部由 random.Random(seed) 决定（默认 seed=42），
先在主线程里把"图 → 坏法"的分配计划定死，再交给线程池并行执行。
"""

from __future__ import annotations

import argparse
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from rvkit.harness.corruptions import CORRUPTION_NAMES, apply_corruption, name_seed

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"
TRAIN_IMAGES = DATA_ROOT / "train" / "images"
TRAIN_LABELS = DATA_ROOT / "train" / "labels"      # 10 类口径（训练用）


def plan_work(names, ratio, mode, seed=42):
    """定死工作计划：抽哪些图、每张做什么坏。返回 [(图名, 坏法或 None), ...]。

    随机全部发生在这一步（单线程、seed 固定）→ 后面 8 个线程只是照单干活，
    谁先谁后都不影响结果——这是可复现的关键。
    """
    rng = random.Random(seed)
    k = len(names) if mode == "copy" or ratio >= 1.0 else round(len(names) * ratio)
    picked = rng.sample(names, min(k, len(names)))
    return [(n, None if mode == "copy" else rng.choice(CORRUPTION_NAMES))
            for n in picked]


def process_one(item, severity, out_root, data_root=DATA_ROOT):
    """处理一张图：corrupt 模式做坏后写图，copy 模式原样复制；标签都原样复制。"""
    name, cname = item
    out_root = Path(out_root)
    src_img = Path(data_root) / "train" / "images" / name

    dst_img = out_root / "images" / name
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    if cname is None:                          # copy 模式：原图直拷
        shutil.copyfile(src_img, dst_img)
    else:                                      # corrupt 模式：做坏后写
        img = cv2.imread(str(src_img))
        if img is None:
            raise FileNotFoundError(f"读不到原图：{src_img}")
        # 随机源按 (图名/坏法) 播种——plan_work 已定死这张图做哪种坏，
        # 这里保证同一张图同一坏法永远长一样（可复现）
        out = apply_corruption(img, cname, severity,
                               np.random.default_rng(name_seed(name, cname)))
        cv2.imwrite(str(dst_img), out)

    txt = Path(name).with_suffix(".txt").name
    dst_img.parent.parent.joinpath("labels").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(data_root) / "train" / "labels" / txt,
                    dst_img.parent.parent / "labels" / txt)
    return name


def self_check(items, out_root):
    """数量自查：images 与 labels 都应恰好等于计划张数。"""
    out_root = Path(out_root)
    n_img = len(list((out_root / "images").glob("*")))
    n_lbl = len(list((out_root / "labels").glob("*")))
    print(f"自查：images={n_img} labels={n_lbl} 计划={len(items)}")
    if n_img != len(items) or n_lbl != len(items):
        raise RuntimeError("产物数量与计划不符，请检查上方数字")


def main():
    parser = argparse.ArgumentParser(description="训练图增强副本：抽比例、随机做坏（或纯拷贝）")
    parser.add_argument("--list", required=True, help="训练名单 txt（如图名列表）")
    parser.add_argument("--ratio", type=float, default=0.3, help="抽多少比例（1.0=全部）")
    parser.add_argument("--out", required=True, help="输出根目录（含 images/ labels/）")
    parser.add_argument("--mode", choices=["corrupt", "copy"], default="corrupt",
                        help="corrupt=每张随机做一种坏；copy=原样复制（拼盘用）")
    parser.add_argument("--severity", type=int, choices=[1, 2, 3], default=2,
                        help="坏法强度档（仅 corrupt 模式）")
    parser.add_argument("--workers", type=int, default=8, help="线程数")
    parser.add_argument("--seed", type=int, default=42, help="抽样/选坏法的随机种子")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="数据根目录")
    args = parser.parse_args()

    from rvkit.harness.datasets import read_names

    names = read_names(args.list)
    items = plan_work(names, args.ratio, args.mode, args.seed)
    n_corrupt = sum(1 for _, c in items if c is not None)
    print(f"名单 {len(names)} 张 → 本次处理 {len(items)} 张"
          f"（corrupt {n_corrupt} / copy {len(items) - n_corrupt}），强度 s{args.severity}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda it: process_one(it, args.severity, args.out, args.data_root),
                      items))
    print(f"完成：{args.out}")
    self_check(items, args.out)


if __name__ == "__main__":
    main()
