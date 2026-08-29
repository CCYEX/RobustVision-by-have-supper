#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generate_corrupt.py - 把 corrupt_base 名单的图批量做坏（D6）。

对名单里每张图 × 7 种坏 × 中档强度（_s2），产出四个平级目录（两套口径各一对）：

    data/corrupt/<坏法>_s2/images/*.jpg   做坏后的图（真像素，只能生成不能链接）
    data/corrupt/<坏法>_s2/labels/*.txt   10 类标签 —— 从 data/val/labels/ 原样复制
    data/corrupt8/<坏法>_s2/images/*.jpg  同一张坏图的硬链接（内容相同 → 不占磁盘）
    data/corrupt8/<坏法>_s2/labels/*.txt  8 类 COCO 对齐标签 —— 从 data/val8/labels/ 复制

为什么搞两套：Ultralytics 找标签的规则是「图片路径里 images → labels」，
10 类口径查 corrupt/.../labels，8 类口径查 corrupt8/.../labels —— 和 val/val8
完全同一套规矩。标签为什么要复制而不是引用：做坏不移动物体 → 框的答案不变，
把源 txt 一个字节不动地拷过来就是正确答案（tests/test_corrupt_bbox.py 验证）。

可复现：每张图的随机源 = md5(图名/坏名) 播种 → 8 线程谁先跑完都改变不了结果。

运行（ThreadPoolExecutor 开 8 个线程，纯 CPU 活，2100 张几分钟内完事）：
    python -m rvkit.harness.generate_corrupt           # 名单前 300 张 × 7 种（迷你版）
    python -m rvkit.harness.generate_corrupt --n 0     # 全量 1,235 张 × 7 种
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from rvkit.harness.corruptions import CORRUPTION_NAMES, apply_corruption, name_seed

# ---- 常量（runner.py 同款目录推导：本文件位于 src/rvkit/harness/ 下）-----------

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"
BASE_LIST = DATA_ROOT / "splits" / "corrupt_base.txt"
N_MINI = 300            # 迷你版规模（计划书：先取前 300 张）
WORKERS = 8             # 线程数（计划书指定）


# ---- 单张图的处理 ------------------------------------------------------------

def corrupt_one(img_name, cname, severity=2, data_root=DATA_ROOT):
    """处理一张图 × 一种坏：写坏图 + 两套标签各归各位。返回坏图路径。

    任意一张图失败都会抛异常 → 主流程停下修数据，绝不悄悄少生成。
    """
    data_root = Path(data_root)
    base = f"{cname}_s{severity}"          # 文件夹名，如 low_light_s2（英文避免编码坑）

    # 1) 读原图 → 做坏 → 写坏图。随机源按 (图名/坏名) 播种：可复现且与线程顺序无关
    src_img = data_root / "val" / "images" / img_name
    img = cv2.imread(str(src_img))
    if img is None:
        raise FileNotFoundError(f"读不到原图：{src_img}")
    rng = np.random.default_rng(name_seed(img_name, cname))
    out = apply_corruption(img, cname, severity, rng)

    dst_img = data_root / "corrupt" / base / "images" / img_name
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_img), out)

    # 2) 10 类标签：原样复制（"复制"本身就是字节级不变的保证，测试再复核一遍）
    txt_name = Path(img_name).with_suffix(".txt").name
    labels10 = data_root / "val" / "labels" / txt_name
    dst_labels10 = dst_img.parent.parent / "labels" / txt_name
    dst_labels10.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(labels10, dst_labels10)

    # 3) 8 类口径：同一张坏图用硬链接共享（改不了坏图本身 → 两口径永远同像素），
    #    标签则换成 val8 的 8 类那份
    mirror_img = data_root / "corrupt8" / base / "images" / img_name
    mirror_img.parent.mkdir(parents=True, exist_ok=True)
    if not mirror_img.exists():            # 可重复运行：已存在就跳过
        try:
            os.link(dst_img, mirror_img)   # 硬链接：同一份数据第二个名字
        except OSError:                    # 文件系统不支持 → 退回复制
            shutil.copy2(dst_img, mirror_img)
    labels8 = data_root / "val8" / "labels" / txt_name
    if not labels8.exists():               # val 每张图都该有 8 类 txt，缺 = 上游有漏
        raise FileNotFoundError(f"缺 8 类标签，请先跑 convert.py --mode bdd8coco：{labels8}")
    labels8_dir = mirror_img.parent.parent / "labels"
    labels8_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(labels8, labels8_dir / txt_name)
    return dst_img


# ---- 批量 + 自查 -------------------------------------------------------------

def generate(names, cname, severity, workers):
    """对一批图施加同一种坏法（线程池并行）。corrupt_one 失败会直接抛异常终止。"""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda n: corrupt_one(n, cname, severity), names))


def self_check(names, severity, data_root, todo=None):
    """数量自查：本次生成的每种坏 × 4 类目录，每个都应恰好 len(names) 个文件。"""
    data_root = Path(data_root)
    todo = todo or CORRUPTION_NAMES
    print("\n===== 产物自查（文件数）=====")
    ok = True
    for cname in todo:
        base = f"{cname}_s{severity}"
        counts = {}
        for root, sub in (("corrupt", ("images", "labels")),
                          ("corrupt8", ("images", "labels"))):
            for d in sub:
                counts[f"{root}/{d}"] = len(list((data_root / root / base / d).glob("*")))
        line_ok = all(v == len(names) for v in counts.values())
        ok &= line_ok
        print(f"  {base:<16} images={counts['corrupt/images']} labels={counts['corrupt/labels']} "
              f"8c-images={counts['corrupt8/images']} 8c-labels={counts['corrupt8/labels']}"
              f"{'  ✓' if line_ok else '  ✗ 应为 ' + str(len(names))}")
    if not ok:
        raise RuntimeError("产物数量与名单不符，请查看上方 ✗ 行")
    print(f"✓ 数量自查通过：{len(todo)} 种坏 × {len(names)} 张，四类目录数量全部一致")


def main():
    parser = argparse.ArgumentParser(description="批量生成做坏图（10 类 + 8 类双口径）")
    parser.add_argument("--n", type=int, default=N_MINI,
                        help="名单前 N 张做坏；0 = 全量（默认 300 的迷你版）")
    parser.add_argument("--severity", type=int, choices=[1, 2, 3], default=2,
                        help="强度档位：1 轻 / 2 中（默认）/ 3 重")
    parser.add_argument("--only", default=None,
                        help="只生成指定坏法（逗号分隔，如 --only rain,jpeg）；调参后重生成某一种时用")
    parser.add_argument("--workers", type=int, default=WORKERS, help="线程数")
    parser.add_argument("--list", default=str(BASE_LIST), help="底图名单 txt")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="数据根目录")
    args = parser.parse_args()

    from rvkit.harness.datasets import make_mini, read_names   # 复用名单工具
    from rvkit.harness.corruptions import OPS                  # 算子登记表（校验 --only 用）

    todo = CORRUPTION_NAMES if not args.only else [s.strip() for s in args.only.split(",")]
    unknown = [c for c in todo if c not in OPS]
    if unknown:
        parser.error(f"未知坏法：{unknown}（可选：{CORRUPTION_NAMES}）")

    names = read_names(args.list)
    names = make_mini(names, args.n) if args.n else names      # n=0 → 全量
    print(f"名单 {Path(args.list).name}：本次做坏 {len(names)} 张 × "
          f"{len(todo)} 种坏，强度 s{args.severity}，{args.workers} 线程")

    for cname in todo:                     # 按坏法分组跑：进度按文件夹报，一眼能对上
        generate(names, cname, args.severity, args.workers)
        print(f"[{cname}_s{args.severity}] {len(names)} 张完成")

    # 把本次生成对应的条件名单落到 data/splits/mini/（checkup 扫描该目录时
    # 自动把坏图条件纳入体检）；名单与实际生成的图永远同步
    from rvkit.harness.datasets import write_names
    mini_dir = Path(args.data_root) / "splits" / "mini"
    for cname in todo:
        write_names(mini_dir / f"{cname}_s{args.severity}.txt", names)
    print(f"条件清单已更新：{mini_dir}/<坏法>_s{args.severity}.txt × {len(todo)}")

    self_check(names, args.severity, args.data_root, todo)


if __name__ == "__main__":
    main()
