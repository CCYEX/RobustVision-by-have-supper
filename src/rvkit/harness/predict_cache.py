#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""predict_cache.py - 把模型的全部答案存成一张表（parquet），D9 第 8 步。

为什么值一条租卡命：9/9 做校准要"每条答案的对/错 + 把握度"。有了这张表，
校准那天直接读表，不用再开显卡重跑一遍模型——这是"校准免费"的保证。

每行一个预测框：
    image   图名（如 b1cebfb7-284f5117.jpg）
    cls_id  类别编号（模型自己脑子里的编号）
    cls     类别名（从权重自带的 names 表查，不用我们操心口径）
    conf    把握度 0~1（校准的主角）
    x1 y1 x2 y2  框的像素坐标

用法（D9 卡片原样）：
    python -m rvkit.harness.predict_cache runs/m0/weights/best.pt \
        data/splits/val_all.txt cache/m0.parquet
可选：--images-dir（默认 data/val/images）、--conf 0.05、--imgsz 640、--device auto
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"


def pick_device(requested="auto"):
    """"auto" → 有 N 卡用 "0"，否则 CPU（与 runner 同款逻辑）。"""
    if requested != "auto":
        return requested
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


def cache_predictions(weights, list_path, out_path, images_dir=None,
                      conf=0.05, imgsz=640, device="auto"):
    """对名单里的每张图跑 predict，把全部框写进 parquet。返回行数。"""
    from ultralytics import YOLO                       # 延迟导入，保持 import 轻快
    from rvkit.harness.datasets import read_names

    images_dir = Path(images_dir or DATA_ROOT / "val" / "images")
    names = read_names(list_path)
    paths = [str(images_dir / n) for n in names]       # predict 直接吃路径列表
    print(f"权重：{weights} | 图像：{images_dir} | {len(paths)} 张 | conf≥{conf}")

    model = YOLO(weights)
    rows = []
    # stream=True：逐张产出结果，1 万张也不会把显存/内存撑爆
    for r in model.predict(source=paths, conf=conf, imgsz=imgsz,
                           device=pick_device(device), stream=True, verbose=False):
        image = Path(r.path).name
        for x1, y1, x2, y2, c, cls_id in r.boxes.data.tolist():
            rows.append({"image": image, "cls_id": int(cls_id),
                         "cls": model.names[int(cls_id)], "conf": round(c, 4),
                         "x1": round(x1, 1), "y1": round(y1, 1),
                         "x2": round(x2, 1), "y2": round(y2, 1)})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"完成：{len(df)} 个框 → {out_path}（来自 {len(paths)} 张图）")
    return len(df)


def main():
    parser = argparse.ArgumentParser(description="把模型答案存成 parquet（校准的原料）")
    parser.add_argument("weights", help="模型权重，如 runs/m0/weights/best.pt")
    parser.add_argument("list", help="图名名单 txt（如 calib+test 合并的 val_all.txt）")
    parser.add_argument("out", help="输出 parquet 路径，如 cache/m0.parquet")
    parser.add_argument("--images-dir", default=None,
                        help="图片目录（默认 data/val/images；名单是不带路径的文件名）")
    parser.add_argument("--conf", type=float, default=0.05,
                        help="把握度门槛（默认 0.05：校准要尽量全的低分答案）")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cache_predictions(args.weights, args.list, args.out, args.images_dir,
                      args.conf, args.imgsz, args.device)


if __name__ == "__main__":
    main()
