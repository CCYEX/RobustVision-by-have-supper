#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_cases.py - 自动挑「模型翻车现场」并画对比图（D10 分析日）。

挑选规则（当前版本）：夜间清单里，GT 有行人、但模型对行人的最高把握 < 0.3
（≈ 夜间漏检行人）的图，按漏检人数从多到少排序取前 N 张。
画法：原图上 GT 框画绿色、模型预测（conf≥0.25）画红色 → 一眼看出"该找到没找到"。

用法：
    D:/Coding/DL_Env/PuTong_P3.11.15/python.exe experiments/plot_cases.py \
        runs/m0/weights/best.pt cache/m0.parquet --condition night --limit 12
产出：results/qualitative/<条件>_<图名>.jpg + cases_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
PERSON_ID = 0          # 类别表 0 号 = person
CONF_DRAW = 0.25       # 画框门槛（低于它的预测也算"模型其实看到了但很虚"）
GREEN, RED = (0, 200, 0), (0, 0, 230)


def gt_boxes(image_name, labels_dir):
    """GT 的 (cls_id, x1, y1, x2, y2) 像素坐标（YOLO 归一化 × 1280×720）。"""
    txt = Path(labels_dir) / (Path(image_name).stem + ".txt")
    out = []
    if txt.exists():
        for line in txt.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            cx, cy, w, h = (float(v) for v in p[1:])
            out.append((int(p[0]),
                        (cx - w / 2) * 1280, (cy - h / 2) * 720,
                        (cx + w / 2) * 1280, (cy + h / 2) * 720))
    return out


def main():
    parser = argparse.ArgumentParser(description="自动挑夜间漏检行人的失败案例")
    parser.add_argument("weights")
    parser.add_argument("cache", help="答案表 parquet")
    parser.add_argument("--condition", default="night", help="条件名单名（data/splits/conditions/<名>.txt）")
    parser.add_argument("--labels-dir", default=str(DATA_ROOT / "val" / "labels"))
    parser.add_argument("--images-dir", default=str(DATA_ROOT / "val" / "images"))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "qualitative"))
    args = parser.parse_args()

    cond_names = [l.strip() for l in (DATA_ROOT / "splits" / "conditions" / f"{args.condition}.txt")
                  .read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.read_parquet(args.cache)
    df = df[df["image"].isin(cond_names)]

    # 候选打分：GT 行人数 − 高把握预测行人数 = 漏检数（越多越"翻车"）
    rows = []
    for image in cond_names:
        gts = gt_boxes(image, args.labels_dir)
        n_gt = sum(1 for g in gts if g[0] == PERSON_ID)
        if n_gt == 0:
            continue
        sub = df[df["image"] == image]
        best_conf = sub[(sub["cls_id"] == PERSON_ID)]["conf"].max() if len(sub) else 0.0
        rows.append({"image": image, "n_gt_person": n_gt,
                     "best_conf": round(float(best_conf), 3),
                     "missed": n_gt if best_conf < 0.3 else 0})
    cand = pd.DataFrame(rows).sort_values("missed", ascending=False).head(args.limit)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 画图：GT 绿框 + 预测红框（conf≥0.25）
    from ultralytics import YOLO
    model = YOLO(args.weights)
    for _, r in cand.iterrows():
        img = cv2.imread(str(Path(args.images_dir) / r["image"]))
        for cls_id, x1, y1, x2, y2 in gt_boxes(r["image"], args.labels_dir):
            if cls_id == PERSON_ID:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 2)
        for _, d in df[(df["image"] == r["image"]) & (df["conf"] >= CONF_DRAW)].iterrows():
            cv2.rectangle(img, (int(d["x1"]), int(d["y1"])), (int(d["x2"]), int(d["y2"])), RED, 2)
            cv2.putText(img, f"{d['conf']:.2f}", (int(d["x1"]), max(12, int(d["y1"]) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1)
        cv2.putText(img, f"GT person: {r['n_gt_person']}  (green=GT, red=pred)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"{args.condition}_{r['image']}"), img)

    cand.to_csv(out_dir / "cases_summary.csv", index=False)
    print(f"挑出 {len(cand)} 张 → {out_dir}")
    print(cand.to_string(index=False))


if __name__ == "__main__":
    main()
