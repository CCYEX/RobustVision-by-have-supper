#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""matching.py - 校准的判卷函数（D12/D14）：给每条检测打"对/错"标签。

校准要回答"模型说 90% 把握时实际有多少是对的"，所以需要把每条检测变成 0/1：
    1. 同一张图的全部检测按置信度从高到低排队；
    2. 轮到某条检测：在标准答案（GT）里找「同类、IoU≥0.5、还没被认领」里 IoU 最大的那个；
       认领成功 → y=1；找不到 → y=0；
    3. 每个真值框只能被认领一次（两个检测不能同时算答对同一个物体）。

这就是 Pascal VOC / COCO 评测的标准贪心匹配——但在这里不用于算 mAP，
只用于给置信度配"这句话对不对"的真值。

labels_dir 的两种口径（函数本体不改，调用方选目录）：
    data/val/labels    10 类原始编号 → 自己训的 M0/M1/M2 的缓存
    data/val8/labels   8 类 COCO 编号 → COCO 现成模型的缓存

GT 标签是 YOLO 归一化格式（cls cx cy w h，除以 1280×720），这里还原成像素坐标。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMG_W, IMG_H = 1280, 720


def box_iou(one, others):
    """一个框对一组框的 IoU。one/others = (x1, y1, x2, y2) 像素坐标。

    返回长度 len(others) 的数组；面积 ≤0 的框 IoU 记 0（防除零）。
    """
    x1 = np.maximum(one[0], others[:, 0])
    y1 = np.maximum(one[1], others[:, 1])
    x2 = np.minimum(one[2], others[:, 2])
    y2 = np.minimum(one[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_one = max(0.0, (one[2] - one[0]) * (one[3] - one[1]))
    area_others = np.clip((others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1]), 0, None)
    union = area_one + area_others - inter
    return np.where(union > 0, inter / union, 0.0)


def match_image(dets, gts, iou_thr=0.5):
    """对一张图的检测做贪心判卷。

    dets: [(cls_id, conf, x1, y1, x2, y2), ...] 顺序任意（内部会按 conf 排队）
    gts:  [(cls_id, x1, y1, x2, y2), ...]
    返回: [0/1, ...] 与 dets 原顺序一一对应（排队只在函数内部做）。
    """
    order = sorted(range(len(dets)), key=lambda i: -dets[i][1])
    claimed = set()
    y = [0] * len(dets)
    if gts:
        gt_arr = np.array([(g[1], g[2], g[3], g[4]) for g in gts], dtype=float)
        gt_cls = [g[0] for g in gts]
        for i in order:
            d = dets[i]
            cand = [j for j in range(len(gts)) if j not in claimed and gt_cls[j] == d[0]]
            if not cand:
                continue
            ious = box_iou(d[2:], gt_arr[cand])
            best = int(np.argmax(ious))
            if ious[best] >= iou_thr:
                claimed.add(cand[best])
                y[i] = 1
    return y


def load_gt(image_name, labels_dir):
    """读一张图的 GT：YOLO txt（归一化）→ [(cls_id, x1, y1, x2, y2) 像素]。

    没有 txt（图没有标注）→ 空列表：该图全部检测都会被判 0（合理——
    没有真值即没有可匹配的物体；校准集里零标注图很少，见 data card）。
    """
    txt = Path(labels_dir) / (Path(image_name).stem + ".txt")
    gts = []
    if txt.exists():
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
            gts.append((cls_id,
                        (cx - w / 2) * IMG_W, (cy - h / 2) * IMG_H,
                        (cx + w / 2) * IMG_W, (cy + h / 2) * IMG_H))
    return gts


def label_cache(df, labels_dir, iou_thr=0.5):
    """给整张答案表打分。df 需有列：image, cls_id, conf, x1, y1, x2, y2。

    返回 df 的副本并新增列 y（0/1）。按图分组 → 组内贪心 → 拼回原顺序。
    """
    df = df.copy()
    ys = np.zeros(len(df), dtype=int)
    for image, idx in df.groupby("image", sort=False).indices.items():
        gts = load_gt(image, labels_dir)
        sub = df.loc[idx]
        dets = list(zip(sub["cls_id"], sub["conf"], sub["x1"], sub["y1"], sub["x2"], sub["y2"]))
        ys[idx] = match_image(dets, gts, iou_thr=iou_thr)
    df["y"] = ys
    return df
