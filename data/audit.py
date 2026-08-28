#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit.py - 数据审计：产出三张表并更新 docs/data_card.md。

① 天气 × 时段交叉表：train / val 各一张，一眼看清天气与时段分布（如 clear∩daytime 数量）
② 框尺寸分布表：遍历 data/yolo/labels/ 的 txt，还原像素尺寸按 COCO 惯例分小/中/大
③ M2 真实恶劣子集抽样：train 中 night 抽 3000 + rainy 抽 1000（雨夜去重）
   → data/splits/m2_real_adverse.txt

运行：python data/audit.py
（train 的 JSON 约 1.45GB，加载需一两分钟属正常；需先跑过 convert.py 才有 yolo/labels）
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path

# ---- 常量 ------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parent                 # data/
TRAIN_ANN = DATA_ROOT / "train" / "annotations" / "bdd100k_labels_images_train.json"
VAL_ANN = DATA_ROOT / "val" / "annotations" / "bdd100k_labels_images_val.json"
YOLO_DIR = DATA_ROOT / "yolo" / "labels"                   # convert.py 的输出目录
SPLITS_DIR = DATA_ROOT / "splits"
DATA_CARD = DATA_ROOT.parent / "docs" / "data_card.md"

IMG_W, IMG_H = 1280, 720                                   # BDD100K 分辨率

CLASSES = ["person", "rider", "car", "truck", "bus",
           "train", "motor", "bike", "traffic light", "traffic sign"]

# 交叉表的行（7 种天气）与列（4 种时段），顺序固定 → 输出一致
WEATHERS = ["clear", "overcast", "partly cloudy", "rainy", "snowy", "foggy", "undefined"]
TIMES = ["daytime", "night", "dawn/dusk", "undefined"]

# COCO 惯例按像素面积分档：小 < 32² ≤ 中 < 96² ≤ 大
SMALL_A, MEDIUM_A = 32 * 32, 96 * 96

# M2 真实恶劣子集规模
M2_NIGHT_N, M2_RAIN_N = 3000, 1000

# ③ 抽样统一使用该 rng（seed 固定 = 结果可复现）
rng = random.Random(42)


# ---- 通用工具 ---------------------------------------------------------------

def load_records(path):
    """读取标注 JSON，返回记录列表（每张图一个 dict）。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_names(path, names):
    """把图名列表写成 txt，每行一个文件名（不带路径）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{n}\n" for n in names), encoding="utf-8")


# ---- ① 天气 × 时段交叉表 -----------------------------------------------------

def cross_counts(records):
    """统计天气×时段交叉，返回 (Counter[(weather, timeofday)], 总图数)。"""
    c = Counter(
        (rec["attributes"].get("weather", "undefined"),
         rec["attributes"].get("timeofday", "undefined"))
        for rec in records
    )
    return c, len(records)


def cross_to_md(counter, total, title):
    """把交叉计数转成 markdown 表格字符串。"""
    row_total = {t: 0 for t in TIMES}
    lines = [f"### {title}（共 {total} 张）", "",
             "| weather \\ timeofday | " + " | ".join(TIMES) + " | 合计 |",
             "| --- |" + " --- |" * (len(TIMES) + 1)]
    for w in WEATHERS:
        cells = []
        for t in TIMES:
            v = counter[(w, t)]
            row_total[t] += v
            cells.append(str(v))
        row_sum = sum(counter[(w, t)] for t in TIMES)
        lines.append(f"| {w} | {' | '.join(cells)} | {row_sum} |")
    lines.append(f"| **合计** | " + " | ".join(str(row_total[t]) for t in TIMES)
                 + f" | **{total}** |")
    return "\n".join(lines)


# ---- ② 框尺寸分布 -----------------------------------------------------------

def scan_yolo_sizes(split):
    """遍历 yolo/labels/<split>/ 的 txt，统计每类小/中/大框数。

    返回 (stats, total)：stats = {cls_id: {"small": n, "medium": n, "large": n}}。
    每行 YOLO 格式为 "cls cx cy w h"，w/h 乘回 1280×720 得像素宽高，面积分档。
    """
    stats = {i: {"small": 0, "medium": 0, "large": 0} for i in range(len(CLASSES))}
    split_dir = YOLO_DIR / split
    if not split_dir.is_dir():
        print(f"[提示] 未找到 {split_dir}，跳过（请先运行 convert.py）")
        return stats, 0
    for txt in split_dir.glob("*.txt"):
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            area = float(parts[3]) * IMG_W * float(parts[4]) * IMG_H
            if area < SMALL_A:
                key = "small"
            elif area < MEDIUM_A:
                key = "medium"
            else:
                key = "large"
            stats[int(parts[0])][key] += 1
    total = sum(sum(v.values()) for v in stats.values())
    return stats, total


def size_to_md(stats, total, title):
    """把每类小/中/大框统计转成 markdown 表格。"""
    lines = [f"### {title}（共 {total} 个框）", "",
             "| 类别 | 小目标 | 中目标 | 大目标 | 小占比 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for i, name in enumerate(CLASSES):
        s, m, l = stats[i]["small"], stats[i]["medium"], stats[i]["large"]
        tot = s + m + l
        pct = s / tot * 100 if tot else 0.0
        lines.append(f"| {name} | {s} | {m} | {l} | {pct:.1f}% |")
    return "\n".join(lines)


# ---- ③ M2 真实恶劣子集抽样 --------------------------------------------------

def sample_m2(records):
    """train 里抽 night 3000 + rainy 1000（雨天先剔除已抽中的雨夜图）。

    返回 (图名列表, night数, rain数)。只基于 JSON 里有记录的图（records 即全部记录）。
    """
    night, rainy = [], []
    for rec in records:
        attr = rec.get("attributes", {})
        if attr.get("timeofday") == "night":
            night.append(rec["name"])
        if attr.get("weather") == "rainy":
            rainy.append(rec["name"])

    n_night = min(M2_NIGHT_N, len(night))
    night_sample = rng.sample(night, n_night)
    night_set = set(night_sample)

    rainy_pool = [n for n in rainy if n not in night_set]   # 雨夜图不重复入组
    n_rain = min(M2_RAIN_N, len(rainy_pool))
    rain_sample = rng.sample(rainy_pool, n_rain)

    return sorted(set(night_sample) | set(rain_sample)), n_night, n_rain


# ---- 更新 data_card.md ------------------------------------------------------

def build_data_card_section(cross_train, total_train, cross_val, total_val,
                            size_train, size_train_total, size_val, size_val_total,
                            m2_names, n_night, n_rain):
    """拼出"条件分布"章节（markdown 文本）。"""
    sec = [f"## 条件分布（audit 实测 · {date.today().isoformat()}）", "",
           "### ① 天气 × 时段交叉表", "",
           cross_to_md(cross_train, total_train, "train"), "",
           cross_to_md(cross_val, total_val, "val"), "",
           "### ② 框尺寸分布（COCO 惯例：面积 <32² 小 / 32²~96² 中 / ≥96² 大）", "",
           size_to_md(size_train, size_train_total, "train"), "",
           size_to_md(size_val, size_val_total, "val"), "",
           "### ③ M2 真实恶劣子集", "",
           f"- night：{n_night} 张（timeofday==night，seed 42 抽样）",
           f"- rainy：{n_rain} 张（weather==rainy，已剔除与 night 重复的雨夜图）",
           f"- 共 {len(m2_names)} 张，清单见 `data/splits/m2_real_adverse.txt`", ""]
    return "\n".join(sec)


def update_data_card(new_section):
    """把模板中"## 条件分布"及其后的内容替换为实测章节。"""
    text = DATA_CARD.read_text(encoding="utf-8")
    # 移除顶部的占位注释（含 "audit" 的 HTML 注释，如 <!-- D3 ... audit.py 跑完后填写 -->）
    text = re.sub(r"<!--[^>]*audit[^>]*-->\n?", "", text)
    marker = "## 条件分布"
    idx = text.find(marker)
    if idx == -1:
        raise RuntimeError(f"{DATA_CARD} 中找不到 '{marker}'，请检查模板")
    DATA_CARD.write_text(text[:idx] + new_section, encoding="utf-8")


# ---- 入口 -------------------------------------------------------------------

def main():
    print(f"加载 {TRAIN_ANN.name} …（约 1.45GB，需一两分钟）")
    train_records = load_records(TRAIN_ANN)
    train_cross, train_total = cross_counts(train_records)
    print(f"train：{train_total} 张")

    print(f"加载 {VAL_ANN.name} …（约 208MB）")
    val_records = load_records(VAL_ANN)
    val_cross, val_total = cross_counts(val_records)
    print(f"val：{val_total} 张")

    print("扫描 yolo/labels（train/val）…")
    size_train, size_train_total = scan_yolo_sizes("train")
    size_val, size_val_total = scan_yolo_sizes("val")
    print(f"train 框数 {size_train_total}，val 框数 {size_val_total}")

    print("M2 真实恶劣子集抽样…")
    m2_names, n_night, n_rain = sample_m2(train_records)
    write_names(SPLITS_DIR / "m2_real_adverse.txt", m2_names)
    print(f"m2_real_adverse.txt：night {n_night} + rainy {n_rain} = {len(m2_names)} 张")

    section = build_data_card_section(
        train_cross, train_total, val_cross, val_total,
        size_train, size_train_total, size_val, size_val_total,
        m2_names, n_night, n_rain)
    update_data_card(section)
    print(f"data_card.md 已更新：{DATA_CARD}")


if __name__ == "__main__":
    main()