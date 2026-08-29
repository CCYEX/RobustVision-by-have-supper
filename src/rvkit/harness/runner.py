#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""runner.py - 「循环机器」：对每个条件名单自动 生成yaml → 考试 → 记一行分（D5）。

流程（计划书 D5）：
    对每个条件（白天晴/夜/雨/雪/晨昏/雾，各取前 300 张的迷你版）：
        datasets.build_eval_yaml() 生成说明书 → adapter.val() 考试 → 表格一行
    全部跑完后加一列「相对白天晴掉了百分之几」：
        rel_drop = (该行map − 白天晴map) / 白天晴map × 100
    存成 <out>.csv 和 <out>.md 两个文件（数字完全一致，一个给程序读、一个给人看）。

运行（本机，模型用 COCO 现成的 yolo11s.pt + 8 类口径）：
    python -m rvkit.harness.runner --model yolo11s.pt
默认设备自动选：有 N 卡用显卡（快几十倍），没有退回 CPU。
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from rvkit.harness import datasets
from rvkit.harness.adapter import UltralyticsAdapter
from rvkit.harness.corruptions import CORRUPTION_NAMES
from rvkit.harness.report import df_to_markdown

# ---- 常量 -------------------------------------------------------------------

# 条件名单顺序 = 成绩表行序；clean_day 排第一（它是 rel_drop 的基准行）
CONDITIONS = ["clean_day", "night", "rain", "snow", "dawn_dusk", "fog"]
BASELINE = "clean_day"          # 「比白天晴掉了几成」的基准条件
N_MINI = 300                    # 迷你版每个条件取前 300 张

# runner.py 位于 src/rvkit/harness/ 下：parents[3] = 仓库根目录 → data/、results/
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

# --mode 的别名：D7 的 cli 用 labels8 这个叫法，这里提前兼容
MODE_ALIASES = {"labels": "bdd10", "labels10": "bdd10", "labels8": "bdd8coco"}


# ---- 打分 -------------------------------------------------------------------

def pick_device(requested="auto"):
    """把 --device 参数翻成 Ultralytics 认的值："auto" → 有显卡用 "0"，否则 "cpu"。"""
    if requested != "auto":
        return requested
    import torch                                  # 只在需要判断时才加载 torch
    return "0" if torch.cuda.is_available() else "cpu"


def run_checkup(model_weights, splits_dir, mode="bdd8coco", out_stem=None,
                data_root=DATA_ROOT, n_mini=N_MINI, imgsz=640, device="auto"):
    """对 splits_dir 下每个条件名单算分，写 csv+md 成绩表。

    返回 (主表 df, 每类分数长表 df)：主表给报告/日志，长表（condition, cls_id,
    cls, ap）给 report.py 找"掉最狠的类"。
    """
    mode = MODE_ALIASES.get(mode, mode)           # labels8 → bdd8coco 等别名归一
    if mode not in datasets.MODE_CLASS_NAMES:
        raise ValueError(f"未知口径：{mode}（可选：{list(datasets.MODE_CLASS_NAMES)}）")
    data_root = Path(data_root)
    splits_dir = Path(splits_dir)
    out_stem = Path(out_stem or (RESULTS_DIR / "mini_coco_checkup"))
    device = pick_device(device)
    print(f"模型：{model_weights} | 口径：{mode} | 设备：{device} | "
          f"迷你规模：{'完整名单' if not n_mini else f'每条件前 {n_mini} 张'}")

    # 第 1 步：发现名单。扫描 --splits 目录下全部 txt——自然条件（clean_day 等）
    # 和坏图条件（low_light_s2 等）混放在一起，来多少算多少。
    # 行序固定保证报告稳定：基准 clean_day 第一 → 其余自然条件按固定顺序 →
    # 坏图按 CORRUPTION_NAMES × 档位顺序 → 其它（如有）按文件名。
    found = {p.stem: p for p in sorted(splits_dir.glob("*.txt"))}
    if BASELINE not in found:
        raise FileNotFoundError(f"{splits_dir} 下找不到基准名单 {BASELINE}.txt")
    corrupt_conds = [f"{c}_s{s}" for c in CORRUPTION_NAMES for s in (1, 2, 3)]
    order = ([BASELINE]
             + [c for c in CONDITIONS if c in found and c != BASELINE]
             + [k for k in corrupt_conds if k in found]
             + sorted(k for k in found
                      if k not in {BASELINE, *CONDITIONS, *corrupt_conds}))
    lists = {}
    for cond in order:
        names = datasets.make_mini(datasets.read_names(found[cond]), n_mini or None)
        if not names:
            print(f"[提示] {cond} 名单为空，跳过")
            continue
        lists[cond] = names
        if n_mini:                                 # n_mini=0 → 用原始名单，不落盘迷你版
            datasets.write_names(data_root / "splits" / "mini" / f"{cond}.txt", names)
    if BASELINE not in lists:                      # 没有基准就算不了 rel_drop
        raise RuntimeError(f"基准条件 {BASELINE} 名单为空，无法计算 rel_drop")

    # 第 2 步：模型只加载一次，六个条件共用（加载要几秒~几十秒，别重复干活）
    adapter = UltralyticsAdapter(model_weights, device=device)

    # 第 3 步：逐条件 生成yaml → 考试 → 记一行（主表一行 + 每类分数各一行）
    class_names = datasets.MODE_CLASS_NAMES[mode]     # id → 类名（每类分数查名字用）
    rows, perclass_rows, counts = [], [], {}
    for cond, names in lists.items():
        if len(names) < 100:                       # 太少则分数抖，提醒读者别过度解读
            print(f"[提示] {cond} 只有 {len(names)} 张，分数参考价值有限（正常现象）")
        yaml_path = datasets.build_eval_yaml(cond, names, mode, data_root)
        print(f"[{cond}] {len(names)} 张，开考 …")
        scores = adapter.val(yaml_path, imgsz=imgsz)
        rows.append({"condition": cond, **{k: scores[k] for k in ("map", "map50", "p", "r")}})
        counts[cond] = len(names)
        for cls_id, ap in zip(scores["per_class_index"], scores["per_class"]):
            perclass_rows.append({"condition": cond, "cls_id": cls_id,
                                  "cls": class_names[cls_id], "ap": round(ap, 4)})
        print(f"[{cond}] mAP50-95={scores['map']:.4f}  mAP50={scores['map50']:.4f}")

    # 第 4 步：rel_drop 列。基准 = clean_day 的 map；负数 = 掉分，越负掉得越狠。
    df = pd.DataFrame(rows)
    base_map = df.loc[df["condition"] == BASELINE, "map"].iloc[0]
    df["rel_drop"] = (df["map"] - base_map) / base_map * 100.0

    # 表格里保留合理位数：分数 4 位、掉分 1 位（再多没有信息量）
    df[["map", "map50", "p", "r"]] = df[["map", "map50", "p", "r"]].round(4)
    df["rel_drop"] = df["rel_drop"].round(1)
    perclass_df = pd.DataFrame(perclass_rows)
    df.attrs["counts"] = counts                    # 各条件样本数挂在元数据上，报告要用

    # 第 5 步：落盘。csv 给程序（D9 拼主表），md 给人（贴日志/README）；
    # 每类分数单独一份（report.py 的"掉最狠的类"就吃它）。
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_stem.with_suffix(".csv"), index=False)
    perclass_df.to_csv(out_stem.with_name(out_stem.stem + "_perclass.csv"), index=False)
    md = [f"# 迷你成绩表（{Path(model_weights).name} × {mode} 口径，{date.today().isoformat()}）",
          "",
          df_to_markdown(df), "",
          f"- rel_drop = (该条件map − {BASELINE}map) / {BASELINE}map × 100%，负数 = 掉分"
          + ("；完整名单口径" if not n_mini else f"；迷你规模：每条件前 {n_mini} 张"),
          "- 全量口径以云电脑复测为准", ""]
    out_stem.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n===== 成绩表 =====\n{df.to_string(index=False)}")
    print(f"\n完成：{out_stem.with_suffix('.csv')} / {out_stem.with_name(out_stem.stem + '_perclass.csv')} "
          f"/ {out_stem.with_suffix('.md')}")
    return df, perclass_df


def main():
    parser = argparse.ArgumentParser(description="对每个条件名单自动算分（迷你版）")
    parser.add_argument("--model", required=True, help="模型权重，如 yolo11s.pt")
    parser.add_argument("--splits", default=str(DATA_ROOT / "splits" / "conditions"),
                        help="条件名单所在目录（默认 data/splits/conditions）")
    parser.add_argument("--mode", default="bdd8coco",
                        help="bdd10=10 类口径（自己的模型）；bdd8coco=8 类 COCO 对齐（现成模型）")
    parser.add_argument("--out", default=None,
                        help="输出文件前缀（默认 results/mini_coco_checkup，生成 .csv/.md）")
    parser.add_argument("--mini", type=int, default=N_MINI,
                        help="每条件取前 N 张；0 = 用完整名单")
    parser.add_argument("--device", default="auto", help='"cpu" / "0" / auto（默认）')
    args = parser.parse_args()

    mode = MODE_ALIASES.get(args.mode, args.mode)   # labels8 → bdd8coco 等别名归一
    if mode not in datasets.MODE_CLASS_NAMES:
        parser.error(f"未知口径：{args.mode}")

    run_checkup(args.model, args.splits, mode=mode, out_stem=args.out,
                n_mini=args.mini, device=args.device)


if __name__ == "__main__":
    main()
