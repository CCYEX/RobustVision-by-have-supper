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


def df_to_markdown(df):
    """DataFrame → markdown 表格（手写五行，省掉 tabulate 依赖；D7 花活再换）。"""
    header = "| " + " | ".join(df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def run_checkup(model_weights, splits_dir, mode="bdd8coco", out_stem=None,
                data_root=DATA_ROOT, n_mini=N_MINI, imgsz=640, device="auto"):
    """对 splits_dir 下每个条件名单算分，写 csv+md 成绩表，返回那张表（DataFrame）。

    这是给 runner 自己和将来的 cli 共用的入口；参数含义见 main() 的 argparse。
    """
    data_root = Path(data_root)
    splits_dir = Path(splits_dir)
    out_stem = Path(out_stem or (RESULTS_DIR / "mini_coco_checkup"))
    device = pick_device(device)
    print(f"模型：{model_weights} | 口径：{mode} | 设备：{device} | "
          f"迷你规模：每条件前 {n_mini} 张")

    # 第 1 步：迷你名单。n_mini>0 → 每个条件取前 n 张，写到 data/splits/mini/。
    # 固定取前 n（名单本身已打乱），重跑完全可复现；每次重写保证与源名单同步。
    lists = {}
    for cond in CONDITIONS:
        src = splits_dir / f"{cond}.txt"
        if not src.exists():                       # 缺名单直接报错，别悄悄跳过
            raise FileNotFoundError(f"找不到条件名单：{src}")
        names = datasets.make_mini(datasets.read_names(src), n_mini or None)
        if not names:
            print(f"[提示] {cond} 名单为空，跳过")
            continue
        lists[cond] = names
        if n_mini:                                 # n_mini=0 → 用原始名单，不落盘迷你版
            mini_path = data_root / "splits" / "mini" / f"{cond}.txt"
            datasets.write_names(mini_path, names)
    if BASELINE not in lists:                      # 没有基准就算不了 rel_drop
        raise RuntimeError(f"缺少基准条件 {BASELINE}，无法计算 rel_drop")

    # 第 2 步：模型只加载一次，六个条件共用（加载要几秒~几十秒，别重复干活）
    adapter = UltralyticsAdapter(model_weights, device=device)

    # 第 3 步：逐条件 生成yaml → 考试 → 记一行
    rows = []
    for cond, names in lists.items():
        if len(names) < 100:                       # 太少则分数抖，提醒读者别过度解读
            print(f"[提示] {cond} 只有 {len(names)} 张，分数参考价值有限（正常现象）")
        yaml_path = datasets.build_eval_yaml(cond, names, mode, data_root)
        print(f"[{cond}] {len(names)} 张，开考 …")
        scores = adapter.val(yaml_path, imgsz=imgsz)
        rows.append({"condition": cond, **{k: scores[k] for k in ("map", "map50", "p", "r")}})
        print(f"[{cond}] mAP50-95={scores['map']:.4f}  mAP50={scores['map50']:.4f}")

    # 第 4 步：rel_drop 列。基准 = clean_day 的 map；负数 = 掉分，越负掉得越狠。
    df = pd.DataFrame(rows)
    base_map = df.loc[df["condition"] == BASELINE, "map"].iloc[0]
    df["rel_drop"] = (df["map"] - base_map) / base_map * 100.0

    # 表格里保留合理位数：分数 4 位、掉分 1 位（再多没有信息量）
    df[["map", "map50", "p", "r"]] = df[["map", "map50", "p", "r"]].round(4)
    df["rel_drop"] = df["rel_drop"].round(1)

    # 第 5 步：落盘。csv 给程序（D9 拼主表），md 给人（贴日志/README）。
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_stem.with_suffix(".csv"), index=False)
    md = [f"# 迷你成绩表（{Path(model_weights).name} × {mode} 口径，{date.today().isoformat()}）",
          "",
          df_to_markdown(df), "",
          f"- rel_drop = (该条件map − {BASELINE}map) / {BASELINE}map × 100%，负数 = 掉分",
          f"- 迷你规模：每条件前 {n_mini} 张；全量口径以云电脑复测为准", ""]
    out_stem.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n===== 成绩表 =====\n{df.to_string(index=False)}")
    print(f"\n完成：{out_stem.with_suffix('.csv')} 与 {out_stem.with_suffix('.md')}")
    return df


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
