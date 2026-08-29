#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""report.py - 吃 runner 的成绩表，吐两样东西（D7）：
    ① markdown 报告：标题 + 主表 + 三行自动总结（最狠的条件 / 掉最狠的类 / 一句人话）
    ② 一张掉分热力图：行=条件、列=指标，数值=相对白天晴掉了几 %，越红掉得越狠
      （cmap=RdYlGn_r，与计划书配色一致），存 results/figures/。

两个实现说明：
    - markdown 表格用手写转换器（df_to_markdown），不依赖 tabulate——计划书里
      "装 tabulate"只在用 pandas 自带 to_markdown 时才需要，我们绕开了它；
    - 绘图后端强制 Agg：无显示器的服务器/云电脑上也能正常出图。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                # 必须在 import pyplot 之前设置后端

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASELINE = "clean_day"               # 与 runner.CONDITIONS 的基准条件一致
SMALL_SAMPLE_N = 100                 # 样本少于这个数的条件，结论加"仅供参考"


# ---- markdown 表格 ----------------------------------------------------------

def df_to_markdown(df):
    """DataFrame → markdown 表格（手写五行，省掉 tabulate 依赖）。"""
    header = "| " + " | ".join(df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


# ---- 两句自动总结的原料 -------------------------------------------------------

def worst_condition(df):
    """返回 (条件名, rel_drop, map)：非基准行里 rel_drop 最负（掉分最狠）的那个。"""
    rows = df[df["condition"] != BASELINE]
    row = rows.sort_values("rel_drop").iloc[0]
    return row["condition"], float(row["rel_drop"]), float(row["map"])


def worst_class(perclass_df, baseline=BASELINE, min_base_ap=0.05):
    """返回 (类名, 平均掉分%)：每类相对基准条件的掉分率，跨条件取平均，取最狠的类。

    基线 AP 太低（<0.05）的类要排除——除以接近 0 的数会让掉分率爆炸失真
    （例如 train 类全库才 151 个框，迷你 300 张里可能只有一两个）。
    """
    pivot = perclass_df.pivot_table(index="cls", columns="condition", values="ap")
    if len(pivot.columns) <= 1:
        return "（只有一个条件，无法判定）", 0.0
    base = pivot[baseline]
    drops = [(pivot[c] - base) / base * 100.0
             for c in pivot.columns if c != baseline]
    mean_drop = pd.concat(drops, axis=1).mean(axis=1)
    mean_drop = mean_drop[base >= min_base_ap]
    return mean_drop.idxmin(), float(mean_drop.min())


# ---- 热力图 -------------------------------------------------------------------

def make_heatmap(df, out_png):
    """掉分热力图：数值 = (该条件 − 白天晴) / 白天晴 × 100，正数越大掉得越狠、越红。"""
    # 先选列再取行并强制 float——整行 Series 混着 condition 字符串（object 类型），
    # 直接拿去参与减法会把整个矩阵污染成 object，seaborn 就画不出颜色了
    base = df.loc[df["condition"] == BASELINE, ["map", "map50"]].iloc[0].astype(float)
    drop = df.set_index("condition")[["map", "map50"]]
    drop = (drop - base) / base * 100.0
    drop.columns = ["mAP50-95", "mAP50"]

    fig, ax = plt.subplots(figsize=(5.2, 0.6 * len(drop) + 1.6))
    # 图内文字用英文：Windows 默认字体没有 CJK 字形（云端 Linux 更没有），
    # 中文会画成方块；条件名本来就是英文，中文说明放在报告正文里
    sns.heatmap(drop, annot=True, fmt=".1f", cmap="RdYlGn_r",
                vmin=-100, vmax=100, cbar_kws={"label": "drop vs clean_day (%)"}, ax=ax)
    ax.set_title("mAP drop vs clean_day (positive = worse)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---- 主入口 -------------------------------------------------------------------

def render(df, perclass_df, model, mode, out):
    """生成完整 markdown 报告（含热力图），返回报告路径。

    df / perclass_df：runner.run_checkup 的两个返回值；
    model / mode：只用于标题文案；out：报告 md 的目标路径。
    """
    out = Path(out)
    fig_path = out.parent / "figures" / f"{out.stem}_heatmap.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    make_heatmap(df, fig_path)

    # 三行总结的原料：最狠条件 / 最狠类 / 基线分数；样本太少的条件加免责括号
    cond, rel_drop, worst_map = worst_condition(df)
    base_map = float(df.loc[df["condition"] == BASELINE, "map"].iloc[0])
    counts = df.attrs.get("counts", {})
    n = counts.get(cond)
    caveat = f"（仅 {n} 张，仅供参考）" if n is not None and n < SMALL_SAMPLE_N else ""
    cls, cls_drop = worst_class(perclass_df)
    model_name = Path(model).name

    lines = [
        f"# 迷你体检报告（{model_name} × {mode} 口径，{date.today().isoformat()}）",
        "",
        "## 主表", "",
        df_to_markdown(df), "",
        "## 三行总结", "",
        f"- 最狠的条件：{cond}（mAP50-95 掉 {abs(rel_drop):.1f}%，{base_map:.4f} → {worst_map:.4f}）{caveat}",
        f"- 掉最狠的类：{cls}（各条件相对白天晴平均掉 {abs(cls_drop):.1f}%）",
        f"- 一句人话：{model_name} 在白天晴天能考 mAP {base_map:.3f}，到 {cond} 只剩 "
        f"{worst_map:.3f}；最受伤的是 {cls} 类——恶劣条件主要伤它。",
        "",
        f"![掉分热力图](figures/{fig_path.name})",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
