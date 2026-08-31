#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_ladder.py - 拼三模型阶梯表（项目主结果，README 头版与简历数字的唯一来源）。

读 results/{m0,m1,m2}_full.csv（rvkit checkup 的产出），按条件合并成：
    results/ladder.csv —— 每行一个条件，列 = M0/M1/M2 的 mAP50-95 + 两级增量
    results/ladder.md  —— 同表的 markdown 版 + 头版摘要（偏移条件平均、夜间专项、clean 牺牲）

用法：python experiments/make_ladder.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

# 自然条件（真实偏移）与合成损坏分开展示；fog 自然样本仅 9 张单独标注
NATURAL = ["clean_day", "night", "rain", "snow", "dawn_dusk", "fog"]
SHIFTED_NATURAL = ["night", "rain", "snow", "dawn_dusk"]      # 算"偏移条件平均"时不含 clean 和 fog(9张)


def main():
    frames = {}
    for m in ("m0", "m1", "m2"):
        df = pd.read_csv(RESULTS / f"{m}_full.csv")[["condition", "map"]]
        frames[m] = df.set_index("condition")["map"]

    lad = pd.concat(frames, axis=1).rename_axis("condition").reset_index()
    lad["d_m1_m0"] = (lad["m1"] - lad["m0"]).round(4)
    lad["d_m2_m1"] = (lad["m2"] - lad["m1"]).round(4)

    order = {c: i for i, c in enumerate(NATURAL)}
    rest = sorted(set(lad["condition"]) - set(NATURAL))
    lad["_o"] = lad["condition"].map(lambda c: order.get(c, 99))
    lad = lad.sort_values(["_o", "condition"]).drop(columns="_o").round(4)
    lad.to_csv(RESULTS / "ladder.csv", index=False)

    # ---- 摘要数字 ----
    def g(cond, col):
        return float(lad.loc[lad["condition"] == cond, col].iloc[0])

    nat = lad[lad["condition"].isin(SHIFTED_NATURAL)]
    cor = lad[lad["_is_cor"] == True] if False else lad[lad["condition"].str.endswith("_s2")]
    clean = g("clean_day", "m0"), g("clean_day", "m1"), g("clean_day", "m2")

    lines = ["# 三模型阶梯表（ladder）—— 主结果", "",
             "| 条件 | M0 | M1 | M2 | M1−M0 | M2−M1 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in lad.iterrows():
        name = r["condition"] + ("（仅9张）" if r["condition"] == "fog" else "")
        lines.append(f"| {name} | {r['m0']:.4f} | {r['m1']:.4f} | {r['m2']:.4f} | "
                     f"{r['d_m1_m0']:+.4f} | {r['d_m2_m1']:+.4f} |")

    lines += ["", "## 头版摘要", "",
              f"- **clean_day（晴天白天）**：M0 {clean[0]:.4f} → M1 {clean[1]:.4f} → M2 {clean[2]:.4f}"
              f"（M2 相对 M0 变化 {clean[2]-clean[0]:+.4f}——clean 牺牲明码标价）",
              f"- **自然偏移条件平均**（夜/雨/雪/晨昏）：M0 {nat['m0'].mean():.4f} → "
              f"M1 {nat['m1'].mean():.4f}（{nat['m1'].mean()-nat['m0'].mean():+.4f}）→ "
              f"M2 {nat['m2'].mean():.4f}（{nat['m2'].mean()-nat['m1'].mean():+.4f}）",
              f"- **合成损坏平均**（7 种）：M0 {cor['m0'].mean():.4f} → "
              f"M1 {cor['m1'].mean():.4f}（{cor['m1'].mean()-cor['m0'].mean():+.4f}）→ "
              f"M2 {cor['m2'].mean():.4f}（{cor['m2'].mean()-cor['m1'].mean():+.4f}）",
              f"- **夜间专项（RQ2b 核心）**：M0 {g('night','m0'):.4f} → M1 {g('night','m1'):.4f} → "
              f"M2 {g('night','m2'):.4f}（M2 相对 M0 {g('night','m2')-g('night','m0'):+.4f}）"]
    (RESULTS / "ladder.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
