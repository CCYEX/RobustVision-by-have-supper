#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""calibrate.py - 检测版温度缩放（D14/D16）：教吹牛的模型说实话。

原理（说人话）：
    把握度 c → 对数几率 logit = ln(c/(1-c)) → 除以温度 T → 再 sigmoid 回概率。
    T 是在「校准集」上学出来的唯一参数：选 T 让"报的概率"最贴近"实际对错"（NLL 最小）。
    T>1 = 模型原来太自信（打折）；T<1 = 原来太谦虚（加码）。
    整个变换是单调的 → 检测的排序不变 → mAP 一点不变（保险丝检查就在这）。

输入：predict_cache 产出的答案表（parquet，含 image/cls_id/conf/x1..y2）。
用法（M0 全流程，约 3 分钟）：
    python -m rvkit.calibration.calibrate cache/m0.parquet \
        --calib-list data/splits/calib.txt --labels-dir data/val/labels \
        --conditions-dir data/splits/conditions --out results/calibration/m0
产出（--out 目录下）：report.md（分条件 ECE/MCE/Brier 前后对照）、
    ece.csv、reliability.png（各条件校准前后可靠性图）、temperature.json（T 与元数据）。
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from rvkit.calibration.matching import label_cache

EPS = 1e-4          # 置信度截断：logit 在 0/1 处发散，先把 c 夹进 [EPS, 1-EPS]
N_BINS = 15         # ECE 分桶数（把 0~1 的置信度均分 15 档）


# ---- 核心数学 ---------------------------------------------------------------

def to_logit(c):
    c = np.clip(np.asarray(c, dtype=float), EPS, 1 - EPS)
    return np.log(c / (1 - c))


def apply_temperature(conf, t):
    """校准变换：sigmoid(logit(c) / T)。T>0 恒为单调增 → 排序不变 → mAP 不变。"""
    return 1.0 / (1.0 + np.exp(-to_logit(conf) / t))


def nll_of_t(t, logit, y):
    """给定 T 的平均负对数似然（越小 = 报的概率越诚实）。"""
    p = 1.0 / (1.0 + np.exp(-logit / t))
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(conf, y):
    """在校准集上找最优 T。返回 (T, nll_before, nll_after)。"""
    from scipy.optimize import minimize_scalar
    logit = to_logit(conf)
    y = np.asarray(y, dtype=float)
    res = minimize_scalar(lambda t: nll_of_t(t, logit, y),
                          bounds=(0.05, 10.0), method="bounded")
    t_best = float(res.x)
    return t_best, nll_of_t(1.0, logit, y), float(res.fun)


# ---- 指标 -------------------------------------------------------------------

def ece_brier_mce(conf, y, n_bins=N_BINS):
    """期望校准误差 / Brier / 最大校准误差（一起算，省一遍循环）。

    ECE = Σ (桶内样本占比 × |桶平均置信度 − 桶实际正确率|)，越小越诚实。
    """
    conf = np.asarray(conf, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    ece = mce = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        gap = abs(conf[m].mean() - y[m].mean())
        ece += m.mean() * gap
        mce = max(mce, gap)
    brier = float(np.mean((conf - y) ** 2))
    return float(ece), brier, float(mce)


def reliability_curve(conf, y, n_bins=N_BINS):
    """可靠性图的数据：每桶 (平均置信度, 实际正确率, 桶占比)。空桶跳过。"""
    conf = np.asarray(conf, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out.append((float(conf[m].mean()), float(y[m].mean()), float(m.mean())))
    return out


# ---- 全流程 -----------------------------------------------------------------

def condition_map(conditions_dir):
    """读各条件名单 → {图名: 条件名}。一张图可属多个条件（雨夜）→ 后到的覆盖，无妨。"""
    mapping = {}
    d = Path(conditions_dir)
    for txt in sorted(d.glob("*.txt")):
        for line in txt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                mapping[line.strip()] = txt.stem
    return mapping


def run(weights_tag, cache_path, calib_list, labels_dir, conditions_dir, out_dir):
    """M0/M1/M2 通用的校准全流程。返回 report 的 md 文本。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(cache_path)
    print(f"答案表：{len(df)} 条检测 / {df['image'].nunique()} 张图")

    # 注意 read_text() 返回整个文件「一个字符串」，直接 for 迭代会逐字符变成
    # 21 个十六进制字符的集合（真实翻车案例）——必须先 splitlines() 按行拆。
    calib_names = {l.strip() for l in Path(calib_list).read_text(encoding="utf-8").splitlines() if l.strip()}
    cmap = condition_map(conditions_dir)

    # 1) 判卷（贪心 IoU 匹配 → 0/1 列）
    df = label_cache(df, labels_dir)
    print(f"判卷完成：整体正确率 {df['y'].mean():.3f}")

    # 2) 在校准集上学 T
    cal = df[df["image"].isin(calib_names)]
    t, nll0, nll1 = fit_temperature(cal["conf"].values, cal["y"].values)
    print(f"温度 T = {t:.4f}（>1 = 原来太自信）；NLL {nll0:.4f} → {nll1:.4f}")

    # 3) 保险丝：单调变换 → 排序必须一位不差（抽样 20 张图验证）
    for image, sub in df.groupby("image", sort=False).head(20).groupby("image"):
        a = sub["conf"].to_numpy()
        b = apply_temperature(a, t)
        assert (np.argsort(a) == np.argsort(b)).all(), f"校准改变了排序！{image}"
    print("保险丝检查通过：排序逐位不变 → mAP 不变")

    # 4) 分条件指标（只看 test 侧：不在 calib 名单里的图）
    test = df[~df["image"].isin(calib_names)].copy()
    test["condition"] = test["image"].map(cmap).fillna("未命名")
    conf_after = apply_temperature(test["conf"].values, t)
    test["conf_cal"] = conf_after

    rows = []
    curves = {}
    for cond in ["未命名", *sorted(set(test["condition"]) - {"未命名"})]:
        sub = test[test["condition"] == cond]
        if len(sub) == 0:
            continue
        e0, b0, m0 = ece_brier_mce(sub["conf"], sub["y"])
        e1, b1, m1 = ece_brier_mce(sub["conf_cal"], sub["y"])
        rows.append({"condition": cond, "n_det": len(sub),
                     "ece_before": round(e0, 4), "ece_after": round(e1, 4),
                     "ece_drop_pct": round((e0 - e1) / e0 * 100, 1) if e0 > 0 else 0.0,
                     "mce_before": round(m0, 4), "mce_after": round(m1, 4),
                     "brier_before": round(b0, 4), "brier_after": round(b1, 4)})
        curves[cond] = (reliability_curve(sub["conf"], sub["y"]),
                        reliability_curve(sub["conf_cal"], sub["y"]))
    tbl = pd.DataFrame(rows).sort_values("ece_before", ascending=False)

    # 5) 产出三件：csv / markdown / 可靠性图
    tbl.to_csv(out / "ece.csv", index=False)
    json.dump({"weights_tag": weights_tag, "temperature": t,
               "nll_before": nll0, "nll_after": nll1,
               "n_calib_det": len(cal), "n_test_det": len(test),
               "date": date.today().isoformat()},
              open(out / "temperature.json", "w"), indent=2)
    _reliability_plot(curves, t, out / "reliability.png")

    md = ["# 置信度校准报告（温度缩放）", "",
          f"模型：{weights_tag}｜温度 **T = {t:.4f}**（>1 = 校准前太自信）"
          f"｜校准集 NLL {nll0:.4f} → {nll1:.4f}｜日期：{date.today().isoformat()}", "",
          "保险丝检查：校准为单调变换，抽样验证排序逐位不变 → mAP 不变。", "",
          "| 条件 | 检测数 | ECE 前 | ECE 后 | 相对下降 | MCE 前→后 | Brier 前→后 |",
          "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for r in rows:
        md.append(f"| {r['condition']} | {r['n_det']} | {r['ece_before']:.4f} | "
                  f"{r['ece_after']:.4f} | {r['ece_drop_pct']}% | "
                  f"{r['mce_before']:.3f}→{r['mce_after']:.3f} | "
                  f"{r['brier_before']:.4f}→{r['brier_after']:.4f} |")
    md += ["", "![可靠性图](reliability.png)", "",
           "读法：曲线越贴对角线越诚实；夜/雾等恶劣条件校准前普遍压在对角线上方"
           "（说 0.8 的把握实际对不到 0.8），校准后应贴回对角线。"]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"完成：{out/'report.md'}")
    return "\n".join(md)


def _reliability_plot(curves, t, png_path):
    """各条件校准前(虚线)后(实线)的可靠性图 + 对角线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = list(curves)
    n = len(conds)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.plot([0, 1], [0, 1], "k:", lw=1, label="完美校准")
    for i, cond in enumerate(conds):
        ax = axes[i // ncol][i % ncol]
        before, after = curves[cond]
        ax.plot([c for c, _, _ in before], [a for _, a, _ in before],
                "o--", color="tab:red", ms=4, lw=1, label="校准前")
        ax.plot([c for c, _, _ in after], [a for _, a, _ in after],
                "o-", color="tab:blue", ms=4, lw=1.2, label="校准后")
        ax.set_title(f"{cond}", fontsize=10)
        ax.set_xlabel("模型报的把握度", fontsize=8)
        ax.set_ylabel("实际正确率", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"温度缩放校准前后（T={t:.3f}）：越贴对角线越诚实", fontsize=12)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="检测版温度缩放校准（对答案表离线跑，不开显卡）")
    parser.add_argument("cache", help="答案表 parquet（predict_cache 的产出）")
    parser.add_argument("--calib-list", default="data/splits/calib.txt",
                        help="校准集名单（只在这些图上学 T）")
    parser.add_argument("--labels-dir", default="data/val/labels",
                        help="GT 标签目录（自训模型 = data/val/labels；COCO 缓存 = data/val8/labels）")
    parser.add_argument("--conditions-dir", default="data/splits/conditions",
                        help="条件名单目录（分条件 ECE 用）")
    parser.add_argument("--tag", default=None, help="模型代号（默认取缓存文件名，如 m0）")
    parser.add_argument("--out", default=None, help="输出目录（默认 results/calibration/<tag>）")
    args = parser.parse_args()
    tag = args.tag or Path(args.cache).stem
    out = args.out or f"results/calibration/{tag}"
    run(tag, args.cache, args.calib_list, args.labels_dir, args.conditions_dir, out)


if __name__ == "__main__":
    main()
