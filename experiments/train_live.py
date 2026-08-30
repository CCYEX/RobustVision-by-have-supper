#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_live.py - 带实时仪表盘的训练引擎（train_m0/m1/m2_live.py 三个入口共用）。

干什么：在子进程里启动训练（复用 train_m0.py 的全部参数与 --resume 逻辑），
实时解析 Ultralytics 的进度输出，在终端刷新一行仪表盘：

    m1 | 轮 8/40 | 步 350/1862 | box 1.349 cls 1.027 dfl 0.971 | 2.8 it/s | 剩 9:14 | mAP50-95 0.281(最佳 0.281)

每完成一轮（results.csv 落盘一次）打一个多行状态块：训练损失、验证 mAP、
历史最佳、早停计数（连续多少轮没刷新最佳 / 上限 15，到 15 会自动停 = 正常结束）。
训练结束自动判定成功/失败，成功时直接给出下一步（考试 + 存答案）的两条命令。

实现要点（两个坑都是实测踩出来的）：
    1. Ultralytics 的进度用 \\r 刷新且一个 epoch 内没有 \\n → 不能用 readline()
       （会整段憋到轮末才返回），必须 os.read 按块读、再按 \\r\\n 切段；
    2. 验证 mAP 不从控制台解析（脆弱），直接读 runs/<name>/results.csv 的最后一行。

用法（三个入口之一）：
    D:/Coding/DL_Env/PuTong_P3.11.15/python.exe experiments/train_m1_live.py
    断点续训：python experiments/train_m1_live.py --resume
注意：Ctrl+C 会连训练一起中断（同一控制台进程组）；中断后加 --resume 接着训。
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PATIENCE = 15  # 早停上限（与 train_m0.py 保持一致，只用于显示计数）

# 训练进度行（取 \r 分段后的最后一段）示例：
# "  1/40  4.26G  1.349  1.028  0.9708  276  640:  18% ━━... 350/1862 2.8it/s 2:13<9:14"
PROGRESS_RE = re.compile(
    r"^\s*(\d+)/(\d+)\s+([\d.]+)G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+\d+:\s*"
    r"(\d+)%\D+?(\d+)/(\d+)\s+([\d.]+)it/s\s+([\d:]+)<([\d:]+)\s*$"
)


def read_results(results_csv):
    """读 results.csv 全部行（dict 列表）；文件不存在/还没写行 → 空列表。"""
    try:
        with open(results_csv, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def fmt_num(v, nd=3):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "?"


def dashboard_line(name, m, last_map, best_map):
    """把一条进度解析结果渲染成单行仪表盘（\\r 原地刷新）。"""
    e, et, box, cls, dfl, pct, si, st, it_s, left = (
        m.group(1), m.group(2), m.group(4), m.group(5), m.group(6),
        m.group(7), m.group(8), m.group(9), m.group(10), m.group(12))
    s = f"{name} | 轮 {e}/{et} | 步 {si}/{st} ({pct}%) | box {box} cls {cls} dfl {dfl}" \
        f" | {it_s} it/s | 剩 {left}"
    if last_map is not None:
        s += f" | mAP50-95 {last_map}(最佳 {best_map})"
    print("\r" + s[:118].ljust(118), end="", flush=True)


def epoch_block(name, rows, epochs):
    """每轮结束时打一个多行状态块：训练损失 + 验证 mAP + 最佳 + 早停计数。"""
    row = rows[-1]
    maps = [float(r["metrics/mAP50-95(B)"]) for r in rows if r.get("metrics/mAP50-95(B)")]
    best = max(maps) if maps else 0.0
    since_best = len(maps) - 1 - maps.index(best) if maps else 0
    print("\n" + "=" * 72)
    print(f"[{time.strftime('%H:%M:%S')}] {name} 第 {int(float(row['epoch'])) + 1}/{epochs} 轮完成"
          f"（早停计数 {since_best}/{PATIENCE}）")
    print(f"  训练损失: box {fmt_num(row.get('train/box_loss'))}  "
          f"cls {fmt_num(row.get('train/cls_loss'))}  dfl {fmt_num(row.get('train/dfl_loss'))}")
    print(f"  验证指标: mAP50-95 {fmt_num(row.get('metrics/mAP50-95(B)'), 4)}  "
          f"mAP50 {fmt_num(row.get('metrics/mAP50(B)'), 4)}  "
          f"P {fmt_num(row.get('metrics/precision(B)'))}  R {fmt_num(row.get('metrics/recall(B)'))}")
    print(f"  历史最佳 mAP50-95: {best:.4f}")
    if since_best >= PATIENCE - 2:
        print(f"  ⚠ 已 {since_best} 轮未刷新最佳，再 {PATIENCE - since_best} 轮不进步将自动早停")
    print("=" * 72)


def final_summary(name, rc, results_csv):
    """训练结束：判定成功/失败，成功则给出下一步命令。"""
    rows = read_results(results_csv)
    best = max((float(r["metrics/mAP50-95(B)"]) for r in rows
                if r.get("metrics/mAP50-95(B)")), default=0.0)
    best_epoch = max((r for r in rows if r.get("metrics/mAP50-95(B)")),
                     key=lambda r: float(r["metrics/mAP50-95(B)"]), default=None)
    print("\n" + "#" * 72)
    weights_ok = (REPO_ROOT / "runs" / name / "weights" / "best.pt").exists()
    if rc == 0 and weights_ok and rows:
        print(f"✅ {name} 训练正常结束")
        e = int(float(best_epoch["epoch"])) + 1 if best_epoch else "?"
        print(f"   最佳 mAP50-95 = {best:.4f}（第 {e} 轮的权重已存为 best.pt）")
        print("   下一步（考试 + 存答案，校准的原料）：")
        print(f"   {sys.executable} -m rvkit.cli checkup --model runs/{name}/weights/best.pt"
              f" --splits data/splits/cloud_eval/ --mode labels10 --out results/{name}_full.md")
        print(f"   {sys.executable} -m rvkit.harness.predict_cache runs/{name}/weights/best.pt"
              f" data/splits/val_all.txt cache/{name}.parquet")
    else:
        print(f"❌ {name} 训练异常退出（returncode={rc}，best.pt 存在={weights_ok}）")
        print(f"   排查：看本文件同目录的运行日志结尾有没有 Traceback / OOM / nan；")
        print(f"   修复后加 --resume 从 runs/{name}/weights/last.pt 续训。")


def run_live(name, data_cfg, epochs, resume=False):
    """启动训练并挂实时仪表盘。三个入口脚本只传不同的 (name, data_cfg, epochs)。"""
    results_csv = REPO_ROOT / "runs" / name / "results.csv"
    cmd = [sys.executable, str(REPO_ROOT / "experiments" / "train_m0.py"),
           "--data", data_cfg, "--name", name, "--epochs", str(epochs)]
    if resume:
        cmd.append("--resume")

    log_path = REPO_ROOT / "runs" / f"trainlog_{name}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8")

    print(f"启动命令：{' '.join(cmd)}")
    print(f"原始输出同时写入：{log_path}")
    print("（Ctrl+C 会连训练一起中断；中断后加 --resume 从断点续训）\n")

    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    last_render, last_csv_mtime, best_map = 0.0, 0.0, None

    def handle(seg):
        """处理一个输出段（\\r 或 \\n 之间的一段）。"""
        nonlocal last_render, last_csv_mtime, best_map
        log_f.write(seg + "\n")
        m = PROGRESS_RE.match(seg)
        if m is None:  # 非进度行：轮末表格 / 保存提示等，透传（过滤掉进度条碎屑）
            s = seg.strip()
            if s and "━" not in s and len(s) < 150:
                print("\r" + " " * 118 + "\r" + "  │ " + s[:110], flush=True)
            return
        # 限频：最多每秒刷新一次仪表盘
        if time.time() - last_render >= 1.0:
            last_render = time.time()
            rows = read_results(results_csv)
            if rows and rows[-1].get("metrics/mAP50-95(B)"):
                cur = fmt_num(rows[-1]["metrics/mAP50-95(B)"], 4)
                best_map = max(best_map or 0, float(rows[-1]["metrics/mAP50-95(B)"]))
                dashboard_line(name, m, cur, f"{best_map:.4f}")
            else:
                dashboard_line(name, m, None, None)
        # 每轮结束的标志：results.csv 落盘了新行
        try:
            mtime = results_csv.stat().st_mtime
        except OSError:
            mtime = 0
        if mtime != last_csv_mtime and mtime > 0:
            last_csv_mtime = mtime
            rows = read_results(results_csv)
            if rows:
                best_map = max(best_map or 0,
                               max(float(r["metrics/mAP50-95(B)"]) for r in rows
                                   if r.get("metrics/mAP50-95(B)")))
                epoch_block(name, rows, epochs)

    buf = ""
    while True:
        chunk = os.read(proc.stdout.fileno(), 4096)  # 有数据就返回，不等整行
        if not chunk:
            break
        buf += chunk.decode("utf-8", errors="replace")
        *parts, buf = re.split(r"[\r\n]", buf)
        for seg in parts:
            if seg:
                handle(seg)
    if buf:
        handle(buf)
    log_f.close()
    proc.wait()
    final_summary(name, proc.returncode, results_csv)
