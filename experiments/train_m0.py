#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_m0.py - 本机 4060 训练入口。

    Ultralytics 本身就是 PyTorch 的官方优化训练器（AMP 混合精度、mosaic 增强、余弦学习率等都已调好）。
    本脚本的价值是把本机 4060（8GB 显存 + Windows 页面文件小）的三个约束一次性打对：

    1. batch=8        8GB 显存跑 YOLO11m@640 的安全值。batch 16 会被 Windows
                      "共享显存"机制悄悄溢出到内存，速度掉 6 倍（实测 5.5 张/秒）。
                      三个模型都必须用同一个 batch，保持可比。
    2. workers=0      数据加载不开子进程（规避 WinError 1455 页面文件过小导致的
                      CUDA DLL 加载失败/卡死）。代价是数据解码在主进程。
    3. cache="ram"    第一轮把图片缓存进内存（1.1 万张约 1GB），之后每轮免磁盘
                      解码——这正是 workers=0 模式下的最大瓶颈，实测可提速 1.5~2 倍。

用法（Git Bash，repo 根目录）：
    D:/Coding/DL_Env/PuTong_P3.11.15/python.exe experiments/train_m0.py
    断线/中断后续训：
    D:/Coding/DL_Env/PuTong_P3.11.15/python.exe experiments/train_m0.py --resume
    训 M1/M2 时换配置：
    D:/Coding/DL_Env/PuTong_P3.11.15/python.exe experiments/train_m0.py --data configs/m1_local.yaml --name m1 --epochs 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __name__ == "__main__":                     # Windows spawn 模式必须有主入口保护
    parser = argparse.ArgumentParser(description="本地GPU训练（M0/M1/M2 通用）")
    parser.add_argument("--data", default="configs/m0_clear_day_local.yaml",
                        help="数据配置 yaml（m0/m1/m2 各一份，见 configs/）")
    parser.add_argument("--name", default="m0", help="本次训练的名字（runs/<name>/）")
    parser.add_argument("--epochs", type=int, default=40, help="最大轮数（早停会提前结束）")
    parser.add_argument("--batch", type=int, default=8, help="批大小（8GB 显存的安全值）")
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续训练")
    args = parser.parse_args()

    from ultralytics import YOLO

    if args.resume:
        # 续训：加载 last.pt，所有原始参数从训练状态里恢复
        model = YOLO(f"runs/{args.name}/weights/last.pt")
        model.train(resume=True)
    else:
        model = YOLO("yolo11m.pt")             # COCO 预训练权重（仓库根目录已有）
        model.train(
            data = args.data,
            epochs = args.epochs,
            patience = 15,                       # 连续 15 轮不进步就自动停
            imgsz = 640,
            batch = args.batch,                  # 8：本机 8GB 显存的安全值（三模型统一）
            amp = True,                          # 混合精度：省显存、加速
            cache = "ram",                       # 图片缓存进内存，第 2 轮起免磁盘解码
            seed = 42,
            workers = 0,                         # 规避 WinError 1455（页面文件太小）
            device = 0,
            # 绝对路径：Ultralytics 会把相对的 project 拼到它自己的 runs_dir 下
            #（M0 实测嵌套成了 runs/detect/runs/m0），绝对路径杜绝这个坑
            project = str(Path("runs").resolve()),
            name = args.name,
            exist_ok=True,                     # 重跑时不改名，直接续用同一目录
        )
