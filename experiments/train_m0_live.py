#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_m0_live.py - 一键训练 M0 基线 + 实时仪表盘。

M0 = 只用「晴天∩白天」11,454 张训练的基线模型（三模型阶梯的第一级）。
用法：
    D:/Coding/DL_Env/PuTong_P3.11.15/python.exe experiments/train_m0_live.py
    断点续训：python experiments/train_m0_live.py --resume
"""

import argparse

from train_live import run_live

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="训练 M0 基线（实时仪表盘）")
    p.add_argument("--resume", action="store_true", help="从 runs/m0/weights/last.pt 续训")
    a = p.parse_args()
    run_live(name="m0", data_cfg="configs/m0_clear_day_local.yaml", epochs=40, resume=a.resume)
