#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_m1_live.py - 一键训练 M1 鲁棒版 + 实时仪表盘。

M1 = 与 M0 完全同配置，唯一区别是练习盘（data/train_m1/，14,890 张）
混入 3,436 张人工做坏的图——回答「合成增强能救回多少分」（RQ2）。
用法：
    D:/Coding/DL_Env/PuTong_P3.11.15/python.exe experiments/train_m1_live.py
    断点续训：python experiments/train_m1_live.py --resume
"""

import argparse

from train_live import run_live

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="训练 M1 鲁棒版（实时仪表盘）")
    p.add_argument("--resume", action="store_true", help="从 runs/m1/weights/last.pt 续训")
    a = p.parse_args()
    run_live(name="m1", data_cfg="configs/m1_local.yaml", epochs=40, resume=a.resume)
